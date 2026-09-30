"""
Privacy, retention, tenant-isolation and federated aggregation hooks for Aegis.

Capabilities provided here (pluggable + safe defaults):
- register_retention_policy: declare a named retention policy (table + timestamp column + action)
- enforce_retention_policies: run periodic retention enforcement (delete or anonymize)
- anonymize_user: safely remove or redact PII for a given user id
- tenant_isolation_check: enforce that requester belongs to same tenant as resource
- federated_aggregation_protect: basic checks and optional DP noise addition for federated aggregation outputs
- utilities to plug into Celery periodic task or a k8s CronJob

Notes:
- This file intentionally keeps DB operations in transactions and does best-effort logging via api.audit.record_audit.
- For production, run enforce_retention_policies from a scheduled runner (Celery beat or k8s CronJob).
"""

from datetime import datetime, timedelta
import logging
import math
import os
import random
from typing import Dict, Any, List, Optional, Callable

from sqlalchemy import text, select, update, delete, and_
from sqlalchemy.exc import SQLAlchemyError

from api.db import SessionLocal
from api.models import User, AuditLog, DataRetentionPolicy, Model, ModelVersion
from api import audit

logger = logging.getLogger("aegis.privacy")

# in-memory registry for policies (also persisted in DataRetentionPolicy table)
_RETENTION_REGISTRY: Dict[str, DataRetentionPolicy] = {}


def register_retention_policy(
    name: str,
    table: str,
    timestamp_column: str = "created_at",
    retention_days: int = 90,
    action: str = "delete",  # "delete" | "anonymize"
    tenant_column: Optional[str] = None,
    filter_sql: Optional[str] = None,
) -> DataRetentionPolicy:
    """
    Register (and persist) a retention policy.
    - table: database table name to operate on (e.g., "safety_events", "jobs")
    - timestamp_column: the column used to determine age
    - retention_days: objects older than now - retention_days are subject to action
    - action: 'delete' or 'anonymize'
    - tenant_column: optional column name for tenant-based scoping
    - filter_sql: optional SQL fragment (WHERE clause) to narrow selection (careful with injection)
    Returns the DataRetentionPolicy ORM instance (persisted).
    """
    session = SessionLocal()
    try:
        # upsert by name
        existing = session.query(DataRetentionPolicy).filter_by(name=name).one_or_none()
        if existing:
            existing.table_name = table
            existing.timestamp_column = timestamp_column
            existing.retention_days = retention_days
            existing.action = action
            existing.tenant_column = tenant_column
            existing.filter_sql = filter_sql
            existing.updated_at = datetime.utcnow()
            session.commit()
            policy = existing
        else:
            policy = DataRetentionPolicy(
                name=name,
                table_name=table,
                timestamp_column=timestamp_column,
                retention_days=retention_days,
                action=action,
                tenant_column=tenant_column,
                filter_sql=filter_sql,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(policy)
            session.commit()
            session.refresh(policy)
        _RETENTION_REGISTRY[name] = policy
        audit.record_audit("retention.register", actor="system", target_type="policy", target_id=policy.id,
                           details={"name": name, "table": table, "action": action, "retention_days": retention_days})
        return policy
    finally:
        session.close()


def _older_than_cutoff(retention_days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=retention_days)


def enforce_retention_policies(dry_run: bool = True, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enforce all registered retention policies.
    - If dry_run=True don't modify DB, just return summary of what would be done.
    - If tenant_id provided, only affect rows that match tenant_column == tenant_id for policies that define tenant_column.
    Returns list of action summaries per policy.
    """
    session = SessionLocal()
    summaries = []
    try:
        policies = session.query(DataRetentionPolicy).all()
        for p in policies:
            cutoff = _older_than_cutoff(p.retention_days)
            where_clauses = [text(f"{p.timestamp_column} < :cutoff")]
            params = {"cutoff": cutoff}
            if p.filter_sql:
                # developer-supplied SQL fragment; pass through but recommend careful use
                where_clauses.append(text(f"({p.filter_sql})"))
            if tenant_id and p.tenant_column:
                where_clauses.append(text(f"{p.tenant_column} = :tenant_id"))
                params["tenant_id"] = tenant_id

            where_sql = " AND ".join([str(c) for c in where_clauses])
            # Count rows affected
            count_sql = f"SELECT count(*) as cnt FROM {p.table_name} WHERE {where_sql}"
            res = session.execute(text(count_sql), params).fetchone()
            to_delete = int(res.cnt) if res else 0

            summary = {"policy": p.name, "table": p.table_name, "action": p.action, "cutoff": cutoff.isoformat(), "rows": to_delete}
            logger.info("Retention policy '%s' would affect %d rows in %s (action=%s)", p.name, to_delete, p.table_name, p.action)
            if dry_run:
                summaries.append(summary)
                continue

            # Apply action
            if p.action == "delete":
                del_sql = f"DELETE FROM {p.table_name} WHERE {where_sql}"
                session.execute(text(del_sql), params)
                session.commit()
                audit.record_audit("retention.delete", actor="system", target_type=p.table_name, target_id=None,
                                   details={"policy": p.name, "rows_deleted": to_delete})
                summary["applied"] = True
            elif p.action == "anonymize":
                # For anonymize we attempt to null PII-ish columns where present: email, username, input_snapshot, etc.
                # This is a best-effort safe redact. For complex schemas, add table-specific anonymizers.
                # We construct an update that sets common column names to redacted values if they exist.
                updates = []
                # best-effort list
                anonymize_cols = ["username", "email", "input_snapshot", "text", "parameters"]
                set_sql_parts = []
                for col in anonymize_cols:
                    # set col = '<redacted>' if column exists; using SQL that's tolerant requires DB-specific introspection.
                    # We'll attempt a simple update and ignore errors (some DBs will fail if column missing).
                    set_sql_parts.append(f"{col} = :redacted")
                set_sql = ", ".join(set_sql_parts)
                update_sql = f"UPDATE {p.table_name} SET {set_sql} WHERE {where_sql}"
                try:
                    session.execute(text(update_sql), {**params, "redacted": "[REDACTED]"})
                    session.commit()
                    audit.record_audit("retention.anonymize", actor="system", target_type=p.table_name, target_id=None,
                                       details={"policy": p.name, "rows_anonymized": to_delete})
                    summary["applied"] = True
                except SQLAlchemyError:
                    session.rollback()
                    logger.exception("Anonymize failed for table %s; consider implementing table-specific anonymizer", p.table_name)
                    summary["applied"] = False
            else:
                logger.warning("Unknown action %s for policy %s", p.action, p.name)
                summary["applied"] = False

            summaries.append(summary)
        return summaries
    finally:
        session.close()


def anonymize_user(user_id: int, actor: str = "system") -> bool:
    """
    Anonymize a user record (PII removal). This operation:
    - sets username -> anonymized placeholder (anon-<id>-<ts>)
    - replaces password_hash with an unusable marker, clears scopes, disables account
    - updates related tables per policy (jobs.user_id -> NULL)
    - records an audit log
    Returns True on success.
    """
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=user_id).one_or_none()
        if not user:
            return False
        anon_name = f"anon-{user_id}-{int(datetime.utcnow().timestamp())}"
        # clear sensitive fields
        user.username = anon_name
        user.password_hash = "!"
        user.scopes = []
        user.disabled = True
        session.commit()

        # Clear user_id references in jobs (best-effort)
        try:
            session.execute(text("UPDATE jobs SET user_id = NULL WHERE user_id = :uid"), {"uid": user_id})
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to clear jobs for user %s", user_id)

        audit.record_audit("user.anonymize", actor=actor, target_type="user", target_id=user_id,
                           details={"anon_username": anon_name})
        return True
    except Exception:
        session.rollback()
        logger.exception("Failed to anonymize user %s", user_id)
        return False
    finally:
        session.close()


def tenant_isolation_check(requesting_tenant: Optional[str], resource_tenant: Optional[str]) -> None:
    """
    Enforce tenant isolation. Raises ValueError if tenants differ.
    - In a multi-tenant system, call this before returning or mutating tenant-scoped resources.
    """
    if requesting_tenant is None or resource_tenant is None:
        # if tenant unknown, be conservative: deny
        raise ValueError("tenant information missing; deny by default")
    if requesting_tenant != resource_tenant:
        raise PermissionError("cross-tenant access denied")


# --- Federated aggregation protection ----------------------------------------------------

def _laplace_noise(scale: float) -> float:
    # Laplace(0, b) where b = scale. Use simple sampling via inverse CDF
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2 * abs(u) + 1e-12)


def federated_aggregation_protect(
    aggregated_value: Dict[str, Any],
    participant_count: int,
    min_participants: int = 5,
    dp_epsilon: Optional[float] = None,
    dp_sensitivity: float = 1.0,
    audit_on_flag: bool = True,
) -> Dict[str, Any]:
    """
    Basic federated aggregation protection:
    - require participant_count >= min_participants else raise PermissionError
    - optionally add Laplace noise to numeric fields if dp_epsilon provided
    - records audit event when protections applied/violated
    Returns a possibly-noised aggregated_value (copy).
    """
    if participant_count < min_participants:
        if audit_on_flag:
            audit.record_audit("federated.block", actor="system", target_type="federated_aggregate", target_id=None,
                               details={"participants": participant_count, "min_required": min_participants})
        raise PermissionError("not enough participants for safe aggregation")

    out = dict(aggregated_value)  # shallow copy
    if dp_epsilon and dp_epsilon > 0:
        # Laplace noise scale b = sensitivity / epsilon
        scale = float(dp_sensitivity) / float(dp_epsilon)
        # apply noise to numeric fields (best-effort)
        for k, v in list(out.items()):
            if isinstance(v, (int, float)):
                noise = _laplace_noise(scale)
                out[k] = v + noise
        audit.record_audit("federated.dp", actor="system", target_type="federated_aggregate", target_id=None,
                           details={"participants": participant_count, "dp_epsilon": dp_epsilon})
    else:
        audit.record_audit("federated.accept", actor="system", target_type="federated_aggregate", target_id=None,
                           details={"participants": participant_count})
    return out
