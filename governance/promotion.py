"""SQLAlchemy-backed MLflow promotion and serving-reference helpers."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from aegis_db.models import AuditLog, Job, Run, Tenant
from aegis_db.session import create_sessionmaker

PROMOTION_KIND = "model_promotion"
LOCAL_DATABASE_URL = "sqlite:///./governance.db"

_engine = None
_session_factory = None
logger = logging.getLogger(__name__)


class PromotionError(ValueError):
    """Raised when an MLflow run cannot be promoted."""


def initialize_promotion_schema(engine) -> None:
    """Create the existing operational tables needed by the promotion flow."""
    for model in (Tenant, Job, Run, AuditLog):
        model.__table__.create(engine, checkfirst=True)


def get_promotion_sessionmaker():
    """Use DATABASE_URL, with a local SQLite database for development."""
    global _engine, _session_factory
    if _session_factory is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            database_url = LOCAL_DATABASE_URL
            logger.warning(
                "DATABASE_URL is unset; using local SQLite for development only"
            )
        _engine, _session_factory = create_sessionmaker(database_url)
        initialize_promotion_schema(_engine)
    return _session_factory


def validate_mlflow_run(client: Any, run_id: str) -> Any:
    """Return a completed MLflow run with a recorded artifact URI."""
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise PromotionError("MLflow run could not be retrieved") from exc

    if run is None:
        raise PromotionError("MLflow run was not found")
    info = getattr(run, "info", None)
    if info is None:
        raise PromotionError("MLflow returned an invalid run")
    if getattr(info, "run_id", run_id) != run_id:
        raise PromotionError("MLflow returned a different run ID")
    if getattr(info, "status", None) != "FINISHED":
        raise PromotionError("Only FINISHED MLflow runs can be promoted")
    if not getattr(info, "artifact_uri", None):
        raise PromotionError("MLflow run has no artifact URI")
    return run


class PersistedModelRegistry:
    """Minimal registry adapter using promotion metadata in the SQL database."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    @staticmethod
    def validate_handoff(model_name: str, version: str, artifact_uri: str) -> None:
        if not model_name or not version or not artifact_uri:
            raise PromotionError("Model name, version, and artifact URI are required")

    def resolve(self, model_name: str, tenant_id: str | None = None):
        with self.session_factory() as session:
            jobs = (
                session.query(Job)
                .filter(Job.kind == PROMOTION_KIND, Job.status == "approved")
                .order_by(Job.created_at.desc())
                .all()
            )
            for job in jobs:
                payload = job.payload_json or {}
                if payload.get("model_name") != model_name:
                    continue
                if tenant_id is not None and job.tenant_id != tenant_id:
                    continue
                run_id = payload.get("run_id")
                run = session.get(Run, run_id) if run_id else None
                if run is None or run.job_id != job.id:
                    continue
                return {
                    "model_name": model_name,
                    "version": payload["version"],
                    "run_id": run.id,
                    "artifact_uri": run.artifacts_uri,
                    "governance_evidence": payload.get("governance_evidence", {}),
                    "tenant_id": job.tenant_id,
                    "job_id": job.id,
                }
        return None


def _timestamp(value):
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _record_decision(
    session_factory,
    run_id: str,
    tenant_id: str,
    tenant_name: str,
    model_name: str,
    version: str,
    actor: str,
    notes: str,
    approved: bool,
    artifact_uri: str | None = None,
    metrics: dict | None = None,
    started_at=None,
    finished_at=None,
    governance_evidence: dict[str, Any] | None = None,
):
    PersistedModelRegistry.validate_handoff(
        model_name, version, artifact_uri or ("unavailable" if not approved else "")
    )
    job_id = uuid.uuid4().hex
    status = "approved" if approved else "rejected"
    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "artifact_uri": artifact_uri,
        "governance_evidence": governance_evidence or {},
    }

    with session_factory() as session:
        tenant = session.get(Tenant, tenant_id)
        if tenant is None:
            session.add(Tenant(id=tenant_id, name=tenant_name or tenant_id))

        job = Job(
            id=job_id,
            tenant_id=tenant_id,
            kind=PROMOTION_KIND,
            status=status,
            payload_json=payload,
        )
        session.add(job)

        if approved:
            run = session.get(Run, run_id)
            if run is None:
                run = Run(
                    id=run_id,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    started_at=_timestamp(started_at),
                    finished_at=_timestamp(finished_at),
                    metrics_json=metrics or {},
                    artifacts_uri=artifact_uri,
                )
                session.add(run)
            else:
                run.tenant_id = tenant_id
                run.job_id = job_id
                run.started_at = _timestamp(started_at)
                run.finished_at = _timestamp(finished_at)
                run.metrics_json = metrics or {}
                run.artifacts_uri = artifact_uri

        session.add(
            AuditLog(
                id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                actor=actor,
                action="model.promotion",
                resource=f"{model_name}:{version}",
                decision=status,
                reason=json.dumps(
                    {
                        "run_id": run_id,
                        "notes": notes,
                        "artifact_uri": artifact_uri,
                        "governance_evidence": governance_evidence or {},
                    },
                    sort_keys=True,
                ),
            )
        )
        session.commit()

    return {
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "artifact_uri": artifact_uri,
        "tenant_id": tenant_id,
        "job_id": job_id,
        "decision": status,
    }


def promote_run(
    run_id: str,
    *,
    client: Any,
    model_name: str | None = None,
    version: str = "1",
    tenant_id: str = "default",
    tenant_name: str | None = None,
    actor: str = "system",
    notes: str = "",
    session_factory=None,
    governance_evidence: dict[str, Any] | None = None,
):
    """Validate a run, record governance metadata, and persist its registry handoff."""
    session_factory = session_factory or get_promotion_sessionmaker()
    registry = PersistedModelRegistry(session_factory)
    model_name = model_name or run_id

    try:
        run = validate_mlflow_run(client, run_id)
        artifact_uri = run.info.artifact_uri
        registry.validate_handoff(model_name, version, artifact_uri)
    except PromotionError:
        _record_decision(
            session_factory,
            run_id,
            tenant_id,
            tenant_name or tenant_id,
            model_name,
            version,
            actor,
            notes,
            False,
            governance_evidence=governance_evidence,
        )
        raise

    return _record_decision(
        session_factory,
        run_id,
        tenant_id,
        tenant_name or tenant_id,
        model_name,
        version,
        actor,
        notes,
        True,
        artifact_uri,
        getattr(run.data, "metrics", {}),
        getattr(run.info, "start_time", None),
        getattr(run.info, "end_time", None),
        governance_evidence,
    )


def resolve_promoted_model(
    model_name: str, *, tenant_id: str | None = None, session_factory=None
):
    registry = PersistedModelRegistry(session_factory or get_promotion_sessionmaker())
    return registry.resolve(model_name, tenant_id)
