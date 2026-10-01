"""
Simple audit logging helper.

- record_audit(action, actor, target_type, target_id, details)
  writes an AuditLog row and logs structured entry.

- Exposes get_audit_events to query recent audit events (for operators).
"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from api.db import SessionLocal
from api.models import AuditLog

logger = logging.getLogger("aegis.audit")


def record_audit(action: str, actor: str, target_type: str, target_id: Optional[int], details: Optional[Dict[str, Any]] = None):
    """
    Persist an audit event and log it in structured logs.
    - action: short action string (e.g., 'user.anonymize', 'retention.delete')
    - actor: who performed the action (username/service)
    - target_type: what type of resource (e.g., 'user', 'safety_events')
    - target_id: optional resource id
    - details: optional dict with extra metadata
    """
    session = SessionLocal()
    try:
        ev = AuditLog(action=action, actor=actor, target_type=target_type, target_id=target_id,
                      details=details or {}, created_at=datetime.utcnow())
        session.add(ev)
        session.commit()
        # structured log line
        logger.info("audit", extra={"action": action, "actor": actor, "target_type": target_type, "target_id": target_id, "details": details or {}})
    except Exception:
        session.rollback()
        logger.exception("failed to write audit event %s", action)
    finally:
        session.close()


def get_audit_events(limit: int = 100) -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        q = session.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        return [{"id": e.id, "action": e.action, "actor": e.actor, "target_type": e.target_type, "target_id": e.target_id, "details": e.details, "created_at": e.created_at.isoformat()} for e in q]
    finally:
        session.close()
