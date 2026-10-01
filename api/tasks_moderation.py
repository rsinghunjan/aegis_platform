  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
"""
Celery moderation tasks: create triage items for flagged SafetyEvents,
notify reviewers via webhook/email placeholder and optionally escalate.

These tasks are intended to run asynchronously when SafetyEvent decision is FLAG
(or BLOCK in some flows) so humans can triage items without blocking inference.
"""
import os
import logging
from celery import shared_task
from datetime import datetime
from typing import Optional, Dict, Any

from api.db import SessionLocal
from api.models import SafetyEvent, TriageItem
from api.audit import record_audit

logger = logging.getLogger("aegis.moderation")

REVIEW_WEBHOOK = os.environ.get("SAFETY_REVIEW_WEBHOOK")  # optional webhook to notify review system


@shared_task(bind=True, name="aegis.create_triage_from_event")
def create_triage_from_event(self, safety_event_id: int, assign_to: Optional[str] = None):
    session = SessionLocal()
    try:
        ev = session.query(SafetyEvent).filter_by(id=safety_event_id).one_or_none()
        if not ev:
            logger.warning("SafetyEvent %s not found", safety_event_id)
            return {"ok": False, "reason": "not found"}
        # create triage item
        ti = TriageItem(
            safety_event_id=ev.id,
            request_id=ev.request_id,
            status="open",
            assigned_to=assign_to,
            reasons=ev.reasons,
            input_snapshot=ev.input_snapshot,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(ti)
        session.commit()
        logger.info("Created triage item %s for safety_event %s", ti.id, ev.id)
        record_audit("triage.create", actor="system", target_type="triage", target_id=ti.id, details={"safety_event_id": ev.id})
        # notify reviewer (best-effort)
        if REVIEW_WEBHOOK:
            try:
                import requests
                payload = {"triage_id": ti.id, "request_id": ti.request_id, "reasons": ti.reasons}
                requests.post(REVIEW_WEBHOOK, json=payload, timeout=5)
            except Exception:
                logger.exception("failed to notify review webhook")
        return {"ok": True, "triage_id": ti.id}
    except Exception:
        session.rollback()
        logger.exception("failed to create triage item")
        raise
    finally:
        session.close()


@shared_task(bind=True, name="aegis.escalate_triage_if_unresolved")
def escalate_triage_if_unresolved(self, triage_id: int):
    session = SessionLocal()
    try:
        ti = session.query(TriageItem).filter_by(id=triage_id).one_or_none()
        if not ti:
            return {"ok": False, "reason": "not found"}
        if ti.status == "open" and (datetime.utcnow() - ti.created_at).total_seconds() > 3600 * 24:
            # escalate (placeholder)
            record_audit("triage.escalate", actor="system", target_type="triage", target_id=ti.id, details={"reason": "stale"})
            # call webhook or send email in production
            return {"ok": True, "escalated": True}
        return {"ok": True, "escalated": False}
    finally:
        session.close()
