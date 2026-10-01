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
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
"""
FastAPI router for triage/human-review operations.

Endpoints (require reviewer/admin scope):
- GET  /v1/safety/triage          -> list open triage items (filterable)
- GET  /v1/safety/triage/{id}     -> get triage item details
- POST /v1/safety/triage/{id}/review -> mark as reviewed (status/resolution), add note
- POST /v1/safety/triage/{id}/reprocess -> re-run safety checks on the original payload

Assumes require_scopes(['admin']) or dedicated 'reviewer' scope via auth.require_scopes.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Optional
from datetime import datetime

from api.auth import require_scopes
from api.db import SessionLocal
from api.models import TriageItem, SafetyEvent
from api.audit import record_audit
from api.tasks_moderation import create_triage_from_event

logger = logging.getLogger("aegis.triage")
router = APIRouter(prefix="/v1/safety", tags=["safety"])

reviewer_scope = require_scopes(["admin", "reviewer"])


@router.get("/triage", dependencies=[Depends(reviewer_scope)])
def list_triage(status: Optional[str] = None, limit: int = 50):
    session = SessionLocal()
    try:
        q = session.query(TriageItem).order_by(TriageItem.created_at.desc())
        if status:
            q = q.filter(TriageItem.status == status)
        items = q.limit(limit).all()
        return [{"id": i.id, "status": i.status, "assigned_to": i.assigned_to, "reasons": i.reasons, "created_at": i.created_at} for i in items]
    finally:
        session.close()


@router.get("/triage/{tid}", dependencies=[Depends(reviewer_scope)])
def get_triage(tid: int):
    session = SessionLocal()
    try:
        t = session.query(TriageItem).filter_by(id=tid).one_or_none()
        if not t:
            raise HTTPException(status_code=404, detail="not found")
        return {
            "id": t.id,
            "status": t.status,
            "assigned_to": t.assigned_to,
            "reasons": t.reasons,
            "input_snapshot": t.input_snapshot,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
    finally:
        session.close()


@router.post("/triage/{tid}/review", dependencies=[Depends(reviewer_scope)])
def review_triage(tid: int, resolution: str = Body(...), note: Optional[str] = Body(None)):
    session = SessionLocal()
    try:
        t = session.query(TriageItem).filter_by(id=tid).one_or_none()
        if not t:
            raise HTTPException(status_code=404, detail="not found")
        t.status = "resolved"
        t.resolution = resolution
        t.review_note = note
        t.resolved_at = datetime.utcnow()
        t.updated_at = datetime.utcnow()
        session.add(t)
        session.commit()
        record_audit("triage.review", actor="reviewer", target_type="triage", target_id=t.id, details={"resolution": resolution, "note": note})
        return {"ok": True}
    finally:
        session.close()


@router.post("/triage/{tid}/reprocess", dependencies=[Depends(reviewer_scope)])
def reprocess_triage(tid: int):
    session = SessionLocal()
    try:
        t = session.query(TriageItem).filter_by(id=tid).one_or_none()
        if not t:
            raise HTTPException(status_code=404, detail="not found")
        # re-run safety checks on the original payload by looking up SafetyEvent
        ev = session.query(SafetyEvent).filter_by(id=t.safety_event_id).one_or_none()
        if not ev:
            raise HTTPException(status_code=404, detail="safety event not found")
        # schedule background re-check: create triage from event again or run safety.check_and_log directly
        create_triage_from_event.delay(ev.id)
        record_audit("triage.reprocess", actor="reviewer", target_type="triage", target_id=t.id, details={})
        return {"ok": True}
    finally:
        session.close()
