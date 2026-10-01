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
"""
Admin router for billing enforcement helpers.

Routes (admin-scoped):
- POST /v1/admin/billing/accounts/{tenant_id}/suspend  -> suspend/un-suspend tenant
- POST /v1/admin/billing/accounts/{tenant_id}/dunning   -> set dunning level & optional suspension expiry

These endpoints are intentionally simple and intended for use by admin UIs or automation
(audit recording should be added where appropriate).
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from datetime import datetime
from typing import Optional

from api.auth import require_scopes
from api.db import SessionLocal
from api.models import BillingAccount
from api.audit import record_audit

router = APIRouter(prefix="/v1/admin/billing", tags=["admin", "billing"])
admin_scope = require_scopes(["admin"])


@router.post("/accounts/{tenant_id}/suspend", dependencies=[Depends(admin_scope)])
def suspend_tenant(tenant_id: str, suspend: bool = Body(...), reason: Optional[str] = Body(None)):
    session = SessionLocal()
    try:
        ba = session.query(BillingAccount).filter_by(tenant_id=tenant_id).one_or_none()
        if not ba:
            raise HTTPException(status_code=404, detail="billing account not found")
        if suspend:
            ba.billing_suspended = True
            ba.billing_suspension_reason = reason
            ba.billing_suspended_at = datetime.utcnow()
        else:
            ba.billing_suspended = False
            ba.billing_suspension_reason = None
            ba.billing_suspended_at = None
            ba.suspension_expires_at = None
            ba.dunning_level = 0
        session.add(ba)
        session.commit()
        record_audit("billing.admin.suspend" if suspend else "billing.admin.unsuspend", actor="admin", target_type="billing_account", target_id=ba.id, details={"tenant": tenant_id, "reason": reason})
        return {"ok": True, "tenant": tenant_id, "suspended": suspend}
    finally:
        session.close()


@router.post("/accounts/{tenant_id}/dunning", dependencies=[Depends(admin_scope)])
def set_dunning(tenant_id: str, dunning_level: int = Body(...), suspension_expires_at: Optional[datetime] = Body(None)):
    session = SessionLocal()
    try:
        ba = session.query(BillingAccount).filter_by(tenant_id=tenant_id).one_or_none()
        if not ba:
            raise HTTPException(status_code=404, detail="billing account not found")
        ba.dunning_level = dunning_level
        if suspension_expires_at:
            ba.suspension_expires_at = suspension_expires_at
        session.add(ba)
        session.commit()
        record_audit("billing.admin.dunning", actor="admin", target_type="billing_account", target_id=ba.id, details={"tenant": tenant_id, "dunning_level": dunning_level})
        return {"ok": True, "tenant": tenant_id, "dunning_level": dunning_level}
    finally:
        session.close()
api/billing_admin_api.py
