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
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
"""
FastAPI router for billing endpoints.

Endpoints:
 - GET  /v1/billing/invoices        -> list invoices (tenant or admin)
 - GET  /v1/billing/invoices/{id}   -> invoice detail
 - POST /v1/billing/invoices/{id}/pay  -> attempt to charge invoice immediately (admin or billing role)
 - POST /v1/billing/webhook         -> payment provider webhook (unauthenticated, verifies provider signature)
"""
import os
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Request, HTTPException, status
from sqlalchemy import desc

from api.db import SessionLocal
from api.models import Invoice, BillingAccount, Tenant
from api.auth import require_scopes
from api.billing_gateway import verify_webhook, charge_customer

logger = logging.getLogger("aegis.billing_api")
router = APIRouter(prefix="/v1/billing", tags=["billing"])

# tenant-scoped read for tenants; admin scope for full access
tenant_scope = require_scopes(["predict"])  # use token tenant claim to filter; adjust as needed
admin_scope = require_scopes(["admin"])


@router.get("/invoices", dependencies=[Depends(tenant_scope)])
def list_invoices(request: Request, limit: int = 50):
    """
    List invoices for the token's tenant (tenant claim in JWT) or all if admin.
    The require_scopes dependency returns the DB user; to simplify, extract tenant from token payload.
    """
    # decode token without re-checking scopes for tenant claim; in production use a proper current_user dependency
    token_payload = request.state._token_payload if hasattr(request.state, "_token_payload") else {}
    tenant = token_payload.get("tenant")
    session = SessionLocal()
    try:
        q = session.query(Invoice).order_by(desc(Invoice.created_at)).limit(limit)
        if tenant:
            q = q.filter(Invoice.tenant_id == tenant)
        rows = q.all()
        return [{"id": r.id, "tenant_id": r.tenant_id, "amount": r.amount, "status": r.status, "period_start": r.period_start, "period_end": r.period_end, "created_at": r.created_at} for r in rows]
    finally:
        session.close()


@router.get("/invoices/{invoice_id}", dependencies=[Depends(tenant_scope)])
def get_invoice(invoice_id: int, request: Request):
    session = SessionLocal()
    try:
        inv = session.query(Invoice).filter_by(id=invoice_id).one_or_none()
        if not inv:
            raise HTTPException(status_code=404, detail="invoice not found")
        # tenant enforcement: only owner or admin may view
        token_payload = request.state._token_payload if hasattr(request.state, "_token_payload") else {}
        tenant = token_payload.get("tenant")
        if tenant and inv.tenant_id != tenant:
            raise HTTPException(status_code=403, detail="forbidden")
        return {"id": inv.id, "tenant_id": inv.tenant_id, "amount": inv.amount, "status": inv.status, "units": inv.units, "created_at": inv.created_at, "paid_at": inv.paid_at}
    finally:
        session.close()


@router.post("/invoices/{invoice_id}/pay", dependencies=[Depends(admin_scope)])
def pay_invoice(invoice_id: int):
    """
    Attempt immediate collection for an invoice. Admin-only operation (or billing role).
    This enqueues or performs a direct charge via the billing gateway.
    """
    session = SessionLocal()
    try:
        inv = session.query(Invoice).filter_by(id=invoice_id).one_or_none()
        if not inv:
            raise HTTPException(status_code=404, detail="invoice not found")
        if inv.status == "paid":
            return {"ok": True, "message": "already paid"}
        # find billing account
        ba = session.query(BillingAccount).filter_by(tenant_id=inv.tenant_id).one_or_none()
        if not ba or not ba.gateway_customer_id:
            raise HTTPException(status_code=400, detail="no billing account configured for tenant")
        cents = int(round(inv.amount * 100))
        res = charge_customer(ba.gateway_customer_id, amount_cents=cents, currency=inv.currency, description=f"Aegis invoice {inv.id}", invoice_id=inv.id)
        if res.get("success"):
            inv.status = "paid"
            inv.paid_at = inv.paid_at or res.get("paid_at")  # provider paid timestamp if present
            # store provider id if available
            if "provider_id" in res:
                inv.provider_charge_id = res["provider_id"]
            session.add(inv)
            session.commit()
            return {"ok": True}
        else:
            # failure; mark as failed and record attempt
            inv.status = "payment_failed"
            session.add(inv)
            session.commit()
            return {"ok": False, "error": res.get("error", "payment_failed")}
    finally:
        session.close()


@router.post("/webhook")
async def payment_webhook(request: Request):
    """
    Generic webhook endpoint. Verifies signature via billing_gateway.verify_webhook and updates invoice state.
    """
    body = await request.body()
    sig = request.headers.get("Stripe-Signature", "")
    event = verify_webhook(body, sig)
    if not event:
        raise HTTPException(status_code=400, detail="invalid webhook signature")
    # stripe style
    typ = getattr(event, "type", event.get("type") if isinstance(event, dict) else None)
    data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else {}
    # Simple handlers: payment_intent.succeeded, payment_intent.payment_failed, charge.succeeded
    session = SessionLocal()
    try:
        if typ in ("payment_intent.succeeded", "charge.succeeded"):
            invoice_meta = (data.get("metadata") or {})
            inv_id = invoice_meta.get("aegis_invoice_id") or invoice_meta.get("invoice_id")
            if inv_id:
                inv = session.query(Invoice).filter_by(id=int(inv_id)).one_or_none()
                if inv:
                    inv.status = "paid"
                    inv.provider_charge_id = data.get("id") or data.get("payment_intent")
                    inv.paid_at = data.get("created")
                    session.add(inv)
                    session.commit()
        elif typ in ("payment_intent.payment_failed", "charge.failed"):
            invoice_meta = (data.get("metadata") or {})
            inv_id = invoice_meta.get("aegis_invoice_id") or invoice_meta.get("invoice_id")
            if inv_id:
                inv = session.query(Invoice).filter_by(id=int(inv_id)).one_or_none()
                if inv:
                    inv.status = "payment_failed"
                    session.add(inv)
                    session.commit()
    finally:
        session.close()
    return {"ok": True}
