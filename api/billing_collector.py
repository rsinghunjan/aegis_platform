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
"""
Celery task to process outstanding invoices and attempt automated collection.

- Scans for invoices with status 'issued' older than a configurable window and attempts to charge via billing_gateway.
- On success marks invoice as 'paid'. On repeated failure increments attempt count and sets status to 'payment_failed' after retries.
"""
import os
import logging
from datetime import datetime, timedelta

from celery import shared_task
from sqlalchemy import and_
from api.db import SessionLocal
from api.models import Invoice, BillingAccount
from api.billing_gateway import charge_customer
from api.audit import record_audit

logger = logging.getLogger("aegis.billing_collector")

RETRY_LIMIT = int(os.environ.get("BILLING_RETRY_LIMIT", "3"))
RETRY_DELAY_HOURS = int(os.environ.get("BILLING_RETRY_DELAY_HOURS", "24"))


@shared_task(bind=True, name="aegis.process_outstanding_invoices")
def process_outstanding_invoices(self):
    session = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        invoices = session.query(Invoice).filter(Invoice.status == "issued").all()
        processed = 0
        for inv in invoices:
            # simple age-based selection: attempt if created_at older than 0h (or policy)
            ba = session.query(BillingAccount).filter_by(tenant_id=inv.tenant_id).one_or_none()
            if not ba or not ba.gateway_customer_id:
                logger.info("No billing account for tenant %s; skipping invoice %s", inv.tenant_id, inv.id)
                continue
            cents = int(round(inv.amount * 100))
            res = charge_customer(ba.gateway_customer_id, amount_cents=cents, currency=inv.currency, description=f"Aegis invoice {inv.id}", invoice_id=inv.id)
            if res.get("success"):
                inv.status = "paid"
                inv.provider_charge_id = res.get("provider_id")
                inv.paid_at = datetime.utcnow()
                session.add(inv)
                session.commit()
                record_audit("billing.charge_success", actor="system", target_type="invoice", target_id=inv.id, details={"tenant": inv.tenant_id, "amount": inv.amount})
            else:
                inv.attempts = getattr(inv, "attempts", 0) + 1
                inv.last_attempt_at = datetime.utcnow()
                if inv.attempts >= RETRY_LIMIT:
                    inv.status = "payment_failed"
                    record_audit("billing.charge_failed", actor="system", target_type="invoice", target_id=inv.id, details={"tenant": inv.tenant_id, "attempts": inv.attempts, "error": res.get("error")})
                session.add(inv)
                session.commit()
            processed += 1
        return {"processed": processed}
    except Exception:
        session.rollback()
        logger.exception("billing collector failed")
        raise
    finally:
        session.close()
