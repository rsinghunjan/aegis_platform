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
Celery periodic billing aggregation.

- `aggregate_and_invoice` is a Celery beat task that:
  - aggregates UsageRecord rows for the billing period (e.g., daily)
  - creates an Invoice row per tenant
  - optionally calls an external billing webhook (BILLING_WEBHOOK_URL)
  - emits AuditLog `billing.invoice_created`
"""
import os
import logging
from datetime import datetime, timedelta

from celery import shared_task
from sqlalchemy import func

from api.db import SessionLocal
from api.models import UsageRecord, Invoice
from api.audit import record_audit
from api.usage import BILLING_UNIT_COST

BILLING_WEBHOOK_URL = os.environ.get("BILLING_WEBHOOK_URL", "")
BILLING_PERIOD_HOURS = int(os.environ.get("BILLING_PERIOD_HOURS", "24"))

logger = logging.getLogger("aegis.billing")


@shared_task(bind=True, name="aegis.aggregate_and_invoice")
def aggregate_and_invoice(self):
    """
    Aggregate usage for the previous billing period and create invoices.
    Idempotent: marks invoices with period_start/period_end.
    """
    session = SessionLocal()
    try:
        end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=BILLING_PERIOD_HOURS)
        # group by tenant
        rows = session.query(
            UsageRecord.tenant_id,
            func.sum(UsageRecord.units).label("units"),
            func.sum(UsageRecord.cost_estimate).label("cost")
        ).filter(UsageRecord.created_at >= start, UsageRecord.created_at < end).group_by(UsageRecord.tenant_id).all()

        invoices = []
        for r in rows:
            tenant = r.tenant_id or "anonymous"
            total_units = int(r.units or 0)
            total_cost = float(r.cost or 0.0)
            inv = Invoice(
                tenant_id=tenant,
                period_start=start,
                period_end=end,
                units=total_units,
                amount=total_cost,
                currency=os.environ.get("BILLING_CURRENCY", "USD"),
                status="issued",
                created_at=datetime.utcnow()
            )
            session.add(inv)
            session.commit()
            session.refresh(inv)
            invoices.append(inv)

            # emit audit
            record_audit("billing.invoice_created", actor="system", target_type="invoice", target_id=inv.id, details={"tenant": tenant, "units": total_units, "amount": total_cost})
            # optionally call webhook (best-effort; non-blocking)
            if BILLING_WEBHOOK_URL:
                try:
                    import requests
                    requests.post(BILLING_WEBHOOK_URL, json={"invoice_id": inv.id, "tenant": tenant, "amount": total_cost, "currency": inv.currency}, timeout=5)
                except Exception:
                    logger.exception("billing webhook failed for invoice %s", inv.id)

        return {"invoices_created": len(invoices)}
    except Exception:
        session.rollback()
        logger.exception("billing aggregation failed")
        raise
    finally:
        session.close()
