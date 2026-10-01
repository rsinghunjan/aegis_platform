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
Usage metering utilities.
- BILLING_CURRENCY (default "USD")
"""
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from prometheus_client import Counter

from api.db import SessionLocal
from api.models import UsageRecord

BILLING_UNIT_COST = float(os.environ.get("BILLING_UNIT_COST", "0.01"))  # dollars per unit
BILLING_CURRENCY = os.environ.get("BILLING_CURRENCY", "USD")

logger = logging.getLogger("aegis.usage")
USAGE_COUNTER = Counter("aegis_usage_records_total", "Total usage records created", ["tenant"])


def record_usage(tenant_id: Optional[str], model: str, version: str, inference_ms: float, units: int = 1, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Persist a usage row. Intended to be called by inference endpoints and background tasks.
    Keep this function fast — in high-throughput systems consider batching writes or using a fast append-only store.
    """
    session = SessionLocal()
    try:
        ur = UsageRecord(
            tenant_id=tenant_id,
            model_name=model,
            version=version,
            units=units,
            inference_ms=inference_ms,
            cost_estimate=units * BILLING_UNIT_COST,
            extra=extra or {},
            created_at=datetime.utcnow()
        )
        session.add(ur)
        session.commit()
        USAGE_COUNTER.labels(tenant=tenant_id or "anonymous").inc(units)
    except Exception:
        session.rollback()
        logger.exception("Failed to record usage for tenant=%s model=%s", tenant_id, model)
    finally:
        session.close()


def get_usage(tenant_id: Optional[str], start_ts, end_ts):
    """
    Aggregate usage for a tenant between timestamps.
    """
    session = SessionLocal()
    try:
        q = session.query(
            UsageRecord.tenant_id,
            UsageRecord.model_name,
            UsageRecord.version
        ).filter(UsageRecord.created_at >= start_ts, UsageRecord.created_at <= end_ts)
        if tenant_id is not None:
            q = q.filter(UsageRecord.tenant_id == tenant_id)
        # aggregate units and cost
        rows = session.query(
            UsageRecord.model_name,
            UsageRecord.version,
            func.sum(UsageRecord.units).label("units"),
            func.sum(UsageRecord.cost_estimate).label("cost")
        ).filter(UsageRecord.created_at >= start_ts, UsageRecord.created_at <= end_ts)
        if tenant_id:
            rows = rows.filter(UsageRecord.tenant_id == tenant_id)
        results = []
        for r in rows.group_by(UsageRecord.model_name, UsageRecord.version):
            results.append({"model": r.model_name, "version": r.version, "units": r.units, "cost": float(r.cost)})
        return results
    finally:
        session.close()
