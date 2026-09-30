"""
Billing metering hooks for hosted inference calls.

This module offers a simple function billing_meter_hf_call which records a usage event.
Integrate with your billing collector / event pipeline to persist invoiceable usage.

- For simple deployments you can write a DB row into usage_events table.
- For high throughput, send events to a metrics stream / Kafka and run a billing batcher.

Provided: a minimal synchronous implementation that writes to SQL via your existing SessionLocal.
"""
import os
import logging
import time
from typing import Optional

try:
    from prometheus_client import Counter, Histogram
    METRICS_AVAILABLE = True
    HF_USAGE_COUNTER = Counter("aegis_hf_usage_total", "Hosted HF inference calls", ["model"])
    HF_USAGE_DURATION = Histogram("aegis_hf_usage_duration_seconds", "Duration of HF inference calls", ["model"])
except Exception:
    METRICS_AVAILABLE = False

from api.db import SessionLocal

logger = logging.getLogger("aegis.billing_metering")


def billing_meter_hf_call(model: str, tenant_id: Optional[str], duration_s: float, request_bytes: int, response_bytes: int):
    """
    Record a single usage event for billing / quota enforcement.

    Implementations should be idempotent and low-latency. This simple version writes a DB row.
    """
    try:
        if METRICS_AVAILABLE:
            HF_USAGE_COUNTER.labels(model=model).inc()
            HF_USAGE_DURATION.labels(model=model).observe(duration_s)
    except Exception:
        logger.exception("prometheus emission failed")

    # Minimal DB write: UsageEvent(table) should be created in api.models with fields:
    # tenant_id, provider, model, duration_s, request_bytes, response_bytes, created_at
    session = None
    try:
        from api.models import UsageEvent

        session = SessionLocal()
        ue = UsageEvent(
            tenant_id=tenant_id or "unknown",
            provider="huggingface",
            model=model,
            duration_seconds=float(duration_s),
            request_bytes=int(request_bytes),
            response_bytes=int(response_bytes),
        )
        session.add(ue)
        session.commit()
    except Exception:
        logger.exception("billing meter DB write failed (non-fatal)")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
