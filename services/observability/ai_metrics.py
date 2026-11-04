"""Prometheus metrics for AI workflow requests."""
from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    AI_REQUESTS = AI_REQUEST_DURATION = AI_COST = None
else:
    AI_REQUESTS = Counter(
        "aegis_ai_requests",
        "AI workflow requests by outcome",
        ("status",),
    )
    AI_REQUEST_DURATION = Histogram(
        "aegis_ai_request_duration_seconds",
        "AI workflow request duration in seconds",
    )
    AI_COST = Counter("aegis_ai_cost_usd", "Total AI provider cost in USD")


def record_ai_request(
    status: str, duration_seconds: float, cost_usd: float = 0.0
) -> None:
    if AI_REQUESTS is None:
        return
    AI_REQUESTS.labels(status=status).inc()
    AI_REQUEST_DURATION.observe(duration_seconds)
    if cost_usd > 0:
        AI_COST.inc(cost_usd)


def render_metrics() -> bytes | None:
    if AI_REQUESTS is None:
        return None
    return generate_latest()
