"""Observability layer: tracing, structured logging, metrics, health checks."""

from .tracing import (
    Tracer,
    Span,
    extracted_trace_context,
    get_trace_id,
    get_tracer,
    inject_trace_context,
    start_span,
    trace_id_context,
    trace_id_from_traceparent,
)
from .logging import configure_structured_logging, get_correlation_id, correlation_id_context
from .metrics import MetricsRegistry, get_metrics_registry
from .health import HealthCheck, HealthRegistry, CircuitBreaker, CircuitOpenError

__all__ = [
    "Tracer",
    "Span",
    "get_tracer",
    "start_span",
    "get_trace_id",
    "trace_id_context",
    "trace_id_from_traceparent",
    "inject_trace_context",
    "extracted_trace_context",
    "configure_structured_logging",
    "get_correlation_id",
    "correlation_id_context",
    "MetricsRegistry",
    "get_metrics_registry",
    "HealthCheck",
    "HealthRegistry",
    "CircuitBreaker",
    "CircuitOpenError",
]
