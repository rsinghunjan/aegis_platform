"""Distributed tracing with an OpenTelemetry-compatible fallback.

When ``opentelemetry-sdk`` is installed and enabled, :func:`start_span`
creates OpenTelemetry spans. Otherwise it records spans in-process, allowing
the same instrumentation paths to work in tests and minimal deployments.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        self.status = status

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Minimal in-process tracer recording completed spans.

    Acts as a safe, dependency-free default when OpenTelemetry is disabled.
    """

    def __init__(self, service_name: str = "aegis"):
        self.service_name = service_name
        self._lock = threading.Lock()
        self._completed_spans: deque[Span] = deque(maxlen=1000)

    @contextmanager
    def start_span(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Iterator[Span]:
        parent = _active_span.get()
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else get_trace_id() or uuid.uuid4().hex,
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes or {}),
        )
        token = _active_span.set(span)
        try:
            yield span
        except Exception:
            span.set_status("error")
            raise
        finally:
            span.end_time = time.time()
            with self._lock:
                self._completed_spans.append(span)
            _active_span.reset(token)

    def completed_spans(self) -> List[Span]:
        with self._lock:
            return list(self._completed_spans)

    def clear(self) -> None:
        with self._lock:
            self._completed_spans.clear()


_active_span: ContextVar[Optional[Span]] = ContextVar("aegis_active_span", default=None)
_trace_id: ContextVar[Optional[str]] = ContextVar("aegis_trace_id", default=None)
_tracer_lock = threading.Lock()
_tracer_instance: Optional[Tracer] = None
_TRACEPARENT_PATTERN = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$", re.IGNORECASE
)


def trace_id_from_traceparent(traceparent: Optional[str]) -> Optional[str]:
    if not traceparent:
        return None
    match = _TRACEPARENT_PATTERN.fullmatch(traceparent.strip())
    if match is None or match.group(1) == "0" * 32 or match.group(2) == "0" * 16:
        return None
    return match.group(1).lower()


@contextmanager
def trace_id_context(trace_id: Optional[str] = None) -> Iterator[str]:
    value = trace_id or uuid.uuid4().hex
    token = _trace_id.set(value)
    try:
        yield value
    finally:
        _trace_id.reset(token)


def get_trace_id() -> Optional[str]:
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            return f"{span_context.trace_id:032x}"
    except ImportError:
        pass
    span = _active_span.get()
    return span.trace_id if span is not None else _trace_id.get()


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    if os.getenv("AEGIS_OTEL_ENABLED", "").lower() in {"1", "true", "yes"} or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    ):
        try:
            from opentelemetry import trace
        except ImportError:
            pass
        else:
            with trace.get_tracer("aegis").start_as_current_span(
                name, attributes=attributes
            ) as span:
                yield span
            return
    with get_tracer().start_span(name, attributes) as span:
        yield span


def inject_trace_context() -> Optional[str]:
    carrier: dict[str, str] = {}
    try:
        from opentelemetry.propagate import inject

        inject(carrier)
    except ImportError:
        pass
    if carrier.get("traceparent"):
        return carrier["traceparent"]
    trace_id = get_trace_id()
    if trace_id is None:
        return None
    span = _active_span.get()
    span_id = span.span_id if span is not None else uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{span_id}-01"


@contextmanager
def extracted_trace_context(traceparent: Optional[str]) -> Iterator[None]:
    trace_id = trace_id_from_traceparent(traceparent)
    with trace_id_context(trace_id):
        try:
            from opentelemetry import context, propagate
        except ImportError:
            yield
        else:
            extracted = propagate.extract(
                {"traceparent": traceparent} if traceparent else {}
            )
            token = context.attach(extracted)
            try:
                yield
            finally:
                context.detach(token)


def get_tracer(service_name: str = "aegis") -> Tracer:
    """Return the process-wide record-only tracer used when OTel is disabled."""
    global _tracer_instance
    with _tracer_lock:
        if _tracer_instance is None:
            _tracer_instance = Tracer(service_name=service_name)
        return _tracer_instance
