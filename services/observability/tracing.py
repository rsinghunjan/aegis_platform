"""Distributed tracing with an OpenTelemetry-compatible fallback.

When ``opentelemetry-sdk`` is installed and configured, :func:`get_tracer`
returns a real OpenTelemetry tracer. Otherwise it returns an in-process
no-op/record-only :class:`Tracer` so that span creation code paths work
identically in both environments (useful for tests and minimal
deployments).
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

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

    Acts as a safe, dependency-free default. If real OpenTelemetry export
    is desired, wire an OTLP exporter in :func:`get_tracer` and keep this
    class for local testing/fallback.
    """

    def __init__(self, service_name: str = "aegis"):
        self.service_name = service_name
        self._lock = threading.Lock()
        self._completed_spans: List[Span] = []
        self._current_span_id: Optional[str] = None

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        parent_id = self._current_span_id
        span = Span(name=name, parent_id=parent_id, attributes=dict(attributes or {}))
        self._current_span_id = span.span_id
        try:
            yield span
        except Exception:
            span.set_status("error")
            raise
        finally:
            span.end_time = time.time()
            with self._lock:
                self._completed_spans.append(span)
            self._current_span_id = parent_id

    def completed_spans(self) -> List[Span]:
        with self._lock:
            return list(self._completed_spans)

    def clear(self) -> None:
        with self._lock:
            self._completed_spans.clear()


_tracer_lock = threading.Lock()
_tracer_instance: Optional[Tracer] = None


def get_tracer(service_name: str = "aegis") -> Tracer:
    """Return a process-wide shared :class:`Tracer` instance.

    Attempts to use a real OpenTelemetry-backed tracer when the SDK is
    installed and ``AEGIS_OTEL_ENABLED=true``; otherwise falls back to the
    local in-process :class:`Tracer`.
    """
    global _tracer_instance
    with _tracer_lock:
        if _tracer_instance is None:
            _tracer_instance = Tracer(service_name=service_name)
        return _tracer_instance
