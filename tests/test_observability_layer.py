"""Tests for services/observability: tracing, logging, metrics, health."""
import json
import logging
import time

import pytest

from services.observability import (
    Tracer,
    configure_structured_logging,
    correlation_id_context,
    get_correlation_id,
    MetricsRegistry,
    HealthCheck,
    HealthRegistry,
    CircuitBreaker,
    CircuitOpenError,
)
from services.observability.logging import JSONFormatter


def test_tracer_records_completed_spans_with_duration():
    tracer = Tracer(service_name="test")
    with tracer.start_span("do-work", attributes={"key": "value"}) as span:
        span.set_attribute("extra", 1)
        time.sleep(0.001)

    spans = tracer.completed_spans()
    assert len(spans) == 1
    assert spans[0].name == "do-work"
    assert spans[0].attributes == {"key": "value", "extra": 1}
    assert spans[0].duration_ms is not None and spans[0].duration_ms >= 0


def test_tracer_marks_span_error_on_exception():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.start_span("failing"):
            raise ValueError("boom")
    assert tracer.completed_spans()[0].status == "error"


def test_tracer_tracks_parent_child_span_ids():
    tracer = Tracer()
    with tracer.start_span("parent") as parent:
        with tracer.start_span("child") as child:
            pass
    spans = {s.name: s for s in tracer.completed_spans()}
    assert spans["child"].parent_id == spans["parent"].span_id


def test_correlation_id_context_sets_and_restores():
    assert get_correlation_id() is None
    with correlation_id_context("abc-123") as cid:
        assert cid == "abc-123"
        assert get_correlation_id() == "abc-123"
    assert get_correlation_id() is None


def test_json_formatter_emits_valid_json_with_correlation_id():
    logger = logging.getLogger("test.json.formatter")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    with correlation_id_context("corr-1"):
        logger.info("hello %s", "world")

    payload = json.loads(stream.getvalue())
    assert payload["message"] == "hello world"
    assert payload["correlation_id"] == "corr-1"
    assert payload["level"] == "INFO"


def test_metrics_registry_counter_gauge_histogram():
    registry = MetricsRegistry()
    counter = registry.counter("requests_total", "Total requests")
    counter.inc(labels={"route": "/health"})
    counter.inc(2.0, labels={"route": "/health"})
    assert counter.value(labels={"route": "/health"}) == 3.0

    gauge = registry.gauge("queue_depth")
    gauge.set(5)
    assert gauge.value() == 5

    histogram = registry.histogram("latency_ms")
    histogram.observe(10.0)
    histogram.observe(20.0)
    assert histogram.observations() == [10.0, 20.0]

    rendered = registry.render()
    assert "requests_total" in rendered
    assert "queue_depth" in rendered
    assert "latency_ms_count" in rendered


def test_health_registry_reports_overall_health():
    registry = HealthRegistry()
    registry.register(HealthCheck("db", lambda: True, critical=True))
    registry.register(HealthCheck("cache", lambda: False, critical=False))
    assert registry.is_healthy()

    registry.register(HealthCheck("critical-down", lambda: False, critical=True))
    assert not registry.is_healthy()


def test_health_registry_handles_check_exceptions():
    registry = HealthRegistry()

    def boom():
        raise RuntimeError("unreachable")

    registry.register(HealthCheck("flaky", boom, critical=True))
    results = registry.run_all()
    assert results[0].healthy is False
    assert "unreachable" in results[0].error


def test_circuit_breaker_opens_after_threshold_and_recovers():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

    def failing():
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(failing)

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not run")

    time.sleep(0.06)
    assert breaker.call(lambda: "recovered") == "recovered"
