"""
OpenTelemetry initialization for Aegis.

- Configures a TracerProvider with OTLP exporter (env-configurable).
- Instruments FastAPI (via ASGI middleware), SQLAlchemy, and Celery (worker).
- Exposes a function `init_otel()` to be called at process startup.
"""
import os
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

logger = logging.getLogger("aegis.otel")


def init_otel(app=None, sqlalchemy_engine=None, celery_app=None):
    """
    Initialize OpenTelemetry tracing.
    - app: optional FastAPI app to auto-instrument
    - sqlalchemy_engine: optional SQLAlchemy engine to instrument
    - celery_app: optional Celery app to instrument
    """
    svc_name = os.environ.get("AEGIS_SERVICE_NAME", "aegis-api")
    svc_ver = os.environ.get("AEGIS_SERVICE_VERSION", "dev")
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", None)  # e.g., "http://otel-collector:4317"

    resource = Resource.create({SERVICE_NAME: svc_name, SERVICE_VERSION: svc_ver})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # OTLP exporter (if configured) + Console exporter fallback for dev
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("OTel: configured OTLP exporter -> %s", otlp_endpoint)
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel: using ConsoleSpanExporter (OTLP endpoint not configured)")

    # Instrument FastAPI app
    if app is not None:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("OTel: FastAPI instrumented")

    # Instrument SQLAlchemy engine
    if sqlalchemy_engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
        logger.info("OTel: SQLAlchemy instrumented")

    # Instrument Celery
    if celery_app is not None:
        CeleryInstrumentor().instrument()
        logger.info("OTel: Celery instrumented")

    # Instrument Redis (used by Celery broker/backend)
    try:
        RedisInstrumentor().instrument()
        logger.info("OTel: Redis instrumented")
    except Exception:
        # optional; redis may not be present in all processes
        logger.debug("OTel: Redis instrumentation not applied")
