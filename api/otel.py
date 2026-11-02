"""
OpenTelemetry initialization for Aegis.

- Configures a TracerProvider with OTLP exporter (env-configurable).
- Instruments FastAPI (via ASGI middleware), SQLAlchemy, and Celery (worker).
- Exposes a function `init_otel()` to be called at process startup.
"""
import os
import logging

logger = logging.getLogger("aegis.otel")
_provider = None


def init_otel(app=None, sqlalchemy_engine=None, celery_app=None):
    """
    Initialize OpenTelemetry tracing.
    - app: optional FastAPI app to auto-instrument
    - sqlalchemy_engine: optional SQLAlchemy engine to instrument
    - celery_app: optional Celery app to instrument
    """
    global _provider
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    enabled = os.environ.get("AEGIS_OTEL_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not enabled and not otlp_endpoint:
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.celery import CeleryInstrumentor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import (
            Resource,
            SERVICE_NAME,
            SERVICE_VERSION,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        logger.warning("OTel enabled but OpenTelemetry dependencies are not installed")
        return None

    if _provider is None:
        svc_name = os.environ.get("AEGIS_SERVICE_NAME", "aegis-api")
        svc_ver = os.environ.get("AEGIS_SERVICE_VERSION", "dev")
        resource = Resource.create({SERVICE_NAME: svc_name, SERVICE_VERSION: svc_ver})
        _provider = TracerProvider(resource=resource)
        if otlp_endpoint:
            _provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=otlp_endpoint,
                        insecure=otlp_endpoint.startswith("http://"),
                    )
                )
            )
            logger.info("OTel: configured OTLP exporter -> %s", otlp_endpoint)
        else:
            _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("OTel: using ConsoleSpanExporter (OTLP endpoint not configured)")
        trace.set_tracer_provider(_provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=_provider)
        logger.info("OTel: FastAPI instrumented")
    if sqlalchemy_engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
        logger.info("OTel: SQLAlchemy instrumented")
    if celery_app is not None:
        CeleryInstrumentor().instrument()
        logger.info("OTel: Celery instrumented")
    try:
        RedisInstrumentor().instrument()
        logger.info("OTel: Redis instrumented")
    except Exception:
        # optional; redis may not be present in all processes
        logger.debug("OTel: Redis instrumentation not applied")
    return _provider
