"""
Structured JSON logging configuration for Aegis services.

- Uses python's logging + python-json-logger for structured logs.
- Provides request_id context via ContextVar and a middleware helper.
- All services should call configure_logging() at startup.
"""
import logging
import os
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

# Context var to carry request id across async contexts
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="unknown")

JSON_LOG_LEVEL = os.environ.get("AEGIS_LOG_LEVEL", "INFO")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = request_id_ctx.get()
        except LookupError:
            record.request_id = "unknown"
        return True


def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        # already configured
        return

    root.setLevel(JSON_LOG_LEVEL)
    handler = logging.StreamHandler()
    fmt_fields = [
        "timestamp", "level", "name", "message", "request_id", "module", "funcName", "lineno"
    ]
    formatter = jsonlogger.JsonFormatter(fmt=",".join(fmt_fields))
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)


def set_request_id(request_id: str):
    request_id_ctx.set(request_id)
