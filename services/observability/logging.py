"""Structured (JSON) logging with correlation IDs."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

_correlation_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "aegis_correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    return _correlation_id_var.get()


@contextmanager
def correlation_id_context(correlation_id: Optional[str] = None) -> Iterator[str]:
    """Set (and restore) the current correlation id for the enclosed block."""
    value = correlation_id or uuid.uuid4().hex
    token = _correlation_id_var.set(value)
    try:
        yield value
    finally:
        _correlation_id_var.reset(token)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra_keys = set(record.__dict__) - _STANDARD_LOGRECORD_KEYS
        for key in extra_keys:
            payload[key] = record.__dict__[key]
        return json.dumps(payload, default=str)


_STANDARD_LOGRECORD_KEYS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def configure_structured_logging(level: int = logging.INFO, logger_name: Optional[str] = None) -> logging.Logger:
    """Configure a logger (root by default) to emit single-line JSON logs."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())
    # Avoid stacking duplicate handlers if called more than once.
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    logger.addHandler(handler)
    return logger
