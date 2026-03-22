
"""
Updated FastAPI server with:
 - structured logging (json)
 - OpenTelemetry init + automatic instrumentation
 - /metrics endpoint for Prometheus
 - request_id middleware that integrates with structured logging and tracing

This file replaces the previous api/api_server.py implementation; ensure you have
api/logging_config.py and api/otel.py present and install opentelemetry deps.
"""
import base64
import time
import uuid
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Summary

from pydantic import BaseModel

# init structured logging and OTEL before other imports that may instrument
from api.logging_config import configure_logging, set_request_id
configure_logging()
logger = logging.getLogger("aegis_api")
logger.setLevel(logging.INFO)

# init OpenTelemetry
from api.otel import init_otel
from api.db import engine as sqlalchemy_engine

app = FastAPI(title="Aegis Model Serving", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# instrument app + SQLAlchemy
init_otel(app=app, sqlalchemy_engine=sqlalchemy_engine)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("aegis_predictions_total", "Total predictions", ["model", "version"])
PREDICTION_LATENCY_MS = Summary("aegis_prediction_latency_ms", "Prediction latency in ms", ["model", "version"])

# simple request-id middleware
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    # attach to logging context via ContextVar
    set_request_id(request_id)
    start = time.time()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.time() - start) * 1000.0
        logger.info("http.request", extra={
            "method": request.method,
            "path": request.url.path,
            "status": getattr(response, "status_code", None),
            "remote": request.client.host if request.client else None,
            "request_id": request_id,
            "latency_ms": elapsed_ms,
        })
    response.headers["x-request-id"] = request_id
    return response

# Expose Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)


# ----------------- rest of your API (model registry, predict endpoints etc.) -----------------
# For brevity, re-use the previously created ModelRegistry and endpoints.
# Paste or import the rest of your existing api_server endpoints below (list_models, predict, etc.).
# Ensure handlers increment Prometheus metrics around inference calls:
#
# Example usage inside prediction handler:
#   with PREDICTION_LATENCY_MS.labels(model=model_name, version=version).time():
#       result = registry.predict(...)
#   PREDICTIONS_TOTAL.labels(model=model_name, version=version).inc()
#
# (The remainder of the implementation is unchanged in behavior but instrumented.)
#
# NOTE: If you already have an api/api_server.py in the repo, merge the middleware,
# init_otel call and metrics endpoint into it. This snippet shows the essential additions.
api/api_server.py

