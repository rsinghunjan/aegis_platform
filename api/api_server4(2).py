diff --git a/api/api_server4.py b/api/api_server4.py
index 85b369d..b7e3c11 100644
--- a/api/api_server4.py
+++ b/api/api_server4.py
@@ -1,3 +1,4 @@
+
 """
 Updated FastAPI server with:
  - structured logging (json)
@@ -15,10 +16,11 @@ import uuid
 import logging
 from typing import Optional, Dict, Any, List
 
-from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
-from fastapi.responses import JSONResponse, PlainTextResponse
+from fastapi import FastAPI, Request
+from fastapi.responses import PlainTextResponse
 from fastapi.middleware.gzip import GZipMiddleware
 from starlette.middleware.cors import CORSMiddleware
 from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Summary
 
-from pydantic import BaseModel
+from api.middleware.billing_enforcement import BillingEnforcementMiddleware
+
+# Routers (platform capabilities)
+from api.auth import router as auth_router
+from api.registry_api import router as registry_router
+from api.billing_api import router as billing_router
+from api.billing_admin_api import router as billing_admin_router
+from api.triage import router as triage_router
 
 # init structured logging and OTEL before other imports that may instrument
 from api.logging_config import configure_logging, set_request_id
 configure_logging()
 logger = logging.getLogger("aegis_api")
 logger.setLevel(logging.INFO)
@@ -33,6 +35,7 @@ from api.otel import init_otel
 from api.db import engine as sqlalchemy_engine
 
 app = FastAPI(title="Aegis Model Serving", version="1.0.0")
+
 app.add_middleware(GZipMiddleware, minimum_size=1000)
 app.add_middleware(
     CORSMiddleware,
@@ -43,6 +46,10 @@ app.add_middleware(
     allow_headers=["*"],
 )
 
+# Billing enforcement (soft fail-open by design)
+# Will return 402 for suspended tenants when tenant_id is available.
+app.add_middleware(BillingEnforcementMiddleware)
+
 # instrument app + SQLAlchemy
 init_otel(app=app, sqlalchemy_engine=sqlalchemy_engine)
 
@@ -73,6 +80,20 @@ async def add_request_id_header(request: Request, call_next):
     response.headers["x-request-id"] = request_id
     return response
 
+# Basic health endpoints
+@app.get("/health")
+async def health():
+    return {"status": "ok"}
+
+@app.get("/ready")
+async def ready():
+    # Keep it simple; DB connectivity checks can be added later.
+    return {"ready": True}
+
 # Expose Prometheus metrics endpoint
 @app.get("/metrics")
 async def metrics():
     data = generate_latest()
     return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
 
-
-# ----------------- rest of your API (model registry, predict endpoints etc.) -----------------
-# For brevity, re-use the previously created ModelRegistry and endpoints.
-# Paste or import the rest of your existing api_server endpoints below (list_models, predict, etc.).
-# Ensure handlers increment Prometheus metrics around inference calls:
-#
-# Example usage inside prediction handler:
-#   with PREDICTION_LATENCY_MS.labels(model=model_name, version=version).time():
-#       result = registry.predict(...)
-#   PREDICTIONS_TOTAL.labels(model=model_name, version=version).inc()
-#
-# (The remainder of the implementation is unchanged in behavior but instrumented.)
-#
-# NOTE: If you already have an api/api_server.py in the repo, merge the middleware,
-# init_otel call and metrics endpoint into it. This snippet shows the essential additions.
-api/api_server.py
+# ----------------- platform routers -----------------
+# Auth + admin + billing + registry + safety triage
+app.include_router(auth_router)
+app.include_router(registry_router)
+app.include_router(billing_router)
+app.include_router(billing_admin_router)
+app.include_router(triage_router)

