from __future__ import annotations
"""
Speculative Gateway Proxy

Small proxy endpoint that acts like inference_gateway but routes speculative requests
to the speculative orchestrator while preserving DLP, budget and audit checks.

- POST /infer_speculative calls the orchestrator at SPEC_ORchestrator_URL
- Config via SPEC_ORCHESTRATOR_URL, SERVICE_API_KEY, etc.
"""
import os, requests, uuid, time
from fastapi import FastAPI, HTTPException, Request
from services.excel.dlp_rules import contains_pii, redact
from services.audit.audit_logger import log_event
from services.cost.budget_manager import check_and_consume

SPEC_ORCHESTRATOR_URL = os.environ.get("SPEC_ORCHESTRATOR_URL", "http://spec-orchestrator:8000/speculative/infer")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "")

app = FastAPI(title="Speculative Gateway Proxy")

def estimate_cost_for_request(req: dict) -> float:
    return float(os.environ.get("EST_COST_PER_REQUEST_USD", "0.001"))

@app.post("/infer_speculative")
async def infer_speculative(req: Request):
    body = await req.json()
    user = body.get("tenant", "unknown")
    if contains_pii(body.get("prompt","")):
        raise HTTPException(status_code=400, detail="Prompt contains PII; blocked")
    # budget check
    if not check_and_consume(user, estimate_cost_for_request(body)):
        raise HTTPException(status_code=429, detail="Budget exceeded")
    # forward to orchestrator
    try:
        r = requests.post(SPEC_ORCHESTRATOR_URL, json=body, headers={"Authorization": f"Bearer {SERVICE_API_KEY}"}, timeout=60)
        r.raise_for_status()
        res = r.json()
        log_event({"user": user, "request_id": res.get("request_id"), "action":"spec_infer_forwarded", "reason":"ok", "model":"spec_orch"}, persist_to_s3=False)
        return res
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
