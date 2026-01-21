from __future__ import annotations
"""
Speculative Decoding Orchestrator

- FastAPI service implementing speculative decoding with batched teacher scoring.
- Configurable via environment variables:
  - STUDENT_ENDPOINT (POST /generate) => returns {"text": "<draft text>"}
  - TEACHER_SCORE_URL (POST /score_batch) => accepts [{"context": "...", "draft": "..."}] and returns [{"token_scores":[...], "tokens":[...]}]
  - TEACHER_GENERATE_URL (POST /generate_one) => returns {"token": "<token>", "token_logprob": -1.2}
  - BATCH_INTERVAL_MS, BATCH_MAX
  - ACCEPT_THRESHOLD (logprob or normalized score)
  - MAX_SPEC_PER_REQ, MAX_SPEC_CHUNK_LEN
- Emits Prometheus metrics on /metrics (via prometheus_client).
- Logs audit/tracing to /var/log/aegis/speculative_events.jsonl
"""
import os
import time
import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import start_http_server, Counter, Gauge

from services.excel.dlp_rules import contains_pii, redact
from services.audit.audit_logger import log_event

# Configuration
STUDENT_ENDPOINT = os.environ.get("STUDENT_ENDPOINT", "http://student.inference:8080/generate")
TEACHER_SCORE_URL = os.environ.get("TEACHER_SCORE_URL", "http://teacher.inference:8080/score_batch")
TEACHER_GENERATE_URL = os.environ.get("TEACHER_GENERATE_URL", "http://teacher.inference:8080/generate_one")
BATCH_INTERVAL_MS = int(os.environ.get("SPEC_BATCH_INTERVAL_MS", "50"))
BATCH_MAX = int(os.environ.get("SPEC_BATCH_MAX", "128"))
ACCEPT_THRESHOLD = float(os.environ.get("SPEC_ACCEPT_THRESHOLD", " -1.0"))  # interpret as min logprob per token (example)
MAX_SPEC_PER_REQ = int(os.environ.get("SPEC_MAX_PER_REQ", "64"))
MAX_SPEC_CHUNK_LEN = int(os.environ.get("SPEC_CHUNK_LEN", "8"))

EVENT_LOG_PATH = os.environ.get("SPEC_EVENT_LOG", "/var/log/aegis/speculative_events.jsonl")
METRICS_PORT = int(os.environ.get("SPEC_METRICS_PORT", "9460"))

# Prometheus metrics
MET_REQS = Counter("aegis_spec_requests_total", "Speculative decode requests received")
MET_ACCEPTED_TOKENS = Counter("aegis_spec_tokens_accepted_total", "Total tokens accepted without teacher autoreg")
MET_TOKENS_PROPOSED = Counter("aegis_spec_tokens_proposed_total", "Total tokens proposed by student")
MET_REQ_FALLBACK = Counter("aegis_spec_fallback_count_total", "Requests that fell back to teacher autoreg at least once")
MET_ACCEPTANCE_RATE = Gauge("aegis_spec_acceptance_rate", "Acceptance rate of student-proposed tokens (rolling, set by app)")
MET_TOKENS_SAVED = Counter("aegis_spec_tokens_saved_total", "Total teacher autoreg steps saved (approx)")
MET_ENERGY_SAVED_KWH = Counter("aegis_spec_energy_saved_kwh_total", "Estimated kWh saved by speculative decoding")

# In-memory batch queue and synchronization
_BATCH_QUEUE: List[Dict[str, Any]] = []
_BATCH_LOCK = asyncio.Lock()
# map request_id -> asyncio.Future where result will be placed
_PENDING_FUTURES: Dict[str, asyncio.Future] = {}

app = FastAPI(title="Aegis Speculative Orchestrator")

# Helpers
async def write_event(ev: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(EVENT_LOG_PATH), exist_ok=True)
        with open(EVENT_LOG_PATH, "a") as f:
            f.write(json.dumps(ev) + "\n")
    except Exception:
        pass

class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    risk_level: str = "default"
    tenant: Optional[str] = None
    request_id: Optional[str] = None

async def call_student(context: str, max_len: int) -> str:
    """Call the student model to propose a draft (returns raw text)."""
    async with aiohttp.ClientSession() as session:
        try:
            resp = await session.post(STUDENT_ENDPOINT, json={"context": context, "max_length": max_len}, timeout=30)
            if resp.status != 200:
                raise RuntimeError(f"student returned {resp.status}")
            j = await resp.json()
            return j.get("text", "")
        except Exception as e:
            raise RuntimeError(f"student error: {e}")

async def call_teacher_generate_one(context: str) -> Dict[str, Any]:
    """Call teacher to generate one token (or short sequence)."""
    async with aiohttp.ClientSession() as session:
        resp = await session.post(TEACHER_GENERATE_URL, json={"context": context}, timeout=30)
        if resp.status != 200:
            raise RuntimeError("teacher generate error")
        return await resp.json()

async def enqueue_for_scoring(context: str, draft: str) -> Dict[str, Any]:
    """Place a single scoring job into the batcher and wait for result."""
    job_id = str(uuid.uuid4())
    fut = asyncio.get_event_loop().create_future()
    job = {"job_id": job_id, "context": context, "draft": draft}
    async with _BATCH_LOCK:
        _BATCH_QUEUE.append(job)
        _PENDING_FUTURES[job_id] = fut
        # If queue large, leave batching to background worker; worker wakes periodically
    # wait for future
    try:
        result = await asyncio.wait_for(fut, timeout=10.0)
        return result
    except asyncio.TimeoutError:
        # remove future
        async with _BATCH_LOCK:
            if job_id in _PENDING_FUTURES:
                _PENDING_FUTURES.pop(job_id, None)
        raise RuntimeError("scoring timeout")

async def _batch_worker():
    """Background worker that batches scoring requests every BATCH_INTERVAL_MS."""
    while True:
        await asyncio.sleep(BATCH_INTERVAL_MS / 1000.0)
        # snapshot queue
        async with _BATCH_LOCK:
            if not _BATCH_QUEUE:
                continue
            batch = _BATCH_QUEUE[:BATCH_MAX]
            del _BATCH_QUEUE[:len(batch)]
        # prepare payload
        payload = [{"job_id": j["job_id"], "context": j["context"], "draft": j["draft"]} for j in batch]
        # call teacher score batch
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(TEACHER_SCORE_URL, json=payload, timeout=30)
                if resp.status != 200:
                    # fail all futures
                    text = await resp.text()
                    for j in batch:
                        fut = _PENDING_FUTURES.pop(j["job_id"], None)
                        if fut and not fut.done():
                            fut.set_exception(RuntimeError(f"teacher score batch failed: {text}"))
                    continue
                results = await resp.json()
                # results should be list of same length with {"job_id":..., "token_scores":[...], "tokens":[...]}
                idx_map = {r["job_id"]: r for r in results}
                for j in batch:
                    fut = _PENDING_FUTURES.pop(j["job_id"], None)
                    r = idx_map.get(j["job_id"])
                    if fut and not fut.done():
                        if r is None:
                            fut.set_exception(RuntimeError("missing result for job"))
                        else:
                            fut.set_result(r)
        except Exception as e:
            for j in batch:
                fut = _PENDING_FUTURES.pop(j["job_id"], None)
                if fut and not fut.done():
                    fut.set_exception(RuntimeError(f"batch error: {e}"))

def acceptance_prefix_len(scores: List[float], threshold: float) -> int:
    """
    Determine longest prefix where each token score >= threshold.
    scores: list of token logprobs or normalized scores
    threshold: acceptance threshold (interpretation depends on teacher scoring)
    """
    cnt = 0
    for s in scores:
        if s >= threshold:
            cnt += 1
        else:
            break
    return cnt

@app.on_event("startup")
async def startup_event():
    # start prometheus metrics server on METRICS_PORT
    start_http_server(METRICS_PORT)
    # start batch worker
    loop = asyncio.get_event_loop()
    loop.create_task(_batch_worker())

@app.post("/speculative/infer")
async def speculative_infer(req: InferRequest):
    MET_REQS.inc()
    # basic DLP / safety: block immediate if PII
    if contains_pii(req.prompt):
        raise HTTPException(status_code=400, detail="Prompt contains PII; blocked")
    # conservative: disallow speculative decoding for high risk
    if req.risk_level == "high_risk":
        raise HTTPException(status_code=403, detail="Speculative decoding disabled for high risk requests")

    context = req.prompt
    max_tokens = min(req.max_tokens, MAX_SPEC_PER_REQ)
    out_tokens: List[str] = []
    teacher_steps_saved = 0
    tokens_proposed = 0
    fell_back = False

    # speculative loop
    while len(out_tokens) < max_tokens:
        # 1) Student proposes up to chunk length
        try:
            draft = await call_student(context, max_len=MAX_SPEC_CHUNK_LEN)
        except Exception:
            # student error -> fallback to teacher generate one
            t = await call_teacher_generate_one(context)
            out_tokens.append(t.get("token"))
            context += t.get("token", "")
            fell_back = True
            MET_REQ_FALLBACK.inc()
            continue

        # naive tokenization: assume teacher scoring returns tokens and scores
        # enqueue for batched teacher scoring
        try:
            score_result = await enqueue_for_scoring(context, draft)
        except Exception:
            # scoring failed -> fallback to teacher generate first token
            try:
                t = await call_teacher_generate_one(context)
                out_tokens.append(t.get("token"))
                context += t.get("token", "")
                fell_back = True
                MET_REQ_FALLBACK.inc()
                continue
            except Exception:
                raise HTTPException(status_code=502, detail="Both student and teacher failed")

        tokens = score_result.get("tokens", [])
        scores = score_result.get("token_scores", [])
        # record metrics
        MET_TOKENS_PROPOSED.inc(len(tokens))
        tokens_proposed += len(tokens)

        # determine acceptance
        pref = acceptance_prefix_len(scores, ACCEPT_THRESHOLD)
        if pref > 0:
            accepted = tokens[:pref]
            out_tokens.extend(accepted)
            teacher_steps_saved += pref
            # update context with accepted tokens (assuming concatenation)
            context += "".join(accepted)
            # continue to propose next chunk
            if len(out_tokens) >= max_tokens:
                break
            else:
                continue
        else:
            # no token accepted, fallback to teacher autoreg for next token
            t = await call_teacher_generate_one(context)
            out_tokens.append(t.get("token"))
            context += t.get("token", "")
            fell_back = True
            MET_REQ_FALLBACK.inc()

    # finalize
    MET_ACCEPTED_TOKENS.inc(teacher_steps_saved)
    MET_TOKENS_SAVED.inc(teacher_steps_saved)
    # rough energy saved estimate (placeholder)
    energy_saved_kwh = float(os.environ.get("EST_KWH_PER_TOKEN", "0.000002")) * teacher_steps_saved
    MET_ENERGY_SAVED_KWH.inc(energy_saved_kwh)
    acceptance_rate = (teacher_steps_saved / tokens_proposed) if tokens_proposed > 0 else 0.0
    try:
        MET_ACCEPTANCE_RATE.set(acceptance_rate)
    except Exception:
        pass

    # audit event
    req_id = req.request_id or str(uuid.uuid4())
    ev = {
        "ts": time.time(),
        "request_id": req_id,
        "tenant": req.tenant,
        "tokens_proposed": tokens_proposed,
        "tokens_accepted": teacher_steps_saved,
        "acceptance_rate": acceptance_rate,
        "fell_back": fell_back,
        "prompt_snippet": req.prompt[:200]
    }
    await write_event(ev)
    # Also log to audit logger
    log_event({"user": req.tenant or "unknown", "request_id": req_id, "action": "speculative_infer", "reason": "completed", "model": "student+teacher"}, persist_to_s3=False)
    return {"request_id": req_id, "text": "".join(out_tokens), "tokens_proposed": tokens_proposed, "tokens_accepted": teacher_steps_saved, "acceptance_rate": acceptance_rate}
