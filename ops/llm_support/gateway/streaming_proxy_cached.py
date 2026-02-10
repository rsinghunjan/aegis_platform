"""
Streaming proxy with prompt/response caching (local LRU, Redis or hybrid).

Environment-driven behavior:
- CACHE_MODE: "local" | "redis" | "hybrid" (default "local")
- LOCAL_CACHE_SIZE, LOCAL_CACHE_TTL
- REDIS_URL
- CACHE_SAFE_SCORE: max safety score allowed to cache (default 0.1)
- CACHE_DP_ALLOW_HITS: "true" if returning cached responses should still attempt to charge DP for token usage (default false)
- TOKENIZER_URL, DP_SERVICE, ORCH_WEBHOOK as earlier implementations

Notes:
- We enforce conservative safety rules for caching:
  - Only deterministic-ish generation ops are cached: temperature <= CACHE_MAX_TEMPERATURE (default 0.0)
  - DP-charged requests are NOT cached unless CACHE_DP_ALLOW_HITS=true (and then DP charge is attempted)
  - Only cache when safety_score <= CACHE_SAFE_SCORE
- In hybrid mode: check local -> redis -> backend; write-through to redis and local on backend miss.
"""
import os
import json
import time
import logging
import asyncio
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from ..cache.cache import LocalCache
from ..cache.redis_cache import RedisCache
from ..cache.key_utils import make_cache_key

LOG = logging.getLogger("aegis.streaming.cache")
logging.basicConfig(level=logging.INFO)

# Config via env
CACHE_MODE = os.environ.get("CACHE_MODE", "local")
LOCAL_CACHE_SIZE = int(os.environ.get("LOCAL_CACHE_SIZE", "1024"))
LOCAL_CACHE_TTL = int(os.environ.get("LOCAL_CACHE_TTL", "3600"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_SAFE_SCORE = float(os.environ.get("CACHE_SAFE_SCORE", "0.1"))
CACHE_DP_ALLOW_HITS = os.environ.get("CACHE_DP_ALLOW_HITS", "false").lower() in ("1", "true")
CACHE_MAX_TEMPERATURE = float(os.environ.get("CACHE_MAX_TEMPERATURE", "0.0"))

LLM_BACKEND_HTTP = os.environ.get("LLM_BACKEND_HTTP", "http://vllm.model-serving.svc.cluster.local:8080/generate_stream")
DP_SERVICE = os.environ.get("DP_SERVICE", "http://dp-service:8084")
ORCH_WEBHOOK = os.environ.get("ORCH_WEBHOOK", "http://aegis-orchestrator.aegis-system.svc.cluster.local:8088/webhook")
TOKENIZER_URL = os.environ.get("TOKENIZER_URL", "http://tokenizer.aegis-system.svc.cluster.local:8090/count")
DP_AUTH = os.environ.get("DP_AUTH_TOKEN", "")

# Metrics
CACHE_HITS = Counter("aegis_cache_hits_total", "Cache hits", ["backend"])
CACHE_MISSES = Counter("aegis_cache_misses_total", "Cache misses", ["backend"])
CACHE_LATENCY = Histogram("aegis_cache_latency_seconds", "Cache get/set latency")
LOCAL_CACHE_SIZE_GAUGE = Gauge("aegis_local_cache_size", "Local cache size")
REDIS_UP = Gauge("aegis_cache_redis_up", "Whether redis is reachable (1/0)")

# Start metrics server (optional; port controlled by METRICS_PORT env var)
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9102"))
try:
    start_http_server(METRICS_PORT)
    LOG.info("Prometheus metrics server started on %s", METRICS_PORT)
except Exception:
    LOG.exception("Failed to start Prometheus metrics HTTP server")

# instantiate caches
_local_cache = LocalCache(maxsize=LOCAL_CACHE_SIZE, ttl=LOCAL_CACHE_TTL)
_redis_cache = None
if CACHE_MODE in ("redis", "hybrid"):
    try:
        _redis_cache = RedisCache(url=REDIS_URL)
        REDIS_UP.set(1)
    except Exception:
        LOG.exception("Failed to init redis cache")
        REDIS_UP.set(0)

app = FastAPI(title="Aegis Streaming Proxy with Cache")


async def count_tokens(text: str, model: str = "default") -> int:
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.post(TOKENIZER_URL, json={"text": text, "model": model})
            r.raise_for_status()
            return int(r.json().get("tokens", 0))
    except Exception:
        LOG.exception("tokenizer call failed; falling back")
        return len(text.split())


async def charge_dp(client_id: str, tokens: int) -> bool:
    if not client_id:
        return False
    headers = {"Authorization": f"Bearer {DP_AUTH}"} if DP_AUTH else {}
    async with httpx.AsyncClient(timeout=3.0) as c:
        try:
            r = await c.post(f"{DP_SERVICE}/charge", json={"client": client_id, "tokens": tokens}, headers=headers)
            return r.status_code == 200
        except Exception:
            LOG.exception("dp charge failed")
            return False


def cacheable_request(body: dict) -> bool:
    """
    Conservative predicate: only cache if temperature <= CACHE_MAX_TEMPERATURE
    and request is marked cacheable or has deterministic params.
    """
    params = body.get("payload", {}).get("params", {}) or {}
    temp = float(params.get("temperature", params.get("temp", 0.0)))
    dp_required = body.get("payload", {}).get("dp_required", False)
    # Do not cache DP-required requests unless explicitly allowed
    if dp_required and not CACHE_DP_ALLOW_HITS:
        return False
    # Only cache deterministic / low-temp requests
    if temp > CACHE_MAX_TEMPERATURE:
        return False
    return True


async def backend_stream(payload: dict) -> AsyncIterator[dict]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", LLM_BACKEND_HTTP, json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    yield {"text": line}


@app.post("/generate")
async def generate(req: Request):
    """
    Entrypoint for generation with caching.

    On cache hit: return cached response (optionally charge DP).
    On miss: stream from backend, accumulate full response, evaluate safety_score and possibly cache final response (only full completions).
    """
    start = time.time()
    body = await req.json()
    client_id = body.get("client_id")
    model = body.get("model", "default")
    model_version = body.get("model_version", "unknown")
    system_prompt = body.get("system_prompt", "")
    prompt_text = body.get("payload", {}).get("inputs", "")
    retrieval_meta = body.get("payload", {}).get("retrieval_meta")
    params = body.get("payload", {}).get("params", {})

    key = make_cache_key(model, model_version, system_prompt, prompt_text, retrieval_meta, params)

    # Attempt cache read according to mode
    # Local-first in hybrid
    if CACHE_MODE in ("local", "hybrid"):
        with CACHE_LATENCY.time():
            v = _local_cache.get(key)
        if v is not None:
            CACHE_HITS.labels(backend="local").inc()
            LOCAL_CACHE_SIZE_GAUGE.set(len(_local_cache))
            # If DP required and allowed, attempt DP charge
            if body.get("payload", {}).get("dp_required", False) and CACHE_DP_ALLOW_HITS:
                tokens = await count_tokens(v.get("response",""))
                ok = await charge_dp(client_id, tokens)
                if not ok:
                    # fail-closed: treat as cache miss (or block). We'll choose block for safety.
                    await notify_orchestrator({"id": f"dp-cache-{int(time.time())}", "model_env": body.get("model_env","prod"), "dp_over_budget": True})
                    return Response(content=json.dumps({"status":"blocked","reason":"dp_over_budget"}), media_type="application/json")
            return Response(content=json.dumps({"status":"ok","cached": True,"response": v["response"], "meta": v.get("meta", {})}), media_type="application/json")

    if CACHE_MODE in ("redis", "hybrid") and _redis_cache:
        with CACHE_LATENCY.time():
            v = _redis_cache.get(key)
        if v:
            CACHE_HITS.labels(backend="redis").inc()
            # populate local cache for hybrid
            if CACHE_MODE == "hybrid":
                _local_cache.set(key, v)
                LOCAL_CACHE_SIZE_GAUGE.set(len(_local_cache))
            # DP handling as above
            if body.get("payload", {}).get("dp_required", False) and CACHE_DP_ALLOW_HITS:
                tokens = await count_tokens(v.get("response",""))
                ok = await charge_dp(client_id, tokens)
                if not ok:
                    await notify_orchestrator({"id": f"dp-cache-{int(time.time())}", "model_env": body.get("model_env","prod"), "dp_over_budget": True})
                    return Response(content=json.dumps({"status":"blocked","reason":"dp_over_budget"}), media_type="application/json")
            return Response(content=json.dumps({"status":"ok","cached": True,"response": v["response"], "meta": v.get("meta", {})}), media_type="application/json")
        else:
            CACHE_MISSES.labels(backend="redis").inc()

    # Cache miss: stream from backend and accumulate full response
    CACHE_MISSES.labels(backend="backend").inc()
    final_chunks = []
    safety_score_max = 0.0
    async for chunk in backend_stream(body):
        # each chunk might be {"text": "..."} or {"text":"...", "score":0.01}
        text = chunk.get("text", "")
        score = float(chunk.get("score", 0.0))
        if score > safety_score_max:
            safety_score_max = score
        # stream chunk to client as incremental NDJSON for low-latency UX
        yield_chunk = json.dumps({"status": "ok", "text": text}) + "\n"
        final_chunks.append(text)
        yield yield_chunk

    # After backend completed, decide caching
    full_response = "".join(final_chunks)
    # only cache if request is cacheable and safety score below threshold
    if cacheable_request(body) and safety_score_max <= CACHE_SAFE_SCORE:
        entry = {"response": full_response, "meta": {"model": model, "model_version": model_version, "cached_at": int(time.time())}}
        # write to redis and local depending on mode
        if CACHE_MODE in ("redis", "hybrid") and _redis_cache:
            try:
                _redis_cache.set(key, entry, ttl=LOCAL_CACHE_TTL)
            except Exception:
                LOG.exception("redis set failed on caching")
        if CACHE_MODE in ("local", "hybrid"):
            _local_cache.set(key, entry)
            LOCAL_CACHE_SIZE_GAUGE.set(len(_local_cache))

    elapsed = time.time() - start
    CACHE_LATENCY.observe(elapsed)
