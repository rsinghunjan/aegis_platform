"""
Enhanced Redis-backed rate limiter with:
 - tenant quota caching (Redis TTL)
 - atomic token-bucket + daily quota check via Lua
 - support for Redis HA configurations (via api.redis_client.get_redis())
 - middleware wrapper compatible with FastAPI app

Environment variables:
 - RATE_LIMIT_REDIS_URL (fallback if no sentinel/cluster set)
 - RATE_LIMIT_REDIS_SENTINELS, RATE_LIMIT_REDIS_SENTINEL_MASTER
 - RATE_LIMIT_REDIS_CLUSTER_NODES
 - DEFAULT_RATE_LIMIT_PER_MIN (int)
 - RATE_LIMIT_BURST (int)
 - TENANT_QUOTA_CACHE_TTL (seconds)
 - DAILY_QUOTA_KEY_TTL (seconds, default 86400)
"""
import os
import time
import json
import logging
import asyncio
from typing import Optional

from fastapi import HTTPException, status, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from api.redis_client import get_redis
from api.db import SessionLocal
from api.models import TenantQuota

logger = logging.getLogger("aegis.rate_limiter")

DEFAULT_RATE = int(os.environ.get("DEFAULT_RATE_LIMIT_PER_MIN", "120"))
DEFAULT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "60"))
TENANT_QUOTA_CACHE_TTL = int(os.environ.get("TENANT_QUOTA_CACHE_TTL", "30"))  # seconds
DAILY_QUOTA_KEY_TTL = int(os.environ.get("DAILY_QUOTA_KEY_TTL", "86400"))  # 24h by default

# Lua script that atomically refills token bucket and checks/updates daily quota counter.
# KEYS: tokens_key, ts_key, daily_key, tenant_quota_cache_key (unused in lua but kept for extensibility)
# ARGV: now, rate_per_min, cap, tokens_req, daily_quota (0 means unlimited), day_ttl
_LUA_TOKEN_AND_DAILY = r"""
local tokens_key = KEYS[1]
local ts_key = KEYS[2]
local daily_key = KEYS[3]

local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cap = tonumber(ARGV[3])
local req = tonumber(ARGV[4])
local daily_quota = tonumber(ARGV[5])
local day_ttl = tonumber(ARGV[6])

-- load state
local last_ts = tonumber(redis.call("GET", ts_key) or "0")
local tokens = tonumber(redis.call("GET", tokens_key) or tostring(cap))

-- refill
local elapsed = math.max(0, now - last_ts)
local per_sec = rate / 60.0
local refill = elapsed * per_sec
tokens = math.min(cap, tokens + refill)

if tokens < req then
  -- persist tokens & ts for next attempt
  redis.call("SET", tokens_key, tokens)
  redis.call("SET", ts_key, now)
  return cjson.encode({ok=0, reason="insufficient_tokens", tokens=tokens})
end

-- check daily quota
local daily_count = tonumber(redis.call("GET", daily_key) or "0")
if daily_quota > 0 then
  if (daily_count + req) > daily_quota then
    -- do not consume tokens if daily quota exceeded
    redis.call("SET", tokens_key, tokens)
    redis.call("SET", ts_key, now)
    return cjson.encode({ok=0, reason="daily_quota_exceeded", daily_count=daily_count, daily_quota=daily_quota})
  end
end

-- consume tokens and increment daily counter atomically
tokens = tokens - req
redis.call("SET", tokens_key, tokens)
redis.call("SET", ts_key, now)

if daily_quota > 0 then
  local new_daily = redis.call("INCRBY", daily_key, req)
  redis.call("EXPIRE", daily_key, day_ttl)
  return cjson.encode({ok=1, tokens=tokens, daily_count=new_daily})
else
  -- unlimited daily quota
  return cjson.encode({ok=1, tokens=tokens, daily_count=0})
end
"""


class RateLimitExceeded(HTTPException):
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)


def _fetch_tenant_quota_from_db(tenant_id: str) -> dict:
    """
    Blocking DB lookup, run off the event loop via run_in_executor (see
    `_get_tenant_quota`) since SQLAlchemy's sync Session would otherwise stall
    the async request path for the duration of the query.
    """
    try:
        session = SessionLocal()
        try:
            tq = session.query(TenantQuota).filter_by(tenant_id=tenant_id).one_or_none()
            if tq:
                return {
                    "rate_per_min": int(tq.rate_per_min) if tq.rate_per_min else DEFAULT_RATE,
                    "burst": int(tq.burst) if tq.burst else DEFAULT_BURST,
                    "daily_quota_units": int(tq.daily_quota_units) if tq.daily_quota_units else 0,
                }
            return {"rate_per_min": DEFAULT_RATE, "burst": DEFAULT_BURST, "daily_quota_units": 0}
        finally:
            session.close()
    except Exception:
        logger.exception("failed to load tenant quota from DB for %s", tenant_id)
        return {"rate_per_min": DEFAULT_RATE, "burst": DEFAULT_BURST, "daily_quota_units": 0}


async def _get_tenant_quota(tenant_id: Optional[str]) -> dict:
    """
    Return quota dict: {"rate_per_min": int, "burst": int, "daily_quota_units": int or 0}
    Uses Redis cache (tenant_quota_cache:<tenant>) with TTL to avoid DB on each request.
    Falls back to DB query if missing.
    """
    if not tenant_id:
        return {"rate_per_min": DEFAULT_RATE, "burst": DEFAULT_BURST, "daily_quota_units": 0}

    cache_key = f"tenant_quota_cache:{tenant_id}"
    r = await get_redis()
    if r:
        try:
            cached = await r.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            logger.exception("failed to read tenant quota cache for %s", tenant_id)

    # Cache miss: load from DB without blocking the event loop.
    loop = asyncio.get_event_loop()
    quota = await loop.run_in_executor(None, _fetch_tenant_quota_from_db, tenant_id)

    # cache in Redis
    if r:
        try:
            await r.set(cache_key, json.dumps(quota), ex=TENANT_QUOTA_CACHE_TTL)
        except Exception:
            logger.exception("failed to set tenant quota cache for %s", tenant_id)
    return quota


async def _atomic_consume(tenant_id: Optional[str], route_key: str, tokens: int, rate_per_min: int, burst: int, daily_quota_units: int) -> dict:
    """
    Run the Lua script to consume tokens and increment daily counter.
    Returns dict with ok:1 or 0 and reason.
    """
    r = await get_redis()
    if not r:
        # Redis down: fail-open
        logger.warning("Redis unavailable for rate limiting; failing open")
        return {"ok": 1, "tokens": 0, "note": "redis_unavailable_fail_open"}

    now = int(time.time())
    tokens_key = f"rl:{tenant_id or 'global'}:{route_key}:tokens"
    ts_key = f"rl:{tenant_id or 'global'}:{route_key}:ts"
    daily_key = f"rl:{tenant_id or 'global'}:{route_key}:daily:{time.strftime('%Y%m%d')}"
    try:
        # eval the script
        res = await r.eval(_LUA_TOKEN_AND_DAILY, 3, tokens_key, ts_key, daily_key, now, rate_per_min, rate_per_min + burst, tokens, daily_quota_units, DAILY_QUOTA_KEY_TTL)
        # res is JSON encoded by script
        if isinstance(res, (bytes, str)):
            try:
                parsed = json.loads(res)
                return parsed
            except Exception:
                logger.exception("failed to parse lua response: %s", res)
                return {"ok": 0, "reason": "lua_parse_error"}
        else:
            logger.warning("unexpected lua response type: %s", type(res))
            return {"ok": 0, "reason": "lua_unexpected"}
    except Exception as exc:
        logger.exception("lua eval failed: %s", exc)
        # fallback: allow (fail-open) but log
        return {"ok": 1, "tokens": 0, "note": "lua_error_fail_open"}


async def enforce_rate_limit(tenant_id: Optional[str], route_key: str, tokens: int = 1) -> None:
    """
    Public entrypoint for endpoints. Raises RateLimitExceeded on violation.
    """
    start = time.monotonic()
    quota = await _get_tenant_quota(tenant_id)
    rate = int(quota.get("rate_per_min", DEFAULT_RATE))
    burst = int(quota.get("burst", DEFAULT_BURST))
    daily_quota = int(quota.get("daily_quota_units", 0) or 0)

    res = await _atomic_consume(tenant_id, route_key, tokens, rate, burst, daily_quota)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    # Lightweight observability: surface slow rate-limit decisions (e.g. Redis
    # latency spikes) without adding a metrics dependency to this hot path.
    if elapsed_ms > 50.0:
        logger.warning("rate limit check for tenant=%s route=%s took %.1fms", tenant_id, route_key, elapsed_ms)
    else:
        logger.debug("rate limit check for tenant=%s route=%s took %.1fms", tenant_id, route_key, elapsed_ms)
    if res.get("ok") == 1:
        # allowed
        return
    reason = res.get("reason") or res.get("note") or "rate_limited"
    if reason == "insufficient_tokens":
        raise RateLimitExceeded(f"Rate limit exceeded for tenant {tenant_id or 'anonymous'} on {route_key}")
    if reason == "daily_quota_exceeded":
        raise RateLimitExceeded(f"Daily quota exceeded for tenant {tenant_id or 'anonymous'} on {route_key}")
    # any other reason -> raise generic
    raise RateLimitExceeded(f"Rate limit blocked: {reason}")


# Middleware wrapper that expects request.state.tenant_id set (or extract in middleware)
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts tenant identity from request.state.tenant_id or Authorization header.
    To use, add app.add_middleware(RateLimitMiddleware).
    """

    def __init__(self, app, route_key_getter=None):
        super().__init__(app)
        self.route_key_getter = route_key_getter or (lambda request: f"{request.method}:{request.url.path}")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # attempt to find tenant in request.state (auth dependency should set it)
        tenant = getattr(request.state, "tenant_id", None)

        # fallback: try token payload in request.state (some auth middleware might store it)
        token_payload = getattr(request.state, "_token_payload", None)
        if not tenant and token_payload and isinstance(token_payload, dict):
            tenant = token_payload.get("tenant")

        route_key = self.route_key_getter(request)
        try:
            await enforce_rate_limit(tenant, route_key, tokens=1)
        except RateLimitExceeded as exc:
            return JSONResponse(status_code=429, content={"code": 429, "message": str(exc.detail)})
        return await call_next(request)
