129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
Redis-backed token-bucket rate limiter and quota enforcement.
    # Determine configured rate for tenant. For simplicity, we fetch per-tenant limit via DB sync call.
    # Avoid blocking DB call per-request at scale; in prod cache tenant limits in Redis or in-memory with TTL.
    from api.db import SessionLocal
    from api.models import TenantQuota

    # default global limit
    rate = DEFAULT_RATE_LIMIT_PER_MIN
    burst = RATE_LIMIT_BURST

    if tenant_id:
        try:
            session = SessionLocal()
            tq = session.query(TenantQuota).filter_by(tenant_id=tenant_id).one_or_none()
            if tq:
                rate = int(tq.rate_per_min or rate)
                burst = int(tq.burst or burst)
        except Exception:
            logger.exception("failed to load tenant quota for %s", tenant_id)
        finally:
            try:
                session.close()
            except Exception:
                pass

    key = f"{tenant_id or 'global'}:{route_key}"
    ok = await _token_bucket_consume(key, rate, burst, tokens=tokens)
    if not ok:
        raise RateLimitExceeded(f"Rate limit exceeded for tenant {tenant_id or 'anonymous'} on {route_key}")


# FastAPI middleware wrapper (optional)
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts tenant identity from request state or headers and enforces rate limit.
    To use, set request.state.tenant_id earlier (for example via auth dependency).
    """

    def __init__(self, app, route_key_getter=None):
        super().__init__(app)
        # route_key_getter: function(request) -> route_key (string)
        self.route_key_getter = route_key_getter or (lambda request: f"{request.method}:{request.url.path}")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        tenant_id = getattr(request.state, "tenant_id", None)
        route_key = self.route_key_getter(request)
        try:
            await enforce_rate_limit(tenant_id, route_key, tokens=1)
        except RateLimitExceeded as exc:
            return JSONResponse(status_code=429, content={"code": 429, "message": str(exc.detail)})
        return await call_next(request)
api/rate_limiter.py
