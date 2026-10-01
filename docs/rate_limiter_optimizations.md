  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
# Rate limiter optimizations — summary & operational guidance

What I changed
- Tenant quota caching:
  - per-tenant quota is cached in Redis (tenant_quota_cache:<tenant>) for TENANT_QUOTA_CACHE_TTL seconds to reduce DB load.
  - fallback to DB when cache miss; caches result back to Redis.
- Atomic token + daily-quota operation:
  - a Lua script atomically refills token bucket, consumes tokens, and increments a daily usage counter.
  - daily counter key uses the UTC date (YYYYMMDD) and an expiry so daily reset is atomic and predictable.
- Redis HA support:
  - redis client helper supports Cluster, Sentinel and single-node URLs.
  - when Redis is unavailable we fail-open (allow requests) but log warnings; you may change to fail-closed if desired.
- Middleware compatibility:
  - RateLimitMiddleware uses request.state.tenant_id or token payload injected by auth to enforce per-tenant limits.

Recommended runtime configuration
- Deploy Redis in HA mode (managed or Redis Cluster). Configure RATE_LIMIT_REDIS_CLUSTER_NODES or RATE_LIMIT_REDIS_SENTINELS accordingly.
- Set TENANT_QUOTA_CACHE_TTL to a small value (e.g., 10–60s) to balance reactivity and DB load.
- Instrument the code with Prometheus metrics:
  - redis_down_events, rate_limiter_rejections, daily_quota_hits, token_bucket_consumed, tenant_quota_cache_hits/misses.

Scaling notes & improvements
- For very high QPS, consider pushing tenant quota cache into local process memory with TTL plus background refresh to avoid hop latency.
- For sliding windows or more complex billing windows, implement a rolling-window counter (Redis sorted sets or time-bucket keys).
- Use Lua scripts & Redis atomic ops for any cross-key checks to avoid race conditions.
- Consider storing tenant quota metadata in a fast key-value store (Redis) as the source of truth for runtime enforcement, and backfill DB asynchronously to keep operational performance.

Next steps I can implement
- Add Prometheus metrics emission in the limiter and a small Grafana panel.
- Patch api/api_server.py to set request.state.tenant_id from the JWT token (so middleware works out-of-the-box) and attach the middleware.
- Add a short integration test that validates atomic daily quota increments (requires Redis running).

If you want I will:
- apply the middleware wiring change to api/api_server.py,
- add the Prometheus metrics counters and a dashboard snippet,
- and add an integration test verifying daily quota enforcement using the integration docker-compose Redis.
Which would you like me to do next?
docs/rate_limiter_optimizations.md
