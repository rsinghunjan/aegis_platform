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
 38
 39
 40
 41
 42
 43
 44
 45
# Rate limiting, quotas, usage metering and billing (Aegis)

Overview
--------
This addition implements:
- Redis-backed token-bucket per-tenant rate limiting (enforced via middleware or dependency).
- Tenant quotas persisted in Postgres (`tenant_quotas`).
- Usage metering via `usage_records` table; usage incremented by inference calls and jobs.
- Periodic billing aggregation (Celery task `aegis.aggregate_and_invoice`) that writes `invoices`.
- Billing webhook support to integrate with external billing systems.

How to enable
1. Start Redis and configure RATE_LIMIT_REDIS_URL to point at it.
2. Run alembic migrations (creates tenants, tenant_quotas, usage_records, invoices).
   DATABASE_URL=... alembic upgrade head
3. Seed tenants and quotas (scripts or DB insert).
4. Ensure `api/usage.record_usage()` is invoked in inference endpoints and long-running tasks — the earlier files added record_usage call sites in examples.
5. Start Celery beat to schedule `aegis.aggregate_and_invoice` (e.g., daily).

Integration points
- Enforce call-level rate limits by adding `RateLimitMiddleware` to your app (import and add to app.add_middleware), or call `await enforce_rate_limit(tenant_id, route_key)` in an auth/dependency after resolving tenant.
- Enforce daily quotas by checking TenantQuota.daily_quota_units before allowing an inference. Example: when recording usage, check sum(today) + units <= daily_quota_units.
- Billing: set `BILLING_WEBHOOK_URL` to notify billing systems when invoices are created.

Config & env
- RATE_LIMIT_REDIS_URL (default redis://localhost:6379/0)
- DEFAULT_RATE_LIMIT_PER_MIN (fallback per-tenant)
- RATE_LIMIT_BURST
- BILLING_UNIT_COST (dollars per unit)
- BILLING_WEBHOOK_URL (optional)

Operational notes
- Cache tenant quotas in Redis for high throughput; the implementation reads DB and should be optimized for production.
- Consider using Redis Lua atomic scripts for more advanced quota semantics (daily counters, sliding windows).
- Use separate Redis DB/cluster for rate limiting to isolate broker traffic from rate-limit storage.
- For multi-tenant billing, integrate tenant metadata (billing account ID, payment method) into `tenants` table and extend `billing.aggregate_and_invoice` to push invoices to the real billing provider.
- Provide per-tenant dashboards in Grafana using `usage_records` / Prometheus metrics.

Next steps I can do for you
- Patch api/api_server.py to:
  - add RateLimitMiddleware to the app startup
  - call `record_usage` inside the predict endpoints (I can submit a PR updating the inference handlers)
- Implement Redis-cached tenant quota lookup with TTL to avoid DB hits on every request.
- Add a small admin UI / API for viewing tenant usage and unpaid invoices.
Which would you like me to do next?
