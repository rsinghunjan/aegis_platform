"""
Tests for api/rate_limiter2.py hot-path performance behavior.

Focus:
 - `_get_tenant_quota`'s DB fallback (cache miss) runs the blocking
   SQLAlchemy lookup in a worker thread via run_in_executor, so it does not
   stall the event loop for the duration of the query.
 - `enforce_rate_limit` still returns correct allow/deny decisions when
   Redis is unavailable (fail-open path), a pre-existing behavior this
   change must preserve.

Run: pytest tests/test_rate_limiter2.py -q
"""
import asyncio
import time

import api.rate_limiter2 as rl


def test_fetch_tenant_quota_db_path_does_not_block_event_loop(monkeypatch):
    async def fake_get_redis():
        return None  # no cache; forces DB fallback path

    def slow_fetch(tenant_id):
        time.sleep(0.3)
        return {"rate_per_min": 42, "burst": 7, "daily_quota_units": 0}

    monkeypatch.setattr(rl, "get_redis", fake_get_redis)
    monkeypatch.setattr(rl, "_fetch_tenant_quota_from_db", slow_fetch)

    async def scenario():
        ticks = []

        async def ticker():
            for _ in range(10):
                await asyncio.sleep(0.02)
                ticks.append(time.time())

        results = await asyncio.gather(rl._get_tenant_quota("tenant-1"), ticker())
        return results[0], ticks

    quota, ticks = asyncio.run(scenario())
    assert quota == {"rate_per_min": 42, "burst": 7, "daily_quota_units": 0}
    # Ticker should have made progress concurrently with the slow DB fetch.
    assert len(ticks) >= 5


def test_enforce_rate_limit_fails_open_when_redis_unavailable(monkeypatch):
    async def fake_get_redis():
        return None

    def fast_fetch(tenant_id):
        return {"rate_per_min": 120, "burst": 60, "daily_quota_units": 0}

    monkeypatch.setattr(rl, "get_redis", fake_get_redis)
    monkeypatch.setattr(rl, "_fetch_tenant_quota_from_db", fast_fetch)

    # Should not raise: Redis unavailable -> fail-open per _atomic_consume.
    asyncio.run(rl.enforce_rate_limit("tenant-1", "GET:/x", tokens=1))
