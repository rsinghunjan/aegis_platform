"""
Tests for OpaHttpEngine.evaluate_async (aegis_policy/engines/opa_http.py).

Focus: policy evaluation over HTTP must not block the event loop when used
from an async request path. `evaluate_async` should offload the blocking
urllib call to a worker thread, so other coroutines can keep making progress
concurrently with a (simulated) slow OPA round-trip.

Run: pytest tests/test_opa_http_async.py -q
"""
import asyncio
import io
import json
import time
from datetime import datetime, timezone

import aegis_policy.engines.opa_http as opa_http_module
from aegis_policy.contracts import (
    EnvironmentContext,
    PolicyContext,
    PolicyInput,
    PrincipalRole,
    RequestActor,
    RequestMeta,
    TypedResource,
)
from aegis_policy.engines.opa_http import OpaHttpEngine


def _make_policy_input() -> PolicyInput:
    actor = RequestActor(
        principal_id="p1",
        principal_type="user",
        org_id="org1",
        roles=(PrincipalRole(role="EnvDeployer", scope_type="environment", scope_id="env1"),),
        claims={},
    )
    req = RequestMeta(request_id="r1", ts=datetime.now(timezone.utc).isoformat(), action="deploy.request", actor=actor)
    res = TypedResource(type="deployment", id="dep1", attributes={})
    env = EnvironmentContext(id="env1", risk_tier="standard", cloud_targets=(), region_constraints=(), budget={})
    pol = PolicyContext(bundle_sha256="sha", pin_scope_used="environment", mode="central", engine="opa-central-http")
    return PolicyInput(request=req, resource=res, environment=env, policy=pol)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_fake_urlopen(sleep_s: float, allow: bool = True):
    def fake_urlopen(req, timeout=None):
        time.sleep(sleep_s)  # simulate a slow (blocking) OPA HTTP round-trip
        body = json.dumps({"result": {"allow": allow, "reason": "ok"}}).encode("utf-8")
        return _FakeResponse(body)

    return fake_urlopen


def test_evaluate_async_returns_expected_decision(monkeypatch):
    monkeypatch.setattr(opa_http_module, "urlopen", _make_fake_urlopen(0.01, allow=True))
    engine = OpaHttpEngine(engine_name="opa-central-http", endpoint_url="http://opa.example/v1/data", bundle_sha256="sha")
    inp = _make_policy_input()

    record = asyncio.run(engine.evaluate_async(inp))
    assert record.decision.allow is True
    assert record.engine == "opa-central-http"


def test_evaluate_async_does_not_block_event_loop(monkeypatch):
    """
    While a (simulated) slow OPA call is in flight via evaluate_async, a
    concurrently-scheduled coroutine should still be able to make progress
    (i.e. the event loop isn't stalled for the duration of the HTTP call).
    """
    slow_s = 0.3
    monkeypatch.setattr(opa_http_module, "urlopen", _make_fake_urlopen(slow_s, allow=True))
    engine = OpaHttpEngine(engine_name="opa-central-http", endpoint_url="http://opa.example/v1/data", bundle_sha256="sha")
    inp = _make_policy_input()

    async def scenario():
        ticks = []

        async def ticker():
            for _ in range(10):
                await asyncio.sleep(0.02)
                ticks.append(time.time())

        start = time.time()
        _, _ = await asyncio.gather(engine.evaluate_async(inp), ticker())
        total = time.time() - start
        return ticks, total

    ticks, total = asyncio.run(scenario())
    # The ticker should have made progress *during* the blocking OPA call,
    # proving the event loop wasn't stalled; if evaluate() were awaited
    # directly (blocking), the ticker would only run after it completed.
    assert len(ticks) >= 5
    # Overall wall time should be close to the slower of the two tasks
    # (evaluate_async ~slow_s, ticker ~0.2s) rather than their sum.
    assert total < slow_s + 0.2
