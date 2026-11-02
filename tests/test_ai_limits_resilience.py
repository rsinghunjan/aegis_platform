from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore
from ai_core import AICore, AIQuotaExceeded, AIQuotaGuard
from production import create_app
from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow
from services.embeddings import LocalHashEmbeddingProvider
from services.inference import EchoProvider, InferenceRequest, ModelRouter
from services.inference.providers import InferenceProvider, ProviderError
from services.inference.router import NoProviderAvailableError


def _workflow(tmp_path, providers=None):
    workflow = AIWorkflow(
        inference_router=ModelRouter(providers or [EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'ai.db'}"),
    )
    workflow.input_cost_per_1k = 1.0
    workflow.output_cost_per_1k = 1.0
    return workflow


def _app(tmp_path, workflow, authorize=lambda *a: True):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'rt.db'}"))
    return create_app(runtime, tenant_authorizer=authorize, ai_workflow=workflow)


def test_rate_limit_returns_429_with_retry_after_and_is_tenant_scoped(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AEGIS_AI_TENANT_REQUESTS_PER_MINUTE", "2")
    with TestClient(_app(tmp_path, _workflow(tmp_path))) as client:
        statuses = [
            client.post(
                "/ai/answer", json={"tenant_id": "a", "query": f"q{index}"}
            ).status_code
            for index in range(3)
        ]
        limited = client.post("/ai/answer", json={"tenant_id": "a", "query": "q"})
        other = client.post("/ai/answer", json={"tenant_id": "b", "query": "q"})
        insights = client.get("/operator/ai/insights", params={"tenant_id": "a"})
    assert statuses == [200, 200, 429]
    assert limited.status_code == 429
    assert limited.json() == {"detail": "AI usage limit exceeded"}
    assert int(limited.headers["Retry-After"]) >= 1
    assert other.status_code == 200
    assert insights.json()["summary"]["quota_rejections"] == 2


def test_daily_budget_blocks_before_provider_call(tmp_path, monkeypatch):
    class CountingProvider(EchoProvider):
        calls = 0

        def generate(self, request):
            type(self).calls += 1
            return super().generate(request)

    monkeypatch.setenv("AEGIS_AI_TENANT_DAILY_BUDGET_USD", "0.01")
    workflow = _workflow(tmp_path, [CountingProvider()])
    core = AICore.durable(workflow)
    core.answer("t", "hello there", max_tokens=0 + 1)
    calls_after_first = CountingProvider.calls
    with pytest.raises(AIQuotaExceeded) as excinfo:
        core.answer("t", "again", max_tokens=1)
    assert excinfo.value.kind == "budget"
    assert CountingProvider.calls == calls_after_first
    core.answer("other", "hello there", max_tokens=1)


def test_projected_cost_blocks_large_requests(tmp_path):
    workflow = _workflow(tmp_path)
    guard = AIQuotaGuard(workflow.data_store, daily_budget_usd=0.5, output_cost_per_1k=1.0)
    with pytest.raises(AIQuotaExceeded):
        guard.check("t", max_tokens=1000)
    guard.check("t", max_tokens=100)


def test_window_excludes_old_usage(tmp_path):
    workflow = _workflow(tmp_path)
    core = AICore(workflow)
    core.answer("t", "hello")
    guard = AIQuotaGuard(workflow.data_store, requests_per_minute=1)
    with pytest.raises(AIQuotaExceeded):
        guard.check("t")
    guard.check("t", now=datetime.now(timezone.utc) + timedelta(minutes=2))


def test_limits_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AEGIS_AI_TENANT_DAILY_BUDGET_USD", raising=False)
    monkeypatch.delenv("AEGIS_AI_TENANT_REQUESTS_PER_MINUTE", raising=False)
    assert AIQuotaGuard.from_env(_workflow(tmp_path)) is None


def test_invalid_limit_config_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_AI_TENANT_REQUESTS_PER_MINUTE", "-1")
    with pytest.raises(ValueError):
        AIQuotaGuard.from_env(_workflow(tmp_path))


class Flaky(InferenceProvider):
    name = "flaky"

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError("boom")
        return EchoProvider().generate(request)


class Down(InferenceProvider):
    name = "down"

    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        raise ProviderError("down")


def test_router_retries_with_exponential_backoff():
    sleeps = []
    provider = Flaky(failures=2)
    router = ModelRouter([provider], max_retries=2, backoff_seconds=0.5, sleep=sleeps.append)
    assert router.generate(InferenceRequest(prompt="hi")).text
    assert provider.calls == 3
    assert sleeps == [0.5, 1.0]


def test_router_opens_circuit_then_probes_after_cooldown():
    now = [0.0]
    down = Down()
    router = ModelRouter(
        [down, EchoProvider()],
        failure_threshold=2,
        cooldown_seconds=30,
        clock=lambda: now[0],
    )
    for _ in range(2):
        router.generate(InferenceRequest(prompt="hi"))
    assert down.calls == 2
    router.generate(InferenceRequest(prompt="hi"))
    assert down.calls == 2
    assert router.last_attempts[0].error == "circuit_open"
    now[0] = 31.0
    router.generate(InferenceRequest(prompt="hi"))
    assert down.calls == 3


def test_router_all_circuits_open_raises_no_provider():
    now = [0.0]
    router = ModelRouter([Down()], failure_threshold=1, clock=lambda: now[0])
    with pytest.raises(NoProviderAvailableError):
        router.generate(InferenceRequest(prompt="hi"))
    with pytest.raises(NoProviderAvailableError) as excinfo:
        router.generate(InferenceRequest(prompt="hi"))
    assert excinfo.value.attempts[0].error == "circuit_open"


def test_provider_outage_degrades_to_fallback_via_api(tmp_path):
    workflow = _workflow(tmp_path, [Down(), EchoProvider()])
    with TestClient(_app(tmp_path, workflow)) as client:
        response = client.post("/ai/answer", json={"tenant_id": "a", "query": "q"})
    assert response.status_code == 200
    assert response.json()["provider"] == "local-echo"
