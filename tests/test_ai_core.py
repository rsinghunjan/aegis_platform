from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore
from agentic.worker import AgentPrincipal
from ai_core import AICore, AICoreStore, AIMemory, AIObservability, AIPolicyEngine
from production import create_app
from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow
from services.embeddings import LocalHashEmbeddingProvider
from services.inference import EchoProvider, ModelRouter


def _workflow(tmp_path, providers=None):
    return AIWorkflow(
        inference_router=ModelRouter(providers or [EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'ai.db'}"),
    )


def test_memory_is_tenant_scoped_and_bounded():
    memory = AIMemory(max_tenants=2, max_events_per_tenant=2)
    for index in range(3):
        memory.remember("a", "answer", {"n": index})
    memory.remember("b", "answer", {})
    memory.remember("c", "answer", {})
    assert memory.recall("a") == []
    assert len(memory.recall("c")) == 1
    memory.remember("c", "answer", {})
    memory.remember("c", "answer", {})
    assert len(memory.recall("c")) == 2


def test_core_records_memory_and_observability(tmp_path):
    core = AICore(_workflow(tmp_path))
    core.ingest("t", "Aegis governs AI workflows.")
    first = core.answer("t", "What does Aegis govern?")
    second = core.answer("t", "What does Aegis govern?")
    assert first["repeated_query"] is False
    assert second["repeated_query"] is True
    assert core.insights("t")["summary"]["answers"] == 2
    assert core.insights("other")["summary"]["answers"] == 0
    assert "What does Aegis" not in str(core.memory.recall("t"))


def test_recommendations_and_policy_suggestions_are_advisory():
    obs = AIObservability()
    for _ in range(6):
        obs.record("t", "answer", "failed")
    assert obs.recommendations("t")[0]["code"] == "high_failure_rate"
    suggestion = AIPolicyEngine(obs).suggest("t")[0]
    assert suggestion["requires_human_approval"] is True
    assert AIPolicyEngine(obs).suggest("quiet") == []


def test_insights_route_is_authorized(tmp_path):
    def authorize(_request, tenant_id, action, _actor):
        return tenant_id == "tenant-a"

    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'rt.db'}"))
    with TestClient(
        create_app(runtime, tenant_authorizer=authorize, ai_workflow=_workflow(tmp_path))
    ) as client:
        client.post(
            "/ai/knowledge", json={"tenant_id": "tenant-a", "document": "Aegis is AI."}
        )
        client.post("/ai/answer", json={"tenant_id": "tenant-a", "query": "What?"})
        ok = client.get("/operator/ai/insights", params={"tenant_id": "tenant-a"})
        denied = client.get("/operator/ai/insights", params={"tenant_id": "tenant-b"})
    assert ok.status_code == 200 and ok.json()["summary"]["answers"] == 1
    assert denied.status_code in (401, 403)


def test_agent_routes_feed_tenant_scoped_ai_insights(tmp_path):
    def authorize(_request, tenant_id, _action, _actor):
        return tenant_id == "tenant-a"

    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'rt.db'}"))
    with TestClient(
        create_app(
            runtime,
            tenant_authorizer=authorize,
            principal_resolver=lambda _request: AgentPrincipal(
                principal_id="operator"
            ),
            execution_dispatcher=lambda _message: "dispatch-1",
            ai_workflow=_workflow(tmp_path),
        )
    ) as client:
        created = client.post(
            "/agent/runs",
            json={"tenant_id": "tenant-a", "goal": "Summarize platform status."},
        )
        run_id = created.json()["run_id"]
        queued = client.post(
            f"/agent/runs/{run_id}/execute",
            json={"tenant_id": "tenant-a"},
        )
        feedback = client.post(
            f"/agent/runs/{run_id}/feedback",
            json={"tenant_id": "tenant-a", "rating": 5},
        )
        insights = client.get(
            "/operator/ai/insights", params={"tenant_id": "tenant-a"}
        )
        other_tenant = client.get(
            "/operator/ai/insights", params={"tenant_id": "tenant-b"}
        )

    assert created.status_code == 200
    assert queued.status_code == 200
    assert feedback.status_code == 200
    assert insights.json()["summary"]["agent_events"] == 3
    assert insights.json()["summary"]["agent_failures"] == 0
    assert other_tenant.status_code == 403


def _store(tmp_path, **kwargs):
    return AICoreStore(f"sqlite:///{tmp_path / 'core.db'}", **kwargs)


def test_state_survives_restart_and_is_shared_between_instances(tmp_path):
    first = AICore(
        _workflow(tmp_path),
        memory=AIMemory(store=_store(tmp_path)),
        observability=AIObservability(store=_store(tmp_path)),
    )
    first.ingest("t", "Aegis governs AI workflows.")
    first.answer("t", "What does Aegis govern?")
    first.record_agent_event("t", "run-1", "create", "created")

    restarted = AICore(
        _workflow(tmp_path),
        memory=AIMemory(store=_store(tmp_path)),
        observability=AIObservability(store=_store(tmp_path)),
    )
    assert restarted.answer("t", "What does Aegis govern?")["repeated_query"] is True
    summary = restarted.insights("t")["summary"]
    assert summary["answers"] == 2
    assert summary["agent_events"] == 1
    assert restarted.insights("other")["summary"]["answers"] == 0


def test_store_retention_is_bounded_per_tenant_and_kind(tmp_path):
    store = _store(tmp_path, max_events_per_tenant_kind=3)
    for index in range(10):
        store.append("a", "mem:answer", {"n": index})
    store.append("b", "mem:answer", {"n": 99})
    assert [e["n"] for e in store.recent("a")] == [7, 8, 9]
    assert [e["n"] for e in store.recent("b")] == [99]
    store.forget("a")
    assert store.recent("a") == []


def test_store_outage_does_not_fail_requests(tmp_path):
    class BrokenStore:
        def append(self, *args, **kwargs):
            raise RuntimeError("db down")

        def recent(self, *args, **kwargs):
            raise RuntimeError("db down")

    core = AICore(
        _workflow(tmp_path),
        memory=AIMemory(store=BrokenStore()),
        observability=AIObservability(store=BrokenStore()),
    )
    core.ingest("t", "Aegis governs AI workflows.")
    assert core.answer("t", "What does Aegis govern?")["repeated_query"] is False
    assert core.answer("t", "What does Aegis govern?")["repeated_query"] is True
    assert core.insights("t")["summary"]["answers"] == 2


def test_persisted_events_contain_no_raw_text(tmp_path):
    store = _store(tmp_path)
    core = AICore(_workflow(tmp_path), memory=AIMemory(store=store))
    core.ingest("t", "Aegis governs AI workflows.")
    core.answer("t", "SECRET-QUERY-TEXT")
    core.record_agent_event("t", "SECRET-RUN-ID", "create", "created")
    dump = str(store.recent("t"))
    assert "SECRET" not in dump
