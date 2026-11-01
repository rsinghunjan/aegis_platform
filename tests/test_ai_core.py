from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore
from ai_core import AICore, AIMemory, AIObservability, AIPolicyEngine
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
