from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore
from ai_core import AICoreStore, AIMemory, AIObservability
from production import create_app
from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow
from services.embeddings import LocalHashEmbeddingProvider
from services.inference import EchoProvider, ModelRouter


def test_persistent_memory_and_telemetry_are_tenant_scoped(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tenant-events.db'}"
    memory_store = AICoreStore(database_url)
    telemetry_store = AICoreStore(database_url)
    memory = AIMemory(store=memory_store)
    telemetry = AIObservability(store=telemetry_store)

    memory.remember("tenant-a", "answer", {"query_sha256": "a"})
    memory.remember("tenant-b", "answer", {"query_sha256": "b"})
    telemetry.record("tenant-a", "answer", "succeeded", citations=1)
    telemetry.record("tenant-b", "answer", "failed")

    assert [event["query_sha256"] for event in memory.recall("tenant-a")] == ["a"]
    assert [event["query_sha256"] for event in memory.recall("tenant-b")] == ["b"]
    assert telemetry.summary("tenant-a")["answers"] == 1
    assert telemetry.summary("tenant-a")["failure_rate"] == 0
    assert telemetry.summary("tenant-b")["answers"] == 1
    assert telemetry.summary("tenant-b")["failure_rate"] == 1


def test_knowledge_isolation_includes_document_deletion(tmp_path):
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'tenant-knowledge.db'}"),
    )
    document = workflow.ingest("tenant-a", "Private knowledge belonging to tenant A.")

    assert workflow.answer("tenant-a", "What private knowledge?")["citations"]
    assert workflow.answer("tenant-b", "What private knowledge?")["citations"] == []
    assert workflow.delete_document("tenant-b", document["document_id"]) is False
    assert workflow.answer("tenant-a", "What private knowledge?")["citations"]


def test_ai_and_agent_routes_do_not_expose_other_tenant_data(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'tenant-agent.db'}"))
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'tenant-api-knowledge.db'}"),
    )

    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True, ai_workflow=workflow)
    ) as client:
        created_run = client.post(
            "/agent/runs",
            json={"tenant_id": "tenant-a", "goal": "Private tenant A goal"},
        )
        run_id = created_run.json()["run_id"]
        indexed = client.post(
            "/ai/knowledge",
            json={"tenant_id": "tenant-a", "document": "Tenant A private material."},
        )
        answer_a = client.post(
            "/ai/answer",
            json={"tenant_id": "tenant-a", "query": "What material?"},
        )
        answer_b = client.post(
            "/ai/answer",
            json={"tenant_id": "tenant-b", "query": "What material?"},
        )
        hidden_run = client.get(
            f"/agent/runs/{run_id}", params={"tenant_id": "tenant-b"}
        )
        insights_a = client.get(
            "/operator/ai/insights", params={"tenant_id": "tenant-a"}
        )
        insights_b = client.get(
            "/operator/ai/insights", params={"tenant_id": "tenant-b"}
        )

    assert created_run.status_code == 200
    assert indexed.status_code == 200
    assert answer_a.json()["citations"][0]["document_id"] == indexed.json()["document_id"]
    assert answer_b.json()["citations"] == []
    assert hidden_run.status_code == 404
    assert insights_a.json()["summary"]["agent_events"] == 1
    assert insights_b.json()["summary"]["agent_events"] == 0
