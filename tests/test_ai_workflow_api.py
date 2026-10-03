from fastapi.testclient import TestClient
import pytest

from agentic.runtime import AgentRuntime, AgentStore
from production import create_app
from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow, AIWorkflowError
from services.embeddings import LocalHashEmbeddingProvider
from services.inference import EchoProvider, ModelRouter


def test_ai_workflow_routes_are_tenant_authorized_and_retrieve_tenant_docs(tmp_path):
    authorizations = []

    def authorize(_request, tenant_id, action, _actor):
        authorizations.append((tenant_id, action))
        return tenant_id == "tenant-a"

    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'workflow-ai.db'}"),
    )
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'workflow.db'}"))
    with TestClient(
        create_app(runtime, tenant_authorizer=authorize, ai_workflow=workflow)
    ) as client:
        indexed = client.post(
            "/ai/knowledge",
            json={
                "tenant_id": "tenant-a",
                "document": "Aegis combines retrieval with governed AI workflows.",
            },
        )
        denied = client.post(
            "/ai/answer",
            json={"tenant_id": "tenant-b", "query": "What does Aegis combine?"},
        )
        answered = client.post(
            "/ai/answer",
            json={"tenant_id": "tenant-a", "query": "What does Aegis combine?"},
        )

    assert indexed.status_code == 200
    assert indexed.json()["chunks_indexed"] == 1
    assert denied.status_code == 403
    assert answered.status_code == 200
    assert "Aegis combines retrieval" in answered.json()["answer"]
    assert answered.json()["citations"][0]["document_id"] == indexed.json()["document_id"]
    assert ("tenant-a", "ai_knowledge_write") in authorizations
    assert ("tenant-b", "ai_generate") in authorizations


def test_ai_workflow_rejects_oversized_document(tmp_path):
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'limits-ai.db'}"),
    )
    with pytest.raises(ValueError, match="maximum length"):
        workflow.ingest("tenant-a", "x" * (workflow.max_document_chars + 1))
    for _ in range(workflow.max_indexed_chars_per_tenant // workflow.max_document_chars):
        workflow.ingest("tenant-a", "x" * workflow.max_document_chars)
    with pytest.raises(ValueError, match="indexed text limit"):
        workflow.ingest("tenant-a", "x")


def test_ai_workflow_keeps_tenant_indexes_isolated(tmp_path):
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'isolation-ai.db'}"),
    )
    workflow.ingest("tenant-a", "Private tenant A knowledge.")
    response = workflow.answer("tenant-b", "What is tenant A knowledge?")
    assert response["citations"] == []


def test_ai_workflow_persists_tenant_documents_and_metered_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_LLM_INPUT_COST_PER_1K", "2")
    monkeypatch.setenv("AEGIS_LLM_OUTPUT_COST_PER_1K", "4")
    database_url = f"sqlite:///{tmp_path / 'ai-data.db'}"
    first = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(database_url),
    )
    indexed = first.ingest("tenant-a", "Aegis stores durable private knowledge.")
    response = first.answer("tenant-a", "What does Aegis store?")
    assert response["cost_usd"] > 0
    assert response["request_id"]

    restarted = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(database_url),
    )
    answer_after_restart = restarted.answer("tenant-a", "What does Aegis store?")
    assert answer_after_restart["citations"][0]["document_id"] == indexed["document_id"]
    assert restarted.answer("tenant-b", "What does tenant A store?")["citations"] == []
    assert restarted.delete_document("tenant-b", indexed["document_id"]) is False
    assert restarted.answer("tenant-a", "What does Aegis store?")["citations"]
    assert restarted.delete_document("tenant-a", indexed["document_id"]) is True
    assert restarted.answer("tenant-a", "What does Aegis store?")["citations"] == []
    usage = restarted.list_usage("tenant-a")
    assert len(usage) == 4
    assert all(record["status"] == "succeeded" for record in usage)
    assert all("request_sha256" in record for record in usage)
    assert all("prompt" not in record and "answer" not in record for record in usage)


def test_ai_workflow_persists_failed_inference_audit(tmp_path):
    class FailedProvider(EchoProvider):
        name = "failed-test"

        def generate(self, _request):
            raise RuntimeError("provider failure")

    workflow = AIWorkflow(
        inference_router=ModelRouter([FailedProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'failed-ai.db'}"),
    )
    with pytest.raises(AIWorkflowError, match="AI workflow provider is unavailable"):
        workflow.answer("tenant-a", "A query with private content")
    usage = workflow.list_usage("tenant-a")
    assert len(usage) == 1
    assert usage[0]["status"] == "failed"
    assert usage[0]["error_type"] == "NoProviderAvailableError"
    assert "private content" not in repr(usage[0])


def test_operator_ai_usage_is_tenant_authorized_and_filtered(tmp_path):
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'usage-api.db'}"),
    )
    workflow.answer("tenant-a", "hello")
    workflow.answer("tenant-b", "world")
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'usage-agent.db'}"))
    with TestClient(
        create_app(
            runtime,
            tenant_authorizer=lambda _request, tenant_id, action, _actor: (
                tenant_id == "tenant-a"
                and action in {"operator_read", "ai_knowledge_write"}
            ),
            ai_workflow=workflow,
        )
    ) as client:
        allowed = client.get("/operator/ai/usage", params={"tenant_id": "tenant-a"})
        denied = client.get("/operator/ai/usage", params={"tenant_id": "tenant-b"})
        deleted = client.delete(
            "/ai/knowledge/" + "not-a-document",
            params={"tenant_id": "tenant-a"},
        )
    assert allowed.status_code == 200
    assert all(item["tenant_id"] == "tenant-a" for item in allowed.json()["records"])
    assert denied.status_code == 403
    assert deleted.status_code == 404


def test_ai_endpoints_fail_closed_without_tenant_authorizer(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'workflow-auth.db'}"))
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/ai/knowledge",
            json={"tenant_id": "tenant-a", "document": "Private knowledge"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "agent API authorization is not configured"


def test_agent_feedback_is_authorized_and_added_to_evidence_chain(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'feedback.db'}"))
    run = runtime.create_run("tenant-a", '{"tool":"unused","input":{}}')
    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True)
    ) as client:
        response = client.post(
            f"/agent/runs/{run.run_id}/feedback",
            json={
                "tenant_id": "tenant-a",
                "rating": 5,
                "note": "Helpful",
            },
        )
    assert response.status_code == 200
    evidence = runtime.list_evidence(run.run_id, "tenant-a")
    feedback = next(item for item in evidence if item.kind == "user_feedback")
    assert feedback.metadata["rating"] == 5
    assert "Helpful" not in repr(feedback.metadata)
    assert runtime.verify_evidence_chain(run.run_id, "tenant-a")["valid"] is True
