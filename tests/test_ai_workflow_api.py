from fastapi.testclient import TestClient
import pytest

from agentic.runtime import AgentRuntime, AgentStore
from production import create_app
from services.ai_workflow import AIWorkflow
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


def test_ai_workflow_rejects_oversized_document():
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(),
    )
    with pytest.raises(ValueError, match="maximum length"):
        workflow.ingest("tenant-a", "x" * (workflow.max_document_chars + 1))


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
