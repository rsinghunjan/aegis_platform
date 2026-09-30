from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore, ToolSpec
from production import create_app


def test_canonical_health_and_readiness(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'health.db'}"))
    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True)
    ) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}


def test_canonical_agent_endpoint_runs_safe_registered_tool(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'api.db'}"))
    runtime.register_tool(
        ToolSpec(name="echo", output_schema={"type": "object", "required": ["ok"]}),
        lambda payload: {"ok": True, "value": payload.get("value")},
    )
    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True)
    ) as client:
        response = client.post(
            "/agent/runs",
            json={
                "tenant_id": "tenant-a",
                "goal": '{"tool":"echo","input":{"value":"api"}}',
                "idempotency_key": "api-request",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"


def test_agent_api_fails_closed_without_tenant_authorizer(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'auth.db'}"))
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/agent/runs",
            json={"tenant_id": "tenant-a", "goal": '{"tool":"echo","input":{}}'},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "agent API authorization is not configured"


def test_agent_approval_requires_authorized_actor(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'approval.db'}"))
    runtime.register_tool(
        ToolSpec(name="release", risk_level="high"),
        lambda _payload: {"released": True},
    )
    authorizations = []

    def authorize(_request, tenant_id, action, actor):
        authorizations.append((tenant_id, action, actor))
        return actor == "release-manager" or actor is None

    with TestClient(create_app(runtime, tenant_authorizer=authorize)) as client:
        created = client.post(
            "/agent/runs",
            json={
                "tenant_id": "tenant-a",
                "goal": '{"tool":"release","input":{}}',
            },
        )
        run_id = created.json()["run_id"]
        assert created.json()["status"] == "WAITING_APPROVAL"
        approved = client.post(
            f"/agent/runs/{run_id}/approve",
            json={
                "tenant_id": "tenant-a",
                "actor": "release-manager",
                "reason": "reviewed",
            },
        )
    assert approved.status_code == 200
    assert approved.json()["status"] == "SUCCEEDED"
    assert authorizations[-1] == ("tenant-a", "approve", "release-manager")


def test_dockerfile_defaults_to_nonroot_uvicorn_service():
    from pathlib import Path

    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    content = dockerfile.read_text()
    dockerignore = Path(__file__).parents[1] / ".dockerignore"
    assert "USER aegis" in content
    assert "EXPOSE 8000" in content
    assert 'uvicorn", "production:app"' in content
    assert "tail -f /dev/null" not in content
    assert "*.patch" in dockerignore.read_text()
