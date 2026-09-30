from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore, ToolSpec
from production import create_app


def test_canonical_health_and_readiness(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'health.db'}"))
    with TestClient(create_app(runtime)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}


def test_canonical_agent_endpoint_runs_safe_registered_tool(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'api.db'}"))
    runtime.register_tool(
        ToolSpec(name="echo", output_schema={"type": "object", "required": ["ok"]}),
        lambda payload: {"ok": True, "value": payload.get("value")},
    )
    with TestClient(create_app(runtime)) as client:
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


def test_dockerfile_defaults_to_nonroot_uvicorn_service():
    from pathlib import Path

    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    content = dockerfile.read_text()
    assert "USER aegis" in content
    assert "EXPOSE 8000" in content
    assert 'uvicorn", "production:app"' in content
    assert "tail -f /dev/null" not in content
