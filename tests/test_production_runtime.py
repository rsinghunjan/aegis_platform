import asyncio
import time

from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore, ToolSpec
from agentic.worker import (
    AgentExecutionMessage,
    AgentPrincipal,
    AgentWorker,
    CeleryExecutionDispatcher,
)
from production import create_app


class CollectingDispatcher:
    def __init__(self):
        self.messages = []

    def dispatch(self, message: AgentExecutionMessage):
        self.messages.append(message)
        return f"dispatch-{len(self.messages)}"


class LocalTestSandboxExecutor:
    async def execute(self, _spec, payload, handler):
        return handler(payload)


class CollectingEvidenceAnchorBackend:
    name = "test-transparency-log"

    def __init__(self):
        self.anchors = []

    def anchor(self, run_id, tenant_id, head_sha256):
        proof = {
            "anchor_id": f"anchor-{len(self.anchors) + 1}",
            "run_id": run_id,
            "tenant_id": tenant_id,
            "head_sha256": head_sha256,
        }
        self.anchors.append(proof)
        return proof

    def list_anchors(self, run_id, tenant_id):
        return [
            proof
            for proof in self.anchors
            if proof["run_id"] == run_id and proof["tenant_id"] == tenant_id
        ]

    def verify(self, proof):
        return proof in self.anchors


def test_celery_dispatcher_sends_run_reference():
    class FakeCelery:
        def __init__(self):
            self.call = None

        def send_task(self, name, args):
            self.call = (name, args)
            return type("Result", (), {"id": "celery-task-1"})()

    celery = FakeCelery()
    message = AgentExecutionMessage(
        run_id="run-1",
        tenant_id="tenant-a",
        principal=AgentPrincipal(principal_id="operator"),
    )
    task_id = CeleryExecutionDispatcher(celery).dispatch(message)
    assert task_id == "celery-task-1"
    assert celery.call[0] == "aegis.execute_agent_run"
    assert celery.call[1][0]["run_id"] == "run-1"


def test_canonical_health_and_readiness(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'health.db'}"))
    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True)
    ) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}


def test_canonical_agent_endpoint_runs_safe_registered_tool(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'api.db'}"))
    calls = []
    runtime.register_tool(
        ToolSpec(name="echo", output_schema={"type": "object", "required": ["ok"]}),
        lambda payload: calls.append(payload) or {
            "ok": True,
            "value": payload.get("value"),
        },
    )
    dispatcher = CollectingDispatcher()
    with TestClient(
        create_app(
            runtime,
            tenant_authorizer=lambda *_args: True,
            execution_dispatcher=dispatcher,
        )
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
        assert response.json()["status"] == "PENDING"
        assert calls == []
        queued = client.post(
            f"/agent/runs/{response.json()['run_id']}/execute",
            json={"tenant_id": "tenant-a", "role": "admin", "scopes": ["deploy"]},
        )
        assert queued.status_code == 200
        assert queued.json()["status"] == "QUEUED"
    assert dispatcher.messages[0].principal.role == "agent"
    assert dispatcher.messages[0].principal.scopes == []
    result = asyncio.run(AgentWorker(runtime).execute(dispatcher.messages.pop()))
    assert result["status"] == "SUCCEEDED"


def test_agent_api_fails_closed_without_tenant_authorizer(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'auth.db'}"))
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/agent/runs",
            json={"tenant_id": "tenant-a", "goal": '{"tool":"echo","input":{}}'},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "agent API authorization is not configured"


def test_operator_read_models_are_authorized_and_return_summaries(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'operator.db'}"))
    runtime.register_tool(
        ToolSpec(name="echo", idempotent=True),
        lambda _payload: {"ok": True},
    )
    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True)
    ) as client:
        created = client.post(
            "/agent/runs",
            json={
                "tenant_id": "tenant-a",
                "goal": '{"tool":"echo","input":{"api_key":"must-not-be-returned"}}',
            },
        )
        run_id = created.json()["run_id"]
        listing = client.get("/operator/agent/runs", params={"tenant_id": "tenant-a"})
        timeline = client.get(
            f"/operator/agent/runs/{run_id}/timeline",
            params={"tenant_id": "tenant-a"},
        )
        catalog = client.get(
            "/operator/agent/capabilities", params={"tenant_id": "tenant-a"}
        )
        summary = client.get(
            "/operator/agent/evidence-summary",
            params={"tenant_id": "tenant-a", "run_id": run_id},
        )
    assert listing.status_code == timeline.status_code == catalog.status_code == 200
    assert "goal" not in listing.json()["items"][0]
    assert "must-not-be-returned" not in repr(listing.json())
    assert catalog.json()["items"][0]["name"] == "echo"
    assert summary.json()["count"] >= 1
    assert summary.json()["anchors"] == []


def test_periodic_evidence_anchor_is_visible_in_operator_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_EVIDENCE_ANCHOR_INTERVAL_SECONDS", "0.01")
    backend = CollectingEvidenceAnchorBackend()
    runtime = AgentRuntime(
        store=AgentStore(f"sqlite:///{tmp_path / 'anchored-operator.db'}"),
        evidence_anchor_backend=backend,
    )
    run = runtime.create_run("tenant-a", "goal")
    with TestClient(
        create_app(runtime, tenant_authorizer=lambda *_args: True)
    ) as client:
        time.sleep(0.08)
        summary = client.get(
            "/operator/agent/evidence-summary",
            params={"tenant_id": "tenant-a", "run_id": run.run_id},
        )

    assert summary.status_code == 200
    data = summary.json()
    assert data["anchors"][0]["proof"] == backend.anchors[0]
    assert data["integrity"]["anchor_status"] == "verified"


def test_run_can_be_created_then_executed_separately(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'separate.db'}"))
    calls = []
    runtime.register_tool(
        ToolSpec(name="echo", idempotent=True),
        lambda _payload: calls.append(True) or {"ok": True},
    )
    dispatcher = CollectingDispatcher()
    with TestClient(
        create_app(
            runtime,
            tenant_authorizer=lambda *_args: True,
            execution_dispatcher=dispatcher,
        )
    ) as client:
        created = client.post(
            "/agent/runs/create",
            json={"tenant_id": "tenant-a", "goal": '{"tool":"echo","input":{}}'},
        )
        assert created.status_code == 200
        assert created.json()["status"] == "PENDING"
        executed = client.post(
            f"/agent/runs/{created.json()['run_id']}/execute",
            json={"tenant_id": "tenant-a"},
        )
    assert executed.json()["status"] == "QUEUED"
    assert calls == []
    result = asyncio.run(AgentWorker(runtime).execute(dispatcher.messages.pop()))
    assert executed.status_code == 200
    assert result["status"] == "SUCCEEDED"
    assert calls == [True]


def test_agent_approval_requires_authorized_actor(tmp_path):
    runtime = AgentRuntime(
        store=AgentStore(f"sqlite:///{tmp_path / 'approval.db'}"),
        sandbox_executor=LocalTestSandboxExecutor(),
    )
    runtime.register_tool(
        ToolSpec(name="release", risk_level="high"),
        lambda _payload: {"released": True},
    )
    authorizations = []
    dispatcher = CollectingDispatcher()

    def authorize(_request, tenant_id, action, actor):
        authorizations.append((tenant_id, action, actor))
        return actor == "release-manager" or actor is None

    with TestClient(
        create_app(
            runtime,
            tenant_authorizer=authorize,
            principal_resolver=lambda _request: AgentPrincipal(
                principal_id="release-manager"
            ),
            execution_dispatcher=dispatcher,
        )
    ) as client:
        created = client.post(
            "/agent/runs",
            json={
                "tenant_id": "tenant-a",
                "goal": '{"tool":"release","input":{}}',
            },
        )
        run_id = created.json()["run_id"]
        assert created.json()["status"] == "PENDING"
        queued = client.post(
            f"/agent/runs/{run_id}/execute",
            json={"tenant_id": "tenant-a"},
        )
        assert queued.status_code == 200
        waiting = asyncio.run(AgentWorker(runtime).execute(dispatcher.messages.pop()))
        assert waiting["status"] == "WAITING_APPROVAL"
        approval_id = runtime.list_approvals(
            "tenant-a", status="pending"
        )[0].approval_id
        wrong_approval = client.post(
            f"/agent/runs/{run_id}/approve",
            json={
                "tenant_id": "tenant-a",
                "approval_id": "different-approval",
                "reason": "should not match",
            },
        )
        assert wrong_approval.status_code == 400
        approved = client.post(
            f"/agent/runs/{run_id}/approve",
            json={
                "tenant_id": "tenant-a",
                "approval_id": approval_id,
                "actor": "attacker-controlled-body-value",
                "reason": "reviewed",
            },
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "APPROVED"
        queued_resume = client.post(
            f"/agent/runs/{run_id}/execute",
            json={"tenant_id": "tenant-a"},
        )
    assert queued_resume.status_code == 200
    result = asyncio.run(AgentWorker(runtime).execute(dispatcher.messages.pop()))
    assert result["status"] == "SUCCEEDED"
    assert ("tenant-a", "approve", "release-manager") in authorizations
    approval = runtime.list_approvals("tenant-a", status="approved")[0]
    assert approval.actor == "release-manager"
    assert approval.plan_hash and approval.policy_version


def test_new_approval_notification_is_injected_and_evidence_safe(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'notify.db'}"))
    runtime.register_tool(ToolSpec(name="release", risk_level="high"), lambda _: {})
    notifications = []
    dispatcher = CollectingDispatcher()
    with TestClient(
        create_app(
            runtime,
            tenant_authorizer=lambda *_args: True,
            approval_notifier=lambda approval: notifications.append(approval),
            execution_dispatcher=dispatcher,
        )
    ) as client:
        response = client.post(
            "/agent/runs",
            json={
                "tenant_id": "tenant-a",
                "goal": '{"tool":"release","input":{}}',
            },
        )
        assert response.json()["status"] == "PENDING"
        queued = client.post(
            f"/agent/runs/{response.json()['run_id']}/execute",
            json={"tenant_id": "tenant-a"},
        )
    result = asyncio.run(AgentWorker(runtime).execute(dispatcher.messages.pop()))
    assert queued.json()["status"] == "QUEUED"
    assert result["status"] == "WAITING_APPROVAL"
    assert len(notifications) == 1
    assert notifications[0].tool_name == "release"
    assert any(
        item.kind == "approval_notification"
        for item in runtime.list_evidence(response.json()["run_id"], "tenant-a")
    )


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
