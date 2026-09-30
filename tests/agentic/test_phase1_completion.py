import asyncio
import json
from datetime import timedelta

from sqlalchemy import select

from agentic.approvals import ApprovalService
from agentic.capabilities import CapabilityCatalog
from agentic.planner import OpenAICompatiblePlanner
from agentic.remediation import (
    DriftFinding,
    DriftRemediationAdapter,
    local_simulation_adapter,
)
from agentic.runtime import (
    AgentRuntime,
    AgentRuntimeError,
    AgentStore,
    ApprovalRow,
    CapabilityRow,
    ToolSpec,
    _now,
)


def make_runtime(path, **kwargs):
    return AgentRuntime(store=AgentStore(f"sqlite:///{path}"), **kwargs)


def _mock_completion(monkeypatch, content):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(content)}}]}
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())


def test_openai_planner_valid_plan_and_strict_invalid_fallback(monkeypatch):
    planner = OpenAICompatiblePlanner(
        "https://planner.example/v1", "model-a", "provider-secret"
    )
    _mock_completion(
        monkeypatch,
        {"steps": [{"tool": "echo", "input": {"message": "hi"}, "acceptance": {}}]},
    )
    plan = planner.plan("raw goal", {"echo"})
    assert plan.steps[0].tool_name == "echo"
    assert plan.steps[0].input == {"message": "hi"}

    _mock_completion(
        monkeypatch, {"steps": [{"tool": "unregistered", "input": {}}]}
    )
    fallback = planner.plan('{"tool":"echo","input":{"message":"safe"}}', {"echo"})
    assert fallback.steps[0].tool_name == "echo"
    assert fallback.steps[0].input == {"message": "safe"}


def test_planner_falls_back_when_provider_unavailable(monkeypatch):
    planner = OpenAICompatiblePlanner(
        "https://planner.example/v1", "model-a", "provider-secret"
    )

    def provider_error(*_args, **_kwargs):
        raise OSError("provider unavailable")

    monkeypatch.setattr("urllib.request.urlopen", provider_error)
    plan = planner.plan('{"tool":"echo","input":{}}', {"echo"})
    assert plan.steps[0].tool_name == "echo"


def test_catalog_hash_version_persistence_and_tamper_enforcement(tmp_path):
    catalog = CapabilityCatalog()
    catalog.register(ToolSpec(name="echo", version="2.0.0"))
    initial = catalog.summary()
    catalog.register(ToolSpec(name="echo", version="2.0.1"))
    assert catalog.version != initial["version"]

    runtime = make_runtime(
        tmp_path / "capability.db", capability_enforcement=True
    )
    called = []
    runtime.register_tool(
        ToolSpec(name="echo", idempotent=True),
        lambda _payload: called.append(True) or {"ok": True},
    )
    run = runtime.create_run("tenant-a", '{"tool":"echo","input":{}}')
    with runtime.store.sessions() as session:
        row = session.get(CapabilityRow, "echo")
        row.version = "tampered"
        session.commit()
    try:
        asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    except AgentRuntimeError as exc:
        assert "inactive capability" in str(exc)
    else:
        raise AssertionError("tampered capabilities must be refused")
    assert called == []


def test_pure_sandbox_rejects_commands_before_tool_execution(tmp_path):
    runtime = make_runtime(tmp_path / "sandbox.db")
    called = []
    runtime.register_tool(
        ToolSpec(name="takes_json"),
        lambda payload: called.append(payload) or {"ok": True},
    )
    run = runtime.create_run(
        "tenant-a", '{"tool":"takes_json","input":{"cmd":"whoami"}}'
    )
    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "BLOCKED"
    assert called == []
    assert any(
        item.kind == "sandbox_block"
        for item in runtime.list_evidence(run.run_id, "tenant-a")
    )


def test_approval_deny_expiration_and_escalation(tmp_path):
    runtime = make_runtime(
        tmp_path / "approvals.db", approval_sla_seconds=120
    )
    runtime.register_tool(
        ToolSpec(name="release", risk_level="high"),
        lambda _payload: {"released": True},
    )
    first = runtime.create_run("tenant-a", '{"tool":"release","input":{}}')
    assert asyncio.run(runtime.execute(first.run_id, "tenant-a"))[
        "status"
    ] == "WAITING_APPROVAL"
    pending = runtime.list_approvals("tenant-a", "pending")[0]
    assert pending.expires_at and pending.sla_seconds == 120
    denied = runtime.deny(first.run_id, "tenant-a", "operator", "risk")
    assert denied.status == "denied"
    assert denied.denial_reason == "risk"
    assert runtime.get_run(first.run_id, "tenant-a").status == "BLOCKED"

    second = runtime.create_run("tenant-b", '{"tool":"release","input":{}}')
    asyncio.run(runtime.execute(second.run_id, "tenant-b"))
    with runtime.store.sessions() as session:
        approval = session.scalar(
            select(ApprovalRow).where(ApprovalRow.run_id == second.run_id)
        )
        approval.expires_at = _now() - timedelta(seconds=1)
        session.commit()
    assert runtime.expire_approvals() == 1
    assert runtime.get_run(second.run_id, "tenant-b").status == "BLOCKED"

    third = runtime.create_run("tenant-c", '{"tool":"release","input":{}}')
    asyncio.run(runtime.execute(third.run_id, "tenant-c"))
    with runtime.store.sessions() as session:
        approval = session.scalar(
            select(ApprovalRow).where(ApprovalRow.run_id == third.run_id)
        )
        approval.requested_at = _now() - timedelta(seconds=70)
        session.commit()
    service = ApprovalService(runtime)
    assert service.escalate() == 1
    assert service.pending("tenant-c")[0].escalation_status == "escalated"


def test_drift_low_risk_simulation_and_high_risk_approval(tmp_path):
    runtime = make_runtime(tmp_path / "drift-flow.db")
    diagnoses = []
    runtime.register_tool(
        ToolSpec(name="diagnose_drift", idempotent=True),
        lambda payload: diagnoses.append(payload) or {"diagnosis": "shift"},
    )
    adapter = DriftRemediationAdapter(
        runtime,
        remediation_adapters={
            "canary_deploy": local_simulation_adapter,
            "rollback": local_simulation_adapter,
        },
    )
    low = DriftFinding(
        tenant_id="tenant-a",
        model_name="model",
        metric="dataset_drift",
        value=0.15,
        threshold=0.1,
        observed_at="2026-09-30T00:00:00Z",
    )
    low_result = asyncio.run(adapter.consume(low))
    assert low_result["remediation"]["status"] == "SUCCEEDED"
    assert low_result["action"] == "canary_deploy"

    high = low.copy(update={"tenant_id": "tenant-b", "value": 0.3})
    high_result = asyncio.run(adapter.consume(high))
    assert high_result["action"] == "rollback"
    assert high_result["remediation"]["status"] == "WAITING_APPROVAL"
    assert len(diagnoses) == 2
    approval = runtime.list_approvals("tenant-b", "pending")[0]
    runtime.approve(approval.run_id, "tenant-b", "operator", "reviewed")
    result = asyncio.run(runtime.resume(approval.run_id, "tenant-b", autonomous=False))
    assert result["status"] == "SUCCEEDED"
    assert any(
        item.kind == "drift_finding"
        for item in runtime.list_evidence(high_result["run_id"], "tenant-b")
    )


def test_local_drift_smoke_registers_diagnosis_and_is_idempotent(tmp_path):
    runtime = make_runtime(tmp_path / "local-drift.db")
    adapter = DriftRemediationAdapter.local(runtime)
    finding = DriftFinding(
        tenant_id="tenant-local",
        model_name="model",
        metric="dataset_drift",
        value=0.05,
        threshold=0.1,
        observed_at="2026-09-30T00:00:00Z",
    )
    first = asyncio.run(adapter.consume(finding))
    second = asyncio.run(adapter.consume(finding))
    assert first["status"] == second["status"] == "SUCCEEDED"
    assert first["run_id"] == second["run_id"]
    assert first["remediation"]["status"] == "SUCCEEDED"
    assert first["remediation"]["result_hash"]
