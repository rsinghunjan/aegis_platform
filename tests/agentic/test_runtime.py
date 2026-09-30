import asyncio

import pytest
from sqlalchemy import select

from agentic.runtime import (
    AgentPolicyGate,
    AgentRuntime,
    AgentRuntimeError,
    AgentStore,
    AgentRunRow,
    PolicyDecisionRow,
    ToolCallRow,
    ToolSpec,
)


def make_runtime(path, **kwargs):
    return AgentRuntime(store=AgentStore(f"sqlite:///{path}"), **kwargs)


def test_create_plan_execute_verify_and_persist(tmp_path):
    runtime = make_runtime(tmp_path / "agent.db")
    runtime.register_tool(
        ToolSpec(
            name="safe_echo",
            input_schema={"type": "object", "required": ["text"]},
            output_schema={"type": "object", "required": ["ok"]},
            idempotent=True,
            max_cost=0.2,
        ),
        lambda payload: {"ok": True, "echo": payload["text"]},
    )
    goal = '{"tool":"safe_echo","input":{"text":"hello"},"acceptance":{"required_keys":["ok"]}}'
    run = runtime.create_run("tenant-a", goal, budget=1, idempotency_key="once")
    duplicate = runtime.create_run("tenant-a", goal, budget=1, idempotency_key="once")
    assert duplicate.run_id == run.run_id

    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "SUCCEEDED"
    assert runtime.get_run(run.run_id, "tenant-a").spent_cost == 0.2
    evidence = runtime.list_evidence(run.run_id, "tenant-a")
    assert {item.kind for item in evidence} >= {
        "plan",
        "policy_decision",
        "tool_result",
        "final_outcome",
    }
    with runtime.store.sessions() as session:
        assert len(session.scalars(select(ToolCallRow)).all()) == 1
        assert len(session.scalars(select(PolicyDecisionRow)).all()) == 1
        row = session.get(AgentRunRow, run.run_id)
        assert row.status == "SUCCEEDED"


def test_high_risk_requires_explicit_approval_and_resumes(tmp_path):
    path = tmp_path / "resume.db"
    calls = []
    runtime = make_runtime(path)
    runtime.register_tool(
        ToolSpec(name="release", risk_level="high", idempotent=True),
        lambda payload: calls.append(payload) or {"released": True},
    )
    run = runtime.create_run(
        "tenant-a", '{"tool":"release","input":{"artifact":"model-v1"}}'
    )
    waiting = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert waiting["status"] == "WAITING_APPROVAL"
    assert calls == []

    resumed_runtime = make_runtime(path)
    resumed_runtime.register_tool(
        ToolSpec(name="release", risk_level="high", idempotent=True),
        lambda payload: calls.append(payload) or {"released": True},
    )
    resumed_runtime.approve(run.run_id, "tenant-a", "release-manager", "approved")
    result = asyncio.run(resumed_runtime.resume(run.run_id, "tenant-a", autonomous=False))
    assert result["status"] == "SUCCEEDED"
    assert len(calls) == 1
    decisions = [
        item.metadata["action"]
        for item in resumed_runtime.list_evidence(run.run_id, "tenant-a")
        if item.kind == "policy_decision"
    ]
    assert decisions == ["review", "allow"]


def test_global_autonomy_kill_switch_blocks(tmp_path):
    runtime = make_runtime(
        tmp_path / "disabled.db",
        policy=AgentPolicyGate(autonomy_enabled=False),
    )
    called = []
    runtime.register_tool(
        ToolSpec(name="inspect", risk_level="low"),
        lambda payload: called.append(payload) or {"ok": True},
    )
    run = runtime.create_run("tenant-a", '{"tool":"inspect","input":{}}')
    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "global_autonomy_disabled"
    assert called == []


def test_policy_restrictions_and_idempotency_key_mismatch(tmp_path):
    runtime = make_runtime(tmp_path / "policy.db")
    runtime.register_tool(
        ToolSpec(
            name="admin_action",
            allowed_roles=["admin"],
            allowed_environments=["prod"],
            allowed_tenants=["tenant-b"],
            allowed_scopes=["deploy"],
        ),
        lambda payload: {"ok": True},
    )
    run = runtime.create_run(
        "tenant-a",
        '{"tool":"admin_action","input":{}}',
        idempotency_key="key",
    )
    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "BLOCKED"
    with pytest.raises(AgentRuntimeError, match="different goal"):
        runtime.create_run("tenant-a", "changed", idempotency_key="key")


def test_idempotent_tool_retry_and_verification_failure(tmp_path):
    runtime = make_runtime(tmp_path / "retry.db", max_retries=1)
    attempts = []

    def flaky(_payload):
        attempts.append(1)
        return {"ok": len(attempts) > 1}

    runtime.register_tool(
        ToolSpec(
            name="flaky",
            idempotent=True,
            output_schema={"type": "object", "required": ["ok"]},
            max_cost=0,
        ),
        flaky,
    )
    run = runtime.create_run(
        "tenant-a",
        '{"tool":"flaky","input":{},"acceptance":{"equals":{"ok":true}}}',
    )
    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "SUCCEEDED"
    assert len(attempts) == 2


def test_unregistered_planner_tool_is_rejected(tmp_path):
    runtime = make_runtime(tmp_path / "invalid.db")
    run = runtime.create_run("tenant-a", '{"tool":"shell","input":{"cmd":"whoami"}}')
    with pytest.raises(AgentRuntimeError, match="unregistered tool"):
        asyncio.run(runtime.execute(run.run_id, "tenant-a"))


def test_failed_non_idempotent_step_uses_bounded_recovery_plan(tmp_path):
    runtime = make_runtime(tmp_path / "recovery.db")
    attempts = []

    def fail_once(_payload):
        attempts.append("initial")
        raise ValueError("cannot safely retry")

    runtime.register_tool(
        ToolSpec(name="unsafe_to_retry", idempotent=False),
        fail_once,
    )
    runtime.register_tool(
        ToolSpec(name="recover", idempotent=True),
        lambda _payload: attempts.append("recovery") or {"recovered": True},
    )
    goal = (
        '{"tool":"unsafe_to_retry","input":{},'
        '"recovery_steps":[{"tool":"recover","input":{}}]}'
    )
    run = runtime.create_run("tenant-a", goal)
    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "REPLANNING"
    assert attempts == ["initial"]

    result = asyncio.run(runtime.resume(run.run_id, "tenant-a"))
    assert result["status"] == "SUCCEEDED"
    assert attempts == ["initial", "recovery"]
    assert any(
        item.kind == "replan"
        for item in runtime.list_evidence(run.run_id, "tenant-a")
    )


def test_persisted_inputs_redact_secret_fields(tmp_path):
    runtime = make_runtime(tmp_path / "redaction.db")
    runtime.register_tool(
        ToolSpec(name="inspect", idempotent=True),
        lambda payload: {"ok": payload["api_key"] == "[REDACTED]"},
    )
    goal = '{"tool":"inspect","input":{"api_key":"never-persist-this"}}'
    run = runtime.create_run("tenant-a", goal)
    result = asyncio.run(runtime.execute(run.run_id, "tenant-a"))
    assert result["status"] == "SUCCEEDED"
    with runtime.store.sessions() as session:
        step = session.scalar(select(AgentRunRow).where(AgentRunRow.run_id == run.run_id))
        assert "never-persist-this" not in step.plan_json["steps"][0]["input"]["api_key"]
        evidence_text = repr(runtime.list_evidence(run.run_id, "tenant-a"))
        assert "never-persist-this" not in evidence_text
