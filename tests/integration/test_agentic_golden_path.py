"""Canonical end-to-end integration test for the governed agent runtime.

This test proves the "golden path" described in
``docs/agentic_runtime.md`` and ``docs/agentic_completion_plan.md``:

* a tenant-scoped run is created with a deterministic JSON goal/plan;
* a low-risk tool with input/output schemas and acceptance criteria is
  registered and executed successfully, with persisted run/step/tool-call
  and evidence state;
* a medium-risk tool pauses for approval, is approved out of band, and
  resumes execution with ``autonomous=False``, recording approval and
  final-outcome evidence bound to the run/step/tool/capability version;
* a tool that fails an acceptance check retries (because it is declared
  idempotent) and the retry succeeds, with both the failure and the
  eventual success recorded as evidence.

It uses the public ``Orchestrator`` facade (``orchestrator.py``) backed by
an isolated on-disk SQLite database created in a pytest ``tmp_path``. No
external services, LLMs, Redis, or cloud credentials are used: the planner
is explicitly ``DeterministicJSONPlanner``, never the optional
OpenAI-compatible adapter.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from agentic.runtime import (
    AgentRunRow,
    AgentStore,
    AgentRuntime,
    DeterministicJSONPlanner,
    PlanStepRow,
    ToolCallRow,
    ToolSpec,
    _hash,
)
from orchestrator import Orchestrator


def make_orchestrator(db_path) -> Orchestrator:
    """Build an Orchestrator with an isolated SQLite file and the
    deterministic (non-LLM) planner, matching production defaults minus
    the optional OpenAI-compatible adapter."""
    runtime = AgentRuntime(
        store=AgentStore(f"sqlite:///{db_path}"),
        planner=DeterministicJSONPlanner(),
    )
    return Orchestrator(runtime)


def test_golden_path_low_risk_tool_executes_and_persists_evidence(tmp_path):
    orchestrator = make_orchestrator(tmp_path / "golden_path.db")
    captured_outputs = []

    def lookup_policy_document(payload):
        output = {
            "ok": True,
            "summary": f"policy {payload['policy_id']} is current",
        }
        captured_outputs.append(output)
        return output

    orchestrator.register_tool(
        ToolSpec(
            name="lookup_policy_document",
            description="Return a redacted summary of a governed policy document.",
            input_schema={"type": "object", "required": ["policy_id"]},
            output_schema={"type": "object", "required": ["ok", "summary"]},
            risk_level="low",
            idempotent=True,
            max_cost=0.05,
        ),
        lookup_policy_document,
    )

    goal = (
        '{"tool":"lookup_policy_document",'
        '"input":{"policy_id":"data-retention-1"},'
        '"acceptance":{"required_keys":["ok","summary"]}}'
    )
    run = orchestrator.create_run(
        "tenant-golden", goal, budget=1.0, idempotency_key="golden-path-once"
    )

    result = asyncio.run(orchestrator.execute(run.run_id, "tenant-golden"))

    assert result["status"] == "SUCCEEDED"
    # The tool output is returned to the caller (via the handler) even though
    # only its hash is persisted to evidence for audit purposes.
    assert captured_outputs == [
        {"ok": True, "summary": "policy data-retention-1 is current"}
    ]

    persisted_run = orchestrator.runtime.get_run(run.run_id, "tenant-golden")
    assert persisted_run.status == "SUCCEEDED"
    assert persisted_run.spent_cost == 0.05

    evidence = orchestrator.runtime.list_evidence(run.run_id, "tenant-golden")
    kinds = {item.kind for item in evidence}
    assert {
        "prompt_context",
        "plan",
        "policy_decision",
        "tool_call",
        "tool_result",
        "final_outcome",
    } <= kinds
    tool_result = next(item for item in evidence if item.kind == "tool_result")
    assert tool_result.metadata["tool_name"] == "lookup_policy_document"
    assert tool_result.metadata["verified"] is True
    assert tool_result.metadata["output_hash"] == _hash(captured_outputs[0])

    with orchestrator.runtime.store.sessions() as session:
        run_row = session.get(AgentRunRow, run.run_id)
        assert run_row.status == "SUCCEEDED"
        steps = session.scalars(
            select(PlanStepRow).where(PlanStepRow.run_id == run.run_id)
        ).all()
        assert len(steps) == 1
        assert steps[0].status == "SUCCEEDED"
        calls = session.scalars(
            select(ToolCallRow).where(ToolCallRow.run_id == run.run_id)
        ).all()
        assert len(calls) == 1
        assert calls[0].status == "SUCCEEDED"
        assert calls[0].output_hash is not None


def test_medium_risk_tool_requires_approval_then_resumes_with_evidence(tmp_path):
    db_path = tmp_path / "approval_path.db"

    def build_orchestrator():
        orchestrator = make_orchestrator(db_path)
        orchestrator.register_tool(
            ToolSpec(
                name="rotate_credential",
                description="Rotate a scoped service credential.",
                input_schema={"type": "object", "required": ["credential_id"]},
                output_schema={"type": "object", "required": ["rotated"]},
                risk_level="medium",
                requires_approval=True,
                idempotent=True,
                max_cost=0.1,
            ),
            lambda payload: captured_outputs.append(
                {"rotated": payload["credential_id"]}
            )
            or captured_outputs[-1],
        )
        return orchestrator

    captured_outputs: list[dict] = []
    orchestrator = build_orchestrator()
    goal = '{"tool":"rotate_credential","input":{"credential_id":"svc-42"}}'
    run = orchestrator.create_run("tenant-approval", goal, budget=1.0)

    waiting = asyncio.run(orchestrator.execute(run.run_id, "tenant-approval"))
    assert waiting["status"] == "WAITING_APPROVAL"

    with orchestrator.runtime.store.sessions() as session:
        run_row = session.get(AgentRunRow, run.run_id)
        assert run_row.status == "WAITING_APPROVAL"

    from agentic.runtime import ApprovalRow

    with orchestrator.runtime.store.sessions() as session:
        approval = session.scalar(
            select(ApprovalRow).where(ApprovalRow.run_id == run.run_id)
        )
        assert approval.status == "pending"
        assert approval.tool_name == "rotate_credential"
        assert approval.step_id
        assert approval.capability_version  # bound to the active capability hash
        step_id = approval.step_id

    # A fresh orchestrator instance simulates approval/resume happening in a
    # separate process against the same durable store.
    resumed_orchestrator = build_orchestrator()
    resumed_orchestrator.approve(
        run.run_id, "tenant-approval", "security-operator", "verified rotation ticket"
    )
    result = asyncio.run(
        resumed_orchestrator.resume(run.run_id, "tenant-approval", autonomous=False)
    )
    assert result["status"] == "SUCCEEDED"
    assert captured_outputs == [{"rotated": "svc-42"}]

    evidence = resumed_orchestrator.runtime.list_evidence(run.run_id, "tenant-approval")
    approval_evidence = next(item for item in evidence if item.kind == "approval")
    assert approval_evidence.metadata["actor"] == "security-operator"
    assert approval_evidence.metadata["status"] == "approved"
    assert approval_evidence.metadata["step_id"] == step_id
    assert approval_evidence.metadata["tool_name"] == "rotate_credential"

    final_outcome = [item for item in evidence if item.kind == "final_outcome"]
    assert final_outcome, "final outcome evidence must be recorded after resume"

    decisions = [
        item.metadata["action"] for item in evidence if item.kind == "policy_decision"
    ]
    assert decisions == ["review", "allow"]


def test_acceptance_failure_triggers_idempotent_retry_then_succeeds(tmp_path):
    orchestrator = make_orchestrator(tmp_path / "retry_path.db")
    attempts = []

    def flaky_health_check(_payload):
        attempts.append(len(attempts) + 1)
        # Fails the acceptance check on the first attempt, then succeeds.
        return {"ok": len(attempts) > 1}

    orchestrator.register_tool(
        ToolSpec(
            name="flaky_health_check",
            output_schema={"type": "object", "required": ["ok"]},
            idempotent=True,
            max_cost=0.0,
        ),
        flaky_health_check,
    )

    goal = (
        '{"tool":"flaky_health_check","input":{},'
        '"acceptance":{"equals":{"ok":true}}}'
    )
    run = orchestrator.create_run("tenant-retry", goal)
    result = asyncio.run(orchestrator.execute(run.run_id, "tenant-retry"))

    assert result["status"] == "SUCCEEDED"
    assert len(attempts) == 2

    evidence = orchestrator.runtime.list_evidence(run.run_id, "tenant-retry")
    failures = [item for item in evidence if item.kind == "tool_failure"]
    assert len(failures) == 1
    assert failures[0].metadata["reason"] == "acceptance_value_mismatch"
    successes = [item for item in evidence if item.kind == "tool_result"]
    assert len(successes) == 1
    assert successes[0].metadata["verified"] is True
