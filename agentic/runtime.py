"""Durable, policy-gated agent execution with a deterministic planner."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from policy.agent_policy import AgentPolicyGate


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentRun(BaseModel):
    run_id: str
    tenant_id: str
    goal: str
    status: str = "PENDING"
    risk: RiskLevel = RiskLevel.LOW
    budget: float = 0.0
    spent_cost: float = 0.0
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    plan: Optional["Plan"] = None
    final_result_hash: Optional[str] = None


class PlanStep(BaseModel):
    step_id: str
    ordinal: int
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    acceptance_criteria: dict[str, Any] = Field(default_factory=dict)
    status: str = "PENDING"
    attempts: int = 0
    result_hash: Optional[str] = None
    error: Optional[str] = None


class Plan(BaseModel):
    plan_id: str
    steps: list[PlanStep]


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    allowed_roles: list[str] = Field(default_factory=list)
    allowed_environments: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    idempotent: bool = False
    max_cost: float = 0.0
    allowed_tenants: list[str] = Field(default_factory=list)
    allowed_scopes: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    call_id: str
    run_id: str
    tenant_id: str
    step_id: str
    tool_name: str
    status: str
    input_hash: str
    output_hash: Optional[str] = None
    result_metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime


class PolicyDecision(BaseModel):
    decision_id: str
    run_id: str
    tenant_id: str
    tool_name: str
    action: str
    reason: str
    created_at: datetime


class Approval(BaseModel):
    approval_id: str
    run_id: str
    tenant_id: str
    step_id: str
    tool_name: str
    status: str
    actor: Optional[str] = None
    reason: Optional[str] = None
    created_at: datetime
    decided_at: Optional[datetime] = None


class Evidence(BaseModel):
    evidence_id: str
    run_id: str
    tenant_id: str
    kind: str
    sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AgentRuntimeError(RuntimeError):
    """Raised for invalid, inaccessible, or unsafe agent operations."""


class AgentBase(DeclarativeBase):
    pass


class AgentRunRow(AgentBase):
    __tablename__ = "agent_runs"
    __table_args__ = (UniqueConstraint("tenant_id", "idempotency_key"),)

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    goal_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    risk: Mapped[str] = mapped_column(String(16), nullable=False)
    budget: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    spent_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(255))
    plan_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    final_result_hash: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PlanStepRow(AgentBase):
    __tablename__ = "agent_plan_steps"

    step_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    acceptance_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_hash: Mapped[Optional[str]] = mapped_column(String(64))
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ToolCallRow(AgentBase):
    __tablename__ = "agent_tool_calls"

    call_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    step_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output_hash: Mapped[Optional[str]] = mapped_column(String(64))
    result_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PolicyDecisionRow(AgentBase):
    __tablename__ = "agent_policy_decisions"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ApprovalRow(AgentBase):
    __tablename__ = "agent_approvals"

    approval_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    step_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[Optional[str]] = mapped_column(String(255))
    reason: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class EvidenceRow(AgentBase):
    __tablename__ = "agent_evidence"

    evidence_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


AgentRun.update_forward_refs()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_SENSITIVE_KEY = re.compile(r"(secret|password|token|credential|authorization|api.?key)", re.I)


def _persistable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _persistable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_persistable(item) for item in value]
    if isinstance(value, str) and len(value) > 4096:
        return {"sha256": _hash(value), "truncated": True}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AgentRuntimeError("Tool inputs and outputs must be JSON-compatible")


class AgentStore:
    """Small SQLAlchemy store with idempotent table initialization."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            "AEGIS_AGENT_DATABASE_URL",
            os.getenv("DATABASE_URL", "sqlite:///./aegis_agent.db"),
        )
        engine_options: dict[str, Any] = {"pool_pre_ping": True}
        if self.database_url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
        elif self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        self._engine = None
        self._sessions = None
        self._engine_options = engine_options

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self.database_url, **self._engine_options)
        return self._engine

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        return self._sessions

    def initialize(self) -> None:
        AgentBase.metadata.create_all(self.engine, checkfirst=True)


class Planner:
    def plan(
        self, goal: str, registered_tools: set[str], recovery: bool = False
    ) -> Plan:
        raise NotImplementedError


class DeterministicJSONPlanner(Planner):
    """Accepts JSON {tool, input, acceptance} or {steps: [...]}; never runs code."""

    def plan(
        self, goal: str, registered_tools: set[str], recovery: bool = False
    ) -> Plan:
        try:
            request = json.loads(goal)
        except (TypeError, json.JSONDecodeError):
            if len(registered_tools) != 1:
                raise AgentRuntimeError(
                    "Provide a JSON plan naming a registered tool when multiple tools exist"
                )
            request = {"tool": next(iter(registered_tools)), "input": {"goal": goal}}
        steps = (
            request.get("recovery_steps")
            if recovery and isinstance(request, dict)
            else request.get("steps") if isinstance(request, dict) else None
        )
        if recovery and steps is None:
            raise AgentRuntimeError("No recovery plan provided")
        if steps is None and isinstance(request, dict) and "tool" in request:
            steps = [request]
        if not isinstance(steps, list) or not steps:
            raise AgentRuntimeError("Planner requires one or more tool steps")
        plan_steps = []
        for ordinal, spec in enumerate(steps):
            if not isinstance(spec, dict):
                raise AgentRuntimeError("Each planned step must be a JSON object")
            tool_name = spec.get("tool")
            if not isinstance(tool_name, str) or tool_name not in registered_tools:
                raise AgentRuntimeError(f"Planner selected an unregistered tool: {tool_name}")
            tool_input = spec.get("input", {})
            acceptance = spec.get("acceptance", {})
            if not isinstance(tool_input, dict) or not isinstance(acceptance, dict):
                raise AgentRuntimeError("Tool input and acceptance criteria must be JSON objects")
            plan_steps.append(
                PlanStep(
                    step_id=str(uuid.uuid4()),
                    ordinal=ordinal,
                    tool_name=tool_name,
                    input=tool_input,
                    acceptance_criteria=acceptance,
                )
            )
        return Plan(plan_id=str(uuid.uuid4()), steps=plan_steps)


class Verifier:
    def verify(
        self,
        result: Any,
        output_schema: dict[str, Any],
        acceptance: dict[str, Any],
    ) -> tuple[bool, str]:
        if not _matches_schema(result, output_schema):
            return False, "tool_output_schema_mismatch"
        required_keys = acceptance.get("required_keys", [])
        if required_keys and (
            not isinstance(result, dict)
            or any(key not in result for key in required_keys)
        ):
            return False, "acceptance_required_key_missing"
        expected = acceptance.get("equals", {})
        if expected and (
            not isinstance(result, dict)
            or any(result.get(key) != value for key, value in expected.items())
        ):
            return False, "acceptance_value_mismatch"
        return True, "verified"


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    if not schema:
        return True
    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if expected_type in type_map and not isinstance(value, type_map[expected_type]):
        if expected_type == "integer" and isinstance(value, bool):
            return False
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(
            key not in value or not _matches_schema(value[key], child)
            for key, child in properties.items()
        ):
            return False
        if any(key not in value for key in schema.get("required", [])):
            return False
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_matches_schema(item, schema["items"]) for item in value)
    return True


class AgentRuntime:
    def __init__(
        self,
        store: Optional[AgentStore] = None,
        planner: Optional[Planner] = None,
        policy: Optional[AgentPolicyGate] = None,
        verifier: Optional[Verifier] = None,
        max_retries: int = 1,
        max_replans: int = 1,
        step_timeout: float = 30.0,
    ):
        self.store = store or AgentStore()
        self.planner = planner or DeterministicJSONPlanner()
        self.policy = policy or AgentPolicyGate()
        self.verifier = verifier or Verifier()
        self.max_retries = max(0, max_retries)
        self.max_replans = max(0, max_replans)
        self.step_timeout = step_timeout
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register_tool(self, spec: ToolSpec, tool: Callable[..., Any]) -> None:
        if not spec.name or not callable(tool):
            raise AgentRuntimeError("A named callable is required to register a tool")
        if spec.max_cost < 0:
            raise AgentRuntimeError("Tool max_cost cannot be negative")
        self._tools[spec.name] = (spec, tool)

    def create_run(
        self,
        tenant_id: str,
        goal: str,
        budget: float = 0.0,
        idempotency_key: Optional[str] = None,
    ) -> AgentRun:
        self.store.initialize()
        if not tenant_id or not goal:
            raise AgentRuntimeError("tenant_id and goal are required")
        if budget < 0:
            raise AgentRuntimeError("budget cannot be negative")
        with self.store.sessions() as session:
            if idempotency_key:
                existing = session.scalar(
                    select(AgentRunRow).where(
                        AgentRunRow.tenant_id == tenant_id,
                        AgentRunRow.idempotency_key == idempotency_key,
                    )
                )
                if existing:
                    if existing.goal_hash != _hash(goal):
                        raise AgentRuntimeError(
                            "Idempotency key was already used for a different goal"
                        )
                    return self._as_run(existing)
            now = _now()
            row = AgentRunRow(
                run_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                goal=goal,
                goal_hash=_hash(goal),
                status="PENDING",
                risk=RiskLevel.LOW.value,
                budget=budget,
                spent_cost=0,
                idempotency_key=idempotency_key,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            self._add_evidence(
                session,
                row,
                "prompt_context",
                {
                    "goal_sha256": row.goal_hash,
                    "context_ref": f"sha256:{row.goal_hash}",
                    "model_version": "deterministic-json-v1",
                    "budget": budget,
                },
            )
            session.commit()
            return self._as_run(row)

    def get_run(self, run_id: str, tenant_id: str) -> AgentRun:
        with self.store.sessions() as session:
            row = session.get(AgentRunRow, run_id)
            if row is None or row.tenant_id != tenant_id:
                raise AgentRuntimeError("Agent run not found")
            return self._as_run(row)

    async def execute(
        self,
        run_id: str,
        tenant_id: str,
        role: str = "agent",
        environment: str = "local",
        scopes: Optional[set[str]] = None,
        autonomous: bool = True,
    ) -> dict[str, Any]:
        scopes = scopes or set()
        with self.store.sessions() as session:
            row = self._owned_run(session, run_id, tenant_id)
            if row.status in {"SUCCEEDED", "BLOCKED", "FAILED"}:
                return self._execution_result(row)
            if row.status in {"RUNNING", "PLANNING"}:
                return self._execution_result(row)
            recovering = row.status == "REPLANNING"
            row.status = "PLANNING"
            row.updated_at = _now()
            session.commit()
            if row.plan_json is None or recovering:
                try:
                    if recovering and self.max_replans == 0:
                        raise AgentRuntimeError("Replanning is disabled")
                    if recovering:
                        count = len(
                            session.scalars(
                                select(EvidenceRow.evidence_id).where(
                                    EvidenceRow.run_id == run_id,
                                    EvidenceRow.tenant_id == tenant_id,
                                    EvidenceRow.kind == "replan",
                                )
                            ).all()
                        )
                        if count >= self.max_replans:
                            raise AgentRuntimeError("Maximum replanning attempts reached")
                    plan = self.planner.plan(
                        row.goal, set(self._tools), recovery=recovering
                    )
                except Exception as exc:
                    row.status = "FAILED"
                    row.error = str(exc)
                    row.updated_at = _now()
                    self._add_evidence(session, row, "planner_error", {"error": str(exc)})
                    session.commit()
                    if recovering:
                        return self._execution_result(row)
                    raise
                if recovering:
                    failed_steps = session.scalars(
                        select(PlanStepRow).where(
                            PlanStepRow.run_id == run_id,
                            PlanStepRow.tenant_id == tenant_id,
                            PlanStepRow.status == "FAILED",
                        )
                    ).all()
                    for failed_step in failed_steps:
                        failed_step.status = "SUPERSEDED"
                    self._add_evidence(
                        session,
                        row,
                        "replan",
                        {"plan_id": plan.plan_id, "step_count": len(plan.steps)},
                    )
                for step in plan.steps:
                    if step.tool_name not in self._tools:
                        raise AgentRuntimeError("Plan references an unregistered tool")
                    if not _matches_schema(step.input, self._tools[step.tool_name][0].input_schema):
                        raise AgentRuntimeError(f"Input schema mismatch for {step.tool_name}")
                    step.input = _persistable(step.input)
                    session.add(
                        PlanStepRow(
                            step_id=step.step_id,
                            run_id=row.run_id,
                            tenant_id=row.tenant_id,
                            ordinal=step.ordinal,
                            tool_name=step.tool_name,
                            input_json=step.input,
                            acceptance_json=step.acceptance_criteria,
                            status="PENDING",
                            attempts=0,
                            created_at=_now(),
                            updated_at=_now(),
                        )
                    )
                row.plan_json = plan.dict()
                row.status = "RUNNING"
                row.updated_at = _now()
                self._add_evidence(
                    session,
                    row,
                    "plan",
                    {
                        "plan_id": plan.plan_id,
                        "step_count": len(plan.steps),
                        "planner_version": "deterministic-json-v1",
                    },
                )
                session.commit()

        for step_id in self._step_ids(run_id, tenant_id):
            result = await self._execute_step(
                step_id, run_id, tenant_id, role, environment, scopes, autonomous
            )
            if result["status"] != "SUCCEEDED":
                return result

        with self.store.sessions() as session:
            row = self._owned_run(session, run_id, tenant_id)
            steps = session.scalars(
                select(PlanStepRow)
                .where(PlanStepRow.run_id == run_id, PlanStepRow.tenant_id == tenant_id)
                .order_by(PlanStepRow.ordinal)
            ).all()
            final_hash = _hash([step.result_hash for step in steps])
            row.status = "SUCCEEDED"
            row.error = None
            row.final_result_hash = final_hash
            row.updated_at = _now()
            self._add_evidence(
                session,
                row,
                "final_outcome",
                {
                    "status": row.status,
                    "result_hash": final_hash,
                    "total_cost": row.spent_cost,
                },
            )
            session.commit()
            return self._execution_result(row)

    async def resume(self, run_id: str, tenant_id: str, **kwargs: Any) -> dict[str, Any]:
        with self.store.sessions() as session:
            run = self._owned_run(session, run_id, tenant_id)
            if run.status == "PLANNING" and run.plan_json is None:
                run.status = "PENDING"
                run.updated_at = _now()
            running_steps = session.scalars(
                select(PlanStepRow).where(
                    PlanStepRow.run_id == run_id,
                    PlanStepRow.tenant_id == tenant_id,
                    PlanStepRow.status == "RUNNING",
                )
            ).all()
            if running_steps:
                all_idempotent = all(
                    step.tool_name in self._tools
                    and self._tools[step.tool_name][0].idempotent
                    for step in running_steps
                )
                call_rows = session.scalars(
                    select(ToolCallRow).where(
                        ToolCallRow.run_id == run_id,
                        ToolCallRow.tenant_id == tenant_id,
                        ToolCallRow.step_id.in_([step.step_id for step in running_steps]),
                        ToolCallRow.status == "RUNNING",
                    )
                ).all()
                for call in call_rows:
                    call.status = "INTERRUPTED"
                    call.error = "execution_interrupted"
                for step in running_steps:
                    if all_idempotent:
                        step.status = "RETRYING"
                        step.attempts = max(0, step.attempts - 1)
                    else:
                        step.status = "FAILED"
                        step.error = "execution_interrupted"
                if all_idempotent:
                    run.status = "RETRYING"
                    run.error = None
                elif any(step.tool_name not in self._tools for step in running_steps):
                    run.status = "FAILED"
                    run.error = "interrupted_tool_unavailable"
                else:
                    run.status = "REPLANNING"
                    run.error = "interrupted_non_idempotent_step"
                run.updated_at = _now()
                self._add_evidence(
                    session,
                    run,
                    "interrupted_execution",
                    {
                        "step_ids": [step.step_id for step in running_steps],
                        "recovery": run.status,
                    },
                )
                session.commit()
        return await self.execute(run_id, tenant_id, **kwargs)

    def approve(self, run_id: str, tenant_id: str, actor: str, reason: str = "") -> Approval:
        if not actor:
            raise AgentRuntimeError("Approval actor is required")
        with self.store.sessions() as session:
            row = self._owned_run(session, run_id, tenant_id)
            pending = session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.run_id == run_id,
                    ApprovalRow.tenant_id == tenant_id,
                    ApprovalRow.status == "pending",
                )
            )
            if pending is None:
                raise AgentRuntimeError("No pending approval for this run")
            now = _now()
            pending.status = "approved"
            pending.actor = actor
            pending.reason = reason
            pending.decided_at = now
            row.status = "PENDING"
            row.updated_at = now
            self._add_evidence(
                session,
                row,
                "approval",
                {
                    "actor": actor,
                    "status": "approved",
                    "step_id": pending.step_id,
                    "tool_name": pending.tool_name,
                },
            )
            session.commit()
            return Approval(
                approval_id=pending.approval_id,
                run_id=run_id,
                tenant_id=tenant_id,
                step_id=pending.step_id,
                tool_name=pending.tool_name,
                status=pending.status,
                actor=pending.actor,
                reason=pending.reason,
                created_at=pending.created_at,
                decided_at=pending.decided_at,
            )

    def list_evidence(self, run_id: str, tenant_id: str) -> list[Evidence]:
        with self.store.sessions() as session:
            self._owned_run(session, run_id, tenant_id)
            rows = session.scalars(
                select(EvidenceRow)
                .where(EvidenceRow.run_id == run_id, EvidenceRow.tenant_id == tenant_id)
                .order_by(EvidenceRow.created_at)
            ).all()
            return [
                Evidence(
                    evidence_id=item.evidence_id,
                    run_id=item.run_id,
                    tenant_id=item.tenant_id,
                    kind=item.kind,
                    sha256=item.sha256,
                    metadata=item.metadata_json,
                    created_at=item.created_at,
                )
                for item in rows
            ]

    async def _execute_step(
        self,
        step_id: str,
        run_id: str,
        tenant_id: str,
        role: str,
        environment: str,
        scopes: set[str],
        autonomous: bool,
    ) -> dict[str, Any]:
        with self.store.sessions() as session:
            run = self._owned_run(session, run_id, tenant_id)
            step = session.get(PlanStepRow, step_id)
            if step is None or step.run_id != run_id or step.tenant_id != tenant_id:
                raise AgentRuntimeError("Plan step not found")
            if step.status == "SUCCEEDED":
                return {"status": "SUCCEEDED", "run_id": run_id}
            spec, tool = self._tools[step.tool_name]
            approved = bool(
                session.scalar(
                    select(ApprovalRow).where(
                        ApprovalRow.run_id == run_id,
                        ApprovalRow.tenant_id == tenant_id,
                        ApprovalRow.step_id == step_id,
                        ApprovalRow.status == "approved",
                    )
                )
            )
            effective_autonomous = autonomous and not approved
            action, reason = self.policy.evaluate(
                run, spec, role, environment, scopes, approved, effective_autonomous
            )
            decision = PolicyDecisionRow(
                decision_id=str(uuid.uuid4()),
                run_id=run_id,
                tenant_id=tenant_id,
                tool_name=step.tool_name,
                action=action,
                reason=reason,
                created_at=_now(),
            )
            session.add(decision)
            self._add_evidence(
                session,
                run,
                "policy_decision",
                {"decision_id": decision.decision_id, "action": action, "reason": reason},
            )
            if action == "block":
                run.status = "BLOCKED"
                run.error = reason
                step.status = "BLOCKED"
                step.error = reason
                session.commit()
                return {"status": "BLOCKED", "run_id": run_id, "reason": reason}
            if action == "review":
                run.status = "WAITING_APPROVAL"
                step.status = "WAITING_APPROVAL"
                if not session.scalar(
                    select(ApprovalRow).where(
                        ApprovalRow.run_id == run_id,
                        ApprovalRow.tenant_id == tenant_id,
                        ApprovalRow.step_id == step_id,
                        ApprovalRow.status == "pending",
                    )
                ):
                    session.add(
                        ApprovalRow(
                            approval_id=str(uuid.uuid4()),
                            run_id=run_id,
                            tenant_id=tenant_id,
                            step_id=step_id,
                            tool_name=spec.name,
                            status="pending",
                            created_at=_now(),
                        )
                    )
                session.commit()
                return {
                    "status": "WAITING_APPROVAL",
                    "run_id": run_id,
                    "decision_id": decision.decision_id,
                }
            claim = session.execute(
                update(PlanStepRow)
                .where(
                    PlanStepRow.step_id == step_id,
                    PlanStepRow.tenant_id == tenant_id,
                    PlanStepRow.status.in_(
                        ["PENDING", "RETRYING", "WAITING_APPROVAL"]
                    ),
                )
                .values(
                    status="RUNNING",
                    attempts=PlanStepRow.attempts + 1,
                    updated_at=_now(),
                )
                .execution_options(synchronize_session=False)
            )
            if claim.rowcount != 1:
                session.refresh(step)
                session.commit()
                return {"status": step.status, "run_id": run_id}
            session.refresh(step)
            run.status = "RUNNING"
            risk_rank = {RiskLevel.LOW.value: 0, RiskLevel.MEDIUM.value: 1, RiskLevel.HIGH.value: 2}
            current_risk = run.risk
            run.risk = max(
                (current_risk, spec.risk_level.value), key=lambda value: risk_rank[value]
            )
            run.updated_at = _now()
            safe_input = step.input_json
            call_id = str(uuid.uuid4())
            input_hash = _hash(safe_input)
            session.add(
                ToolCallRow(
                    call_id=call_id,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    step_id=step_id,
                    tool_name=spec.name,
                    status="RUNNING",
                    input_hash=input_hash,
                    result_metadata={},
                    created_at=_now(),
                )
            )
            self._add_evidence(
                session,
                run,
                "tool_call",
                {
                    "tool_name": spec.name,
                    "call_id": call_id,
                    "input_hash": input_hash,
                },
            )
            session.commit()

        try:
            if inspect.iscoroutinefunction(tool):
                raw_result = await asyncio.wait_for(tool(safe_input), self.step_timeout)
            else:
                raw_result = await asyncio.wait_for(
                    asyncio.to_thread(tool, safe_input), self.step_timeout
                )
            result = _persistable(raw_result)
            valid, verification_reason = self.verifier.verify(
                result, spec.output_schema, step.acceptance_json
            )
        except asyncio.TimeoutError:
            result = None
            valid = False
            verification_reason = "tool_timeout"
        except Exception as exc:
            result = None
            valid = False
            verification_reason = f"tool_error:{type(exc).__name__}"

        with self.store.sessions() as session:
            run = self._owned_run(session, run_id, tenant_id)
            step = session.get(PlanStepRow, step_id)
            call = session.get(ToolCallRow, call_id)
            if valid:
                output_hash = _hash(result)
                step.status = "SUCCEEDED"
                step.result_hash = output_hash
                step.error = None
                run.spent_cost += spec.max_cost
                run.updated_at = _now()
                call.status = "SUCCEEDED"
                call.output_hash = output_hash
                call.result_metadata = {
                    "type": type(result).__name__,
                    "size_bytes": len(json.dumps(result, default=str).encode()),
                }
                self._add_evidence(
                    session,
                    run,
                    "tool_result",
                    {
                        "tool_name": spec.name,
                        "call_id": call_id,
                        "output_hash": output_hash,
                        "verified": True,
                        "estimated_cost": spec.max_cost,
                        "budget_remaining": max(0.0, run.budget - run.spent_cost),
                    },
                )
                session.commit()
                return {"status": "SUCCEEDED", "run_id": run_id, "result": result}

            error = verification_reason
            call.status = "FAILED"
            call.error = error
            step.error = error
            if step.attempts <= self.max_retries and spec.idempotent:
                step.status = "RETRYING"
                run.status = "RETRYING"
            else:
                step.status = "FAILED"
                run.status = "REPLANNING"
                run.error = error
            self._add_evidence(
                session,
                run,
                "tool_failure",
                {"tool_name": spec.name, "call_id": call_id, "reason": error},
            )
            session.commit()
            if step.status == "RETRYING":
                return await self._execute_step(
                    step_id, run_id, tenant_id, role, environment, scopes, autonomous
                )
            return {"status": run.status, "run_id": run_id, "error": error}

    def _step_ids(self, run_id: str, tenant_id: str) -> list[str]:
        with self.store.sessions() as session:
            return list(
                session.scalars(
                    select(PlanStepRow.step_id)
                    .where(
                        PlanStepRow.run_id == run_id,
                        PlanStepRow.tenant_id == tenant_id,
                        PlanStepRow.status.in_(
                            ["PENDING", "RETRYING", "WAITING_APPROVAL"]
                        ),
                    )
                    .order_by(PlanStepRow.ordinal)
                ).all()
            )

    @staticmethod
    def _owned_run(session: Any, run_id: str, tenant_id: str) -> AgentRunRow:
        row = session.get(AgentRunRow, run_id)
        if row is None or row.tenant_id != tenant_id:
            raise AgentRuntimeError("Agent run not found")
        return row

    @staticmethod
    def _add_evidence(session: Any, run: AgentRunRow, kind: str, metadata: dict[str, Any]) -> None:
        safe_metadata = _persistable(metadata)
        session.add(
            EvidenceRow(
                evidence_id=str(uuid.uuid4()),
                run_id=run.run_id,
                tenant_id=run.tenant_id,
                kind=kind,
                sha256=_hash(safe_metadata),
                metadata_json=safe_metadata,
                created_at=_now(),
            )
        )

    def _as_run(self, row: AgentRunRow) -> AgentRun:
        plan = Plan.parse_obj(row.plan_json) if row.plan_json else None
        return AgentRun(
            run_id=row.run_id,
            tenant_id=row.tenant_id,
            goal=row.goal,
            status=row.status,
            risk=RiskLevel(row.risk),
            budget=row.budget,
            spent_cost=row.spent_cost,
            error=row.error,
            created_at=row.created_at,
            updated_at=row.updated_at,
            plan=plan,
            final_result_hash=row.final_result_hash,
        )

    @staticmethod
    def _execution_result(row: AgentRunRow) -> dict[str, Any]:
        return {
            "status": row.status,
            "run_id": row.run_id,
            "error": row.error,
            "result_hash": row.final_result_hash,
        }
