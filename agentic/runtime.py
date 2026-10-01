"""Durable, policy-gated agent execution with a deterministic planner."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
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
    inspect as sqlalchemy_inspect,
    select,
    text,
    update,
)

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from policy.agent_policy import AgentPolicyGate
from agentic.capabilities import CapabilityCatalog
from agentic.sandbox import SandboxBoundary, SandboxProfile


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
    version: str = "1.0.0"
    sandbox_profile: str = SandboxProfile.PURE.value
    timeout_seconds: Optional[float] = None
    approval_sla_seconds: Optional[int] = None
    max_input_bytes: int = 65536
    max_output_bytes: int = 65536


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
    requested_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    sla_seconds: int = 3600
    escalation_status: str = "none"
    denial_reason: Optional[str] = None
    capability_version: Optional[str] = None
    plan_hash: Optional[str] = None
    policy_version: Optional[str] = None


class Evidence(BaseModel):
    evidence_id: str
    run_id: str
    tenant_id: str
    kind: str
    sha256: str
    previous_sha256: Optional[str] = None
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
    requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    sla_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=3600)
    escalation_status: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    denial_reason: Mapped[Optional[str]] = mapped_column(Text)
    capability_version: Mapped[Optional[str]] = mapped_column(String(64))
    plan_hash: Mapped[Optional[str]] = mapped_column(String(64))
    policy_version: Mapped[Optional[str]] = mapped_column(String(64))


class CapabilityRow(AgentBase):
    __tablename__ = "agent_capabilities"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    spec_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EvidenceRow(AgentBase):
    __tablename__ = "agent_evidence"

    evidence_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_sha256: Mapped[Optional[str]] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


AgentRun.update_forward_refs()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_SENSITIVE_KEY = re.compile(
    r"(secret|password|token|"
    r"credential(?![_-]?id\b)|authorization|api.?key)",
    re.I,
)


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
    if isinstance(value, float) and not math.isfinite(value):
        raise AgentRuntimeError("Tool inputs and outputs must use finite numbers")
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
        existing = {
            column["name"]
            for column in sqlalchemy_inspect(self.engine).get_columns("agent_approvals")
        }
        evidence_columns = {
            column["name"]
            for column in sqlalchemy_inspect(self.engine).get_columns("agent_evidence")
        }
        additions = {
            "requested_at": "TIMESTAMP",
            "expires_at": "TIMESTAMP",
            "sla_seconds": "INTEGER NOT NULL DEFAULT 3600",
            "escalation_status": "VARCHAR(32) NOT NULL DEFAULT 'none'",
            "denial_reason": "TEXT",
            "capability_version": "VARCHAR(64)",
            "plan_hash": "VARCHAR(64)",
            "policy_version": "VARCHAR(64)",
        }
        with self.engine.begin() as connection:
            for name, sql_type in additions.items():
                if name not in existing:
                    connection.execute(
                        text(f"ALTER TABLE agent_approvals ADD COLUMN {name} {sql_type}")
                    )
            if "previous_sha256" not in evidence_columns:
                connection.execute(
                    text("ALTER TABLE agent_evidence ADD COLUMN previous_sha256 VARCHAR(64)")
                )
        with self.sessions() as session:
            legacy_pending = session.scalars(
                select(ApprovalRow).where(
                    ApprovalRow.status == "pending",
                    ApprovalRow.expires_at.is_(None),
                )
            ).all()
            for approval in legacy_pending:
                approval.requested_at = approval.created_at
                approval.sla_seconds = 3600
                approval.expires_at = approval.created_at + timedelta(hours=1)
                approval.status = "expired"
                approval.decided_at = _now()
                approval.denial_reason = "legacy_approval_failed_closed"
                run = session.get(AgentRunRow, approval.run_id)
                if run and run.tenant_id == approval.tenant_id:
                    run.status = "BLOCKED"
                    run.error = "approval_expired"
            session.commit()


class Planner:
    def plan(
        self, goal: str, registered_tools: set[str], recovery: bool = False
    ) -> Plan:
        raise NotImplementedError


class DeterministicJSONPlanner(Planner):
    """Accepts JSON {tool, input, acceptance} or {steps: [...]}; never runs code."""

    max_steps = 32

    def plan(
        self, goal: str, registered_tools: set[str], recovery: bool = False
    ) -> Plan:
        try:
            request = json.loads(
                goal,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite JSON number")
                ),
            )
        except (TypeError, json.JSONDecodeError):
            if len(registered_tools) != 1:
                raise AgentRuntimeError(
                    "Provide a JSON plan naming a registered tool when multiple tools exist"
                )
            request = {"tool": next(iter(registered_tools)), "input": {"goal": goal}}
        allowed_request_fields = {"tool", "input", "acceptance", "steps", "recovery_steps"}
        if isinstance(request, dict) and set(request) - allowed_request_fields:
            raise AgentRuntimeError("Planner returned unsupported request fields")
        steps = (
            request.get("recovery_steps")
            if recovery and isinstance(request, dict)
            else request.get("steps") if isinstance(request, dict) else None
        )
        if recovery and steps is None:
            raise AgentRuntimeError("No recovery plan provided")
        if steps is None and isinstance(request, dict) and "tool" in request:
            steps = [
                {key: request[key] for key in ("tool", "input", "acceptance") if key in request}
            ]
        if not isinstance(steps, list) or not steps or len(steps) > self.max_steps:
            raise AgentRuntimeError("Planner requires one or more tool steps")
        plan_steps = []
        for ordinal, spec in enumerate(steps):
            if not isinstance(spec, dict):
                raise AgentRuntimeError("Each planned step must be a JSON object")
            if set(spec) - {"tool", "input", "acceptance"}:
                raise AgentRuntimeError("Planner returned unsupported step fields")
            tool_name = spec.get("tool")
            if not isinstance(tool_name, str) or tool_name not in registered_tools:
                raise AgentRuntimeError(f"Planner selected an unregistered tool: {tool_name}")
            tool_input = spec.get("input", {})
            acceptance = spec.get("acceptance", {})
            if not isinstance(tool_input, dict) or not isinstance(acceptance, dict):
                raise AgentRuntimeError("Tool input and acceptance criteria must be JSON objects")
            if set(acceptance) - {"required_keys", "equals"}:
                raise AgentRuntimeError("Unsupported acceptance criteria")
            if not isinstance(acceptance.get("required_keys", []), list) or any(
                not isinstance(key, str) for key in acceptance.get("required_keys", [])
            ):
                raise AgentRuntimeError("Acceptance required_keys must be strings")
            if not isinstance(acceptance.get("equals", {}), dict):
                raise AgentRuntimeError("Acceptance equals must be a JSON object")
            try:
                json.dumps([tool_input, acceptance], allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise AgentRuntimeError("Planner values must be strict JSON") from exc
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
        if expected_type in {"integer", "number"} and isinstance(value, bool):
            return False
        return False
    if expected_type in {"integer", "number"} and isinstance(value, bool):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
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
        sandbox: Optional[SandboxBoundary] = None,
        capability_enforcement: Optional[bool] = None,
        approval_sla_seconds: Optional[int] = None,
    ):
        self.store = store or AgentStore()
        if planner is None:
            from agentic.planner import OpenAICompatiblePlanner

            planner = OpenAICompatiblePlanner.from_environment()
        self.planner = planner
        self.policy = policy or AgentPolicyGate()
        self.verifier = verifier or Verifier()
        self.max_retries = max(0, max_retries)
        self.max_replans = max(0, max_replans)
        self.step_timeout = step_timeout
        self.sandbox = sandbox or SandboxBoundary()
        self.capability_enforcement = (
            capability_enforcement
            if capability_enforcement is not None
            else os.getenv("AEGIS_CAPABILITY_ENFORCEMENT", "").lower()
            in {"1", "true", "yes"}
        )
        try:
            sla = (
                approval_sla_seconds
                if approval_sla_seconds is not None
                else int(os.getenv("AEGIS_APPROVAL_SLA_SECONDS", "3600"))
            )
        except ValueError:
            sla = 3600
        self.approval_sla_seconds = max(1, int(sla))
        self.catalog = CapabilityCatalog()
        self.approval_notifier: Optional[Callable[[Approval], Any]] = None
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register_tool(self, spec: ToolSpec, tool: Callable[..., Any]) -> None:
        if not spec.name or len(spec.name) > 128 or not callable(tool):
            raise AgentRuntimeError("A named callable is required to register a tool")
        if not math.isfinite(spec.max_cost) or spec.max_cost < 0:
            raise AgentRuntimeError("Tool max_cost cannot be negative")
        if spec.max_input_bytes <= 0 or spec.max_output_bytes <= 0:
            raise AgentRuntimeError("Tool payload limits must be positive")
        if spec.timeout_seconds is not None and (
            not math.isfinite(spec.timeout_seconds) or spec.timeout_seconds <= 0
        ):
            raise AgentRuntimeError("Tool timeout must be positive")
        if spec.sandbox_profile not in {profile.value for profile in SandboxProfile}:
            raise AgentRuntimeError("Unknown sandbox profile")
        try:
            safe_spec = _persistable(spec.dict())
            spec_size = len(
                json.dumps(safe_spec, default=str, allow_nan=False).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise AgentRuntimeError("Tool metadata must be JSON-compatible") from exc
        if spec_size > 32768:
            raise AgentRuntimeError("Tool metadata exceeds the catalog size limit")
        self._tools[spec.name] = (spec, tool)
        self.catalog.register(spec)
        self.store.initialize()
        with self.store.sessions() as session:
            session.merge(
                CapabilityRow(
                    name=spec.name,
                    version=spec.version,
                    spec_json=safe_spec,
                    spec_hash=_hash(safe_spec),
                    updated_at=_now(),
                )
            )
            session.commit()

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
        principal_id: Optional[str] = None,
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
                    spec = self._tools[step.tool_name][0]
                    if self.capability_enforcement and not self._capability_is_active(spec):
                        raise AgentRuntimeError("Plan references an inactive capability")
                    if not _matches_schema(step.input, spec.input_schema):
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
                        "planner_version": getattr(
                            self.planner, "provider_name", "deterministic-json-v1"
                        ),
                        "model": getattr(self.planner, "model", None),
                        "catalog_hash": self.catalog.sha256,
                    },
                )
                session.commit()

        for step_id in self._step_ids(run_id, tenant_id):
            result = await self._execute_step(
                step_id,
                run_id,
                tenant_id,
                role,
                environment,
                scopes,
                autonomous,
                principal_id,
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
            if pending.expires_at and pending.expires_at <= now:
                pending.status = "expired"
                pending.decided_at = now
                row.status = "BLOCKED"
                row.error = "approval_expired"
                session.commit()
                raise AgentRuntimeError("Approval has expired")
            spec = self._tools.get(pending.tool_name)
            if (
                spec is None
                or self._active_capability_version(spec[0]) != pending.capability_version
            ):
                raise AgentRuntimeError("Approval capability version is no longer active")
            if (
                pending.plan_hash != _hash(row.plan_json or {})
                or pending.policy_version != self.policy.version
            ):
                raise AgentRuntimeError("Approval context is no longer active")
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
                requested_at=pending.requested_at,
                expires_at=pending.expires_at,
                sla_seconds=pending.sla_seconds,
                escalation_status=pending.escalation_status,
                denial_reason=pending.denial_reason,
                capability_version=pending.capability_version,
            )

    def deny(self, run_id: str, tenant_id: str, actor: str, reason: str) -> Approval:
        if not actor or not reason:
            raise AgentRuntimeError("Approval actor and denial reason are required")
        with self.store.sessions() as session:
            run = self._owned_run(session, run_id, tenant_id)
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
            pending.status = "denied"
            pending.actor = actor
            pending.reason = reason
            pending.denial_reason = reason
            pending.decided_at = now
            run.status = "BLOCKED"
            run.error = "approval_denied"
            self._add_evidence(
                session,
                run,
                "approval",
                {
                    "status": "denied",
                    "actor": actor,
                    "step_id": pending.step_id,
                    "tool_name": pending.tool_name,
                },
            )
            session.commit()
            return self._as_approval(pending)

    def list_approvals(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Approval]:
        with self.store.sessions() as session:
            query = select(ApprovalRow)
            if tenant_id is not None:
                query = query.where(ApprovalRow.tenant_id == tenant_id)
            if status is not None:
                query = query.where(ApprovalRow.status == status)
            rows = session.scalars(
                query.order_by(ApprovalRow.created_at)
                .offset(max(0, offset))
                .limit(max(1, min(limit, 100)))
            ).all()
            return [
                self._as_approval(row)
                for row in rows
            ]

    def list_runs(
        self, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> list[AgentRun]:
        with self.store.sessions() as session:
            rows = session.scalars(
                select(AgentRunRow)
                .where(AgentRunRow.tenant_id == tenant_id)
                .order_by(AgentRunRow.created_at.desc())
                .offset(max(0, offset))
                .limit(max(1, min(limit, 100)))
            ).all()
            return [self._as_run(row) for row in rows]

    def list_timeline(
        self, run_id: str, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        entries = [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind,
                "sha256": item.sha256,
                "metadata": item.metadata,
                "created_at": item.created_at,
            }
            for item in self.list_evidence(run_id, tenant_id)
        ]
        page_offset = max(0, offset)
        page_limit = max(1, min(limit, 100))
        return {
            "items": entries[page_offset:page_offset + page_limit],
            "offset": page_offset,
            "limit": page_limit,
            "next_offset": (
                page_offset + page_limit if page_offset + page_limit < len(entries) else None
            ),
        }

    def record_evidence(
        self, run_id: str, tenant_id: str, kind: str, metadata: dict[str, Any]
    ) -> str:
        safe_metadata = _persistable(metadata)
        with self.store.sessions() as session:
            run = self._owned_run(session, run_id, tenant_id)
            matching = session.scalars(
                select(EvidenceRow).where(
                    EvidenceRow.run_id == run_id,
                    EvidenceRow.tenant_id == tenant_id,
                    EvidenceRow.kind == kind,
                )
            ).all()
            for item in matching:
                if item.metadata_json == safe_metadata:
                    return item.evidence_id
            evidence_id = self._add_evidence(session, run, kind, safe_metadata)
            session.commit()
            return evidence_id

    def has_evidence(
        self,
        run_id: str,
        tenant_id: str,
        kind: str,
        metadata_key: Optional[str] = None,
        metadata_value: Optional[Any] = None,
    ) -> bool:
        return any(
            item.kind == kind
            and (
                metadata_key is None
                or item.metadata.get(metadata_key) == metadata_value
            )
            for item in self.list_evidence(run_id, tenant_id)
        )

    def list_remediation_events(
        self, tenant_id: str, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        events = []
        for run in self.list_runs(tenant_id, 0, 100):
            for item in self.list_evidence(run.run_id, tenant_id):
                if item.kind in {"drift_finding", "remediation_proposal"}:
                    events.append(
                        {
                            "run_id": run.run_id,
                            "status": run.status,
                            "evidence_id": item.evidence_id,
                            "kind": item.kind,
                            "sha256": item.sha256,
                            "metadata": item.metadata,
                            "created_at": item.created_at,
                        }
                    )
        page_offset = max(0, offset)
        page_limit = max(1, min(limit, 100))
        return {
            "items": events[page_offset:page_offset + page_limit],
            "offset": page_offset,
            "limit": page_limit,
            "next_offset": (
                page_offset + page_limit if page_offset + page_limit < len(events) else None
            ),
        }

    def list_blocked_decisions(self, tenant_id: str) -> list[dict[str, Any]]:
        with self.store.sessions() as session:
            rows = session.scalars(
                select(PolicyDecisionRow)
                .where(
                    PolicyDecisionRow.tenant_id == tenant_id,
                    PolicyDecisionRow.action == "block",
                )
                .order_by(PolicyDecisionRow.created_at.desc())
                .limit(100)
            ).all()
            return [
                {
                    "decision_id": row.decision_id,
                    "run_id": row.run_id,
                    "tool_name": row.tool_name,
                    "reason": row.reason,
                    "created_at": row.created_at,
                }
                for row in rows
            ]

    def expire_approvals(self) -> int:
        now = _now()
        expired = 0
        with self.store.sessions() as session:
            pending = session.scalars(
                select(ApprovalRow).where(
                    ApprovalRow.status == "pending",
                    ApprovalRow.expires_at <= now,
                )
            ).all()
            for approval in pending:
                approval.status = "expired"
                approval.decided_at = now
                approval.denial_reason = "approval_sla_expired"
                run = self._owned_run(session, approval.run_id, approval.tenant_id)
                run.status = "BLOCKED"
                run.error = "approval_expired"
                self._add_evidence(
                    session,
                    run,
                    "approval",
                    {
                        "status": "expired",
                        "step_id": approval.step_id,
                        "tool_name": approval.tool_name,
                    },
                )
                expired += 1
            session.commit()
        return expired

    def escalate_approvals(self) -> int:
        now = _now()
        with self.store.sessions() as session:
            pending = session.scalars(
                select(ApprovalRow).where(
                    ApprovalRow.status == "pending",
                    ApprovalRow.expires_at > now,
                    ApprovalRow.escalation_status == "none",
                )
            ).all()
            pending = [
                approval
                for approval in pending
                if approval.requested_at
                and approval.requested_at
                + timedelta(seconds=max(1, approval.sla_seconds // 2))
                <= now
            ]
            for approval in pending:
                approval.escalation_status = "escalated"
            session.commit()
            return len(pending)

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
                    previous_sha256=item.previous_sha256,
                    metadata=item.metadata_json,
                    created_at=item.created_at,
                )
                for item in rows
            ]

    def verify_evidence_chain(self, run_id: str, tenant_id: str) -> dict[str, Any]:
        """Verify the run's append-only hash links and return its current chain head."""
        entries = self.list_evidence(run_id, tenant_id)
        if not entries:
            return {"valid": True, "count": 0, "head_sha256": None}
        by_hash = {entry.sha256: entry for entry in entries}
        if len(by_hash) != len(entries):
            return {"valid": False, "count": len(entries), "head_sha256": None}
        roots = [entry for entry in entries if entry.previous_sha256 is None]
        if len(roots) != 1:
            return {"valid": False, "count": len(entries), "head_sha256": None}
        current = roots[0]
        visited: set[str] = set()
        while True:
            if current.sha256 in visited:
                return {"valid": False, "count": len(entries), "head_sha256": None}
            visited.add(current.sha256)
            expected = _hash(
                {
                    "previous_sha256": current.previous_sha256,
                    "kind": current.kind,
                    "metadata": current.metadata,
                }
            )
            if expected != current.sha256:
                return {"valid": False, "count": len(entries), "head_sha256": None}
            children = [
                entry for entry in entries if entry.previous_sha256 == current.sha256
            ]
            if not children:
                break
            if len(children) != 1:
                return {"valid": False, "count": len(entries), "head_sha256": None}
            current = children[0]
        return {
            "valid": len(visited) == len(entries),
            "count": len(entries),
            "head_sha256": current.sha256 if len(visited) == len(entries) else None,
        }

    async def _execute_step(
        self,
        step_id: str,
        run_id: str,
        tenant_id: str,
        role: str,
        environment: str,
        scopes: set[str],
        autonomous: bool,
        principal_id: Optional[str],
    ) -> dict[str, Any]:
        with self.store.sessions() as session:
            run = self._owned_run(session, run_id, tenant_id)
            step = session.get(PlanStepRow, step_id)
            if step is None or step.run_id != run_id or step.tenant_id != tenant_id:
                raise AgentRuntimeError("Plan step not found")
            if step.status == "SUCCEEDED":
                return {"status": "SUCCEEDED", "run_id": run_id}
            spec, tool = self._tools[step.tool_name]
            try:
                if self.capability_enforcement and not self._capability_is_active(spec):
                    raise AgentRuntimeError("inactive_capability")
                if len(
                    json.dumps(step.input_json, separators=(",", ":"), default=str).encode()
                ) > min(spec.max_input_bytes, self.sandbox.max_payload_bytes):
                    raise AgentRuntimeError("sandbox_input_too_large")
                self.sandbox.check_input(
                    step.input_json, spec.sandbox_profile, environment
                )
            except (AgentRuntimeError, ValueError) as exc:
                run.status = "BLOCKED"
                run.error = str(exc)
                step.status = "BLOCKED"
                step.error = str(exc)
                self._add_evidence(
                    session,
                    run,
                    "sandbox_block",
                    {
                        "tool_name": spec.name,
                        "sandbox_profile": spec.sandbox_profile,
                        "reason": str(exc),
                    },
                )
                session.commit()
                return {"status": "BLOCKED", "run_id": run_id, "reason": str(exc)}
            active_capability = self._active_capability_version(spec)
            plan_hash = _hash(run.plan_json or {})
            policy_version = self.policy.version
            approved = bool(
                session.scalar(
                    select(ApprovalRow).where(
                        ApprovalRow.run_id == run_id,
                        ApprovalRow.tenant_id == tenant_id,
                        ApprovalRow.step_id == step_id,
                        ApprovalRow.status == "approved",
                        ApprovalRow.capability_version == active_capability,
                        ApprovalRow.plan_hash == plan_hash,
                        ApprovalRow.policy_version == policy_version,
                    )
                )
            )
            effective_autonomous = autonomous and not approved
            action, reason = self.policy.evaluate(
                run,
                spec,
                role,
                environment,
                scopes,
                approved,
                effective_autonomous,
                principal_id,
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
                {
                    "decision_id": decision.decision_id,
                    "action": action,
                    "reason": reason,
                    "policy_version": self.policy.version,
                    "principal_id": principal_id,
                    "engines": [
                        {
                            "name": getattr(engine, "engine_name", type(engine).__name__),
                            "bundle_sha256": getattr(engine, "bundle_sha256", ""),
                        }
                        for engine in self.policy.engines
                    ],
                },
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
                pending_approval = session.scalar(
                    select(ApprovalRow).where(
                        ApprovalRow.run_id == run_id,
                        ApprovalRow.tenant_id == tenant_id,
                        ApprovalRow.step_id == step_id,
                        ApprovalRow.status == "pending",
                    )
                )
                new_approval = None
                if pending_approval is None:
                    new_approval = ApprovalRow(
                        approval_id=str(uuid.uuid4()),
                        run_id=run_id,
                        tenant_id=tenant_id,
                        step_id=step_id,
                        tool_name=spec.name,
                        status="pending",
                        created_at=_now(),
                        requested_at=_now(),
                        expires_at=_now()
                        + timedelta(
                            seconds=spec.approval_sla_seconds
                            or self.approval_sla_seconds
                        ),
                        sla_seconds=spec.approval_sla_seconds
                        or self.approval_sla_seconds,
                        escalation_status="none",
                        capability_version=active_capability,
                        plan_hash=plan_hash,
                        policy_version=policy_version,
                    )
                    session.add(new_approval)
                session.commit()
                if new_approval and self.approval_notifier:
                    try:
                        approval_payload = self._as_approval(new_approval)
                        if inspect.iscoroutinefunction(self.approval_notifier):
                            await asyncio.wait_for(
                                self.approval_notifier(approval_payload),
                                timeout=self.step_timeout,
                            )
                        else:
                            callback_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    self.approval_notifier, approval_payload
                                ),
                                timeout=self.step_timeout,
                            )
                            if inspect.isawaitable(callback_result):
                                await asyncio.wait_for(
                                    callback_result, timeout=self.step_timeout
                                )
                        notification_status = "sent"
                    except Exception:
                        notification_status = "failed"
                    self.record_evidence(
                        run_id,
                        tenant_id,
                        "approval_notification",
                        {
                            "approval_id": new_approval.approval_id,
                            "status": notification_status,
                        },
                    )
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
                    result_metadata={
                        "sandbox_profile": spec.sandbox_profile,
                        "max_input_bytes": min(
                            spec.max_input_bytes, self.sandbox.max_payload_bytes
                        ),
                        "max_output_bytes": min(
                            spec.max_output_bytes, self.sandbox.max_output_bytes
                        ),
                        "timeout_seconds": spec.timeout_seconds or self.step_timeout,
                    },
                    created_at=_now(),
                )
            )
            return evidence_id
            self._add_evidence(
                session,
                run,
                "tool_call",
                {
                    "tool_name": spec.name,
                    "call_id": call_id,
                    "input_hash": input_hash,
                    "sandbox_profile": spec.sandbox_profile,
                    "max_input_bytes": min(
                        spec.max_input_bytes, self.sandbox.max_payload_bytes
                    ),
                    "max_output_bytes": min(
                        spec.max_output_bytes, self.sandbox.max_output_bytes
                    ),
                    "timeout_seconds": spec.timeout_seconds or self.step_timeout,
                },
            )
            session.commit()

        try:
            if inspect.iscoroutinefunction(tool):
                raw_result = await asyncio.wait_for(
                    tool(safe_input), spec.timeout_seconds or self.step_timeout
                )
            else:
                raw_result = await asyncio.wait_for(
                    asyncio.to_thread(tool, safe_input),
                    spec.timeout_seconds or self.step_timeout,
                )
            result = _persistable(raw_result)
            self.sandbox.check_output(result)
            output_size = len(json.dumps(result, default=str).encode())
            if output_size > spec.max_output_bytes:
                raise ValueError("sandbox_output_too_large")
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
                    "size_bytes": output_size,
                    "sandbox_profile": spec.sandbox_profile,
                    "max_input_bytes": min(
                        spec.max_input_bytes, self.sandbox.max_payload_bytes
                    ),
                    "max_output_bytes": min(
                        spec.max_output_bytes, self.sandbox.max_output_bytes
                    ),
                    "timeout_seconds": spec.timeout_seconds or self.step_timeout,
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
                    step_id, run_id, tenant_id, role, environment, scopes, autonomous,
                    principal_id,
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
    def _add_evidence(
        session: Any, run: AgentRunRow, kind: str, metadata: dict[str, Any]
    ) -> str:
        safe_metadata = _persistable(metadata)
        previous = session.scalar(
            select(EvidenceRow)
            .where(
                EvidenceRow.run_id == run.run_id,
                EvidenceRow.tenant_id == run.tenant_id,
            )
            .order_by(EvidenceRow.created_at.desc(), EvidenceRow.evidence_id.desc())
            .limit(1)
        )
        previous_sha256 = previous.sha256 if previous else None
        digest = _hash(
            {
                "previous_sha256": previous_sha256,
                "kind": kind,
                "metadata": safe_metadata,
            }
        )
        evidence_id = str(uuid.uuid4())
        session.add(
            EvidenceRow(
                evidence_id=evidence_id,
                run_id=run.run_id,
                tenant_id=run.tenant_id,
                kind=kind,
                sha256=digest,
                previous_sha256=previous_sha256,
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
    def _as_approval(row: ApprovalRow) -> Approval:
        return Approval(
            approval_id=row.approval_id,
            run_id=row.run_id,
            tenant_id=row.tenant_id,
            step_id=row.step_id,
            tool_name=row.tool_name,
            status=row.status,
            actor=row.actor,
            reason=row.reason,
            created_at=row.created_at,
            decided_at=row.decided_at,
            requested_at=row.requested_at,
            expires_at=row.expires_at,
            sla_seconds=row.sla_seconds,
            escalation_status=row.escalation_status,
            denial_reason=row.denial_reason,
            capability_version=row.capability_version,
            plan_hash=row.plan_hash,
            policy_version=row.policy_version,
        )


    def _active_capability_version(self, spec: ToolSpec) -> str:
        with self.store.sessions() as session:
            row = session.get(CapabilityRow, spec.name)
            return row.spec_hash if row else _hash(_persistable(spec.dict()))

    def _capability_is_active(self, spec: ToolSpec) -> bool:
        with self.store.sessions() as session:
            row = session.get(CapabilityRow, spec.name)
            return bool(
                row
                and row.version == spec.version
                and row.spec_hash == _hash(row.spec_json)
                and row.spec_hash == _hash(_persistable(spec.dict()))
            )

    @staticmethod
    def _execution_result(row: AgentRunRow) -> dict[str, Any]:
        return {
            "status": row.status,
            "run_id": row.run_id,
            "error": row.error,
            "result_hash": row.final_result_hash,
        }
