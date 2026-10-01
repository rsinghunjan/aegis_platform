from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


DecisionEngine = Literal[
    "rbac",
    "opa-central-http",
    "opa-sidecar-http",
    "opa-embedded-wasm",
    "opa-embedded-subprocess",
]

PinScopeUsed = Literal["environment", "project", "global"]

Decision = Literal["allow", "deny"]


@dataclass(frozen=True)
class Obligation:
    """
    OPA Pattern-2 obligation object.
    Obligation semantics must be enforced by the caller (control plane / runner).
    """

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """
    Canonical Pattern-2 decision returned by policy engines.
    """

    allow: bool
    reason: str = ""
    policy_id: str | None = None
    policy_version: str | None = None
    obligations: Sequence[Obligation] = field(default_factory=tuple)
    labels: Mapping[str, Any] = field(default_factory=dict)

    def decision(self) -> Decision:
        return "allow" if self.allow else "deny"


@dataclass(frozen=True)
class PrincipalRole:
    role: str
    scope_type: Literal["org", "project", "environment"]
    scope_id: str


@dataclass(frozen=True)
class RequestActor:
    principal_id: str
    principal_type: Literal["user", "service_account"]
    org_id: str
    roles: Sequence[PrincipalRole] = field(default_factory=tuple)
    claims: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestMeta:
    request_id: str
    ts: str  # ISO8601 string (caller supplies); keep as str to avoid tz pitfalls
    action: str  # domain-action e.g. "job.submit"
    actor: RequestActor


@dataclass(frozen=True)
class TypedResource:
    type: str  # e.g. "job", "deployment", "model_version"
    id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentContext:
    id: str
    risk_tier: str
    cloud_targets: Sequence[str] = field(default_factory=tuple)
    region_constraints: Sequence[str] = field(default_factory=tuple)
    budget: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyContext:
    bundle_sha256: str
    pin_scope_used: PinScopeUsed
    mode: str  # e.g. "central+sidecar"
    engine: str  # e.g. "opa-central-http"


@dataclass(frozen=True)
class PolicyInput:
    """
    Canonical input document for OPA Pattern-2 decision.
    """

    request: RequestMeta
    resource: TypedResource
    environment: EnvironmentContext
    policy: PolicyContext

    def to_json(self) -> dict[str, Any]:
        def role_to_json(r: PrincipalRole) -> dict[str, Any]:
            return {"role": r.role, "scope_type": r.scope_type, "scope_id": r.scope_id}

        return {
            "request": {
                "request_id": self.request.request_id,
                "ts": self.request.ts,
                "action": self.request.action,
                "actor": {
                    "principal_id": self.request.actor.principal_id,
                    "principal_type": self.request.actor.principal_type,
                    "org_id": self.request.actor.org_id,
                    "roles": [role_to_json(r) for r in self.request.actor.roles],
                    "claims": dict(self.request.actor.claims),
                },
            },
            "resource": {
                "type": self.resource.type,
                "id": self.resource.id,
                "attributes": dict(self.resource.attributes),
            },
            "environment": {
                "id": self.environment.id,
                "risk_tier": self.environment.risk_tier,
                "cloud_targets": list(self.environment.cloud_targets),
                "region_constraints": list(self.environment.region_constraints),
                "budget": dict(self.environment.budget),
            },
            "policy": {
                "bundle_sha256": self.policy.bundle_sha256,
                "pin_scope_used": self.policy.pin_scope_used,
                "mode": self.policy.mode,
                "engine": self.policy.engine,
            },
        }


@dataclass(frozen=True)
class EngineDecisionRecord:
    """
    Captures per-engine decision for evidence and deny-on-disagree.
    """

    engine: DecisionEngine
    decision: PolicyDecision
    bundle_sha256: str
    policy_id: str | None = None
    policy_version: str | None = None
    latency_ms: int | None = None
    raw: Mapping[str, Any] | None = None


def obligations_equal(a: Sequence[Obligation], b: Sequence[Obligation]) -> bool:
    """
    Strict equality (type + params) with stable ordering.
    Callers should order obligations deterministically.
    """
    if len(a) != len(b):
        return False
    for oa, ob in zip(a, b):
        if oa.type != ob.type:
            return False
        if dict(oa.params) != dict(ob.params):
            return False
    return True
