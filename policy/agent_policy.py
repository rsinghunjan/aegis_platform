"""Deterministic policy gate for registered agent tool actions."""
from __future__ import annotations

import os
import hashlib
import json
from typing import Any, Optional

from aegis_policy.contracts import (
    EnvironmentContext,
    PolicyContext,
    PolicyInput,
    PrincipalRole,
    RequestActor,
    RequestMeta,
    TypedResource,
)
from aegis_policy.deny_on_disagree import deny_on_disagree


class AgentPolicyGate:
    """Apply identity, scope, environment, budget, risk, and autonomy controls."""

    def __init__(
        self,
        autonomy_enabled: Optional[bool] = None,
        medium_requires_approval: bool = True,
        autonomy_mode: Optional[str] = None,
        engines: tuple[Any, ...] = (),
        obligation_handler: Optional[Any] = None,
    ):
        self.autonomy_enabled = (
            autonomy_enabled
            if autonomy_enabled is not None
            else os.getenv("AEGIS_AUTONOMY_ENABLED", "true").lower() in {"1", "true", "yes"}
        )
        self.autonomy_mode = autonomy_mode or os.getenv(
            "AEGIS_AUTONOMY_MODE",
            "autonomous-for-low-risk" if self.autonomy_enabled else "disabled",
        )
        if self.autonomy_mode not in {
            "disabled",
            "advisory",
            "supervised",
            "autonomous-for-low-risk",
        }:
            raise ValueError("Unsupported AEGIS_AUTONOMY_MODE")
        self.medium_requires_approval = medium_requires_approval
        self.engines = engines
        self.obligation_handler = obligation_handler

    @property
    def version(self) -> str:
        configuration = {
            "autonomy_enabled": self.autonomy_enabled,
            "autonomy_mode": self.autonomy_mode,
            "medium_requires_approval": self.medium_requires_approval,
            "engines": [
                {
                    "name": getattr(engine, "engine_name", type(engine).__name__),
                    "bundle_sha256": getattr(engine, "bundle_sha256", ""),
                }
                for engine in self.engines
            ],
        }
        canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def evaluate(
        self,
        run: Any,
        spec: Any,
        role: str,
        environment: str,
        scopes: set[str],
        approved: bool,
        autonomous: bool,
        principal_id: Optional[str] = None,
    ) -> tuple[str, str]:
        local_decision = self._evaluate_local(
            run, spec, role, environment, scopes, approved, autonomous
        )
        if local_decision[0] == "block" or not self.engines:
            return local_decision

        try:
            records = [
                engine.evaluate(
                    self._policy_input(
                        run, spec, role, environment, scopes, engine, principal_id
                    )
                )
                for engine in self.engines
            ]
            final = deny_on_disagree(records)
        except Exception:
            return "block", "policy_engine_error"
        if not final.allow:
            return "block", final.reason
        for record in final.engine_records:
            for obligation in record.decision.obligations:
                if self.obligation_handler is None:
                    return "block", f"unhandled_obligation:{obligation.type}"
                try:
                    satisfied = self.obligation_handler(obligation)
                except Exception:
                    satisfied = False
                if not satisfied:
                    return "block", f"obligation_failed:{obligation.type}"
        return local_decision

    def _policy_input(
        self, run: Any, spec: Any, role: str, environment: str,
        scopes: set[str], engine: Any, principal_id: Optional[str],
    ) -> PolicyInput:
        aliases = {
            "agent": "ProjectOperator",
            "operator": "ProjectOperator",
            "admin": "ProjectOwner",
            "reader": "ProjectReader",
        }
        resolved_role = aliases.get(role, role)
        roles = [
            PrincipalRole(role=resolved_role, scope_type="project", scope_id=run.tenant_id)
        ]
        roles.extend(
            PrincipalRole(role=scope, scope_type="environment", scope_id=environment)
            for scope in sorted(scopes)
        )
        bundle_sha256 = str(getattr(engine, "bundle_sha256", ""))
        return PolicyInput(
            request=RequestMeta(
                request_id=f"agent:{run.run_id}:{spec.name}",
                ts=run.updated_at.isoformat(),
                action="agent.run",
                actor=RequestActor(
                    principal_id=principal_id or role,
                    principal_type="service_account",
                    org_id=run.tenant_id,
                    roles=roles,
                    claims={"scopes": sorted(scopes)},
                ),
            ),
            resource=TypedResource(
                type="agent_capability",
                id=spec.name,
                attributes={
                    "org_id": run.tenant_id,
                    "project_id": run.tenant_id,
                    "environment_id": environment,
                    "risk": getattr(spec.risk_level, "value", spec.risk_level),
                    "cost": spec.max_cost,
                },
            ),
            environment=EnvironmentContext(
                id=environment,
                risk_tier=getattr(spec.risk_level, "value", spec.risk_level),
                budget={"remaining": max(0.0, run.budget - run.spent_cost)},
            ),
            policy=PolicyContext(
                bundle_sha256=bundle_sha256,
                pin_scope_used="project",
                mode="multi",
                engine=str(getattr(engine, "engine_name", type(engine).__name__)),
            ),
        )

    def _evaluate_local(
        self,
        run: Any,
        spec: Any,
        role: str,
        environment: str,
        scopes: set[str],
        approved: bool,
        autonomous: bool,
    ) -> tuple[str, str]:
        if self.autonomy_mode == "disabled" or not self.autonomy_enabled:
            return "block", "global_autonomy_disabled"
        if spec.allowed_tenants and run.tenant_id not in spec.allowed_tenants:
            return "block", "tenant_not_allowed"
        if spec.allowed_roles and role not in spec.allowed_roles:
            return "block", "role_not_allowed"
        if spec.allowed_environments and environment not in spec.allowed_environments:
            return "block", "environment_not_allowed"
        if spec.allowed_scopes and not set(spec.allowed_scopes).issubset(scopes):
            return "block", "scope_not_allowed"
        if spec.max_cost > max(0.0, run.budget - run.spent_cost):
            return "block", "budget_exceeded"
        risk_level = getattr(spec.risk_level, "value", spec.risk_level)
        requires_approval = (
            self.autonomy_mode in {"advisory", "supervised"}
            or spec.requires_approval
            or risk_level == "high"
            or (
                risk_level == "medium"
                and self.medium_requires_approval
            )
        )
        if requires_approval and not approved:
            return "review", "explicit_approval_required"
        return "allow", "policy_allow"
