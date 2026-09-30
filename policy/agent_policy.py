"""Deterministic policy gate for registered agent tool actions."""
from __future__ import annotations

import os
from typing import Any, Optional

class AgentPolicyGate:
    """Apply identity, scope, environment, budget, risk, and autonomy controls."""

    def __init__(
        self,
        autonomy_enabled: Optional[bool] = None,
        medium_requires_approval: bool = True,
    ):
        self.autonomy_enabled = (
            autonomy_enabled
            if autonomy_enabled is not None
            else os.getenv("AEGIS_AUTONOMY_ENABLED", "true").lower() in {"1", "true", "yes"}
        )
        self.medium_requires_approval = medium_requires_approval

    def evaluate(
        self,
        run: Any,
        spec: Any,
        role: str,
        environment: str,
        scopes: set[str],
        approved: bool,
        autonomous: bool,
    ) -> tuple[str, str]:
        if autonomous and not self.autonomy_enabled:
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
            spec.requires_approval
            or risk_level == "high"
            or (
                risk_level == "medium"
                and self.medium_requires_approval
            )
        )
        if requires_approval and not approved:
            return "review", "explicit_approval_required"
        return "allow", "policy_allow"
