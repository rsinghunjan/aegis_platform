from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from aegis_policy.contracts import EngineDecisionRecord, Obligation, PolicyDecision, PolicyInput, PrincipalRole


@dataclass(frozen=True)
class RbacEngine:
    """
    Minimal RBAC engine placeholder.

    NOTE: This is intentionally conservative and should be replaced with:
    - proper role resolution from DB
    - environment risk-tier specific roles
    - optional ABAC checks (classification, region)
    """

    bundle_sha256: str

    def _has_role(self, roles: Iterable[PrincipalRole], role: str, scope_type: str, scope_id: str) -> bool:
        for r in roles:
            if r.role == role and r.scope_type == scope_type and r.scope_id == scope_id:
                return True
        return False

    def evaluate(self, policy_input: PolicyInput) -> EngineDecisionRecord:
        action = policy_input.request.action
        actor = policy_input.request.actor
        res = policy_input.resource

        # Always require that actor org matches resource org when present.
        org_id = str(res.attributes.get("org_id", actor.org_id))
        if org_id != actor.org_id:
            d = PolicyDecision(allow=False, reason="cross_org_denied")
            return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)

        project_id = str(res.attributes.get("project_id", ""))
        env_id = str(res.attributes.get("environment_id", ""))

        roles: Sequence[PrincipalRole] = actor.roles

        # Read-only actions: allow if user has any project membership role.
        if action in ("job.read", "deploy.read", "model.read", "evidence.read"):
            if project_id and (
                self._has_role(roles, "ProjectOwner", "project", project_id)
                or self._has_role(roles, "ProjectReader", "project", project_id)
                or self._has_role(roles, "EnvDeployer", "environment", env_id)
            ):
                d = PolicyDecision(allow=True, reason="rbac_read_allowed")
                return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)
            d = PolicyDecision(allow=False, reason="rbac_read_denied")
            return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)

        # Job submission: require ProjectOwner or ProjectOperator in project scope.
        if action in ("job.submit", "rag.index.submit", "agent.run"):
            if project_id and (
                self._has_role(roles, "ProjectOwner", "project", project_id)
                or self._has_role(roles, "ProjectOperator", "project", project_id)
            ):
                d = PolicyDecision(allow=True, reason="rbac_job_submit_allowed")
                return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)
            d = PolicyDecision(allow=False, reason="rbac_job_submit_denied")
            return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)

        # Deploy request: require EnvDeployer at environment scope.
        if action in ("deploy.request", "deploy.rollback"):
            if env_id and self._has_role(roles, "EnvDeployer", "environment", env_id):
                # Add an obligation placeholder for approvals in high-risk tiers (OPA will enforce).
                obligations = (Obligation(type="record_change_ticket", params={}),)
                d = PolicyDecision(allow=True, reason="rbac_deploy_allowed", obligations=obligations)
                return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)
            d = PolicyDecision(allow=False, reason="rbac_deploy_denied")
            return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)

        # Default deny.
        d = PolicyDecision(allow=False, reason="rbac_default_deny")
        return EngineDecisionRecord(engine="rbac", decision=d, bundle_sha256=self.bundle_sha256)
