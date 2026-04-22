from __future__ import annotations

import os
from datetime import datetime, timezone

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
from aegis_policy.engines.opa_http import OpaHttpEngine
from aegis_policy.engines.rbac import RbacEngine


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    bundle_sha = "dev-bundle-sha"

    actor = RequestActor(
        principal_id="p1",
        principal_type="user",
        org_id="org1",
        roles=(
            PrincipalRole(role="EnvDeployer", scope_type="environment", scope_id="env_pci"),
            PrincipalRole(role="ProjectOwner", scope_type="project", scope_id="proj1"),
        ),
        claims={"email": "user@example.com", "mfa": True},
    )
    req = RequestMeta(request_id="r1", ts=now_iso(), action="deploy.request", actor=actor)
    res = TypedResource(
        type="deployment",
        id="dep1",
        attributes={
            "org_id": "org1",
            "project_id": "proj1",
            "environment_id": "env_pci",
            "data_classification": "pci",
            "region": "us-east-1",
        },
    )
    env = EnvironmentContext(
        id="env_pci",
        risk_tier="pci",
        cloud_targets=("kubernetes", "aws"),
        region_constraints=("us-east-1",),
        budget={"monthly_usd_limit": 50000, "monthly_usd_spent": 1000},
    )
    pol = PolicyContext(bundle_sha256=bundle_sha, pin_scope_used="environment", mode="central+sidecar", engine="opa-central-http")
    inp = PolicyInput(request=req, resource=res, environment=env, policy=pol)

    records = []
    records.append(RbacEngine(bundle_sha256=bundle_sha).evaluate(inp))

    opa_url = os.getenv("OPA_URL", "http://localhost:8181/v1/data/aegis/decision")
    records.append(OpaHttpEngine(engine_name="opa-central-http", endpoint_url=opa_url, bundle_sha256=bundle_sha).evaluate(inp))

    final = deny_on_disagree(records)
    print("final:", {"allow": final.allow, "reason": final.reason, "disagree": final.disagree})
    for r in final.engine_records:
        print(r.engine, r.decision.allow, r.decision.reason, [o.type for o in r.decision.obligations])
    return 0 if final.allow else 2


if __name__ == "__main__":
    raise SystemExit(main())
