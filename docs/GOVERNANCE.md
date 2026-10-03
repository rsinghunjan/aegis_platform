# Governance overview

Aegis is a governed AI platform: every tenant-facing AI or agent action runs
through identity, policy, and evidence controls before and after execution.
This document summarizes where each governance control lives in the codebase
and how an operator can inspect the live posture of a running deployment.

## 1. Tenant isolation and trusted identity

- `production.py` builds the control-plane app via `create_app(...)`. Every
  tenant-scoped route calls `authorize(request, tenant_id, action, actor)`,
  which requires an injected `tenant_authorizer` callback. If no authorizer is
  configured, the route **fails closed** with `503`, never `200`.
- Identity for approvals and dispatch comes from `principal_resolver`, not
  from request-body fields. `resolve_principal` raises `503` when no trusted
  principal is available. Approval and approval-decision routes additionally
  reject a missing `principal.principal_id` with `503` before authorization is
  even attempted.
- `AgentExecutionMessage` (see `agentic/worker.py`) carries the
  identity-resolver's principal and scopes into the worker boundary; the
  worker never trusts body-supplied role/scope values.

## 2. Policy and approval gates

- `policy/agent_policy.py` (`AgentPolicyGate`) evaluates identity, scope,
  environment, budget, risk, and autonomy for every tool call, and composes
  any configured `aegis_policy` engines using deny-on-disagreement semantics
  (`aegis_policy/deny_on_disagree.py`). Unsatisfied or unknown obligations fail
  closed to `block`.
- Autonomy modes are `disabled`, `advisory`, `supervised`, and
  `autonomous-for-low-risk` (default), controlled by `AEGIS_AUTONOMY_MODE` /
  `AEGIS_AUTONOMY_ENABLED`. Advisory and supervised modes require explicit
  human approval before any execution; medium/high-risk tools require
  approval even in autonomous-for-low-risk mode.
- Approvals bind to the plan hash, policy version, capability version, and
  tenant/run/step/tool, so a later change to policy or the tool catalog cannot
  silently reuse a stale approval. Approval expiration (`AEGIS_APPROVAL_SLA_SECONDS`)
  fails closed.

## 3. Evidence and audit trail

- `agentic/runtime.py` links every run's plan, approval, execution, and
  feedback events into a hash-chained evidence trail
  (`record_evidence` / `verify_evidence_chain`).
- `agentic/evidence_anchor.py` defines a pluggable `EvidenceAnchorBackend`
  (for example `HttpTransparencyLogBackend`) so the chain head can be
  anchored outside the primary database, protecting against deletion or
  wholesale database replacement. `production.py`'s lifespan periodically
  calls `anchor_pending_evidence` when a backend is configured.
- `GET /operator/agent/evidence-summary` returns the evidence count, hash,
  kinds, anchors, and chain-integrity verdict for a run.

## 4. Risk-tiered execution and autonomy controls

- Tool metadata (`ToolSpec`) declares `risk_level` (`low`, `medium`, `high`)
  and an optional `sandbox_required` flag. High-risk tools always execute
  through `GVisorSandboxExecutor`; medium-risk tools opt in; low-risk tools
  run in-process. See `docs/agentic_runtime.md` for sandbox profile and
  resource-limit details.
- `AEGIS_CAPABILITY_ENFORCEMENT=true` requires the actively registered
  capability version/hash at plan and execution time, preventing a plan from
  invoking a tool version that is no longer registered.

## 5. Compliance-oriented reporting and operator review

- Operator-only, tenant-scoped read models are mounted under `/operator/...`
  and require the `operator_read` action in the injected `tenant_authorizer`:
  `/operator/agent/runs`, `/operator/agent/runs/{run_id}/timeline`,
  `/operator/agent/blocked-decisions`, `/operator/agent/evidence-summary`,
  `/operator/agent/capabilities`, `/operator/agent/capabilities/version`,
  `/operator/agent/capabilities/{name}`, and
  `/operator/agent/remediation-events`.
- `GET /operator/governance/status` reports the live governance posture for a
  tenant in one call: whether a trusted tenant authorizer and principal
  resolver are configured, whether an execution dispatcher is wired up,
  capability enforcement and approval SLA settings, the active policy's
  autonomy mode and policy version, and whether evidence anchoring is
  configured (and with which backend). This endpoint is intended for
  compliance dashboards and operator review; like every other tenant-scoped
  route it fails closed (`503`) when no `tenant_authorizer` is configured.

## Bootable default

The importable `production:app` has no authorizer or dispatcher configured.
`/healthz` and `/readyz` work out of the box; every protected AI, agent, and
operator/governance route fails closed with `503` until the embedding service
supplies trusted `tenant_authorizer`, `principal_resolver`, and
`execution_dispatcher` implementations. See the main [README](../README.md)
and [`docs/agentic_runtime.md`](agentic_runtime.md) for the full state machine
and configuration reference.
