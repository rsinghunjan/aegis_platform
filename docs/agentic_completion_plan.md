# Phase 1 agentic-completion implementation

## Implemented

- `production:app` is a lightweight FastAPI control plane with SQLite fallback,
  health/readiness probes, fail-closed tenant authorization, run/approval
  operations, and authorized operator read models.
- Durable runs, plans, tool calls, decisions, step-bound approvals, capability
  records, and evidence use SQLAlchemy's idempotent table initialization. The
  existing Alembic graph has unresolved roots/references and is not extended.
- Deterministic JSON planning is the default. An optional OpenAI-compatible
  planner validates bounded structured plans and always falls back safely.
- Tool capabilities have schemas, policy metadata, bounded execution settings,
  version binding, and deterministic catalog hashing. Optional signing is an
  injected callback.
- Approval decisions support approve, deny, expiry, SLA metadata, and escalation
  callbacks. A scheduler is intentionally not bundled.
- Drift findings can create deduplicated diagnosis runs and bounded remediation
  proposals. Optional execution callbacks are not connected to real cloud
  systems by default.
- The built-in sandbox boundary enforces profiles, JSON payload/output limits,
  timeout evidence, and network-policy hooks. It is not OS-level isolation.
- The default Docker image installs only FastAPI, Pydantic, SQLAlchemy, and
  Uvicorn. GPU, ML, cloud, Redis, and legacy integrations remain optional.

## Environment variables

See `.env.example` for safe defaults. Key options:

| Variable | Purpose |
| --- | --- |
| `AEGIS_AGENT_DATABASE_URL` | Agent database, default local SQLite |
| `DATABASE_URL` | Compatibility fallback when agent URL is unset |
| `AEGIS_AUTONOMY_MODE` | `disabled`, `advisory`, `supervised`, or `autonomous-for-low-risk` |
| `AEGIS_AUTONOMY_ENABLED` | Legacy global kill switch |
| `AEGIS_CAPABILITY_ENFORCEMENT` | Enforce active capability registration/version |
| `AEGIS_APPROVAL_SLA_SECONDS` | Default approval lifetime |
| `AEGIS_LLM_PLANNER_ENABLED` | Opt in to the provider adapter |
| `AEGIS_LLM_BASE_URL`, `AEGIS_LLM_MODEL` | OpenAI-compatible endpoint and model |
| `AEGIS_LLM_API_KEY` | Provider credential (never committed or persisted) |
| `AEGIS_LLM_TIMEOUT_SECONDS`, `AEGIS_LLM_MAX_STEPS` | Planner bounds |

Agent HTTP APIs remain unavailable (503) until the embedding app injects a
trusted `tenant_authorizer`. Do not trust request-body tenant or actor fields.

## Production enablement checklist

- [ ] Configure a durable managed database and install its SQLAlchemy driver.
- [ ] Reconcile the existing migration graph before relying on schema migrations;
      current runtime tables use `create_all` plus additive compatibility columns.
- [ ] Inject an authentication-backed tenant authorizer and operator RBAC policy.
- [ ] Register reviewed adapter callables and set schemas, tenant/scope limits,
      cost ceilings, risk, idempotency, timeout, and sandbox profile.
- [ ] Set the intended autonomy mode; retain high-risk approval gates.
- [ ] Configure an approval expiry/escalation scheduler and notification callbacks.
- [ ] If using an LLM, configure a trusted HTTPS endpoint, secret delivery,
      network egress controls, and model allowlisting; test fallback behavior.
- [ ] Implement real remediation adapters separately, with canary/rollback,
      least privilege, and independent verification.
- [ ] Use OS/container isolation for untrusted handlers; built-in profiles alone
      are not a security sandbox.
- [ ] Run the local test suite and verify readiness, evidence retention, backups,
      and operational monitoring before production enablement.

This is a governed automation foundation, not literal full autonomy or a claim
that external provider integrations are production-ready.
