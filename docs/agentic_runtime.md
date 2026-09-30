# Agent runtime

`agentic/runtime.py` is the canonical runtime. `Orchestrator` in the repository
root exposes it to other services. The runtime uses a separate SQLAlchemy
metadata registry to avoid importing the optional pgvector model definitions.
`AgentStore.initialize()` idempotently creates runtime tables in the configured
database. This is deliberate until the repository's existing Alembic graph is
reconciled.

## State machine

```text
PENDING -> PLANNING -> RUNNING -> SUCCEEDED
                          |  \-> WAITING_APPROVAL -> RUNNING
                          |  \-> RETRYING -> RUNNING
                          |  \-> REPLANNING -> PLANNING -> RUNNING
                          |                    \--------> FAILED
                          \----> BLOCKED
```

The deterministic JSON planner accepts either a `tool` step or an ordered
`steps` list. When `AEGIS_LLM_PLANNER_ENABLED=true` and endpoint, model, and API
key are configured, `OpenAICompatiblePlanner` sends an OpenAI-compatible
chat-completions request. The adapter accepts only bounded JSON steps naming
registered tools with object inputs and acceptance criteria. Provider errors or
invalid output fall back to the deterministic planner. Prompts and credentials
are not written to evidence; model/provider identifiers and hashes are.

## Capability catalog and sandbox

`CapabilityCatalog` exposes canonical metadata, a deterministic SHA-256 hash,
and an optional signer callback. Runtime registrations are persisted in the
`agent_capabilities` table. `AEGIS_CAPABILITY_ENFORCEMENT=true` requires the
active registered version/hash at plan and execution time. Tool metadata
includes version, risk, schemas, role/environment/scope/tenant allowlists,
approval policy, idempotency, cost, timeouts, payload limits, and sandbox
profile.

Profiles are `pure` (default), `network` (requires an injected network-policy
hook), `storage`, and `restricted-subprocess`. The built-in boundary rejects
oversized payloads, code-like pure-profile fields, unconfigured network access,
and subprocess profiles without an external adapter. Sync handlers run in a
worker thread. This is a policy boundary around trusted registered handlers,
not kernel/container isolation; do not register untrusted Python callables.

Tools should be registered with their input/output schemas, risk level, roles,
environments, approval setting, idempotency declaration, maximum cost, tenant
allowlist, and required scopes. Synchronous handlers run in a worker thread;
async handlers are awaited. Both have a configurable timeout.

## Policy and approval behavior

- Low-risk tools run when identity/scope/environment/tenant and budget checks
  pass.
- Medium-risk tools require approval by default. This can be changed when
  constructing `AgentPolicyGate`.
- High-risk tools always require a persisted explicit approval.
- Approval rows bind tenant, run, step, tool, and capability version. The default
  SLA is 3600 seconds; expiration blocks execution. Halfway escalation is
  available via `ApprovalService`, with optional notification/escalation
  callbacks. A scheduler must invoke expiry/escalation methods; the core server
  does not run a background scheduler.
- Approvals are bound to the individual plan step and tool; approving one action
  does not authorize later high-risk actions in the same run.
- `AEGIS_AUTONOMY_ENABLED=false` blocks autonomous actions globally. A recorded
  approval is resumed through the explicit approval endpoint/facade.
- A denied or review decision is persisted with a decision ID and reason before
  the executor can run.
- Retries are bounded and only repeated for tools marked idempotent. Non-
  idempotent failures are not replayed automatically.
  On explicit resume after a process interruption, an in-flight idempotent step
  may be retried; an in-flight non-idempotent step is marked interrupted and
  sent to recovery planning instead of being silently replayed or marked
  successful.

`/agent/runs/{run_id}/approve` records the actor and reason, then resumes that
run. Agent HTTP routes return 503 until `create_app` receives a
`tenant_authorizer(request, tenant_id, action, actor)` callback; rejected access
returns 403. The embedding service must derive actor and tenant membership from
trusted authentication state. A body tenant or actor ID is not a credential.
The in-process runtime approval method is intended to be called only after the
embedding service has authorized the human approver.

## Persistence and evidence

Agent runs, plan steps, tool-call metadata, policy decisions, approvals,
capabilities, and evidence are durable. The idempotency key is unique within a
tenant. Stored tool input redacts keys that look like credentials and truncates
oversized strings; tool results are reduced to hashes and small type/size
metadata in audit rows. Prompt/context references and final outcome hashes are
audit evidence; raw prompts and result payloads are not copied into evidence.

Configure storage with `AEGIS_AGENT_DATABASE_URL`; `DATABASE_URL` is accepted
as a fallback. SQLite is the local default. PostgreSQL deployments must install
the matching SQLAlchemy driver. The `AgentStore` initializer is safe to rerun,
so a process can restart and resume persisted pending steps.

## Memory and integrations

`PersistentConversationMemory(session_id, tenant_id, ...)` stores retained
messages by tenant and session, compacts older history, and bounds retrieved
context. `ConversationMemory` remains available for simple in-process examples.
Vector retrieval, hosted planners, secret managers, cloud execution, and model
serving are adapter integration points and are not installed or enabled by the
minimal server image.

`DriftRemediationAdapter` accepts a normalized `DriftFinding`, creates an
idempotent diagnosis run, stores a normalized event hash, and records a bounded
remediation proposal. Optional injected adapters can execute canary deployment
or rollback actions; rollback requires approval. `local_simulation_adapter`
verifies a proposal without contacting a cloud provider. Retraining, promotion,
rollback, and notification integrations remain optional callbacks.

## Operator API

Authorized `/operator/agent/*` read models provide paginated run summaries, run
timelines, blocked policy decisions, evidence summaries, capability
inspection/hash, and remediation events. `/agent/approvals` lists tenant-scoped
approvals and `/agent/approvals/{approval_id}/decision` accepts authorized
approve/deny decisions. Responses avoid raw prompts and tool results.

## Autonomy configuration

Set `AEGIS_AUTONOMY_MODE` to `disabled`, `advisory`, `supervised`, or
`autonomous-for-low-risk`. Advisory and supervised modes hold all actions for an
explicit human approval; disabled blocks even approved execution. The legacy
`AEGIS_AUTONOMY_ENABLED=false` setting remains a global kill switch.
