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
                          |  \-> REPLANNING -> FAILED
                          \----> BLOCKED
```

The JSON planner accepts either a `tool` step or an ordered `steps` list. Each
step names an already registered `ToolSpec` and contains JSON input and optional
acceptance criteria. Unregistered tools, malformed steps, and schema-invalid
inputs fail closed. Model output is never interpreted as shell, Python, or
another executable language. A hosted LLM planner can implement `Planner`, but
must continue to produce this constrained plan representation.

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
- `AEGIS_AUTONOMY_ENABLED=false` blocks autonomous actions globally. A recorded
  approval is resumed through the explicit approval endpoint/facade.
- A denied or review decision is persisted with a decision ID and reason before
  the executor can run.
- Retries are bounded and only repeated for tools marked idempotent. Non-
  idempotent failures are not replayed automatically.

`/agent/runs/{run_id}/approve` records the actor and reason, then resumes that
run. Tenant IDs are checked for run, memory, and evidence reads. Authentication
for these new endpoints is expected to be provided by the deployment gateway
or the embedding service; the tenant ID is not itself an authentication
credential.

## Persistence and evidence

Agent runs, plan steps, tool-call metadata, policy decisions, approvals, and
evidence are durable. The idempotency key is unique within a tenant. Stored
tool input redacts keys that look like credentials and truncates oversized
strings; tool results are reduced to hashes and small type/size metadata in
audit rows. Prompt/context references and final outcome hashes are audit
evidence; raw prompt/result payloads are not copied into evidence rows.

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
idempotent diagnosis run, and optionally calls an injected promotion adapter
after successful verification. Monitoring libraries and external governance
systems remain outside the core runtime.
