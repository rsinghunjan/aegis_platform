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
worker thread only for in-process tools. High-risk tools always use the
`GVisorSandboxExecutor`; medium-risk tools can opt in with
`sandbox_required=True`. Low-risk tools run in-process. A missing image,
entrypoint, gVisor runtime, or container backend fails the tool execution; the
runtime never falls back to calling a high-risk handler in-process.

For each sandboxed tool, configure `sandbox_image` with a preloaded, preferably
digest-pinned OCI image and `sandbox_entrypoint` as `module:function`. The image
must contain Python, this package's `agentic.sandbox_runner`, and the
entrypoint's implementation. The host requires Docker Engine with the gVisor
`runsc` runtime installed and configured. The executor runs one container per
tool call with networking disabled, a read-only root filesystem, dropped Linux
capabilities, no-new-privileges, a non-root UID, a PID limit, and per-tool CPU,
memory, and `/tmp` limits (`sandbox_cpu_limit`, `sandbox_memory_limit_mb`, and
`sandbox_disk_limit_mb`). Images are not pulled during execution, avoiding
unbounded startup/network behavior; operators must fetch approved images before
dispatching runs.

Use `sandbox_secret_environment` to map container variable names to host
environment variable names. Only the mapped values are sent to the container
over stdin; values are not part of tool metadata, command-line arguments, or
execution logs. The container receives no host mounts or network access. Keep
the host environment and Docker daemon restricted to trusted operators: the
daemon remains a privileged host service.

The executor logs container creation, terminal state, duration, configured
resource limits, and per-run CPU time, peak memory, and `/tmp` usage. It bounds
JSON output and force-removes containers after success, failure, or timeout.
Image preloading avoids pull latency, but container startup overhead depends on
the host and image; benchmark deployments against the production latency budget.
The sandbox integration tests use a mocked container client and do not require
Docker or gVisor.

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
  callbacks. `create_app(..., approval_notifier=...)` can notify when a new
  request is persisted. A scheduler must invoke expiry/escalation methods; the
  core server does not run a background scheduler.
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

`/agent/runs/{run_id}/approve` records the trusted actor and reason; execution
must subsequently be queued through the worker boundary. Agent HTTP routes return 503 until `create_app` receives a
`tenant_authorizer(request, tenant_id, action, actor)` callback; rejected access
returns 403. The embedding service must derive actor and tenant membership from
trusted authentication state. A body tenant or actor ID is not a credential.
The in-process runtime approval method is intended to be called only after the
embedding service has authorized the human approver.

`principal_resolver` supplies the trusted principal, role, environment, and
scopes used for execution and approval. Body-supplied role, environment, scopes,
and actor fields are not used as authorization context. Approval rows additionally
bind to the immutable plan hash and current policy version, as well as the
capability version, tenant, run, step, and tool. A capability or policy change
therefore requires a new approval.
Approval decisions can include the persisted `approval_id`; the runtime rejects
an ambiguous run-level decision if more than one pending approval is present.

## Control plane and worker boundary

`POST /agent/runs` creates a durable run and does not execute tools.
`POST /agent/runs/{run_id}/execute` submits an `AgentExecutionMessage` to the
injected `execution_dispatcher`; the API returns `QUEUED` and does not call
`AgentRuntime.execute`. A separate worker process can consume that message using
`AgentWorker` and an independently configured runtime with the same database,
policy, and registered trusted handlers. Dispatchers should durably queue and
preserve the `run_id` as the execution idempotency key. The provided
`CeleryExecutionDispatcher` and `register_celery_agent_worker` helpers integrate
with an application-owned Celery instance and broker. Do not expose the worker
consumer directly to untrusted clients. A missing dispatcher fails closed with
HTTP 503.

## Unified policy engines

`AgentPolicyGate` keeps its local identity, tenant, scope, environment, budget,
risk, and autonomy checks, then can evaluate the same action through injected
typed policy engines such as `aegis_policy` RBAC and OPA. Any engine denial,
material disagreement, evaluation error, or obligation without a successful
injected handler blocks execution. The active configuration and engine bundle
hashes are included in the policy version and decision evidence. With no
external engines configured, the local gate remains the policy authority.

## Persistence and evidence

Agent runs, plan steps, tool-call metadata, policy decisions, approvals,
capabilities, and evidence are durable. The idempotency key is unique within a
tenant. Stored tool input redacts keys that look like credentials and truncates
oversized strings; tool results are reduced to hashes and small type/size
metadata in audit rows. Prompt/context references and final outcome hashes are
audit evidence; raw prompts and result payloads are not copied into evidence.
Each evidence row links to the previous evidence digest, and
`verify_evidence_chain` validates the run's hash chain and returns its head.
This detects alteration, insertion, and internal deletion when the head is
known, but does not prevent an attacker with database write access from
truncating or replacing the entire chain. Export or sign the head in an
independent audit system for that guarantee.

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
