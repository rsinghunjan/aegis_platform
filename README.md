# Aegis Platform

Aegis is a repository of ML/AI serving, governance, monitoring, and deployment
components. The canonical lightweight control-plane entry point is
`production:app`; optional cloud, GPU/TPU, model, and legacy API integrations
are not required to start it.

## Local quickstart

```bash
python -m pip install -r requirements-control-plane.txt
uvicorn production:app --host 127.0.0.1 --port 8000
```

The default agent database is `sqlite:///./aegis_agent.db`. Configure
`AEGIS_AGENT_DATABASE_URL` to override it. `DATABASE_URL` is used as a
compatibility fallback. The application provides `/healthz`, `/readyz`,
`/agent/runs`, and tenant-scoped run/evidence/approval endpoints. A safe tool
must be registered by the embedding application before an agent plan can call
it; planner output cannot execute shell or Python code.

Agent HTTP routes fail closed until `create_app` is given a
`tenant_authorizer(request, tenant_id, action, actor)` callback. The embedding
service must derive identity and tenant membership from its trusted auth
context; request-body tenant and actor fields are not credentials.

Run tests with:

```bash
python -m pytest
```

Build and run the container with:

```bash
docker build -t aegis-platform .
docker run --rm -p 8000:8000 -v aegis-data:/data aegis-platform
```

## Canonical runtime and safety

- `production.py` is the FastAPI application entry point.
- `orchestrator.py` is the reusable agent orchestration facade.
- `agentic/runtime.py` owns typed run/plan/tool/policy/approval/evidence models,
  SQLite/PostgreSQL persistence, idempotency, execution, verification, and
  audit hashes.
- `api/memory.py` retains the compatible in-process `ConversationMemory` and
  adds a tenant/session-scoped persistent backend with retention and bounded
  compaction.
- `api/tasks.py` is a Celery lifecycle adapter. It executes only explicitly
  registered handlers; retries are enabled only for handlers registered as
  idempotent. It does not return simulated inference output.

Risk levels `low`, `medium`, and `high` are declared on tool metadata. Explicit
autonomy modes are `disabled`, `advisory`, `supervised`, and
`autonomous-for-low-risk` (the default). Advisory and supervised modes require
explicit approval before any execution; autonomous-for-low-risk still requires
approval for medium/high-risk tools. `AEGIS_AUTONOMY_ENABLED=false` is a global
kill switch. Approval expiration fails closed. The optional OpenAI-compatible
planner always validates plans and falls back to deterministic JSON planning.
The catalog is hash-versioned; signatures require an injected signer. Tool
handlers are trusted adapters: the sandbox boundary limits payloads/timeouts
and rejects code-like pure-profile inputs, but is not OS-level isolation.
Approvals and policy decisions are persisted. Evidence stores hashes and
redacted metadata, not raw tool results. See
[`docs/agentic_runtime.md`](docs/agentic_runtime.md) for the state machine and
configuration details.

## Implemented versus optional

**Implemented here:** health/readiness, a durable local agent runtime, registered
tool execution, deterministic and optional LLM planning, policy and approval
gates, hash-versioned capabilities, bounded tool payloads, verification, bounded
idempotent retries, tenant-scoped memory, evidence records, operator read models,
and pluggable job/remediation adapters.

**Adapter-backed or optional:** cloud deployment, secret managers, hosted LLM
planning, vector retrieval, model loading/inference, external approval systems,
promotion/canary systems, and GPU/TPU support. The optional legacy routes are
mounted only when `AEGIS_MOUNT_LEGACY_API=true`; their dependencies and
configuration must be installed separately. The default image intentionally
does not install large model or accelerator packages.

The pre-existing Alembic history contains multiple roots and an unresolved
revision reference. Agent and memory tables therefore use idempotent SQLAlchemy
schema initialization rather than extending a currently inconsistent migration
chain.

See [`docs/repository-hygiene.md`](docs/repository-hygiene.md) for the
historical patch/diff artifacts and canonical paths.
