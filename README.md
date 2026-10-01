# Aegis: Production Platform for AI Systems

Aegis is an AI-first production platform for building, deploying, and operating
AI/ML/LLM systems. It brings model and inference workflows, retrieval-augmented
generation (RAG), multimodal AI, and governed agent execution together with the
security, governance, observability, and deployment controls needed to run them
in production.

The repository includes focused components for model training and registry
workflows, inference providers, embeddings and RAG, multimodal systems, and
agent orchestration. Production controls—including tenant isolation, policy and
approval gates, evidence and audit trails, monitoring, and deployment
automation—are designed to support those AI workflows. The canonical lightweight
control-plane entry point is `production:app`; optional cloud, GPU/TPU, model,
and legacy API integrations are not required to start it.

## AI production capabilities

- **Model lifecycle:** training examples and pipelines, model registry,
  artifact verification, and promotion workflows.
- **Inference and serving:** provider abstractions and inference routing, with
  optional hosted, local, and accelerator-backed integrations.
- **RAG and knowledge workflows:** document chunking, embedding, vector
  retrieval, ranking, and prompt augmentation.
- **Agents and multimodal AI:** durable, tool-based agent runs and modular
  multimodal workflows for capabilities such as vision and speech.
- **Evaluation and operations:** evaluation and drift-monitoring utilities,
  operational analytics, and evidence-backed rollout controls.
- **Responsible production operations:** identity, tenant-scoped authorization,
  safety and policy enforcement, human approvals, auditability, and deployment
  controls apply around AI workloads.

These capabilities are provided by modular components and integrations rather
than one monolithic service. The canonical runtime below implements durable,
governed agent orchestration; model serving, vector retrieval, hosted planners,
and accelerator support are integration points and may require additional
dependencies or deployment configuration.

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

`POST /agent/runs` only creates a durable run. Execution is queued separately
through `/agent/runs/{run_id}/execute` using the injected
`execution_dispatcher`; worker processes use `AgentWorker` and register their
own trusted tool handlers. Dispatch messages carry identity-resolver output,
not body-supplied role or scope values. Approval routes record decisions using
the principal returned by `principal_resolver`; the request body's `actor` field
is ignored.

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
Approvals bind to the plan hash, policy version, capability version, and
tenant/run/step/tool. Optional `aegis_policy` engines are composed deny-on-
disagreement; unknown or unsatisfied obligations fail closed. Evidence hashes
are linked into a verifiable per-run chain, but the chain head should be
externally anchored to protect against deletion or wholesale database
replacement. See
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
