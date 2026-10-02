# Aegis: Production Platform for AI Systems

Aegis is an AI-first production platform for building, deploying, and operating
AI/ML/LLM systems. Its supported workflow connects tenant-authorized knowledge
ingestion and retrieval-augmented inference with durable, policy-governed agent
runs, operator review, and evidence. Model lifecycle, multimodal, evaluation,
security, observability, and deployment components support this workflow.

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
- **RAG and knowledge workflows:** the canonical app exposes tenant-authorized
  document ingestion and retrieval-augmented inference through `/ai/knowledge`
  and `/ai/answer`. Local hash embeddings and an in-process vector store provide
  a dependency-light reference; configured OpenAI-compatible inference and
  OpenAI embeddings are supported options.
- **Agents and multimodal AI:** durable, tool-based agent runs and modular
  multimodal workflows for capabilities such as vision and speech.
- **Evaluation and operations:** evaluation and drift-monitoring utilities,
  operational analytics, and evidence-backed rollout controls.
- **Responsible production operations:** identity, tenant-scoped authorization,
  safety and policy enforcement, human approvals, auditability, and deployment
  controls apply around AI workloads.

The product is organized around a practical lifecycle: index tenant knowledge
and query it with an AI provider; create durable agent runs using registered
capabilities; review approvals and evidence; and operate workloads using the
existing monitoring, promotion, and deployment integrations. The local
knowledge index is a development/reference implementation, not durable shared
production storage. See [`docs/ai_workflows.md`](docs/ai_workflows.md) for setup,
API examples, and production boundaries.

## Local quickstart

```bash
python -m pip install -r requirements-control-plane.txt
uvicorn production:app --host 127.0.0.1 --port 8000
```

For the OpenAI-compatible inference and embedding adapter, install the
`ai` extra with `python -m pip install -e '.[ai]'`.

The default agent database is `sqlite:///./aegis_agent.db`. Configure
`AEGIS_AGENT_DATABASE_URL` to override it. `DATABASE_URL` is used as a
compatibility fallback. The application provides `/healthz`, `/readyz`, `/ai/knowledge`, `/ai/answer`,
`/agent/runs`, and tenant-scoped run/evidence/approval endpoints. A safe tool
must be registered by the embedding application before an agent plan can call
it; planner output cannot execute shell or Python code.

`GET /operator/governance/status` reports the live governance posture for a
tenant (identity/authorizer wiring, policy autonomy mode and version, evidence
anchoring, and execution dispatcher configuration) for compliance dashboards
and operator review. See [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) for the
full governance control map.

Agent HTTP routes fail closed until `create_app` is given a
`tenant_authorizer(request, tenant_id, action, actor)` callback. The embedding
service must derive identity and tenant membership from its trusted auth
context; request-body tenant and actor fields are not credentials.
The importable `production:app` is a bootable default with no authorizer or
dispatcher configured: health/readiness work, while protected AI/agent requests
fail closed. Production integrations must construct the app with trusted
authorization, and configure an execution dispatcher and worker to run agent
tools.

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
  adds an optional tenant/session-scoped persistent backend with retention and
  bounded compaction; the control plane does not mount it automatically.
- `agentic/worker.py` is the separate agent execution boundary. `api/tasks.py`
  is a distinct Celery lifecycle adapter for registered jobs; it executes only
  explicitly registered handlers and retries only handlers declared idempotent.

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
configuration details. The consolidated environment and integration settings
reference is in that guide. See [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) for
a control-by-control map of identity, policy/approval, evidence, risk-tiered
execution, and operator/compliance reporting.

## Implemented versus optional

**Implemented here:** health/readiness, tenant-authorized knowledge ingestion and
retrieval-augmented inference, provider and embedding adapters, a durable local
agent runtime, registered tool execution, deterministic and optional LLM
planning, policy and approval gates, hash-versioned capabilities, bounded tool
payloads, verification, bounded idempotent retries, agent feedback evidence,
tenant-scoped memory, operator read models, and pluggable job/remediation
adapters.

**Adapter-backed or optional:** hosted LLM planning and inference, semantic
embeddings, durable/shared vector storage, cloud deployment, secret managers,
external approval systems, promotion/canary systems, and GPU/TPU support. The
local inference fallback is an echo provider for smoke testing, not an AI model.
The default image intentionally does not install large model or accelerator
packages. The historical multipart prediction route is not mounted or
production-supported; see the
[legacy API status and migration guide](docs/legacy_api.md).

The pre-existing Alembic history contains multiple roots and an unresolved
revision reference. Agent and memory tables therefore use idempotent SQLAlchemy
schema initialization rather than extending a currently inconsistent migration
chain.

See [`docs/repository-hygiene.md`](docs/repository-hygiene.md) for the
historical patch/diff artifacts and canonical paths.
