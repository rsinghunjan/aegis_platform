# Aegis platform architecture

## Supported runtime

The supported lightweight control plane is the FastAPI application exported as
`production:app`. The root `Dockerfile` starts that app and installs
`requirements-control-plane.txt`. Health/readiness, tenant-authorized AI
knowledge/answer routes, agent control-plane routes, and operator read models
are defined in `production.py`.

The API composes:

- `services/ai_workflow.py` and `services/embeddings/` for reference
  knowledge ingestion, retrieval, and inference.
- `agentic/runtime.py` for durable agent runs, plans, registered capabilities,
  policy decisions, approvals, verification, and evidence.
- `agentic/worker.py` for separately dispatched agent execution; the API does
  not execute agent tools in its request process.
- `policy/agent_policy.py` and `aegis_policy/` for local and optional external
  policy checks.
- `agentic/evidence_anchor.py` for optional external anchoring of agent
  evidence-chain heads.

`orchestrator.py` is a facade over the durable runtime. `frontend/` is an
operator client of the tenant-scoped routes; it does not provide trusted
identity. The embedding application must inject trusted principal resolution
and tenant authorization, and execution additionally requires a dispatcher and
a separately configured worker. Missing authorization/dispatch integrations
fail closed.

## AI platform components and boundaries

`services/inference/` defines provider, routing, batching, and lightweight
model metadata abstractions. `services/embeddings/` defines embedding,
chunking, vector-store, and RAG interfaces. The canonical AI workflow currently
uses local in-memory tenant indexes by default; that implementation is a
single-process reference, not shared durable production storage.

Model lifecycle capabilities are not yet one registry of record.
`services/inference/registry.py` holds in-memory model metadata;
`model_registry/` contains artifact loading and signature verification;
`api/mlflow_registry.py` and `governance/promotion.py` implement separate
promotion paths. Treat these as distinct integrations until a shared
model-version and provenance contract is established.

`services/observability/` provides reusable logging, metrics, tracing, and
health abstractions. External Prometheus, OpenTelemetry, and deployment
integrations exist, but AI answers and agent runs do not yet share a durable
workflow identifier that connects traces, policy evidence, model selection,
and operational telemetry.

## Canonical configuration and deployment

- Control-plane entry point: `production:app`
- Control-plane image/dependencies: root `Dockerfile` and
  `requirements-control-plane.txt`
- Agent runtime settings and embedding integrations: configuration reference
  in `agentic_runtime.md`
- Candidate production control-plane chart: `ops/production/helm/aegis/`

The wider repository retains multiple deployment manifests, dependency files,
legacy servers, and subsystem-specific configuration. They are not
interchangeable with the supported control plane. Before retiring any variant,
trace its references from CI, scripts, and deployment environments. In
particular, the migration documentation in `agentic_runtime.md` and `README.md`
must be reconciled: verify the Alembic revision graph and actual initialization
behavior before declaring a database migration path canonical.

## Migration sequence

1. Keep the canonical entry point, runtime, worker, and API contracts explicit;
   keep dashboard request/response types aligned with `production.py`.
2. Reconcile database migration and configuration ownership, including
   dependency and deployment sources of truth.
3. Add durable tenant-filtered knowledge storage and link retrieval/inference
   requests to durable workflow records and evidence.
4. Define one model-version/provenance contract that connects training,
   artifact verification, approval, serving, and promotion.
5. Correlate policy outcomes, model/provider versions, cost, latency, evidence,
   logs, metrics, and traces across AI requests and agent runs.
6. Retire legacy or duplicated variants only after confirming they have no
   supported consumers.

See `repository-hygiene.md`, `ai_workflows.md`, and `agentic_runtime.md` for
the current support boundaries and operational details.
