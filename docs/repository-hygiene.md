# Repository hygiene and AI runtime

## Patch backlog disposition

The root previously contained 896 tracked patch/diff files (831 `.patch` and
65 `.diff`). They covered overlapping AI/agentic/RAG, production and hardening,
governance/security, data/MLOps, and edge/quantum initiatives. Six pairs had
identical contents; many others used versioned or “final” names without a
reliable relationship to the current source tree. Patch filenames do not
establish that a change is applicable, complete, or newer than the code.

The root-level artifacts were removed rather than applying or archiving a
guessed “latest” subset. They were proposals, not executable configuration or
the source of truth. No patch backlog should be added to the repository root;
maintain changes in the canonical source and tests below. Patch files that
remain outside the repository root are not part of this cleanup.

## Canonical AI/control-plane paths

- `production.py` — FastAPI application, tenant authorization, RAG endpoints,
  agent control-plane routes, and lifecycle-managed evidence anchoring.
- `services/ai_workflow.py` and `services/embeddings/` — tenant-scoped reference
  knowledge ingestion, retrieval, and inference.
- `orchestrator.py` and `agentic/runtime.py` — reusable orchestration and
  durable run planning, policy, execution, verification, approvals, and evidence.
- `agentic/worker.py` — queue-safe execution boundary and optional Celery adapter.
- `agentic/planner.py`, `agentic/capabilities.py`, `agentic/approvals.py`,
  `agentic/sandbox.py`, and `agentic/oci_sandbox.py` — planner, capability,
  approval, and tool execution boundaries.
- `agentic/remediation.py` — normalized monitoring finding adapter.
- `api/memory.py` — in-process and persistent memory.
- `api/tasks.py` — registered job handler lifecycle.
- `requirements-control-plane.txt` — minimal API image dependencies;
  `requirements.txt` and `pyproject.toml` cover wider repository components.
- `Dockerfile` — non-root server image and container startup.

The supported workflow is tenant-authorized knowledge ingestion and retrieval
through `/ai/knowledge` and `/ai/answer`, plus separately dispatched durable
agent runs through `/agent/runs` and the worker. Agent runs include policy,
approval, verification, and evidence controls; see
[`ai_workflows.md`](ai_workflows.md) and
[`agentic_runtime.md`](agentic_runtime.md).

The RAG index is currently process-local and volatile. AI answer requests and
durable agent runs do not yet share one workflow ID, and the model registry,
feature store, evaluation, and promotion components are not a unified runtime
pipeline. These are explicit follow-ups; do not infer they are implemented from
the former patch backlog. Legacy service variants should not be imported as the
canonical application.
