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

- `README.md` — authoritative repository overview and control-plane quickstart.
  The superseded `README2.md`/`README3.md` drafts have been removed.
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
- `api/memory.py` — compatible in-process and optional tenant/session-scoped
  persistent memory; it is not mounted automatically by the control plane.
- `api/tasks.py` — registered job handler lifecycle for API jobs; this is
  separate from agent execution in `agentic/worker.py`.
- `requirements-control-plane.txt` — minimal API image dependencies;
  `requirements.txt` and `pyproject.toml` cover wider repository components.
- `Dockerfile` — non-root server image and container startup.

The supported workflow is tenant-authorized knowledge ingestion and retrieval
through `/ai/knowledge` and `/ai/answer`, plus separately dispatched durable
agent runs through `/agent/runs` and the worker. Agent runs include policy,
approval, verification, and evidence controls; see
[`ai_workflows.md`](ai_workflows.md) and
[`agentic_runtime.md`](agentic_runtime.md).

The root Dockerfile and `uvicorn production:app` command are the canonical
control-plane boot path. `production:app` starts with no tenant authorizer or
execution dispatcher; protected AI/agent routes fail closed until an embedding
application supplies trusted authorization and, for execution, a dispatcher and
worker. The [runtime configuration reference](agentic_runtime.md#configuration-reference)
collects the related integration requirements and optional environment settings.

Consolidating similarly named utility scripts and suffixed deployment or
dependency files remains follow-up work: first map references from CI,
automation, and subsystem-specific guides, then retire only verified-unused
variants. This pass intentionally leaves those files unchanged.

The RAG index is currently process-local and volatile. AI answer requests and
durable agent runs do not yet share one workflow ID, and the model registry,
feature store, evaluation, and promotion components are not a unified runtime
pipeline. These are explicit follow-ups; do not infer they are implemented from
the former patch backlog. Legacy service variants should not be imported as the
canonical application.

## Canonical paths (AI-native hardening)

Removed numbered/duplicate workflows (`ci2`, `fully_ai_readiness_gate2`, `mlflow-training2`,
`sign_with_vault2`, `security_scan`), stray `.rtf`/`.patch3`/`.path` scaffolds, legacy
`mmaisys_*` stubs, and duplicate pre-commit configs. The canonical entry point is
`production.py` (`production:app`); the single pre-commit config is `.pre-commit-config.yaml`.

## Retraining promotion path

`GET /operator/ai/improvements/retraining-manifest` -> reviewer approves the manifest SHA-256 ->
`promote_model_if_valid_signed.yml` (protected `model-promotion` environment) re-fetches and
re-verifies the manifest, validates the candidate, signs it (`api/model_signing.py`, fail-closed),
registers it with the manifest hash as provenance, and verifies the signature. The previous
`mlflow-training*` example workflows were truncated and non-functional and were removed.

## Known corruption

Many files committed before this cleanup (about 117 `.py` and 30+ workflow files) begin with a
block of bare line numbers (a pasted editor gutter); about 30 also lost their head and cannot be
repaired mechanically. Repaired so far: `model_registry/signing.py`, `scripts/promote_mlflow_run.py`,
`api/mlflow_registry.py`. Find the rest with `python -m compileall -q . ` and fix or delete them.
