# Aegis AI-native foundation

The AI-native foundation is a shared software layer around selected Aegis AI
workflows, not a replacement for the platform's existing inference, retrieval,
governance, or agent-runtime components.

## Components

- `AICore` wraps `AIWorkflow` for knowledge ingestion and answering, records
  tenant-scoped memory and observations, and produces operator insights.
- `AIMemory` keeps bounded interaction metadata, durable in SQL via `AICoreStore`. Query and run
  identifiers are hashed; raw prompts, goals, and answers are not retained.
- `AIObservability` summarizes answer failures, citations, latency, cost, and
  successful agent lifecycle events.
- `AIPolicyEngine` creates heuristic policy suggestions. Suggestions require
  human approval and do not mutate enforcement policy.
- `AgentRuntime` remains the durable system of record for agent runs, evidence,
  approvals, and execution policy. The shared core only records a limited
  summary of successful create, dispatch, feedback, and approval operations.

## HTTP surface

The existing `/ai/knowledge` and `/ai/answer` endpoints are routed through
`AICore`. Agent run endpoints continue to use `AgentRuntime` and the configured
execution dispatcher. `GET /operator/ai/insights?tenant_id=...` returns
tenant-scoped persisted metrics, heuristic recommendations, and advisory
policy suggestions. It requires the existing `operator_read` authorization.
The existing authorization callbacks remain required; a tenant ID is not
authentication.

The answer response includes `repeated_query`, indicating whether the same
tenant asked the same exact query within the retained memory window.
It is a best-effort signal, not durable deduplication or idempotency.

## Production-readiness assessment

This foundation does **not** make Aegis fully AI-native or establish that a
deployment is production-ready. Important limitations include:

- Memory and telemetry persist in `ai_core_events` (hash-only metadata, newest
  500 per tenant and kind), shared across replicas on a shared database. During
  a database outage they degrade to per-process state. Agent lifecycle events
  here are observability hints, not the durable audit record.
- Policy and operational recommendations are simple thresholds, not evaluated
  policy decisions. They must not be used as automatic enforcement.
- The AI core currently wraps the knowledge workflow; it does not unify every
  inference, training, governance, and agent decision path behind one planner.
- The default local-hash embeddings and echo provider support offline
  development but are not semantic retrieval or useful generated answers.
- Production deployment still depends on trusted tenant authorization, durable
  database configuration and backups, provider credentials and rate/cost
  controls, worker/dispatcher operations, monitoring, and security review.
- Durable AI usage rows record model/token/latency/cost metadata and hashes,
  while the agent runtime maintains its own durable evidence. Cross-service
  trace correlation and unified durable AI telemetry are not implemented here.

Before calling a deployment production-ready, validate its threat model,
tenant isolation, retention and deletion requirements, provider behavior,
database failover/backup, worker recovery, capacity limits, alerting, evaluation
quality, and rollback procedures in that deployment's environment.

## Golden evaluation and feedback loop

`data/golden_datasets/ai_workflow.json` is the versioned golden set for workflow
answer quality. Run `python scripts/evaluate_ai_workflow.py` to evaluate the
configured inference route. The report includes expected-fact recall,
source-groundedness, citation coverage, regressions against dataset thresholds,
and approval-gated tuning recommendations. It does not retain generated answers.
Configure the same inference provider/model settings used by the deployment when
running an evaluation intended to measure that deployment.

Changes to prompts, retrieval, model routing, or the golden set run this
evaluation tests as a pull-request gate. The provider-backed evaluation also
runs when the `AEGIS_LLM_API_KEY` repository secret is configured; without it,
CI skips the external-model call. Feedback submitted through
`POST /agent/runs/{id}/feedback` remains in the tenant-scoped evidence chain and
can be reviewed using `GET /operator/agent/runs/{id}/feedback-summary`. Pass
reviewed ratings with `--feedback-rating <1-5>` to include them in an
evaluation's feedback summary and tuning recommendations. A failing score or a
low rating can suggest reviewing retrieval, prompts, or routing, but never
changes those settings automatically. Tune only in a proposed change, rerun the
golden evaluation, and require the existing human approval process before
promotion.
