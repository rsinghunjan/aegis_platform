# Aegis AI-native foundation

The AI-native foundation is a shared software layer around selected Aegis AI
workflows, not a replacement for the platform's existing inference, retrieval,
governance, or agent-runtime components.

## Components

- `AICore` wraps `AIWorkflow` for knowledge ingestion and answering, records
  tenant-scoped memory and observations, and produces operator insights.
- `AIMemory` keeps bounded, process-local interaction metadata. Query and run
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
tenant-scoped in-process metrics, heuristic recommendations, and advisory
policy suggestions. It requires the existing `operator_read` authorization.
The existing authorization callbacks remain required; a tenant ID is not
authentication.

The answer response includes `repeated_query`, indicating whether the same
tenant asked the same exact query during this process's retained memory window.
It is a best-effort signal, not durable deduplication or idempotency.

## Production-readiness assessment

This foundation does **not** make Aegis fully AI-native or establish that a
deployment is production-ready. Important limitations include:

- Memory and insight counters are in-process, bounded, and lost on restart;
  replicas do not share them. Agent lifecycle events in these counters are
  observability hints, not the durable audit record.
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
