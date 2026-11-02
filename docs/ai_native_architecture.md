# Aegis AI-native foundation

The AI-native foundation is a shared software layer around selected Aegis AI
workflows, not a replacement for the platform's existing inference, retrieval,
governance, or agent-runtime components.

## Components

- `AICore` wraps `AIWorkflow` for knowledge ingestion, answering, and agent
  planning inference, records tenant-scoped memory and observations, and
  produces operator insights.
- `AIMemory` keeps bounded interaction metadata, durable in SQL via `AICoreStore`. Query and run
  identifiers are hashed; raw prompts, goals, and answers are not retained.
- `AIObservability` summarizes answer and planning inference, latency, cost,
  agent planning, policy decisions, and lifecycle events.
- `AIPolicyEngine` creates heuristic policy suggestions. Suggestions require
  human approval and do not mutate enforcement policy.
- `AgentRuntime` remains the durable system of record for agent runs, evidence,
  approvals, and execution policy. Planning inference and tool policy decisions
  are observed by the shared core; the runtime's `AgentPolicyGate` remains the
  enforcement authority for every selected tool.

## HTTP surface

The existing `/ai/knowledge` and `/ai/answer` endpoints and agent planning
inference are routed through `AICore`. Agent run endpoints continue to use
`AgentRuntime` and the configured execution dispatcher. Worker processes should
configure `AgentWorker` with the same durable `AICore` so planning and policy
decisions are included in shared telemetry. `GET /operator/ai/insights?tenant_id=...` returns
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
- The AI core unifies knowledge answers and agent planning, but does not unify
  every inference, training, governance, or agent decision path behind one
  planner. Tool execution policy remains a separate, enforced runtime gate.
- The default local-hash embeddings and echo provider support offline
  development but are not semantic retrieval or useful generated answers.
- Production deployment still depends on trusted tenant authorization, durable
  database configuration and backups, provider credentials and rate/cost
  controls, worker/dispatcher operations, monitoring, and security review.
- Durable AI usage rows record model/token/latency/cost metadata and hashes,
  while the agent runtime maintains its own durable evidence. Cross-service
  trace correlation is not implemented here.

Before calling a deployment production-ready, validate its threat model,
tenant isolation, retention and deletion requirements, provider behavior,
database failover/backup, worker recovery, capacity limits, alerting, evaluation
quality, and rollback procedures in that deployment's environment.
