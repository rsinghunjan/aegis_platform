# Aegis AI workflow

The canonical `production:app` provides a connected, tenant-authorized
knowledge-to-answer workflow alongside the durable agent lifecycle. The
operator UI's **AI Workflows** tab offers the same document-ingest and question
flow as the HTTP API.

## Run the reference workflow

Install and start the control plane:

```bash
python -m pip install -r requirements-control-plane.txt
uvicorn production:app --host 127.0.0.1 --port 8000
```

Agent and AI endpoints fail closed unless the embedding application constructs
the app with a `tenant_authorizer` that validates the caller's trusted identity
and tenant membership. The authorization actions for this workflow are
`ai_knowledge_write` and `ai_generate`. A tenant ID in a request is a resource
selector, not proof of access.
The default `production:app` starts without an authorizer or execution
dispatcher, so its protected AI/agent routes return 503 until an embedding
application supplies those dependencies. See the [runtime configuration
reference](agentic_runtime.md#configuration-reference) for required integration
callbacks and optional environment settings.

Without external credentials, Aegis uses local hash embeddings and an echo
inference provider. These make the workflow exercisable offline but do not
provide semantic embeddings or generated answers. Configure one or more
providers to use the unified inference gateway:

```bash
python -m pip install -e '.[ai]'
export AEGIS_LLM_API_KEY='...'
export AEGIS_LLM_MODEL='gpt-4o-mini'
# Optional for an OpenAI-compatible host (the value should include /v1 as needed).
export AEGIS_LLM_BASE_URL='https://api.openai.com/v1'
# Optional ordered fallback chain: openai, anthropic, ollama, vllm, echo.
export AEGIS_INFERENCE_PROVIDERS='openai,ollama'
```

The default provider selection preserves OpenAI-compatible inference when
credentials and the SDK are available, otherwise it uses the offline echo
provider. An explicit `AEGIS_INFERENCE_PROVIDERS` list is attempted in order;
it does not silently add echo. Set provider credentials/base URLs using the
provider-specific environment variables. The OpenAI-compatible key and base URL
are also shared with the optional planner. Set `AEGIS_LLM_PLANNER_ENABLED=true`
to use that planner; otherwise agent planning stays deterministic. For
model-backed document embeddings, set
`AEGIS_EMBEDDING_PROVIDER=openai`; the default `local-hash` embedding provider
is deterministic but not semantically trained. Defaults and all supported
runtime environment settings are listed in the [runtime configuration
reference](agentic_runtime.md#configuration-reference).

Index a tenant document and ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ai/knowledge \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","document":"Aegis protects AI workloads with policy and evidence."}'

curl -X POST http://127.0.0.1:8000/ai/answer \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","query":"How does Aegis protect AI workloads?"}'
```

Knowledge documents, chunks, and embeddings are persisted in SQL storage
(`AEGIS_AI_DATABASE_URL`, default `sqlite:///./aegis_ai.db`) with tenant-bound
vector queries. Use PostgreSQL for a shared multi-worker service; schema tables
are initialized idempotently by the application. Every answer writes an
inference audit row containing provider/model, token counts, latency, estimated
cost, status, and SHA-256 request/response fingerprints—not raw prompts or
answers. Set `AEGIS_LLM_INPUT_COST_PER_1K` and
`AEGIS_LLM_OUTPUT_COST_PER_1K` to the deployment's USD price per 1,000 tokens;
the default is zero until rates are configured. The tenant-authorized
`GET /operator/ai/usage?tenant_id=...` endpoint requires the `operator_read`
action and returns bounded usage records.

The answer response includes provider/model, token and latency signals,
estimated cost, request ID, and document/chunk citations. Requests are bounded (64,000 document characters,
8,000 query characters, at most 4,096 output tokens, 100 documents and 256,000
indexed characters per tenant, and 100 cached tenant pipeline objects per
process). Database quotas survive restarts. Storage operations and retrieval
are tenant-filtered; all AI endpoints still require the trusted injected
tenant authorizer. Configure database backups, access controls, and retention
for the knowledge and audit tables as part of deployment operations.

## Governed agent operations and feedback

For tool execution, create a durable run with `/agent/runs`, queue it with
`/agent/runs/{run_id}/execute`, then follow it through the operator run,
approval, and evidence endpoints. Execution requires a separately configured
dispatcher and worker with trusted registered tools. The worker policy gate
checks identity, tenant, scopes, risk, budget, and approval before invoking a
tool; tool results are verified against output schemas and acceptance criteria.
See [the runtime guide](agentic_runtime.md) for the state machine, capability
catalog, policy integrations, retries, and worker boundary.

Submit a 1–5 rating for a run's outcome through the feedback endpoint:

```bash
curl -X POST http://127.0.0.1:8000/agent/runs/RUN_ID/feedback \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","rating":4,"note":"Useful answer"}'
```

This endpoint requires the `ai_feedback` authorization action. Feedback is
stored as a hash-chained run evidence entry; it records the rating and a hash of
the optional note, not the note text. Agent output verification and user ratings
provide quality signals, but this implementation does not automatically change
prompts, retrain models, or promote a model based on feedback.

## Operations and scope

The answer endpoint reports per-request tokens, latency, cost, citations, and
request ID. Durable AI usage/audit records are available in the operator usage
endpoint. Agent runs remain a separate lifecycle; cross-service trace
correlation, provider-specific price catalogs, budget enforcement for direct
AI workflow calls, model evaluation pipelines, and automated feedback loops
remain follow-on work.
