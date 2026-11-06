# Production agent and AI deployment

The API and agent runtime intentionally fail closed when tenant identity or agent
dispatch is not configured. When `AEGIS_ENV=production` (or
`AEGIS_PRODUCTION=true`), startup also requires trusted principal resolution,
tenant authorization, an execution dispatcher, development-provider fallbacks
disabled, and PostgreSQL URLs for both the agent store and AI/vector store.

## Kubernetes and Helm

Start from `helm/aegis-api/values.production.example.yaml`. It demonstrates a
shared external secret reference for the database, model API key, Celery broker,
and Celery result backend; it contains no credentials. Provision a secret named
`aegis-runtime` through the cluster's secret manager integration with these keys:

- `database-url`: PostgreSQL URL for a database with the `vector` extension
- `llm-api-key`: API key for the configured OpenAI-compatible planner, chat, and
  embedding provider
- `celery-broker-url` and `celery-result-backend`: URLs for the production queue
  and result backend

Configure `OIDC_ISSUER`, `OIDC_JWKS_URI`, and `OIDC_AUDIENCE` for the trusted
identity provider. Its signed access token must include a stable `sub` and the
tenant claim named by `AEGIS_OIDC_TENANT_CLAIM` (default `tenant_id`); configure
that claim to contain the tenant ID or an array of authorized tenant IDs. The
chart enables a separate Celery worker deployment, sharing database, provider,
and queue configuration with the API.

The provider cost-rate settings
(`AEGIS_LLM_INPUT_COST_PER_1K` and `AEGIS_LLM_OUTPUT_COST_PER_1K`) default to
zero. Set both to the current model/provider rates to make persisted usage
estimates meaningful. These values support attribution; they are not a hard
provider-side spend cap. Configure hard usage limits with the model provider and
cloud account as well.

## Verifying the deployment

1. Confirm API and worker pods use the same release image, PostgreSQL URL,
   broker, and result backend. Verify the API is ready and the worker consumes
   the configured Celery queue.
2. With valid and invalid OIDC tokens, verify tenant-scoped API requests are
   authorized or rejected as expected; never use caller-supplied tenant headers
   as identity.
3. Submit an agent run in a non-production test tenant and verify it is queued,
   executed once, persisted in PostgreSQL, and visible through the API.
4. Ingest and retrieve a document with the production embedding provider, then
   verify tenant isolation and provider/model/cost audit data.
5. Configure `AEGIS_OTEL_ENABLED=true`, `OTEL_EXPORTER_OTLP_ENDPOINT`, and a
   distinct `AEGIS_SERVICE_NAME` for API and worker. Agent execution messages
   carry W3C `traceparent`; check that the collector links API and worker spans.
6. Configure a scheduled PostgreSQL backup to durable object storage, restore it
   into an isolated database, and run migrations and integration checks against
   the restored copy before relying on the backup.
7. Run provider-backed evaluation with explicit evaluation credentials and
   budget limits. Keep feedback-derived changes human-reviewed: the current
   feedback loop proposes candidates and does not automatically alter prompts
   or models.
8. Configure the GitHub `model-promotion` environment with required reviewers,
   `MLFLOW_TRACKING_URI` and `AEGIS_BASE_URL` variables, and the required
   operator/signing secrets. Verify a signed promotion and audit artifact before
   enabling production promotion.

The chart example uses placeholder identity endpoints and image coordinates;
replace them for the deployment. CI and local container tests cannot verify
external identity, provider, broker, object-store, monitoring, or signing
credentials. Keep those as explicit environment-specific release gates.
