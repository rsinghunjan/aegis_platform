# AI Observability Runbook

Install `requirements-observability.txt` in the API and worker environments, set
`AEGIS_OTEL_ENABLED=true`, and set
`OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`. Prometheus scrapes
`/metrics`; traces are available in Jaeger at `http://localhost:16686`.
AI audit rows retain the trace ID alongside the request hash and provider data.
Set `AEGIS_SERVICE_NAME=aegis-api` for the API and `AEGIS_SERVICE_NAME=aegis-worker`
for workers so spans identify their originating service.
Start the local observability stack with
`docker compose -f docker-compose.observability.yaml up -d`, and connect the API
to the same Docker network so Prometheus can scrape `api:8000`.

## Backup and restore

1. Confirm the AI database backup job completed and retain its timestamp and
   checksum with the normal database backup evidence.
2. For PostgreSQL, take a consistent database snapshot with the deployment's
   approved `pg_dump`/snapshot procedure. For SQLite, stop writers and use the
   SQLite backup API; do not copy a live database file.
3. Restore the snapshot into an isolated recovery database using the matching
   database procedure. Verify `ai_inference_audit` and
   `ai_knowledge_documents` are present and that recent audit rows retain
   `trace_id`, status, and request hashes.
4. Run a synthetic `/ai/answer` request against the recovery deployment and
   verify its response trace ID matches the new audit row. Do not use customer
   prompts as test data.

## Provider failover

1. Open the alert's runbook link and search Jaeger by the `X-Trace-ID` response
   header. Check whether retrieval, provider, or audit persistence is failing.
2. Confirm the configured fallback is available and has current credentials,
   quota, and policy approval.
3. Change `AEGIS_INFERENCE_PROVIDERS` to the approved fallback order and roll
   out the API and worker configuration together.
4. Send a synthetic request, verify a successful trace through retrieval,
   provider, and audit persistence, then monitor the failure-rate and latency
   alerts for 15 minutes.
5. If the fallback is unhealthy, restore the previous provider configuration
   and escalate rather than disabling audit persistence or tenant quotas.

## Cost response

1. Inspect `aegis_ai_cost_usd_total` alongside request volume and provider spans.
2. Check for unexpected traffic, retry loops, and quota configuration changes.
3. Apply the approved tenant quota or provider budget change; do not log
   prompts, model responses, credentials, or tenant identifiers in metrics.
4. Confirm the cost increase has stopped and record the alert, change, and
   recovery evidence.

## Rollback and rehearsal

1. Record the current deployment revision and provider configuration before a
   rollout.
2. If a release causes elevated AI failures, route traffic to the last known
   good API/worker revision using the deployment system's rollback procedure.
   Keep API and worker versions aligned.
3. Verify `/healthz`, `/metrics`, Jaeger ingestion, and one synthetic
   request-to-audit trace before restoring normal traffic.
4. In staging each quarter, rehearse a provider outage, a database restore,
   and a deployment rollback using synthetic data. Capture time to detect,
   fail over, restore, and rollback; update this runbook with any gaps.
