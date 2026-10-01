# Disaster Recovery (DR) Playbook — Aegis

Scope
-----
Covers full recovery from major outages: database loss, model artifact loss, region failure, and catastrophic cluster failure.

Priorities
1. Restore control plane (kube API) and ensure access to secrets & kubeconfig.
2. Restore database (RDS snapshot / pg_dump restore) and verify integrity.
3. Restore object store (model artifacts) from snapshots.
4. Boot application as read-only for verification, then re-enable write traffic.
5. Run smoke tests & validate SLOs before promoting to prod traffic.

RTO / RPO guidance
- Target RTO for dev: 1-2 hours; for prod: < 60 minutes (depends on SLA)
- RPO depends on backup cadence:
  - daily pg_dump -> RPO = 24h
  - for lower RPO enable WAL archiving or managed DB PITR (recommended)

Inventory (what you need to recover)
- Latest DB backups (S3 keys) + checksum
- Model artifact snapshots (S3 keys)
- Kubernetes manifests / Helm charts + image registry access
- Secrets (Vault or K8s secrets); ensure access to Vault/key to decrypt sealed-secrets
- CI/CD credentials (GHCR push/pull tokens)
- Contact list: infra on-call, DB admin, security

High-level recovery procedure
1. Gain access to cluster or create a new one.
2. Restore secrets:
   - If using Vault: ensure Vault is available and secrets are present
   - If using sealed-secrets: decrypt with controller private key
3. Restore Postgres:
   - Locate latest backup in S3 (verify checksum)
   - Launch Postgres instance (managed RDS or self-hosted)
   - Run `pg_restore` or `pg_dump` restore procedures
4. Restore model artifacts:
   - Copy model tarballs from S3 to model registry storage (S3 bucket or PV)
   - Re-run model registration script (or registry API) to re-populate DB metadata if needed
5. Deploy services:
   - Helm install aegis-api
   - Run DB migration job (alembic) only if schema is older than backups
   - Start workers and ensure Celery broker is available
6. Verification:
   - Run smoke tests (scripts/deploy_smoke_test.sh)
   - Validate metrics and traces
   - Run QA tests for models (sample inputs)

Rollback procedure (application-level)
- If a new deployment causes failures, roll back via Helm:
  - `helm rollback aegis-api <previous_revision>`
- If model canary fails, promote previous stable model via registry promotion (promote stage back to prod for previous version)

WAL / PITR recommendation
- For production, enable WAL shipping or use a managed DB with PITR (Point-in-Time Recovery) to reduce RPO to minutes.
- Store WAL in durable object storage and have automatic restore script to replay WAL up to a timestamp.

Postmortem checklist
- Capture timeline, root cause, impacted customers
- Review backups & retention policies; adjust schedule as needed
- Test the DR runbook in a scheduled game-day exercise at least annually

Contacts
- ops-oncall: ops@example.com
- db-admin: db-admin@example.com
- security: security@example.com
