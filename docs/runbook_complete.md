  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
```markdown
# Runbook: achieving the completion criteria for Aegis (practical wiring)

This runbook explains the exact actions and configuration to reach the completion criteria:
- No direct provider SDKs in runtime code
- CI remote verification with short‑lived creds and cosign
- Terraform overlays that provision staging infra & Helm values that work
- Orchestrator enforces verification and fails-closed
- Canary rollout automation + alerts
- DB backups & restores validated
- OIDC auth, secrets in ExternalSecrets/Vault, and audit logging

1) Enforce "no direct SDKs"
 - The CI job `.github/workflows/sdk-scan.yml` runs `scripts/scan_cloud_sdk_usage.py --strict`.
 - Make sure the scanner is kept up-to-date; add `--strict` in CI to fail PRs.

2) CI remote verification via OIDC + cosign (AWS example)
 - On Github: add secret `AWS_ROLE_TO_ASSUME` (ARN of role with S3 read-only), `AWS_REGION`, optional `COSIGN_PUB_KEY`.
 - AWS: create IAM role trusting GitHub OIDC (`token.actions.githubusercontent.com`) with a condition for repository and environment.
 - Role policy should include minimal S3 GetObject / ListBucket for staging buckets.
 - The workflow `verify-model-signatures-aws-oidc.yml` will assume the role and run verifier.

3) Terraform overlays & Helm values
 - Apply overlay: `cd infra/terraform/overlays/aws && terraform init && terraform apply -var="db_password=..." -auto-approve`
 - Capture outputs (bucket_name, database_url) and substitute into `helm/values.aws.yaml`, then:
    `helm upgrade --install aegis ./helm -n aegis -f helm/values.aws.yaml`
 - For production-style infra, add VPC, subnets, and DB subnet groups.

4) Orchestrator verification & audit
 - Orchestrator must call `fetch_and_prepare_model_for_load(model_name)` from `aegis_multimodal_ai_system.orchestrator_verify_loader`.
 - On success, record audit via `aegis_multimodal_ai_system.orchestrator.audit_helper.record_audit_event`.
 - On failure, mark the model as unverified and alert.

5) Canary + monitoring
 - Install Argo Rollouts and Prometheus in cluster.
 - Apply `k8s/argo-rollouts/example-rollout.yaml` and `k8s/prometheus/alert-rules.yaml`.
 - Configure Prometheus scrape for model metrics and tests that simulate degraded canary to validate rollback.

6) Backups & restores
 - Regularly run `scripts/backup/backup_db.py` and store dumps in the object store.
 - Test restore with `scripts/backup/restore_db.py <s3://bucket/path>` into a staging DB and run smoke tests.

7) Secrets & OIDC for API
 - Deploy ExternalSecrets operator and configure `k8s/external-secrets/external-secret-cosign.yaml` (or similar) to populate cosign public key and other secrets.
 - Configure API to validate OIDC tokens via `aegis_multimodal_ai_system.api.auth.validate_oidc_token` and use `get_actor_from_token` for audit actor.

8) Acceptance criteria
 - Run scanner: zero findings.
 - CI verifier (OIDC) can download staging artifacts and cosign-verify them on PRs.
 - Terraform overlay provisions resources and Helm chart installs without modifying chart templates.
 - Orchestrator refuses to load unverified models and audit entries exist.
 - Canary test demonstrates automatic rollback on metric breach.
 - Backup/restore test passes and artifacts are present in object store.

If you want, I will:
 - A) Open a PR that wires orchestrator verification + audit calls in the primary load path (recommended).
 - B) Run the scanner, create a branch with refactors for discovered files and open a PR.
 - C) Prepare Terraform overlay wiring for a chosen provider (AWS or OCI) in a runnable state (I will not apply infra, you or infra team must approve).

Which would you like me to start implementing now?
```
docs/runbook_complete.md
