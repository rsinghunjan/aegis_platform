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
# Finalize last ~12% runbook (quick checklist)

This runbook ties together the generated code and the final actions to complete Aegis.

1) Enforce no direct SDK imports
 - Commit `.pre-commit-config.yaml` and install pre-commit locally:
   pip install pre-commit
   pre-commit install
 - Ensure `.github/workflows/premerge-checks.yml` is enabled so PRs run scanner + tests.

2) Refactor remaining direct SDK usages
 - Run candidate refactors:
   python3 scripts/auto_refactor.py --apply
 - Review `.refactored.py` files and adapt replacements.
 - Run tests and open PR with changes.

3) Enforce verification at runtime (incremental)
 - Option A (recommended immediate): import patcher early in app startup:
     import aegis_multimodal_ai_system.orchestrator.enforce_verification_patch as ev
     ev.patch_loaders()
   This provides runtime fail-closed protection while callsites are migrated.
 - Option B (preferred long-term): replace loaders to call fetch_and_prepare_model_for_load explicitly.

4) Wire cosign & public key
 - Use `.github/workflows/build-and-sign.yml` to sign artifacts.
 - Store cosign public key in ExternalSecrets/Vault (external secret example provided).
 - Ensure verifier CI picks up COSIGN public key from mounted secret path.

5) Provision staging infra
 - Run Terraform overlay (example for AWS present). Use `scripts/terraform/output_to_env.sh` to export outputs to env file consumed by Helm/CI.
 - Update `helm/values.aws.yaml` with outputs and run helm install.

6) Canary test & automation
 - Use `k8s/argo-rollouts/example-rollout.yaml`, Prometheus rules, and `.github/workflows/canary-test.yml` to simulate canary traffic and verify promotion/rollback.

7) Backups & restore
 - Deploy CronJob `k8s/cronjobs/db-backup-cronjob.yaml` and test one run.
 - Use `k8s/jobs/restore-db-job.yaml` for a restore into staging DB and validate.

8) OIDC & secrets hardening
 - Configure ExternalSecrets to pull cosign key and object store credentials from Vault.
 - Ensure API uses `aegis_multimodal_ai_system.api.auth.validate_oidc_token` and use `get_actor_from_token` for audit entries.

9) Acceptance checks (after above)
 - Scanner CI returns zero findings.
 - Premerge checks pass.
 - A model PR referencing a staged artifact triggers verify-model-signatures-aws-oidc.yml and verification passes.
 - Orchestrator refuses to load manipulated artifact (fail-closed).
 - Canary test triggers rollback when errors injected.
 - Backup/restore test passes.

If you want, I will:
 - (A) Run the scanner across the repo and prepare a PR with `.refactored.py` candidates and suggested replacements.
 - (B) Add the enforce_verification_patch import into your orchestration/entrypoint files in a PR (so patching applies automatically on startup).
 - (C) Prepare a PR to add the pre-commit config and pre-merge CI workflow.
 - (D) Prepare a PR with k8s CronJob/Job manifests and ExternalSecrets wiring.

Pick one or more (A/B/C/D) and I’ll prepare branches and PRs immediately.
docs/finalize_12pct_runbook.md
