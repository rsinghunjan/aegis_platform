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
# Staging runbook — automated tasks & prerequisites
  - COSIGN_PRIVATE_KEY_B64 (base64 of private key)  -- required for sign job
  - OBJECT_STORE_BUCKET (optional, for storing signed artifacts)
  - TF_VAR_db_password (or supply in workspace variables)

Steps (high-level)
1. Run the scanner locally and inspect suggested refactors:
   python3 scripts/auto_refactor.py --root .    # dry run
   # optionally:
   python3 scripts/auto_refactor.py --apply     # writes .refactored.py files for review

2. Review .refactored.py edits, run unit tests, and commit replacements in a PR.

3. Provision staging infra (manual or CI):
   - Option A (recommended manual first): run terraform locally:
       cd infra/terraform/overlays/aws
       terraform init
       terraform apply -var="db_password=..." -auto-approve
   - Option B (CI): use GitHub Actions terraform-apply-staging.yml after confirming role & secrets.

4. Build & sign model artifacts:
   - In GitHub Actions run build-and-sign.yml with model_dir pointing to the model you want to sign.
   - Save the cosign public key in ExternalSecrets or Vault as cosign public key used by verifier.

5. Deploy Helm chart to staging:
   - Use helm-deploy-staging.yml workflow or run locally using terraform outputs and helm upgrade --install.

6. Run canary test:
   - Use canary-test.yml workflow to update image_tag and run simulate_canary_load.
   - Observe Prometheus alerts and ensure rollback triggers if alert conditions are met.

7. Verify DB backups & restore:
   - Use scripts/backup/backup_db.py to back up DB to staging object store.
   - Run scripts/backup/restore_db.py to restore into a staging DB and run smoke tests.

Security notes
- Use short-lived credentials via OIDC for CI where possible.
- Store cosign private keys offline or in a secure key manager; if you must store private key in GitHub secrets, rotate frequently and restrict repository access.

If you want, I will:
- A) Run the scanner and open a refactor PR replacing SDK usages (I will produce .refactored.py and commit after your approval).
- B) Prepare the terraform OIDC wiring & a non-production apply PR for infra (I will not run apply).
- C) Open PRs to add the build/sign and helm deploy workflows and provide a step-by-step checklist for you/infra to enable and run them.

Which action would you like me to take next (A/B/C or combination)? If you want me to start with A, I’ll run the scanner now and prepare a refactor branch. ```
