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
```markdown
# Production Readiness Evidence Template (collect & attach to sign-off)

Use this template to collect evidence for production sign-off.

1) Code lock
 - Scanner output (must be empty): attach `scanner_output.txt` from `scripts/scan_cloud_sdk_usage.py --strict`

2) Infra provisioning
 - Terraform apply log: attach `terraform_apply_log.txt`
 - terraform outputs (tf_outputs.json)

3) CI OIDC setup
 - AWS role ARN used by GitHub Actions
 - IAM policy JSON attached to role (s3 read-only)
 - Screenshot or diff of role trust policy showing token.actions.githubusercontent.com trust

4) Cosign signing
 - Model archive sha256 and cosign signature artifact (upload artifact links)
 - Public key stored in Vault/ExternalSecrets (path)
 - CI run id that produced signed artifact

5) Verifier CI
 - GitHub Actions run id for `verify-model-signatures-aws-oidc.yml` that successfully verified staged artifact
 - verifier logs

6) Orchestrator enforcement
 - Log lines showing verification success/failure during load
 - DB audit table entries (SELECT * FROM model_audit WHERE ...), attach CSV

7) Canary test
 - Argo Rollouts events showing canary promotion or rollback
 - Prometheus alert firing logs & Alertmanager events
 - Load generator logs

8) Backup & restore
 - Backup job logs showing upload URI
 - Restore job logs showing successful restore
 - Post-restore smoke test results

9) Security
 - Location of cosign private key (storage policy)
 - ExternalSecrets configuration (k8s manifest)
 - OIDC / IdP integration docs (endpoint, client id, audience)

Attach the above artifacts to your release ticket for sign-off.
```
docs/production_evidence_template.md
