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
```markdown

High-level steps
1. Vault admin: run vault/setup_transit_oidc.sh after editing OWNER/REPO, VAULT_ADDR and VAULT_AUDIENCE.
2. Ensure transit key exists and is non-exportable (exportable=false).
3. Configure audit logging in Vault (file or syslog) to capture transit/sign calls.
4. Add repository secrets:
   - VAULT_ADDR: https://vault.example.internal:8200
   - VAULT_AUDIENCE: <audience used in Vault role> e.g., aegis-vault
   - OBJECT_STORE_BUCKET: s3 bucket for artifacts (if using S3 uploads)
5. Update .github/workflows/sign_with_vault.yml or call scripts/package_and_sign_vault.sh in your Argo job; the Actions workflow uses OIDC to login and obtain a short-lived token.

Testing (staging)
- Create a small artifact (echo "test" > test.tar.gz).
- Run the GitHub Action (workflow dispatch) with artifact_path pointing to the artifact file.
- Verify Vault audit log contains a transit/sign call with the login subject matching GitHub OIDC sub claim.
- Verify the signature file <artifact>.sig.json is produced and uploaded to S3 if configured.

Rollout plan
- Stage 1: deploy Vault role + use sign_with_vault.yml in a non-critical repo/workflow to validate flow.
- Stage 2: migrate one CI workflow (audit-only) to sign_with_vault flow and validate.
- Stage 3: replace COSIGN_PRIVATE_KEY_B64 usage across workflows, keep repo linter until no occurrences remain.
- Stage 4: remove any remaining private-key secrets from GitHub and add monitoring/alerts for re-introductions.

Notes / troubleshooting
- If Vault login fails, dump the login JSON to inspect claims: the OIDC sub must match bound_claims repository value.
- For Argo / in-cluster jobs, prefer Kubernetes auth method for Vault (k8s auth) instead of embedding tokens into pods.
- Consider backing the transit key with HSM/Cloud KMS if required by compliance.
```
vault/README.md
