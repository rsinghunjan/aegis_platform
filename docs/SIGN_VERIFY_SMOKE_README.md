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
```markdown
# Sign & Verify Smoke Test (Vault transit)

Purpose
- Provide an automated CI smoke test that ensures Vault transit signing + verification works end-to-end.
- Runs in CI (scheduled and manual dispatch) and fails if transit/verify does not return valid.

Files added
- .github/workflows/sign_verify_smoke_test.yml — GitHub Actions workflow to produce (or accept) an artifact, obtain OIDC token, login to Vault, sign the artifact, verify signature via transit/verify, and upload artifact+sig for inspection.
- scripts/sign_and_verify_vault.sh — helper script invoked by the workflow performing sign + verify logic.
- docs/SIGN_VERIFY_SMOKE_README.md — short runbook.

Configuration (placeholders — do not commit secrets)
- Ensure repository secrets:
  - VAULT_ADDR — URL of Vault (e.g., https://vault.example.internal:8200)
  - VAULT_AUDIENCE — the audience value configured in your Vault jwt role
  - OBJECT_STORE_BUCKET — optional S3 bucket for artifact upload in other workflows (not required for smoke test)
- Ensure Vault has the transit key aegis-cosign and a jwt role (aegis-sign-role) bound to this repo and the VAULT_AUDIENCE (see vault/setup_transit_oidc.sh).

How to run manually
1. Create a quick artifact (or let the workflow create one):
   echo "aegis smoke $(date -u)" > smoke-test.txt

2. Trigger workflow (gh CLI):
   gh workflow run sign_verify_smoke_test.yml -f artifact_path=smoke-test.txt

3. Inspect run logs:
   - Confirm "Request OIDC token" and "Vault login" steps succeed.
   - Confirm signature created and stored as smoke-test.txt.sig.json in artifacts.
   - Verify the job succeeded.

Notes
- The smoke test uses Vault transit/verify rather than exporting a public key. This avoids requiring key export while still proving that the transit key signs/verifies correctly.
- Keep the workflow in CI (scheduled weekly) to catch regressions (token/audience misconfigurations, Vault auth changes, etc).
docs/SIGN_VERIFY_SMOKE_README.md
