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
# Vault OIDC / GitHub Actions setup (summary)

Goal
- Allow GitHub Actions to authenticate to Vault using OIDC and an assigned Vault role, without storing long‑lived VAULT_TOKEN in repo secrets.

High-level steps (operator)
1. Enable OIDC auth method in Vault (cluster admin):
   vault auth enable oidc

2. Configure OIDC provider with GitHub as IDP (Vault docs):
   vault write auth/oidc/config \
     oidc_discovery_url="https://token.actions.githubusercontent.com" \
     oidc_client_id="vault" \
     oidc_client_secret="..."

   # For GitHub, see HashiCorp docs: you usually configure Vault OIDC with GitHub Actions issuer.

3. Create a Vault policy for the CI role (store as file, e.g. vault/policies/aegis-oidc.hcl) and apply it:
   vault policy write aegis-ci vault/policies/aegis-oidc.hcl

4. Create a role that maps GitHub OIDC claims to the policy:
   vault write auth/oidc/role/github-actions \
     bound_audiences="vault" \
     user_claim="sub" \
     bound_subject="repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main" \
     policies="aegis-ci" \
     ttl="1h"

   - bound_subject can be more permissive (e.g., "*") but prefer narrow bindings for least privilege.
   - Alternatively create separate roles per repo/env.

5. In GitHub Actions workflow:
   - request id-token permission (permissions: id-token: write).
   - use hashicorp/vault-action@v2 with method: oidc and role: github-actions (or store role in a secret).
   - vault-action will mint an ephemeral VAULT_TOKEN for the job.

6. Rotate keys:
   - The workflow runs scripts/vault_rotate_model_sign_key.py (which writes keys to KV v2).
   - The workflow uses the ephemeral VAULT_TOKEN, so no long-lived tokens are stored.

Verification
- Use scripts/vault_verify_rotation.py to confirm active_key_id points to a valid key entry.
- Optionally sync to Kubernetes secrets and verify the secret update.

Security notes
- Use a narrow role bound_subject restricting repo, branch, or environment to limit what CI can do.
- Monitor Vault audit logs for rotation events.
- Prefer Vault Transit for signing operations in production (avoid writing raw key material to KV).
