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
 62
 63
 64
 65
 66
 67
 68
 69
 70
#!/usr/bin/env bash
#
# scripts/vault_oidc_github_setup.sh
#
# Operator script: configure Vault OIDC auth role for GitHub Actions and attach the minimal policy
#
# Usage (operator):
#   export VAULT_ADDR='https://vault.example:8200'
#   vault login <operator-token>
#   ./scripts/vault_oidc_github_setup.sh \
#       --policy-name=aegis-transit-policy \
#       --role-name=aegis-github-role \
#       --github-repo=org/repo
#
# Notes:
# - This script configures Vault auth method for OIDC (or JWT depending on Vault version).
# - Adjust bound_audiences and claim mappings to match your GitHub Actions OIDC token audience and claims.
set -euo pipefail

POLICY_NAME="${1:-aegis-transit-policy}"
ROLE_NAME="${2:-aegis-github-role}"
GITHUB_REPO="${3:-your-org/your-repo}"  # e.g., "myorg/aegis"
OIDC_DISCOVERY="https://token.actions.githubusercontent.com"

if [ -z "${VAULT_ADDR:-}" ]; then
  echo "VAULT_ADDR must be set"
  exit 2
fi

echo "Creating OIDC auth method (if not present)"
if ! vault auth list -format=json | jq -e 'has("oidc/")' >/dev/null 2>&1; then
  vault auth enable oidc || true
fi

echo "Configure OIDC with GitHub discovery endpoint"
vault write auth/oidc/config \
  oidc_discovery_url="${OIDC_DISCOVERY}" \
  default_role="${ROLE_NAME}" || true

echo "Create role ${ROLE_NAME} bound to GitHub repo ${GITHUB_REPO}"
# These bound_claims will depend on Vault version and the JWT shape; adjust if your token uses sub/aud/iss claims differently.
vault write auth/oidc/role/"${ROLE_NAME}" \
  role_type="oidc" \
  bound_audiences="vault" \
  user_claim="sub" \
  allowed_redirect_uris="http://localhost" \
  oidc_scopes="openid" \
  bound_claims='{"repository": "'"${GITHUB_REPO}"'"}' \
  policies="${POLICY_NAME}" \
  ttl="1h"

echo "Ensure policy ${POLICY_NAME} exists (create if missing). Policy should include transit sign/verify rights."
if ! vault policy read "${POLICY_NAME}" >/dev/null 2>&1; then
  echo "Writing minimal policy ${POLICY_NAME}"
  cat > /tmp/${POLICY_NAME}.hcl <<'EOF'
path "transit/sign/aegis-model-sign" {
  capabilities = ["update"]
}
path "transit/verify/aegis-model-sign" {
  capabilities = ["update"]
}
path "transit/keys/aegis-model-sign" {
  capabilities = ["read"]
}
EOF
  vault policy write "${POLICY_NAME}" /tmp/${POLICY_NAME}.hcl
  rm -f /tmp/${POLICY_NAME}.hcl
fi

echo "OIDC role ${ROLE_NAME} created. In GitHub Actions workflow use hashicorp/vault-action or manual OIDC token exchange to obtain an ephemeral VAULT_TOKEN scoped to this role."
scripts/vault_oidc_github_setup.sh
