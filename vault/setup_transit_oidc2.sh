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
#!/usr/bin/env bash
# Run as Vault admin to:
#  - enable transit engine and create signing key
#  - create policy allowing transit/sign for aegis-cosign
#  - enable jwt (OIDC) auth and create a role bound to a GitHub Actions repo.
#
# Replace OWNER/REPO, VAULT_ADDR, VAULT_AUDIENCE with your values before running.
set -euo pipefail

export VAULT_ADDR="${VAULT_ADDR:-https://vault.example.internal:8200}"
export ROLE_NAME="${ROLE_NAME:-aegis-sign-role}"
export POLICY_NAME="${POLICY_NAME:-aegis-transit-policy}"
export TRANSIT_KEY="${TRANSIT_KEY:-aegis-cosign}"
export VAULT_AUDIENCE="${VAULT_AUDIENCE:-aegis-vault}"

echo "VAULT_ADDR: $VAULT_ADDR"
vault secrets enable -path=transit transit || true
vault write -f transit/keys/"$TRANSIT_KEY" type=rsa-4096 exportable=false

cat > /tmp/${POLICY_NAME}.hcl <<'EOF'
path "transit/sign/aegis-cosign" {
  capabilities = ["update"]
}
path "transit/keys/aegis-cosign" {
  capabilities = ["read"]
}
EOF
vault policy write "$POLICY_NAME" /tmp/${POLICY_NAME}.hcl
rm -f /tmp/${POLICY_NAME}.hcl

vault auth enable jwt || true
vault write auth/jwt/config oidc_discovery_url="https://token.actions.githubusercontent.com" oidc_client_id="$VAULT_AUDIENCE"

# Replace bound_claims repository value below with your OWNER/REPO (e.g., rsinghunjan/aegis_multimodal_ai_system)
vault write auth/jwt/role/"$ROLE_NAME" \
  role_type="jwt" \
  bound_issuer="https://token.actions.githubusercontent.com" \
  user_claim="sub" \
  bound_audiences="$VAULT_AUDIENCE" \
  token_ttl="5m" \
  token_max_ttl="10m" \
  policies="$POLICY_NAME" \
  bound_claims='{"repository": "OWNER/REPO"}'

echo "Created transit key $TRANSIT_KEY and role $ROLE_NAME bound to OWNER/REPO. Enable audit logging on Vault and test with a GitHub Actions run."
vault/setup_transit_oidc.sh
