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
#!/usr/bin/env bash
# Run as Vault admin. Quick-setup script to:
#  - enable transit engine (aegis-cosign key)
#  - create a minimal policy for signing
#  - enable JWT/OIDC auth (GitHub Actions OIDC) and create a role bound to a specific repo
#
# Replace placeholders below: OWNER/REPO, VAULT_ADDR, VAULT_AUD, ROLE_NAME
set -euo pipefail

export VAULT_ADDR="${VAULT_ADDR:-https://vault.example.internal:8200}"
export ROLE_NAME="${ROLE_NAME:-aegis-sign-role}"
export POLICY_NAME="${POLICY_NAME:-aegis-transit-policy}"
export TRANSIT_KEY="${TRANSIT_KEY:-aegis-cosign}"
# Audience used when requesting the OIDC token from GitHub Actions. Use same audience in GH Actions step.
export VAULT_AUDIENCE="${VAULT_AUDIENCE:-aegis-vault}"

echo "VAULT_ADDR: $VAULT_ADDR"
echo "Creating transit key: $TRANSIT_KEY"
vault secrets enable -path=transit transit || true
vault write -f transit/keys/"$TRANSIT_KEY" type=rsa-4096 exportable=false

echo "Writing policy: $POLICY_NAME"
cat > /tmp/${POLICY_NAME}.hcl <<'EOF'
path "transit/sign/aegis-cosign" {
  capabilities = ["update"]
}
path "sys/mounts" {
  capabilities = ["read", "list"]
}
EOF
vault policy write "$POLICY_NAME" /tmp/${POLICY_NAME}.hcl
rm -f /tmp/${POLICY_NAME}.hcl

echo "Enabling jwt auth (OIDC discovery for GitHub Actions)"
vault auth enable jwt || true
# Configure discovery URL for GitHub Actions OIDC
vault write auth/jwt/config oidc_discovery_url="https://token.actions.githubusercontent.com" oidc_client_id="$VAULT_AUDIENCE"

echo "Creating role $ROLE_NAME bound to repo and audience"
# Bind claims to restrict only the repository and optionally branch/ref.
# Update 'bound_claims' to your repository (OWNER/REPO).
vault write auth/jwt/role/"$ROLE_NAME" \
  role_type="jwt" \
  bound_issuer="https://token.actions.githubusercontent.com" \
  user_claim="sub" \
  bound_audiences="$VAULT_AUDIENCE" \
  token_ttl="5m" \
  token_max_ttl="10m" \
  policies="$POLICY_NAME" \
  bound_claims='{"repository": "OWNER/REPO"}'

echo "Done. Transit key created: $TRANSIT_KEY. Role created: $ROLE_NAME (bound to OWNER/REPO)."
echo "Next: enable audit logging (vault audit enable file /var/log/vault_audit.log) and configure CI to request OIDC token w/ audience=$VAULT_AUDIENCE"
vault/setup_transit_oidc.sh
