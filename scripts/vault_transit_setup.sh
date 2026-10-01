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
#!/usr/bin/env bash
# Create transit key and least-privilege policy for Aegis model signing.
# Requires `vault` CLI authenticated as an operator (root/token with policies to create).
#
# Usage:
#   export VAULT_ADDR=https://vault.example:8200
#   vault login <operator-token>
#   ./scripts/vault_transit_setup.sh aegis-model-sign
set -euo pipefail

KEY_NAME="${1:-aegis-model-sign}"
TRANSIT_MOUNT="${VAULT_TRANSIT_MOUNT:-transit}"
POLICY_NAME="${POLICY_NAME:-aegis-transit-policy}"
echo "Creating transit key: ${KEY_NAME} at mount ${TRANSIT_MOUNT}"

# enable transit mount if not exists
if ! vault secrets list -format=json | jq -e "has(\"${TRANSIT_MOUNT}/\")" >/dev/null 2>&1; then
  echo "Enabling transit secrets engine at ${TRANSIT_MOUNT}"
  vault secrets enable -path="${TRANSIT_MOUNT}" transit
fi

# create key (supports automatic key rotation if desired)
vault write -f "${TRANSIT_MOUNT}/keys/${KEY_NAME}"

# write minimal policy for CI/token role usage
cat > /tmp/${POLICY_NAME}.hcl <<'EOF'
# Least-privilege policy to allow signing & verifying with a specific key
path "transit/sign/${KEY_NAME}" {
  capabilities = ["update"]
}
path "transit/verify/${KEY_NAME}" {
  capabilities = ["update"]
}
# optionally allow reading key metadata
path "transit/keys/${KEY_NAME}" {
  capabilities = ["read"]
}
EOF

echo "Writing policy ${POLICY_NAME}"
vault policy write "${POLICY_NAME}" /tmp/${POLICY_NAME}.hcl
rm -f /tmp/${POLICY_NAME}.hcl

echo "Done. Created transit key ${KEY_NAME} and policy ${POLICY_NAME}."
echo "Next: create an AppRole or OIDC role that attaches policy ${POLICY_NAME} and use OIDC/AppRole in CI to get short-lived VAULT_TOKEN."
