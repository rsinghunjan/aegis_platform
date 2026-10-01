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
#!/usr/bin/env bash
#
# scripts/vault_create_approle.sh
#
# Operator script: create an AppRole bound to a policy for long-running services / automation
# Use AppRole only for workloads that cannot use OIDC or Kubernetes auth.
#
# Usage:
#   ./scripts/vault_create_approle.sh aegis-transit-role aegis-transit-policy
set -euo pipefail

ROLE_NAME="${1:-aegis-approle-role}"
POLICY_NAME="${2:-aegis-transit-policy}"

if [ -z "${VAULT_ADDR:-}" ]; then
  echo "VAULT_ADDR must be set"
  exit 2
fi

echo "Enabling approle auth method if missing"
if ! vault auth list -format=json | jq -e 'has("approle/")' >/dev/null 2>&1; then
  vault auth enable approle || true
fi

echo "Creating role ${ROLE_NAME} with policy ${POLICY_NAME}"
vault write auth/approle/role/"${ROLE_NAME}" \
    token_ttl="1h" \
    token_max_ttl="4h" \
    secret_id_ttl="10m" \
    policies="${POLICY_NAME}"

echo "Fetching role_id (store securely)"
ROLE_ID=$(vault read -field=role_id auth/approle/role/"${ROLE_NAME}"/role-id)
echo "ROLE_ID=${ROLE_ID}"

echo "Create a wrapped secret_id (operator should deliver to the service out-of-band)"
SECRET_ID_WRAPPED=$(vault write -wrap-ttl=60 -format=json auth/approle/role/"${ROLE_NAME}"/secret-id | jq -r '.wrap_info.token')
echo "WRAPPED_SECRET_ID_TOKEN=${SECRET_ID_WRAPPED}"
echo "Deliver the wrapped secret token to the service owner (it unwraps to get the secret_id)."

echo "AppRole ${ROLE_NAME} created. Keep ROLE_ID and unwrapping steps secure."
