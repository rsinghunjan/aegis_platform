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
#!/usr/bin/env bash
# Pull AEGIS secrets from Vault and apply to Kubernetes as a generic secret.
# Requires: vault CLI, kubectl configured or KUBECONFIG env.
#
# Usage:
#  VAULT_ADDR=... VAULT_TOKEN=... ./scripts/k8s_sync_secrets_from_vault.sh --namespace aegis --secret-name aegis-secrets
set -euo pipefail

NAMESPACE=${1:-aegis}
SECRET_NAME=${2:-aegis-secrets}

VAULT_MOUNT=${VAULT_MOUNT:-secret}
# Read active key pointer
ACTIVE_PATH="${VAULT_MOUNT}/data/aegis/keys/model_sign/active"

if ! command -v vault >/dev/null 2>&1; then
  echo "vault CLI required"
  exit 2
fi
if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl required"
  exit 2
fi

echo "Reading active model sign key id from Vault..."
ACTIVE_JSON=$(vault kv get -format=json "aegis/keys/model_sign/active")
ACTIVE_KEY_ID=$(echo "$ACTIVE_JSON" | jq -r '.data.data.active_key_id')
echo "Active key id: $ACTIVE_KEY_ID"

echo "Fetching key payload..."
KEY_JSON=$(vault kv get -format=json "aegis/keys/model_sign/${ACTIVE_KEY_ID}")
KEY_VALUE=$(echo "$KEY_JSON" | jq -r '.data.data.value.key')

# Fetch other secrets: JWT SECRET
JWT_JSON=$(vault kv get -format=json "aegis/secrets/jwt" 2>/dev/null || true)
JWT_SECRET=$(echo "$JWT_JSON" | jq -r '.data.data.value.SECRET_KEY' 2>/dev/null || echo "")

# prepare kubectl secret manifest (stringData so kubectl encodes)
TMPFILE=$(mktemp)
cat > "$TMPFILE" <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: ${SECRET_NAME}
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  AEGIS_MODEL_SIGN_KEY: "${KEY_VALUE}"
  AEGIS_MODEL_SIGN_KEY_ID: "${ACTIVE_KEY_ID}"
  AEGIS_SECRET_KEY: "${JWT_SECRET}"
EOF

echo "Applying secret to Kubernetes..."
kubectl apply -f "$TMPFILE"

echo "Secret updated. Cleanup."
rm -f "$TMPFILE"
