#!/usr/bin/env bash
set -euo pipefail
#
# grant_keyvault_key_permissions_idempotent.sh
#
# Grant wrapKey/unwrapKey/get to a principal (UAMI objectId). Idempotent (will tolerate existing settings).
#
# Usage:
#  ./scripts/grant_keyvault_key_permissions_idempotent.sh --vault my-keyvault --principal-id <objectId> --key-name my-key
#
VAULT=""
PRINCIPAL_ID=""
KEY_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --vault) VAULT="$2"; shift 2;;
    --principal-id) PRINCIPAL_ID="$2"; shift 2;;
    --key-name) KEY_NAME="$2"; shift 2;;
    -h|--help) echo "Usage: $0 --vault VAULT --principal-id PRINCIPAL_ID --key-name KEY"; exit 0;;
    *) echo "Unknown $1"; exit 2;;
  esac
done

if [ -z "$VAULT" ] || [ -z "$PRINCIPAL_ID" ] || [ -z "$KEY_NAME" ]; then
  echo "Missing required args"
  exit 2
fi

echo "[azure] Ensuring Key Vault key exists..."
if ! az keyvault key show --vault-name "$VAULT" --name "$KEY_NAME" >/dev/null 2>&1; then
  echo "[azure] Key $KEY_NAME not found in vault $VAULT; create it first or run kms_rotate_azure_key.sh to provision a key."
  exit 1
fi

echo "[azure] Fetching current key access policies (if any)..."
# We will call set-policy which is idempotent: it will add the permissions for the provided principal.
echo "[azure] Granting wrapKey/unwrapKey/get to principal ${PRINCIPAL_ID} on vault ${VAULT}"
az keyvault set-policy --name "$VAULT" --object-id "$PRINCIPAL_ID" --key-permissions wrapKey unwrapKey get || {
  echo "[azure] az keyvault set-policy returned non-zero. You may not have permission to modify key policies."
  exit 1
}
echo "[azure] Key Vault policy updated (or already applied)."
