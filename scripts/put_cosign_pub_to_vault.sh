
#!/usr/bin/env bash
set -euo pipefail
if [ -z "${VAULT_ADDR:-}" ]; then echo "VAULT_ADDR not set"; exit 2; fi
if [ -z "${VAULT_TOKEN:-}" ]; then echo "VAULT_TOKEN not set"; exit 2; fi
if [ $# -lt 2 ]; then
  echo "Usage: $0 <cosign_pub.pem> <vault_path (eg secret/data/aegis/cosign)>"
  exit 2
fi
PUB="$1"
VAULT_PATH="$2"
jq -n --arg pk "$(cat $PUB)" '{"data": {"public_key": $pk}}' | vault write "$VAULT_PATH" -
echo "Wrote public key to $VAULT_PATH"
