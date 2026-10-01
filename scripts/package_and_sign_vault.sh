 url=https://github.com/rsinghunjan/aegis_multimodal_ai_system/blob/main/scripts/package_and_sign_vault.sh
#!/usr/bin/env bash
# Package a model dir deterministically (calls existing script) and ask Vault transit to sign the artifact.
# Usage:
#   ./scripts/package_and_sign_vault.sh <model_dir> <artifact_out_path>
# Requires: VAULT_ADDR + VAULT_TOKEN env set; transit key 'aegis-cosign' exists.
set -euo pipefail

MODEL_DIR="${1:-}"
OUT_TAR="${2:-}"

if [ -z "$MODEL_DIR" ] || [ -z "$OUT_TAR" ]; then
  echo "Usage: $0 <model_dir> <artifact_out_path>"
  exit 2
fi

if [ ! -f scripts/make_deterministic_archive.py ]; then
  echo "make_deterministic_archive.py not found; ensure scripts exist" >&2
  exit 3
fi

python3 scripts/make_deterministic_archive.py "$MODEL_DIR" "$OUT_TAR"

# compute digest base64
DIGEST_B64=$(openssl dgst -sha256 -binary "$OUT_TAR" | base64 -w 0)

if [ -z "${VAULT_ADDR:-}" ] || [ -z "${VAULT_TOKEN:-}" ]; then
  echo "VAULT_ADDR or VAULT_TOKEN not set; cannot sign with Vault transit" >&2
  exit 4
fi

SIGN_JSON=$(curl -sS --header "X-Vault-Token: $VAULT_TOKEN" --request POST --data "{\"input\":\"${DIGEST_B64}\"}" "${VAULT_ADDR%/}/v1/transit/sign/aegis-cosign")
echo "$SIGN_JSON" > "${OUT_TAR}.sig.json"
# store human-friendly .sig (may need processing depending on transit config)
echo "$SIGN_JSON" > "${OUT_TAR}.sig"
echo "Signed artifact and saved ${OUT_TAR}.sig (raw Vault response in ${OUT_TAR}.sig.json)"
