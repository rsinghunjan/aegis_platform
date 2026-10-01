#!/usr/bin/env bash
# Wrap packaging + Vault sign + Rekor entry (simple)
# Usage: ./scripts/package_and_attest.sh <model_dir> <artifact_out_path>
set -euo pipefail
MODEL_DIR="${1:-}"; OUT_TAR="${2:-}"
if [ -z "$MODEL_DIR" ] || [ -z "$OUT_TAR" ]; then
  echo "Usage: $0 <model_dir> <artifact_out_path>"
  exit 2
fi

# Create deterministic archive (assumes script available)
if [ -f scripts/make_deterministic_archive.py ]; then
  python3 scripts/make_deterministic_archive.py "$MODEL_DIR" "$OUT_TAR"
else
  tar -czf "$OUT_TAR" -C "$MODEL_DIR" .
fi

# Sign with Vault transit (requires VAULT_ADDR + VAULT_TOKEN in env)
if [ -z "${VAULT_ADDR:-}" ] || [ -z "${VAULT_TOKEN:-}" ]; then
  echo "VAULT_ADDR/VAULT_TOKEN not set; cannot sign" >&2
  exit 3
fi

DIGEST_B64=$(openssl dgst -sha256 -binary "$OUT_TAR" | base64 -w 0)
SIGN_JSON=$(curl -sS --header "X-Vault-Token: $VAULT_TOKEN" --request POST --data "{\"input\":\"${DIGEST_B64}\"}" "${VAULT_ADDR%/}/v1/transit/sign/aegis-cosign")
echo "$SIGN_JSON" > "${OUT_TAR}.sig.json"
echo "Signed artifact -> ${OUT_TAR}.sig.json"

# Post to Rekor (transparency log) if REKOR_URL and REKOR_API_KEY set
if [ -n "${REKOR_URL:-}" ]; then
  # Minimal Rekor POST: we submit a simple entry with artifact digest and signature
  rekor_entry=$(jq -n --arg d "$DIGEST_B64" --arg s "$(echo $SIGN_JSON | jq -r '.data.signature // .signature // empty')" '{kind:"aegis:artifact", digest:$d, signature:$s}')
  curl -sS -X POST -H "Content-Type: application/json" -d "$rekor_entry" "${REKOR_URL%/}/api/v1/log"
  echo "Posted minimal Rekor entry (if REKOR_URL configured)."
else
  echo "REKOR_URL not set; skipping Rekor transparency log post."
fi

# Optionally upload to S3 if OBJECT_STORE_BUCKET set
if [ -n "${OBJECT_STORE_BUCKET:-}" ]; then
  aws s3 cp "$OUT_TAR" "s3://${OBJECT_STORE_BUCKET}/model-archives/$(basename $OUT_TAR)"
  aws s3 cp "${OUT_TAR}.sig.json" "s3://${OBJECT_STORE_BUCKET}/model-archives/$(basename ${OUT_TAR}.sig.json)"
fi

echo "Package & attest complete."

