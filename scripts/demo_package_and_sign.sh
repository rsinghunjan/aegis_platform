
#!/usr/bin/env bash
# Wrapper to package a model dir deterministically and sign with cosign (local dev).
# Usage: scripts/demo_package_and_sign.sh <model_dir> <artifact_out_path>
set -eu
MODEL_DIR=${1:-model_registry/demo-models/cifar_demo/0.1}
OUT_TAR=${2:-artifacts/cifar-demo-0.1.tar.gz}
SIG_PATH=${OUT_TAR}.sig

python3 scripts/make_deterministic_archive.py "$MODEL_DIR" "$OUT_TAR"
python3 scripts/create_model_signature.py "$OUT_TAR" "$MODEL_DIR/model_signature.json" --artifact-uri "file://$(pwd)/$OUT_TAR"

if [ -n "${COSIGN_PRIVATE_KEY_B64:-}" ]; then
  echo "$COSIGN_PRIVATE_KEY_B64" | base64 -d > /tmp/cosign.key
  cosign sign --key /tmp/cosign.key "$OUT_TAR"
  echo "Signed $OUT_TAR -> $SIG_PATH"
else
  echo "COSIGN_PRIVATE_KEY_B64 not set; skipping signing (set env or CI secret to sign)."
fi
