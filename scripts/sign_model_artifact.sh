 

#!/usr/bin/env bash
# Sign a model artifact using GPG (fallback) or Cosign if you prefer container signing
# Usage: ./scripts/sign_model_artifact.sh path/to/artifact
set -euo pipefail

ARTIFACT=${1:-}
if [ -z "$ARTIFACT" ]; then
  echo "Usage: $0 path/to/artifact"
  exit 2
fi

# Prefer cosign (if installed) with keyless signing for public registries (example for files)
if command -v cosign >/dev/null 2>&1; then
  echo "Using cosign keyless to sign file (requires cosign vX and Rekor access)"
  cosign sign-blob --keyless "$ARTIFACT" > "${ARTIFACT}.cosign.sig"
  echo "Wrote ${ARTIFACT}.cosign.sig"
  exit 0
fi

# Fallback: GPG local signing (requires a GPG key in the runner)
if command -v gpg >/dev/null 2>&1; then
  echo "Using gpg to sign artifact"
  gpg --batch --yes --armor --output "${ARTIFACT}.asc" --detach-sign "$ARTIFACT"
  echo "Wrote ${ARTIFACT}.asc"
  exit 0
fi

echo "No signing tool found (cosign or gpg). Install cosign or gpg."
exit 3
scripts/sign_model_artifact.sh
