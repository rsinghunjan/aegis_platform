
#!/usr/bin/env bash
# Simple model signing helper using OpenSSL RSA private key.
# Produces a detached signature file (artifact.sig) using SHA256.
# Usage:
#   ./scripts/sign_model.sh /path/to/model.tar /path/to/private_key.pem
set -euo pipefail

ARTIFACT=${1:-}
PRIVKEY=${2:-}
OUTSIG=${3:-"${ARTIFACT}.sig"}

if [ -z "$ARTIFACT" ] || [ -z "$PRIVKEY" ]; then
  echo "Usage: $0 /path/to/artifact /path/to/private_key.pem [out.sig]"
  exit 2
fi

openssl dgst -sha256 -sign "$PRIVKEY" -out "${OUTSIG}.bin" "$ARTIFACT"
# Convert to PEM-like base64 for transport (optional). Keep raw binary for runtime verify.
mv "${OUTSIG}.bin" "${OUTSIG}"
echo "Signature written to ${OUTSIG}"
