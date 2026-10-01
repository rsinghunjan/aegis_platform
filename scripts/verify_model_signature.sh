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
#!/usr/bin/env bash
# Verify model artifact signature
# Usage: ./scripts/verify_model_signature.sh path/to/artifact [signature_file]
set -euo pipefail

ARTIFACT=${1:-}
SIG=${2:-}

if [ -z "$ARTIFACT" ]; then
  echo "Usage: $0 path/to/artifact [signature_file]"
  exit 2
fi

if [ -z "$SIG" ]; then
  if [ -f "${ARTIFACT}.cosign.sig" ]; then
    SIG="${ARTIFACT}.cosign.sig"
  elif [ -f "${ARTIFACT}.asc" ]; then
    SIG="${ARTIFACT}.asc"
  else
    echo "No signature found (expected ${ARTIFACT}.cosign.sig or ${ARTIFACT}.asc)."
    exit 3
  fi
fi

if command -v cosign >/dev/null 2>&1 && [[ "$SIG" == *.cosign.sig ]]; then
  echo "Verifying with cosign keyless..."
  cosign verify-blob --keyless --signature "$SIG" "$ARTIFACT"
  exit $?
fi

if command -v gpg >/dev/null 2>&1 && [[ "$SIG" == *.asc ]]; then
  echo "Verifying with gpg..."
  gpg --verify "$SIG" "$ARTIFACT"
  exit $?
fi

echo "No suitable verifier found for $SIG"
exit 4
