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
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
#!/usr/bin/env bash
# verify_and_fetch.sh
# InitContainer helper: download an artifact from S3, verify signature, and extract to destination.
#
# Behavior:
# - If /etc/cosign/pubkey.pem exists, use cosign verify-blob --key to verify the artifact using the public key.
# - Else, if VAULT_ADDR and VAULT_TOKEN are present, call Vault transit/verify to validate the signature.
# - On verification success, extract artifact (tar.gz) to destination.
#
# Usage:
#   ./scripts/verify_and_fetch.sh <s3_uri_to_artifact> <dest_dir>
set -euo pipefail

ARTIFACT_URI="${1:-}"
DEST_DIR="${2:-}"

if [ -z "$ARTIFACT_URI" ] || [ -z "$DEST_DIR" ]; then
  echo "Usage: $0 <s3_uri_to_artifact> <dest_dir>"
  exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

ARTFILE="$tmpdir/$(basename "$ARTIFACT_URI")"
SIGFILE="${ARTFILE}.sig.json"

echo "Downloading artifact: $ARTIFACT_URI"
# relies on AWS creds available to pod (IRSA / mounted creds / env)
aws s3 cp "$ARTIFACT_URI" "$ARTFILE"

# Attempt to find signature at same path with .sig.json suffix
SIG_S3_URI="${ARTIFACT_URI}.sig.json"
if aws s3 cp "$SIG_S3_URI" "$SIGFILE" 2>/dev/null; then
  echo "Downloaded signature from $SIG_S3_URI"
else
  echo "Signature not found at ${SIG_S3_URI}; exiting"
  exit 3
fi

# If cosign public key mounted, prefer local cosign verification
if [ -f /etc/cosign/pubkey.pem ]; then
  echo "Found local cosign public key at /etc/cosign/pubkey.pem — using cosign verify-blob"
  if ! command -v cosign >/dev/null 2>&1; then
    echo "cosign not present in image; please include cosign in initContainer image" >&2
    exit 4
  fi
  # cosign verify-blob expects the signature (raw) OR signature file depending on format.
  # If SIGFILE is Vault JSON, try to extract a raw signature field 'signature' first.
  SIG_RAW=$(jq -r '.signature // .data.signature // empty' "$SIGFILE" || true)
  if [ -z "$SIG_RAW" ]; then
    # If no raw signature field, attempt to treat SIGFILE as cosign-compatible signature file
    echo "No raw signature field in JSON; attempting cosign verify-blob --signature $SIGFILE"
    cosign verify-blob --key /etc/cosign/pubkey.pem --signature "$SIGFILE" "$ARTFILE"
  else
    # Write raw signature to file in a format cosign accepts (base64 -> binary)
    echo "$SIG_RAW" | awk '{print}' > "$tmpdir/sig.b64"
    base64 -d "$tmpdir/sig.b64" > "$tmpdir/sig.bin" || true
    cosign verify-blob --key /etc/cosign/pubkey.pem --signature "$tmpdir/sig.bin" "$ARTFILE"
  fi
  echo "cosign verification succeeded"
else
  # Fall back to Vault transit/verify (requires VAULT_TOKEN env to be present — prefer k8s auth)
  if [ -n "${VAULT_ADDR:-}" ] && [ -n "${VAULT_TOKEN:-}" ]; then
    echo "Verifying signature via Vault transit/verify"
    DIGEST_B64=$(openssl dgst -sha256 -binary "$ARTFILE" | base64 -w 0)
    SIGNATURE=$(jq -r '.signature // .data.signature // empty' "$SIGFILE" || true)
    if [ -z "$SIGNATURE" ]; then
      echo "No signature field found in signature JSON for Vault verification" >&2
      exit 5
    fi
    VERIFY_JSON=$(curl -sS --header "X-Vault-Token: $VAULT_TOKEN" --request POST \
      --data "{\"input\":\"${DIGEST_B64}\",\"signature\":\"${SIGNATURE}\"}" "${VAULT_ADDR%/}/v1/transit/verify/aegis-cosign")
    VALID=$(echo "$VERIFY_JSON" | jq -r '.data.valid // false' || true)
    if [ "$VALID" != "true" ]; then
      echo "Vault transit/verify returned invalid: $VERIFY_JSON" >&2
      exit 6
    fi
    echo "Vault transit verification succeeded"
  else
    echo "No cosign public key and VAULT_ADDR/VAULT_TOKEN not available — cannot verify" >&2
    exit 7
  fi
fi

# At this point verification passed; extract artifact to destination
mkdir -p "$DEST_DIR"
echo "Extracting $ARTFILE to $DEST_DIR"
tar -xzf "$ARTFILE" -C "$DEST_DIR"
echo "Model extracted to $DEST_DIR"
scripts/verify_and_fetch.sh
