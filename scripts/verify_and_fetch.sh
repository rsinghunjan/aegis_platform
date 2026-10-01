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
 url=https://github.com/rsinghunjan/aegis_multimodal_ai_system/blob/main/scripts/verify_and_fetch.sh
#!/usr/bin/env bash
# Verify a signed artifact in object storage and unpack it to a destination directory.
# Usage:
#   verify_and_fetch.sh <artifact_s3_uri> <local_dest_path>
# Behavior:
# - Downloads artifact and .sig from S3 (requires aws cli & credentials available to pod).
# - Uses cosign verify-blob with the public key at /etc/cosign/cosign.pub (mounted via ExternalSecret).
# - On verification success, extracts artifact (assumes tar.gz) to <local_dest_path> parent and exits 0.
# - On failure, exits non-zero (fail-closed).
set -euo pipefail

ARTIFACT_S3="${1:-}"
DEST_PATH="${2:-}"

if [ -z "$ARTIFACT_S3" ] || [ -z "$DEST_PATH" ]; then
  echo "Usage: $0 s3://bucket/path/to/artifact.tar.gz /dest/path/model.tar.gz"
  exit 2
fi

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

ART_NAME="$(basename "$ARTIFACT_S3")"
SIG_NAME="${ART_NAME}.sig"
ART_LOCAL="${TMPDIR}/${ART_NAME}"
SIG_LOCAL="${TMPDIR}/${SIG_NAME}"

# Derive signature S3 path: assume sibling .sig in same prefix; allow override via env ARTIFACT_SIG_S3_URI
if [ -n "${ARTIFACT_SIG_S3_URI:-}" ]; then
  ARTIFACT_SIG_S3_URI="$ARTIFACT_SIG_S3_URI"
else
  ARTIFACT_SIG_S3_URI="$(dirname "$ARTIFACT_S3")/${SIG_NAME}"
fi

echo "Downloading artifact: $ARTIFACT_S3 -> $ART_LOCAL"
aws s3 cp "$ARTIFACT_S3" "$ART_LOCAL"

echo "Downloading signature: $ARTIFACT_SIG_S3_URI -> $SIG_LOCAL"
aws s3 cp "$ARTIFACT_SIG_S3_URI" "$SIG_LOCAL"

PUB_KEY="/etc/cosign/cosign.pub"
if [ ! -f "$PUB_KEY" ]; then
  echo "Cosign public key not found at $PUB_KEY. Ensure ExternalSecrets created secret 'cosign-public-key' with key 'cosign.pub'." >&2
  exit 3
fi

# Verify the artifact blob with cosign
if ! command -v cosign >/dev/null 2>&1; then
  echo "cosign CLI not found in image. Please include cosign in the initContainer image." >&2
  exit 4
fi

echo "Verifying signature..."
# use cosign verify-blob with signature file
if ! cosign verify-blob --key "$PUB_KEY" --signature "$SIG_LOCAL" "$ART_LOCAL"; then
  echo "Signature verification failed! Aborting (fail-closed)." >&2
  exit 5
fi

echo "Signature verified successfully. Extracting artifact..."
mkdir -p "$(dirname "$DEST_PATH")"
# Support tar.gz or plain file
case "$ART_LOCAL" in
  *.tar.gz|*.tgz)
    tar -xzf "$ART_LOCAL" -C "$(dirname "$DEST_PATH")"
    ;;
  *)
    # just move into place
    mv "$ART_LOCAL" "$DEST_PATH"
    ;;
esac

echo "Artifact placed at: $DEST_PATH"
exit 0
scripts/verify_and_fetch.shsc
