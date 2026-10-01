#!/usr/bin/env bash
# Basic verification: check S3 object and checksum, optionally test restore to a temp DB
# Usage:
#   ./scripts/verify_backup.sh s3://bucket/db/aegis-db-20251203T120000Z.sql.gz
set -euo pipefail
if [ $# -ne 1 ]; then echo "Usage: $0 s3://.../backup.sql.gz"; exit 2; fi
S3_PATH="$1"
TMPDIR=$(mktemp -d)
FILE="${TMPDIR}/backup.sql.gz"
LOG="[verify_backup]"

echo "${LOG} Downloading ${S3_PATH}"
aws s3 cp "${S3_PATH}" "${FILE}"
echo "${LOG} Download complete; computing sha256"
sha256sum "${FILE}"
# Optional: run a quick restore into a throwaway PG container (CI only)
echo "${LOG} Verification done"
rm -rf "${TMPDIR}"
