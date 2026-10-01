#!/usr/bin/env bash
# Restore Postgres DB from S3 backup (destructive!)
# Usage:
#   ./scripts/pg_restore_from_s3.sh s3://bucket/db/aegis-db-20251203T120000Z.sql.gz
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 s3://bucket/path/to/backup.sql.gz"; exit 2
fi

BACKUP_S3_PATH="$1"
TMPDIR=$(mktemp -d)
LOCAL_FILE="${TMPDIR}/backup.sql.gz"
LOGPREFIX="[pg_restore]"

echo "${LOGPREFIX} Downloading ${BACKUP_S3_PATH}"
aws s3 cp "${BACKUP_S3_PATH}" "${LOCAL_FILE}"
echo "${LOGPREFIX} Restoring to DATABASE_URL=${DATABASE_URL:-postgresql://postgres:password@localhost:5432/aegis}"

gunzip -c "${LOCAL_FILE}" | pg_restore --clean --no-owner --dbname="${DATABASE_URL:-postgresql://postgres:password@localhost:5432/aegis}"
echo "${LOGPREFIX} Restore complete"
rm -rf "${TMPDIR}"
