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
#!/usr/bin/env bash
# Push a base backup using wal-g to the configured object store (S3 / MinIO)
# Requires:
#   - wal-g installed in PATH
#   - environment variables:
#       WALG_S3_PREFIX (e.g. s3://bucket/path)
#       AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role)
#       PGHOST, PGPORT, PGUSER, PGPASSWORD (or .pgpass)
#
# Usage:
#   ./scripts/pg_backup_walg.sh
set -euo pipefail

echo "Starting wal-g base backup push..."

if ! command -v wal-g >/dev/null 2>&1; then
  echo "wal-g not found in PATH. Install wal-g and try again." >&2
  exit 2
fi

: "${WALG_S3_PREFIX:?Need to set WALG_S3_PREFIX env (e.g. s3://bucket/aegis)}"

export PGPASSWORD="${PGPASSWORD:-}"
# Run base backup
wal-g backup-push /var/lib/postgresql/data || {
  echo "wal-g base backup failed" >&2
  exit 3
}

echo "Base backup pushed to ${WALG_S3_PREFIX}"
