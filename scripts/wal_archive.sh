#!/usr/bin/env bash
# wal_archive.sh — push WAL files to S3 via wal-g
# Prereqs: wal-g installed and configured (WALG_S3_PREFIX, AWS creds or IAM role)
# Usage: run from Postgres host or container as archive_command or cron job
set -euo pipefail

WALG_S3_PREFIX=${WALG_S3_PREFIX:-}
if [ -z "$WALG_S3_PREFIX" ]; then
  echo "WALG_S3_PREFIX not set"
  exit 2
fi

# rotate base backup every N days via cron (pg_basebackup) and rely on wal-g for WALs
# Example: wal-g backup-push /var/lib/postgresql/data
echo "Starting wal-g backup push..."
wal-g backup-push /var/lib/postgresql/data
echo "Finished base backup push"

# Optional: remove old backups based on retention policy
# wal-g delete before TIMESTAMP
# wal-g delete retain <N> --confirm
