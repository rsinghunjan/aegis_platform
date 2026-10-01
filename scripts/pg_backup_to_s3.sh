#!/usr/bin/env bash
# Dump Postgres DB and upload to S3 (atomic, gzipped)
# Usage:
#   export DATABASE_URL=postgresql://user:pass@host:5432/db
#   export BACKUP_S3_BUCKET=s3://my-bucket/aegis-backups
#   ./scripts/pg_backup_to_s3.sh
set -euo pipefail

# Config (override env)
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:password@localhost:5432/aegis}"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
TMPDIR=$(mktemp -d)
OUTFILE="${TMPDIR}/aegis-db-${TIMESTAMP}.sql.gz"
LOGPREFIX="[pg_backup]"

if [ -z "${S3_BUCKET}" ]; then
  echo "${LOGPREFIX} BACKUP_S3_BUCKET not set"; exit 2
fi

echo "${LOGPREFIX} Starting pg_dump at ${TIMESTAMP}"
# Use pg_dump (requires pg client in image)
# If DATABASE_URL contains special characters, ensure it's quoted
pg_dump "${DATABASE_URL}" -Fc -f - | gzip -c > "${OUTFILE}"
echo "${LOGPREFIX} Dump complete, uploading to ${S3_BUCKET}"
aws s3 cp "${OUTFILE}" "${S3_BUCKET}/db/aegis-db-${TIMESTAMP}.sql.gz" --storage-class STANDARD_IA
# Optionally write a manifest with checksum
sha256sum "${OUTFILE}" > "${TMPDIR}/aegis-db-${TIMESTAMP}.sha256"
aws s3 cp "${TMPDIR}/aegis-db-${TIMESTAMP}.sha256" "${S3_BUCKET}/db/aegis-db-${TIMESTAMP}.sha256"
echo "${LOGPREFIX} Upload complete"

# Prune older backups using aws s3api (or set S3 lifecycle rules)
echo "${LOGPREFIX} Pruning backups older than ${BACKUP_RETENTION_DAYS} days"
# This is best-effort; prefer S3 lifecycle policies for reliability.
aws s3 ls "${S3_BUCKET}/db/" | awk '{print $4, $1" "$2" "$3}' | while read -r file rest; do
  # extract timestamp from filename if present or fall back to LastModified parsing
  if echo "${file}" | grep -qE 'aegis-db-[0-9T]+\.sql\.gz'; then
    # parse timestamp
    ts=$(echo "${file}" | sed -E 's/.*aegis-db-([0-9T]+Z)\.sql\.gz/\1/')
    # convert to epoch; if parse fails, skip
    if date -d "${ts}" >/dev/null 2>&1; then
      file_date_epoch=$(date -d "${ts}" +%s)
      now_epoch=$(date +%s)
      age_days=$(( (now_epoch - file_date_epoch) / 86400 ))
      if [ "${age_days}" -ge "${BACKUP_RETENTION_DAYS}" ]; then
        echo "${LOGPREFIX} Deleting old backup ${file}"
        aws s3 rm "${S3_BUCKET}/db/${file}" || true
        aws s3 rm "${S3_BUCKET}/db/$(basename ${file} .sql.gz).sha256" || true
      fi
    fi
  fi
done

echo "${LOGPREFIX} Backup job finished"
rm -rf "${TMPDIR}"script
