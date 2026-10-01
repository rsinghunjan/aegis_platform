
#!/usr/bin/env bash
# Postgres backup -> upload to S3 (uses pg_dump)
# Usage: ./scripts/pg_backup.sh <DATABASE_URL> <S3_BUCKET> <S3_PREFIX>
set -euo pipefail

DATABASE_URL="${1:-$DATABASE_URL}"
S3_BUCKET="${2:-$AUDIT_S3_BUCKET}"
S3_PREFIX="${3:-aegis/backups/postgres}"

if [ -z "$DATABASE_URL" ] || [ -z "$S3_BUCKET" ]; then
  echo "Usage: $0 <DATABASE_URL> <S3_BUCKET> [S3_PREFIX]" >&2
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTFILE="/tmp/aegis_pg_backup_${TS}.sql.gz"

echo "Dumping database..."
PGPASSWORD="$(python -c "import os,urllib.parse as u; print(u.urlparse('$DATABASE_URL').password or '')")"

# Use pg_dump (assumes pg tools installed)
pg_dump "$DATABASE_URL" | gzip -c > "$OUTFILE"

echo "Uploading to s3://${S3_BUCKET}/${S3_PREFIX}/${TS}.sql.gz"
aws s3 cp "$OUTFILE" "s3://${S3_BUCKET}/${S3_PREFIX}/${TS}.sql.gz" --acl private
if [ $? -ne 0 ]; then
  echo "Failed to upload backup" >&2
  exit 3
fi

echo "Backup uploaded successfully: s3://${S3_BUCKET}/${S3_PREFIX}/${TS}.sql.gz"
rm -f "$OUTFILE"
