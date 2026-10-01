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
#!/usr/bin/env bash
# Simple logical backup (pg_dump) and upload to S3.
# Usage: ./scripts/pg_logical_backup.sh <PG_CONNSTRING> <S3_BUCKET> <PREFIX>
# Example:
#   ./scripts/pg_logical_backup.sh "postgresql://aegis:password@db-host:5432/aegisdb" my-bucket backups/aegis
set -euo pipefail
PG_CONN="${1:-}"
S3_BUCKET="${2:-}"
PREFIX="${3:-backups}"

if [ -z "$PG_CONN" ] || [ -z "$S3_BUCKET" ]; then
  echo "Usage: $0 <PG_CONNSTRING> <S3_BUCKET> [PREFIX]" >&2
  exit 2
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="backup-${TIMESTAMP}.sql.gz"
echo "Creating logical dump to $OUT ..."
pg_dump --format=custom --dbname="$PG_CONN" | gzip -c > "$OUT"

KEY="$PREFIX/$OUT"
echo "Uploading to s3://$S3_BUCKET/$KEY ..."
aws s3 cp "$OUT" "s3://$S3_BUCKET/$KEY" --only-show-errors
echo "Uploaded. Removing local file."
rm -f "$OUT"
echo "Backup complete: s3://$S3_BUCKET/$KEY"
