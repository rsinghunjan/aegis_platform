
#!/usr/bin/env bash
# Restore a logical backup into a running Postgres instance.
# Usage: ./scripts/pg_logical_restore.sh <S3_BUCKET> <S3_KEY> <PG_CONN>
# Example:
#  ./scripts/pg_logical_restore.sh my-bucket backups/backup-20250101T120000Z.sql.gz "postgresql://postgres:password@127.0.0.1:5432/postgres"
set -euo pipefail
S3_BUCKET="${1:-}"
S3_KEY="${2:-}"
PG_CONN="${3:-}"

if [ -z "$S3_BUCKET" ] || [ -z "$S3_KEY" ] || [ -z "$PG_CONN" ]; then
  echo "Usage: $0 <S3_BUCKET> <S3_KEY> <PG_CONN>" >&2
  exit 2
fi

TMP="/tmp/pg_restore_$$.sql.gz"
echo "Downloading s3://$S3_BUCKET/$S3_KEY to $TMP ..."
aws s3 cp "s3://$S3_BUCKET/$S3_KEY" "$TMP" --only-show-errors

echo "Restoring into Postgres ($PG_CONN) ..."
gunzip -c "$TMP" | pg_restore --dbname="$PG_CONN" --clean --no-owner
echo "Restore complete."
rm -f "$TMP"
scripts/pg_logical_restore.sh
