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
 31
 32
 33
 34
 35
 36
 37
 38
#!/usr/bin/env bash
# Restore a backup to a temporary DB to verify integrity. Requires DOCKER and postgres image or a test DB.
# Usage: ./scripts/pg_restore_verify.sh s3://bucket/prefix/2025...sql.gz
set -euo pipefail

S3_URI="${1:-}"
if [ -z "$S3_URI" ]; then
  echo "Usage: $0 s3://bucket/path/to/backup.sql.gz" >&2
  exit 2
fi

TMP_FILE="/tmp/aegis_restore_verify.sql.gz"
aws s3 cp "$S3_URI" "$TMP_FILE"
if [ $? -ne 0 ]; then
  echo "Failed to download backup" >&2
  exit 3
fi

# Start temporary Postgres container
CONTAINER="aegis_pg_verify_$$"
docker run --name "$CONTAINER" -e POSTGRES_PASSWORD=pass -d -p 5433:5432 postgres:15
sleep 5

# Restore into temp DB
gunzip -c "$TMP_FILE" | docker exec -i "$CONTAINER" psql -U postgres
RC=$?

docker stop "$CONTAINER" >/dev/null
docker rm "$CONTAINER" >/dev/null
rm -f "$TMP_FILE"

if [ "$RC" -eq 0 ]; then
  echo "Restore verify: SUCCESS"
  exit 0
else
  echo "Restore verify: FAILED (psql returned $RC)" >&2
  exit 4
fi
scripts/pg_restore_verify.sh
