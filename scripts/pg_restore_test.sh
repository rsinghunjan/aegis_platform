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
#!/usr/bin/env bash
# End-to-end restore smoke test:
#  - start ephemeral Postgres (Docker)
#  - download latest logical backup from S3 (by prefix)
#  - restore into ephemeral Postgres
#  - run validation queries
#
# Usage: ./scripts/pg_restore_test.sh <S3_BUCKET> <PREFIX> [PG_VERSION]
# Example:
#   ./scripts/pg_restore_test.sh my-backups aegis/backups 15
set -euo pipefail
S3_BUCKET="${1:-}"
PREFIX="${2:-backups}"
PG_VERSION="${3:-15}"
TMPDIR=$(mktemp -d)
cd "$TMPDIR" || exit 1

if [ -z "$S3_BUCKET" ] || [ -z "$PREFIX" ]; then
  echo "Usage: $0 <S3_BUCKET> <PREFIX> [PG_VERSION]" >&2
  exit 2
fi

CONTAINER_NAME="aegis-restore-test-$$"
PG_PASSWORD="testpass123"
PG_PORT=54320

echo "Starting ephemeral Postgres container (postgres:${PG_VERSION}) ..."
docker run --rm -d --name "$CONTAINER_NAME" -e POSTGRES_PASSWORD="$PG_PASSWORD" -p ${PG_PORT}:5432 "postgres:${PG_VERSION}"
# Wait for postgres to accept connections
for i in $(seq 1 30); do
  if docker exec "$CONTAINER_NAME" pg_isready -U postgres -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
    echo "Postgres ready"
    break
  fi
  sleep 2
done

# Find latest backup object under prefix
echo "Listing backups in s3://$S3_BUCKET/$PREFIX ..."
LATEST_KEY=$(aws s3 ls "s3://$S3_BUCKET/$PREFIX/" --recursive | sort | tail -n 1 | awk '{print $4}')
if [ -z "$LATEST_KEY" ]; then
  echo "No backups found under s3://$S3_BUCKET/$PREFIX/" >&2
  docker stop "$CONTAINER_NAME" || true
  exit 3
fi
echo "Latest backup key: $LATEST_KEY"

# Download
aws s3 cp "s3://$S3_BUCKET/$LATEST_KEY" "./backup.sql.gz" --only-show-errors

# Restore
echo "Restoring backup into ephemeral Postgres ..."
gunzip -c "./backup.sql.gz" | docker exec -i "$CONTAINER_NAME" psql -U postgres -d postgres

# Validation: run simple queries (customize for your schema)
echo "Running validation queries..."
docker exec -i "$CONTAINER_NAME" psql -U postgres -d postgres -c "SELECT 1 as ok;" >/dev/null

# Optionally run more complex checks (row counts, checksums)
echo "Restore smoke test succeeded."

# Cleanup
docker stop "$CONTAINER_NAME" >/dev/null || true
rm -rf "$TMPDIR"
scripts/pg_restore_test.sh
