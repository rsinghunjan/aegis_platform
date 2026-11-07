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
#!/usr/bin/env bash
# Smoke-test the restored Postgres instance.
# Usage:
#   ./scripts/verify_restore.sh --host localhost --port 55432 --user postgres
set -euo pipefail

HOST="localhost"
PORT="55432"
USER="postgres"
DB="postgres"
QUERY_TIMEOUT=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --db) DB="$2"; shift 2;;
    *) echo "Unknown arg $1"; exit 2;;
  esac
done

if ! command -v psql >/dev/null 2>&1; then
  echo "psql not found. Install libpq / psql to run verification queries." >&2
  exit 3
fi

echo "Running verification queries against ${HOST}:${PORT}..."
export PGPASSWORD="${PGPASSWORD:-}"

# Example smoke checks (customize to your schema)
set -o pipefail
psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -c "SELECT 1;" -t -A >/dev/null 2>&1 || {
  echo "Basic connection test failed" >&2
  exit 4
}

# Check presence of important tables
# adapt these checks to your core tables
for tbl in users models jobs; do
  echo "Checking table ${tbl} exists..."
  psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -c "SELECT 1 FROM information_schema.tables WHERE table_name='${tbl}' LIMIT 1;" -t -A | grep -q 1 || {
    echo "Expected table ${tbl} missing in restore" >&2
    exit 5
  }
done

echo "Smoke verification passed."
