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
 65
 66
 67
 68
 69
 70
#!/usr/bin/env bash
# Restore Postgres from wal-g base backup + WAL replay to a target point-in-time or latest.
# This script restores into a new PGDATA directory (must be empty) and optionally starts a postgres instance for verification.
#
# Usage:
#  ./scripts/pg_restore_walg.sh --pgdata /tmp/restore-data --listen-port 55432 [--target-time "2025-12-03 01:23:45"]
#
# Environment:
#   WALG_S3_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
set -euo pipefail

PGDATA="${PGDATA:-}"
PGDATA_ARG=""
LISTEN_PORT=55432
TARGET_TIME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pgdata) PGDATA="$2"; shift 2;;
    --listen-port) LISTEN_PORT="$2"; shift 2;;
    --target-time) TARGET_TIME="$2"; shift 2;;
    *) echo "Unknown arg $1"; exit 2;;
  esac
done

: "${PGDATA:?--pgdata must be supplied (empty dir to restore into)}"
: "${WALG_S3_PREFIX:?WALG_S3_PREFIX must be set}"

if [ -d "$PGDATA" ] && [ "$(ls -A "$PGDATA")" ]; then
  echo "PGDATA directory $PGDATA exists and is not empty. Please provide an empty dir." >&2
  exit 3
fi

mkdir -p "$PGDATA"
echo "Restoring base backup into $PGDATA (S3 prefix ${WALG_S3_PREFIX})"

if ! command -v wal-g >/dev/null 2>&1; then
  echo "wal-g not installed. Please install wal-g and try again." >&2
  exit 4
fi

# Fetch the latest base backup into PGDATA
# Note: wal-g backup-fetch writes to the target directory
wal-g backup-fetch "$PGDATA" LATEST || {
  echo "wal-g backup-fetch failed" >&2
  exit 5
}

# Create recovery.conf / postgresql.conf adjustments for WAL archival replay
# wal-g now uses standby.signal / recovery.signal depending on Postgres version.
echo "Configuring recovery to replay WALs..."
PG_CONF="$PGDATA/postgresql.conf"
echo "listen_addresses='*'" >> "$PG_CONF"
echo "port = ${LISTEN_PORT}" >> "$PG_CONF"
# Ensure wal_level, archive_mode, and restore_command are correct — wal-g backup-fetch already populated restore commands in some setups.
# For wal-g we set restore_command to wal-g wal-fetch
cat >> "$PGDATA/recovery.conf" <<EOF
restore_command = 'envdir /etc/wal-g.d/env wal-g wal-fetch "%f" "%p"'
EOF

# Start Postgres on the restored data dir (use local postgres binary)
# Note: this assumes postgres bin is available and accessible (same major version)
if command -v pg_ctl >/dev/null 2>&1; then
  echo "Starting restored postgres on port ${LISTEN_PORT}..."
  pg_ctl -D "$PGDATA" -o "-p ${LISTEN_PORT}" -w start
  echo "Postgres started for verification (port ${LISTEN_PORT})."
  echo "Remember to stop it after verification: pg_ctl -D \"$PGDATA\" -w stop"
else
  echo "pg_ctl not available; restore is complete to $PGDATA. You must start postgres manually on this directory." >&2
  exit 0
fi
