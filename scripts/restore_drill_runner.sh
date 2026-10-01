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
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
#!/usr/bin/env bash
# Restore drill runner - local or k8s mode
#
# Usage:
#  ./scripts/restore_drill_runner.sh --mode local --pgdata /tmp/restore --listen-port 55432
#
# Modes:
#  - local : fetch latest wal-g base backup to PGDATA and start postgres locally (requires pg_ctl in runner)
#  - k8s   : create a Kubernetes Job that restores into a pod (requires kubectl context)
#
set -euo pipefail

MODE="local"
PGDATA="/tmp/aegis_restore_pgdata"
LISTEN_PORT="55432"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --pgdata) PGDATA="$2"; shift 2;;
    --listen-port) LISTEN_PORT="$2"; shift 2;;
    *) echo "Unknown arg $1"; exit 2;;
  esac
done

: "${WALG_S3_PREFIX:?WALG_S3_PREFIX must be set in env (s3://bucket/path)}"

if [ "$MODE" = "local" ]; then
  echo "Running local restore into ${PGDATA}"
  mkdir -p "$PGDATA"
  if ! command -v wal-g >/dev/null 2>&1; then
    echo "wal-g not installed in runner. Attempting to run restore inside a container..."
    docker run --rm -v "${PGDATA}:/var/lib/postgresql/data" -e WALG_S3_PREFIX="${WALG_S3_PREFIX}" -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}" -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}" walgalpine/wal-g:latest sh -c "wal-g backup-fetch /var/lib/postgresql/data LATEST"
  else
    export WALG_S3_PREFIX="${WALG_S3_PREFIX}"
    wal-g backup-fetch "$PGDATA" LATEST
  fi

  # configure minimal postgresql.conf adjustments for restored instance
  echo "listen_addresses='*'" >> "${PGDATA}/postgresql.conf"
  echo "port = ${LISTEN_PORT}" >> "${PGDATA}/postgresql.conf"

  if command -v pg_ctl >/dev/null 2>&1; then
    echo "Starting postgres for verification"
    pg_ctl -D "$PGDATA" -o "-p ${LISTEN_PORT}" -w start
    echo "Postgres started on port ${LISTEN_PORT} (PID $(cat ${PGDATA}/postmaster.pid 2>/dev/null || echo '?'))"
  else
    echo "pg_ctl not available - you must start postgres manually using restored PGDATA"
