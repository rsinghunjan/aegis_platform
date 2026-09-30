#!/usr/bin/env bash
set -euo pipefail

# Simple smoke runner for CI/local:
# - starts the server in background
# - waits for the server to accept connections
# - starts 2 clients (background)
# - waits for the server to exit (or fails after timeout)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Start server in background
python -m federated.server &
SERVER_PID=$!
echo "Started federated server (pid=${SERVER_PID})"

# Wait for server to accept connections on 127.0.0.1:8080
echo "Waiting for server to be reachable on 127.0.0.1:8080..."
MAX_WAIT=60
i=0
while true; do
  if python - <<PY >/dev/null 2>&1
import socket, sys, time
s = socket.socket()
try:
    s.settimeout(1.0)
    s.connect(("127.0.0.1", 8080))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    echo "Server reachable"
    break
  fi
  i=$((i + 1))
  if [ "$i" -ge "$MAX_WAIT" ]; then
    echo "Server did not become reachable within ${MAX_WAIT}s; killing server and failing"
    kill -9 "${SERVER_PID}" || true
    exit 1
  fi
  sleep 1
done

# Start two clients (background)
python -m federated.client --cid 1 --host 127.0.0.1:8080 &
C1_PID=$!
echo "Started client 1 (pid=${C1_PID})"

python -m federated.client --cid 2 --host 127.0.0.1:8080 &
C2_PID=$!
echo "Started client 2 (pid=${C2_PID})"

# Wait for server to exit, with an overall timeout
MAX_SERVER_WAIT=180
SECONDS_WAITED=0
while kill -0 "${SERVER_PID}" >/dev/null 2>&1; do
  if [ "${SECONDS_WAITED}" -ge "${MAX_SERVER_WAIT}" ]; then
    echo "Server did not finish within ${MAX_SERVER_WAIT}s; killing processes and failing"
    kill -9 "${SERVER_PID}" "${C1_PID}" "${C2_PID}" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 1
  SECONDS_WAITED=$((SECONDS_WAITED + 1))
done

# Reap clients and server
wait "${SERVER_PID}"
wait "${C1_PID}"
wait "${C2_PID}"

echo "Federated smoke test completed successfully"
