
#!/usr/bin/env bash
# Wait for API (http://localhost:8081/health) and other services to be ready.
set -euo pipefail

API_URL="${1:-http://localhost:8081/health}"
RETRIES=${2:-60}
SLEEP_SEC=${3:-2}

i=0
while [ $i -lt $RETRIES ]; do
  if curl -sSf "${API_URL}" >/dev/null 2>&1; then
    echo "Service healthy: ${API_URL}"
    exit 0
  fi
  i=$((i+1))
  echo "Waiting for ${API_URL} (${i}/${RETRIES})..."
  sleep ${SLEEP_SEC}
done

echo "Timed out waiting for ${API_URL}"
exit 1
