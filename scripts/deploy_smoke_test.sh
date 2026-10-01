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
#!/usr/bin/env bash
# Simple smoke test against the deployed service
# Usage: ./scripts/deploy_smoke_test.sh http://<cluster-ip>:8080
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
AUTH_URL="${BASE_URL}/auth/token"
PREDICT_URL="${BASE_URL}/v1/models/multimodal_demo/versions/v1/predict"
HEALTH_URL="${BASE_URL}/health"

echo "Checking health endpoint..."
curl -sf "${HEALTH_URL}" | jq .

echo "Requesting token (seeded admin user: admin/adminpass)..."
TOKEN_RESPONSE=$(curl -s -X POST "${AUTH_URL}" -F "username=admin" -F "password=adminpass")
ACCESS_TOKEN=$(echo "${TOKEN_RESPONSE}" | jq -r .access_token)

if [ -z "${ACCESS_TOKEN}" ] || [ "${ACCESS_TOKEN}" = "null" ]; then
  echo "Failed to obtain token. Response:"
  echo "${TOKEN_RESPONSE}"
  exit 1
fi

echo "Token obtained, running predict..."
cat > /tmp/payload.json <<'JSON'
{"text":"smoke test from CI"}
JSON

PREDICT_RESPONSE=$(curl -s -X POST "${PREDICT_URL}" -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" -d @/tmp/payload.json)
echo "Predict response:"
echo "${PREDICT_RESPONSE}" | jq .

echo "Smoke test completed successfully."
scripts
