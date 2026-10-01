#!/usr/bin/env bash
# Helper to run the k6 load test in tests/load/k6/aegis_load_test.js
# Example:
#   API_BASE=http://localhost:8081 ADMIN_USER=admin ADMIN_PASS=adminpass ./scripts/run_k6_load.sh
set -euo pipefail

K6_SCRIPT="tests/load/k6/aegis_load_test.js"
K6_IMAGE="loadimpact/k6:latest"

if command -v k6 >/dev/null 2>&1; then
  echo "Running k6 locally..."
  k6 run "$K6_SCRIPT"
else
  echo "k6 not found locally; running in docker..."
  docker run --rm -i -v "$PWD":/app -w /app -e API_BASE -e ADMIN_USER -e ADMIN_PASS $K6_IMAGE run "$K6_SCRIPT"
fi
