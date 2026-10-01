#!/usr/bin/env bash
# Simple chaos helper: choose a random pod in the specified namespace and delete it.
# Requires kubectl context pointed at the cluster where Aegis is running.
#
# Usage: ./scripts/chaos_kill_random_pod.sh default api
# Params:
#  $1 - namespace (default: default)
#  $2 - label selector (required, e.g., app=aegis-api or app=aegis-worker)
set -euo pipefail

NS=${1:-default}
if [ -z "${2:-}" ]; then
  echo "Usage: $0 <namespace> <label-selector>" >&2
  exit 2
fi
SELECTOR="$2"

echo "Selecting a random pod in namespace ${NS} with selector ${SELECTOR}..."
PODS=$(kubectl get pods -n "$NS" -l "$SELECTOR" -o jsonpath='{.items[*].metadata.name}')
if [ -z "$PODS" ]; then
  echo "No pods found for selector ${SELECTOR} in ${NS}" >&2
  exit 1
fi
ARR=($PODS)
R=${ARR[$((RANDOM % ${#ARR[@]}))]}
echo "Killing pod: $R"
kubectl delete pod "$R" -n "$NS" --grace-period=5 || true
echo "Killed $R"
