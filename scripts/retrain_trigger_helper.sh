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
# retrain_trigger_helper.sh <event_json_path> <out_dir>
# Simple helper: reads an Alertmanager webhook payload (or generic JSON),
# extracts useful labels (alertname, severity, dataset_uri if provided),
# and invokes the training script with MLflow env pointing to cluster tracking server.
#
# This runs inside the workflow container (expects /workspace/training/train_deepspeed.py present in image).
set -euo pipefail
EVENT_JSON="${1:-}"
OUT_DIR="${2:-/workspace/model_registry/demo-models/cifar_deepspeed/tmp}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://mlflow.aegis.svc.cluster.local:5000}"

if [ -z "$EVENT_JSON" ] || [ ! -f "$EVENT_JSON" ]; then
  echo "Event JSON not provided or not found: $EVENT_JSON" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

# Extract possible dataset URI or labels from Alertmanager payload (best-effort)
ALERT_NAME=$(jq -r '.alerts[0].labels.alertname // "drift_alert"' "$EVENT_JSON" 2>/dev/null || true)
SEVERITY=$(jq -r '.alerts[0].labels.severity // "warning"' "$EVENT_JSON" 2>/dev/null || true)
DATASET_URI=$(jq -r '.alerts[0].annotations.dataset_uri // empty' "$EVENT_JSON" 2>/dev/null || true)

echo "Retrain helper triggered by alert: $ALERT_NAME (severity=$SEVERITY)"
if [ -n "$DATASET_URI" ]; then
  echo "Using DATASET_URI: $DATASET_URI"
fi

export MLFLOW_TRACKING_URI="$MLFLOW_URI"

# Call the training script; adapt args as your training supports
python3 /workspace/training/train_deepspeed.py --out-dir "${OUT_DIR}" --max-epochs 1

# After training finished, write a small metadata file for packaging step to pick up
jq -n --arg alert "$ALERT_NAME" --arg sev "$SEVERITY" --arg ds "$DATASET_URI" '{alert: $alert, severity: $sev, dataset: $ds, out_dir: "'"${OUT_DIR}"'"}' > "${OUT_DIR}/retrain_event_meta.json"

echo "Training complete; artifacts in ${OUT_DIR}"
scripts/retrain_trigger_helper.sh
