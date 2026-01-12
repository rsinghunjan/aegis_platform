#!/usr/bin/env bash
set -euo pipefail
#
# ops/onboarding/onboard_run.sh
#
# Operator helper script to run HIL onboarding (Acme example) and RT rollout + WCET validation.
# This script wraps and sequences certify_rack.py and rt_rollout_script.sh, signs artifacts if KMS is configured,
# and summarises outputs in /tmp/aegis_onboard_summary.json
#
# Usage:
#   export ACME_HIL_URL=... ACME_HIL_TOKEN=...
#   export EVIDENCE_BUCKET=...
#   export AWS_KMS_KEY_ID=...
#   ./ops/onboarding/onboard_run.sh --rack acme-rack-1 --manifest ops/hil/playbook/example_replay_manifest.json --repeats 12
#

RACK_NAME="acme-rack-1"
MANIFEST="ops/hil/playbook/example_replay_manifest.json"
REPEATS=12
S3_BUCKET="${EVIDENCE_BUCKET:-}"
RT_IMAGE="${RT_IMAGE:-}"
SUMMARY_OUT="/tmp/aegis_onboard_summary.json"

print_help() {
  cat <<EOF
Usage: $0 [--rack name] [--manifest path] [--repeats N] [--s3-bucket bucket] [--rt-image image]
Example:
  ./ops/onboarding/onboard_run.sh --rack acme-rack-1 --manifest ops/hil/playbook/example_replay_manifest.json --repeats 12 --s3-bucket my-evidence
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rack) RACK_NAME="$2"; shift 2;;
    --manifest) MANIFEST="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --s3-bucket) S3_BUCKET="$2"; shift 2;;
    --rt-image) RT_IMAGE="$2"; shift 2;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg $1"; print_help; exit 2;;
  esac
done

echo "Aegis Onboard Run"
echo "Rack: $RACK_NAME"
echo "Manifest: $MANIFEST"
echo "Repeats: $REPEATS"
echo "S3 bucket: ${S3_BUCKET:-<none>}"
echo "RT image: ${RT_IMAGE:-<none>}"
echo

# Preflight: basic checks
missing=()
command -v python >/dev/null || missing+=("python")
command -v kubectl >/dev/null || echo "kubectl not found; ensure you have cluster access if running RT jobs"
if [ ${#missing[@]} -ne 0 ]; then
  echo "Missing commands: ${missing[*]}. Install before continuing."
  exit 2
fi

# 1) HIL smoke & simulate missing hardware if not configured
if [ -z "${ACME_HIL_URL:-}" ] || [ -z "${ACME_HIL_TOKEN:-}" ]; then
  echo "ACME_HIL_URL or ACME_HIL_TOKEN not set. Running hardware gate simulator (will simulate HIL)."
  python ops/ci/hardware_gate_simulator.py --check hil || true
  SIMULATED_HIL=true
else
  SIMULATED_HIL=false
  echo "Running HIL smoke test against vendor..."
  python - <<PY
from ops.hil.adapters.acme_hil_adapter import AcmeHILAdapter
print("Backends:", AcmeHILAdapter().list_backends())
PY
fi

# 2) Run certify_rack (this runs repeats and produces summary)
echo "Running certify_rack.py (this may take a while)..."
CERTIFY_CMD=(python ops/hil/certify_rack.py --adapter ops.hil.adapters.acme_hil_adapter.AcmeHILAdapter --manifest "$MANIFEST" --name "$RACK_NAME" --repeats "$REPEATS")
if [ -n "$S3_BUCKET" ]; then
  CERTIFY_CMD+=(--s3-bucket "$S3_BUCKET")
fi
set +e
"${CERTIFY_CMD[@]}"
CERT_EXIT=$?
set -e
if [ $CERT_EXIT -ne 0 ]; then
  echo "certify_rack failed (exit $CERT_EXIT). Check logs. If running in simulation this is expected."
fi

# Find determinism summary
SUMMARY_CANDIDATES=(/tmp/hil_replay_summary*.json)
SUMMARY_FILE=""
for f in "${SUMMARY_CANDIDATES[@]}"; do
  [ -f "$f" ] || continue
  SUMMARY_FILE="$f"
  break
done

if [ -n "$SUMMARY_FILE" ]; then
  echo "Found determinism summary: $SUMMARY_FILE"
  SUMMARY_SIGNED=false
  if [ -n "${AWS_KMS_KEY_ID:-}" ] && command -v python >/dev/null ; then
    echo "Attempting to sign determinism summary with KMS..."
    python ops/governance/kms_sign_helper.py --file "$SUMMARY_FILE" --key-id "$AWS_KMS_KEY_ID" || echo "KMS sign failed"
    if [ -f "${SUMMARY_FILE}.sig" ]; then SUMMARY_SIGNED=true; fi
  fi
else
  echo "No determinism summary found; continuing (maybe simulated run)."
fi

# 3) RT rollout & WCET
if [ -n "$RT_IMAGE" ]; then
  echo "Starting RT rollout using image: $RT_IMAGE"
  ./ops/rt/rt_rollout_script.sh "$RT_IMAGE" || echo "RT rollout script returned non-zero"
else
  echo "No RT image provided. To run RT rollout, re-run with --rt-image <registry/repo:tag>"
fi

# 4) If WCET aggregate exists, sign & upload
if [ -f /tmp/wcet_aggregate_remote.json ]; then
  echo "Found WCET aggregate: /tmp/wcet_aggregate_remote.json"
  if [ -n "${AWS_KMS_KEY_ID:-}" ]; then
    python ops/governance/kms_sign_helper.py --file /tmp/wcet_aggregate_remote.json --key-id "$AWS_KMS_KEY_ID" || echo "KMS sign failed for WCET"
  fi
  if [ -n "${S3_BUCKET:-}" ]; then
    python ops/evidence/s3_retention_uploader.py --pattern /tmp/wcet_aggregate_remote.json --bucket "$S3_BUCKET" || echo "S3 upload failed"
  fi
fi

# 5) Summarize and write summary JSON
python - <<PY
import json, os, datetime
out = {
  "rack": "${RACK_NAME}",
  "manifest": "${MANIFEST}",
  "summary_file": "${SUMMARY_FILE or ''}",
  "summary_signed": ${SUMMARY_SIGNED:-False},
  "rt_image": "${RT_IMAGE}",
  "wcet_aggregate": os.path.exists("/tmp/wcet_aggregate_remote.json"),
  "ts": datetime.datetime.utcnow().isoformat()+"Z"
}
open("${SUMMARY_OUT}","w").write(json.dumps(out, indent=2))
print("Wrote final onboard summary to ${SUMMARY_OUT}")
PY

echo "Onboarding run complete. Inspect ${SUMMARY_OUT} and artifacts under /tmp. Upload ticket with artifacts and next steps."
