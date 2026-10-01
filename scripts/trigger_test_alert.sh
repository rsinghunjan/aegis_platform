
#!/usr/bin/env bash
# Trigger a synthetic test alert via Alertmanager API
# Usage: ./scripts/trigger_test_alert.sh <alertmanager_url> <alertname> <severity>
AM_URL="${1:-http://localhost:9093}"
ALERTNAME="${2:-AegisTestAlert}"
SEV="${3:-warning}"

cat <<EOF | curl -s -XPOST -H "Content-Type: application/json" --data @- "${AM_URL}/api/v1/alerts"
[{
  "labels": {
    "alertname": "${ALERTNAME}",
    "severity": "${SEV}"
  },
  "annotations": {
    "summary": "Test alert ${ALERTNAME}",
    "description": "This is a test alert for Alertmanager routing validation",
    "runbook": "docs/runbook_oncall.md#test-alert"
  },
  "startsAt": "$(date --iso-8601=seconds)",
  "endsAt": "$(date --iso-8601=seconds --date='+5 minutes')"
}]
EOF
echo "Triggered test alert ${ALERTNAME} -> ${AM_URL}"
