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
#!/usr/bin/env bash
# Render alertmanager/alertmanager.yml from alertmanager/alertmanager.yml.template using envsubst.
# This avoids committing secrets into the repo and allows CI/k8s to inject secrets at deploy time.
#
# Usage:
#   SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..." \
#   PAGERDUTY_ROUTING_KEY="abc123" \
#   SAFETY_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..." \
#   ./scripts/render_alertmanager_config.sh
#
# The script writes ./alertmanager/alertmanager.yml (overwrites).
set -euo pipefail

TEMPLATE_FILE="alertmanager/alertmanager.yml.template"
OUTPUT_FILE="alertmanager/alertmanager.yml"

if [ ! -f "$TEMPLATE_FILE" ]; then
  echo "Template not found: $TEMPLATE_FILE" >&2
  exit 2
fi

: "${SLACK_WEBHOOK_URL:?Environment variable SLACK_WEBHOOK_URL must be set (do NOT commit this to repo)}"
: "${PAGERDUTY_ROUTING_KEY:?Environment variable PAGERDUTY_ROUTING_KEY must be set (do NOT commit this to repo)}"
: "${SAFETY_SLACK_WEBHOOK_URL:?Environment variable SAFETY_SLACK_WEBHOOK_URL must be set (do NOT commit this to repo)}"

# Use envsubst to replace variables safely. Limit substitution to the variables we expect.
export SUBST_VARS='${SLACK_WEBHOOK_URL} ${PAGERDUTY_ROUTING_KEY} ${SAFETY_SLACK_WEBHOOK_URL}'

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Render
envsubst "$SUBST_VARS" < "$TEMPLATE_FILE" > "$OUTPUT_FILE".tmp

# Validate YAML minimally (yq preferred if available)
if command -v python >/dev/null 2>&1; then
  python - <<PY
import sys, yaml
try:
  yaml.safe_load(open("$OUTPUT_FILE.tmp"))
except Exception as e:
  print("Rendered config is invalid YAML:", e, file=sys.stderr)
  sys.exit(3)
print("Rendered alertmanager config validated as YAML")
PY
fi

mv "$OUTPUT_FILE".tmp "$OUTPUT_FILE"
chmod 600 "$OUTPUT_FILE"
echo "Rendered $OUTPUT_FILE (permissions set to 600). Do not commit this file with secrets."
scripts/render_alertmanager_config.sh
