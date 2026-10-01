
#!/usr/bin/env sh
#
# Convert terraform tf_outputs.json -> staging.env with exported env vars.
# Usage: ./scripts/terraform/output_to_env.sh infra/terraform/overlays/aws/tf_outputs.json > staging.env
set -eu

TF_JSON=${1:-infra/terraform/overlays/aws/tf_outputs.json}
if [ ! -f "$TF_JSON" ]; then
  echo "tf_outputs.json not found at $TF_JSON" >&2
  exit 2
fi

# Handle keys with hyphens; uppercase them and replace - with _
# Example output: export BUCKET_NAME="value"
jq -r 'to_entries[] | "export \(.key | gsub(\"-\"; \"_\") | ascii_upcase)=\(.value.value)"' "$TF_JSON"
