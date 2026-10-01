
#!/usr/bin/env bash
# Read terraform output JSON and write env file for downstream workflows or local helm use.
# Usage:
#   ./scripts/terraform/output_to_env.sh infra/terraform/overlays/aws/tf_outputs.json > staging.env
TF_JSON="$1"
if [ -z "$TF_JSON" ]; then
  echo "Usage: $0 <terraform_outputs.json>" >&2
  exit 2
fi
jq -r 'to_entries[] | "export " + .key + "=\"" + (.value.value|tostring) + "\""' "$TF_JSON"
