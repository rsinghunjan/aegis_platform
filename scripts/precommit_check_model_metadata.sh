#!/usr/bin/env bash
# Local pre-commit helper: fails commit if staged additions under model_registry/ lack metadata or model card.
# Recommended to add to .pre-commit-config.yaml as a local hook.
set -euo pipefail
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^model_registry/' || true)
if [ -z "$STAGED" ]; then
  exit 0
fi

PASS=0
for f in $STAGED; do
  # find top-level model dir (first two segments)
  dir=$(echo "$f" | awk -F/ '{print $1 "/" $2}')
  if [ -d "$dir" ]; then
    if [ ! -f "$dir/metadata.yaml" ] && [ ! -f "$dir/metadata.yml" ] && [ ! -f "$dir/metadata.json" ]; then
      echo "ERROR: $dir is missing metadata.yaml/json. Add metadata.yaml according to MODEL_METADATA_SCHEMA.json"
      PASS=1
    fi
    if [ ! -f "$dir/MODEL_CARD.md" ]; then
      echo "ERROR: $dir is missing MODEL_CARD.md. Add a model card using docs/model_card_template.md"
      PASS=1
    fi
  fi
done

if [ $PASS -ne 0 ]; then
  echo "Model registry commit checks failed. Fix metadata and model card before committing."
  exit 1
fi
exit 0
