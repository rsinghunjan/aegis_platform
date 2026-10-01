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
#!/usr/bin/env bash
#
# Helper to set repository secrets using gh CLI.
# Usage:
#   ./scripts/gh_set_repo_secrets.sh owner/repo
#
# The script prompts for values; you can also provide env vars before running.
set -euo pipefail

REPO=${1:-}
if [ -z "$REPO" ]; then
  echo "Usage: $0 owner/repo" >&2
  exit 2
fi

echo "Setting GitHub Actions secrets for $REPO"
# Example secrets - adjust names/values as needed
read -p "AWS_ROLE_TO_ASSUME (role ARN): " AWS_ROLE
read -p "AWS_REGION (e.g. us-east-1): " AWS_REGION
read -p "OBJECT_STORE_BUCKET (staging bucket name): " OBJECT_STORE_BUCKET
read -p "COSIGN_PRIVATE_KEY_B64 (base64, optional; leave empty to skip): " COSIGN_PRIVATE_KEY_B64

echo "Setting secrets via gh..."
gh secret set AWS_ROLE_TO_ASSUME --repo "$REPO" --body "$AWS_ROLE"
gh secret set AWS_REGION --repo "$REPO" --body "$AWS_REGION"
gh secret set OBJECT_STORE_BUCKET --repo "$REPO" --body "$OBJECT_STORE_BUCKET"
if [ -n "$COSIGN_PRIVATE_KEY_B64" ]; then
  gh secret set COSIGN_PRIVATE_KEY_B64 --repo "$REPO" --body "$COSIGN_PRIVATE_KEY_B64"
fi

echo "Secrets set. Double-check them in the GitHub repo Settings -> Secrets -> Actions."
scripts/gh_set_repo_secrets.sh
