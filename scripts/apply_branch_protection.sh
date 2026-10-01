 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
#!/usr/bin/env bash
#
# Apply branch protection settings (main by default) using the GitHub REST API.
#
# Usage:
#   REPO=owner/repo BRANCH=main GITHUB_TOKEN=ghp_... ./scripts/apply_branch_protection.sh
#
# The script configures:
#  - required status checks (strict/up-to-date)
#  - required PR reviews (1 approval, dismiss stale)
#  - enforce admins
#  - require linear history
#  - require conversation resolution
#
set -euo pipefail

REPO="${REPO:-}"
BRANCH="${BRANCH:-main}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [ -z "$REPO" ]; then
  echo "Set REPO=owner/repo" >&2
  exit 2
fi
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Set GITHUB_TOKEN with a token that has repo admin permissions" >&2
  exit 2
fi

API="https://api.github.com/repos/${REPO}/branches/${BRANCH}/protection"

# Required status check contexts - update these strings to match your exact workflow names
CONTEXTS=(
  "GCP Finalize & Ops Validation (on-demand + PR / push)"
  "Release with Ops Validation Gate"
  "Alibaba ACK GPU Smoke Test"
)

# Build JSON array for contexts
contexts_json=$(printf '%s\n' "${CONTEXTS[@]}" | jq -R -s -c 'split("\n")[:-1]')

read -r -d '' PAYLOAD <<EOF || true
{
  "required_status_checks": {
    "strict": true,
    "contexts": $contexts_json
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
EOF

echo "Applying branch protection to ${REPO}@${BRANCH}..."
resp=$(curl -sS -w "\n%{http_code}" -X PUT "$API" \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d "$PAYLOAD")

# Split body and status
http_code=$(echo "$resp" | tail -n1)
body=$(echo "$resp" | sed '$d')

if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
  echo "Branch protection applied successfully."
  echo "$body" | jq .
  exit 0
else
  echo "Failed to apply branch protection: HTTP $http_code" >&2
  echo "$body" | jq . >&2 || echo "$body" >&2
  exit 3
fi
