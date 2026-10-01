
#!/usr/bin/env bash
# Convenience script using gh CLI to set branch protection on 'main'.
# Requires: gh (GitHub CLI) authenticated and repo owner permissions.
# This sets:
#  - require PR reviews (1)
#  - require status checks to pass: "build-and-test", "staged-integration"
#
# Adjust 'required_status_checks' contexts to match your workflow job names.
set -euo pipefail

OWNER="$(git remote get-url origin | sed -n 's#.*/\([^/]*\)/\(.*\)\.git#\1#p')"
REPO="$(git remote get-url origin | sed -n 's#.*/\([^/]*\)/\(.*\)\.git#\2#p')"

# contexts must match the exact check run names (job names). Adjust if your CI uses different names.
REQUIRED_CHECKS='["Build,staged-integration","staged-integration","build-and-test"]'

echo "Setting branch protection on ${OWNER}/${REPO}:main"
# Use GitHub REST API via gh api because `gh api` accepts JSON body
gh api --method PUT /repos/${OWNER}/${REPO}/branches/main/protection -f required_status_checks='{"strict":true,"contexts":["build-and-test","staged-integration"]}' -f required_pull_request_reviews='{"required_approving_review_count":1}' -f enforce_admins=true

echo "Branch protection applied. Please verify in repository settings."
