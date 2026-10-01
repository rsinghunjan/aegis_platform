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
```markdown
# Branch protection settings for Aegis (exact config to apply)

Apply these settings to the `main` branch (or another protected branch) to ensure the GCP finalize + ops validation and the ops-gate workflows block merges until checks pass.

Recommended, exact settings
- Branch: main
- Require pull request reviews before merging: true
  - Required approving reviews count: 1
  - Dismiss stale pull request approvals when new commits are pushed: true
  - Require review from Code Owners: false (optional)
- Require status checks to pass before merging: true
  - Require branches to be up-to-date before merging (strict): true
  - Required status checks (contexts):
    - "GCP Finalize & Ops Validation (on-demand + PR / push)"
    - "Release with Ops Validation Gate"
    - "Alibaba ACK GPU Smoke Test" (optional — remove if you don't want this to block)
  - Note: add or remove contexts to match the exact workflow names that appear in your repo's checks UI.
- Enforce for administrators: true
- Require linear history (no merge commits): true
- Require conversation resolution before merging: true
- Allow force pushes: false
- Allow branch deletion: false
- Restrict who can push: (leave unset unless you want to lock pushes to specific teams/users)

Required GitHub permissions / token to apply programmatically
- If applying via the GitHub API (script below) use a token with `repo` scope and repository admin permissions.
- If using an organization token, ensure it has the privilege to modify branch protection for the target repo.

How to use
- Option A (UI):
  1. Go to GitHub → Settings → Branches → Branch protection rules.
  2. Edit or add a rule for the `main` branch and set the exact options above.
  3. For required status checks, enter the workflow names exactly as they appear in the checks list.

- Option B (scripted via API):
  - Use the provided script `scripts/apply_branch_protection.sh`.
  - Ensure `GITHUB_TOKEN` is exported and REPO (owner/repo) is provided.
  - Example:
    REPO=your-org/aegis BRANCH=main GITHUB_TOKEN=${GH_TOKEN} ./scripts/apply_branch_protection.sh

Notes & troubleshooting
- Workflow names must match the status check context strings shown in the PR checks. If you change a workflow name, update the required contexts accordingly.
- If a required context never appears in the checks list (e.g., the workflow never runs on PRs), it will prevent merging — ensure the workflows run on PR/push or remove them from required contexts.
- To debug, check the branch protection API response for errors (insufficient permissions, invalid contexts).
```
