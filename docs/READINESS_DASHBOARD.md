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
```markdown
# Aegis Readiness Dashboard / Notifier

This small system periodically runs the repository readiness checker and publishes a compact status summary:
- Posts the summary to a Slack webhook.
- Upserts (create/update) a comment with the same summary on every open PR in the repository.

Files added
- scripts/post_readiness_to_slack_and_prs.sh — runs the readiness check, posts to Slack, upserts PR comments.
- .github/workflows/readiness_poller.yml — GitHub Actions workflow that runs on a daily schedule and on-demand.

Prerequisites
- Ensure scripts/repo_readiness_check.sh (already added earlier) is present and executable.
- Add a repository secret named `SLACK_WEBHOOK_URL` containing an Incoming Webhook URL (Slack).
- The workflow uses the repository's GITHUB_TOKEN to post comments on PRs. The default token is sufficient for repo-level comments.

Configuration
- Modify the cron schedule in .github/workflows/readiness_poller.yml if you want a different cadence.
- The script looks for /tmp/repo_readiness.json produced by scripts/repo_readiness_check.sh — the workflow runs the checker first.
- The PR comment uses a marker <!-- aegis-readiness-check --> so subsequent runs will update the same comment instead of posting new ones.

How to run manually
- Locally (requires jq and a Slack webhook):
  SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ" \
    GITHUB_REPOSITORY="owner/repo" \
    GITHUB_TOKEN="<token-with-repo-permissions>" \
    ./scripts/post_readiness_to_slack_and_prs.sh

Security & best practices
- Store SLACK_WEBHOOK_URL in GitHub repo secrets or organization secrets — never hard-code it.
- The GITHUB_TOKEN provided by Actions is scoped to the workflow run and is preferred for posting comments.
- If you want to restrict PR commenting to particular branches or PR labels, enhance the script's PR filtering logic.

Extensibility ideas
- Provide a small web dashboard rendering /tmp/repo_readiness.json as a status page (simple static site).
- Post richer Slack blocks (colors/emoji) and include links to the latest workflow run artifacts.
- Only comment on PRs that touch infra/ops files (reduce noise).
docs/READINESS_DASHBOARD.md
