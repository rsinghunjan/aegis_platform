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
```markdown
# Security scanning (Trivy, dependency audits, repo linter)

This document explains the new CI jobs added to run automated security checks on pull requests and on a weekly schedule.

What runs
- Repo secret & key linter (scripts/repo_secret_lint.py)
  - Fails PRs that add likely private keys or signing secrets (COSIGN_PRIVATE_KEY_B64, PEM private keys, etc).
- Trivy filesystem scan
  - Scans repository files for OS / language vulnerabilities and IaC misconfigurations.
  - Produces an HTML report artifact (trivy-report.html).
  - Fails the job if HIGH or CRITICAL vulnerabilities are detected.
- Dependency audits
  - Python: pip-audit (reads requirements.txt or poetry-exported requirements).
  - Node: npm audit (uses package-lock.json).
  - Fails if HIGH/CRITICAL vulnerabilities are uncovered.

How to run locally (quick)
- Run the linter:
  python3 scripts/repo_secret_lint.py

- Run Trivy (requires trivy installed):
  trivy fs --severity CRITICAL,HIGH --exit-code 1 --cache-dir /tmp/trivy-cache .

- Run pip-audit for requirements.txt:
  python3 -m pip install pip-audit
  pip-audit -r requirements.txt

What to do on failures
- Linter finds patterns:
  - Inspect the file; if it's a false positive, update the linter patterns or move the file to a protected secret store.
  - If it’s an actual secret, remove it from the repo history, rotate the secret, and update the PR.
- Trivy / dependency audit finds issues:
  - Triage vulnerabilities by severity and fix/update packages.
  - For IaC failures, fix misconfigurations flagged by Trivy (e.g., insecure security group rules).
  - If a vulnerability is accepted as risk-tolerated, document the rationale in the PR and mark as exception in tracking system.

Notes & recommendations
- Do not add COSIGN private keys or other signing private key material to GitHub secrets / repo files. Use Vault transit (already implemented).
- Consider adding allowlist exceptions for known false positives with clear justification and tracking.
- Periodically (monthly) review vulnerability findings and upgrade dependencies.

If you'd like, I can:
- Add an “allowlist” mechanism that stores accepted CVE exceptions in a YAML file and makes the workflow emit warnings rather than fail for those items.
- Extend the Trivy job to scan built images by building Dockerfiles found in the repo (requires more permissions / runner time).
- Add automated GitHub Issue creation for high severity findings.
```
docs/SECURITY_SCAN_README.md
