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
 51
 52
 53
 54
 55
 56
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
 81
 82
```markdown
# Secret removal & rotation playbook

IMPORTANT: Do this in a coordinated maintenance window. Rewrite of history requires everyone to re-clone.

Summary steps (recommended order)
1. Rotate/Cleanup exposed credentials (immediately).
2. Generate detect-secrets baseline and commit it.
3. Remove secrets from history using git-filter-repo or BFG (after rotation).
4. Garbage-collect the repo and force-push the cleaned mirror.
5. Ask all collaborators to re-clone repository.
6. Enforce pre-commit detect-secrets and scheduled scans to prevent regression.

A. Rotate & revoke (examples)
- AWS (access/secret keys)
  - Identify key: AWS Console > IAM > Access keys for user
  - Create a new key, update CI/hosts to use new key, then delete the old key.
  - Docs: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html
- Google Cloud (service account keys)
  - Create new key, update secrets, delete old key from IAM > Service accounts.
  - Docs: https://cloud.google.com/iam/docs/creating-managing-service-account-keys
- GitHub token
  - Revoke the token at https://github.com/settings/tokens and create a new one.
- Other API keys (Stripe, Twilio, Slack, SendGrid)
  - Login to provider console, rotate keys, update CI and secrets.

B. Use git-filter-repo (recommended)
- Install:
  - pip install git-filter-repo
- Workflow (mirror + filter)
  1) Mirror the repo:
     git clone --mirror git@github.com:OWNER/REPO.git
     cd REPO.git
  2) Remove a path (example: secrets/ or a file):
     git filter-repo --path secrets/ --invert-paths
     # or remove specific file:
     git filter-repo --path path/to/file --invert-paths
  3) For replace-text (redact specific strings):
     Create `replacements.txt` with lines like:
       password==>REDACTED_PASSWORD
     Then:
       git filter-repo --replace-text ../replacements.txt
  4) Cleanup and push:
     git reflog expire --expire=now --all
     git gc --prune=now --aggressive
     git push --force --mirror origin
- After push: inform all contributors to delete local clones and re-clone.

C. Or use BFG (alternate)
- Download BFG jar: https://rtyley.github.io/bfg-repo-cleaner/
- Example remove files:
  git clone --mirror git@github.com:OWNER/REPO.git
  java -jar bfg.jar --delete-files id_rsa REPO.git
  cd REPO.git
  git reflog expire --expire=now --all && git gc --prune=now --aggressive
  git push --force

D. After history rewrite
- All collaborators: re-clone repo:
  git clone git@github.com:OWNER/REPO.git
- Re-apply any local branches by creating fresh branches off new origin.

E. Move secrets into a secret manager
- GitHub Actions secrets:
  - Using gh CLI:
    gh secret set VAR_NAME --body "value" --repo OWNER/REPO
  - Or via repository Settings → Secrets.
- HashiCorp Vault (example):
  - vault kv put secret/aegis/<env> API_KEY="..."
  - Update CI to fetch secrets via login method (AppRole/GitHub/OIDC).
- Avoid storing secrets in files; read from env vars and mount managed secrets in runtime.

F. Prevention & monitoring
- Commit the detect-secrets baseline (.secrets.baseline)
- Add pre-commit hook for detect-secrets (see .pre-commit-config.yaml)
- Add the scheduled GitHub Action for periodic scanning
- Add an incident template for secret leaks (rotate + remediate steps)

G. Communications
- Add a short notice in the PR that rewrites history and instructs team to re-clone.
- Document which credentials were rotated and who did the rotation (audit).
```
