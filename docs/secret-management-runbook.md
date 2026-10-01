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
# Aegis Secret Management & Rotation Runbook
- Track metrics and alert if auth fails or models break.

Automation overview included in this repo
- `scripts/vault_rotate_model_sign_key.py` - creates a new key in Vault and marks it active
- `scripts/k8s_sync_secrets_from_vault.sh` - pulls Vault secrets and updates K8s secret
- GitHub Actions scheduled workflow to rotate keys regularly (optional) — see `.github/workflows/scheduled-rotate-keys.yml`

Best practices
- Rotate symmetric keys every 30–90 days depending on sensitivity; rotate ephemeral API credentials more frequently.
- Use asymmetric signing (e.g., GPG / KMS) for long-lived attestations if audits require non-repudiation.
- Use Vault Transit for signing operations (avoid exporting raw keys to apps).
- Secure Vault with RBAC: only designated service accounts can read specific paths.
- Never store secrets in plaintext in repo; use sealed-secrets or reference secrets via environment at runtime.
- Maintain a secrets inventory and expiration timeline.

CI/CD & GitOps
- Avoid storing real secrets in CI. Use short-lived tokens or OIDC to access Vault.
- For GitOps, store sealed-secrets (controller holds private key) rather than plaintext K8s secrets.
- Automation should post rotation audit metadata to a secure audit log.

Retention and key retirement
- Keep previous key for verification until all artifacts signed with it are re-signed or expired.
- Maintain `previous_keys` list in Vault metadata with `retire_after` timestamp.

Contacts & escalation
- On-call: ops-team@example.com
- Security/incident: security@example.com
- Vault admin: vault-admin@example.com

CI / automation variables (must be configured as repo secrets / cluster secrets)
- VAULT_ADDR, VAULT_ROLE_ID / VAULT_SECRET_ID or VAULT_TOKEN (prefer OIDC)
- KUBECONFIG or KUBE_SERVICEACCOUNT_TOKEN for updating k8s secrets
- ROTATION_SCHEDULE (cron)
docs/secret-management-runbook.md
