
```markdown
Notes & recommendations (concise)

- Prefer managed DB (RDS / Cloud SQL) for production unless you need self‑managed Postgres.
- If self‑managed, use a battle‑tested operator (Patroni/bitnami-postgresql-ha/Zalando) instead of rolling your own StatefulSet.
- Use physical (pgbackrest/pg_basebackup) + WAL archive for true PITR. Logical dumps are easier but only provide point snapshots.
- Secure backups: S3 bucket policy, KMS encryption, IAM least privilege.
- Test restores frequently (weekly or after any backup/automation change).
- Keep Postgres versions aligned between backup/test runners to avoid restore incompatibilities.

Quick checklist before running a production restore test
- Do not run automated restore against production DB or production credentials.
- Use an isolated ephemeral environment (CI runner or staging project).
- Ensure AWS credentials used in CI have only list/get permissions for backup bucket (no write unless deliberately uploading test artifacts).
- Remove or rotate any credentials if accidentally logged or exposed.

