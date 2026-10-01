# Backup Retention & Object-store Lifecycle

S3 lifecycle (example)
- Use S3 lifecycle rules to transition older backups to GLACIER or delete them automatically.
- Example policy (JSON) to keep daily backups for 90 days and then archive:

```json
{
  "Rules": [
    {
      "ID": "db-backups-retention",
      "Prefix": "db/",
      "Status": "Enabled",
      "Transitions": [
        { "Days": 30, "StorageClass": "STANDARD_IA" },
        { "Days": 90, "StorageClass": "GLACIER" }
      ],
      "Expiration": { "Days": 365 }
    },
    {
      "ID": "models-retention",
      "Prefix": "models/",
      "Status": "Enabled",
      "Expiration": { "Days": 3650 }
    }
  ]
}
```

Recommendations
- DB backups: daily pg_dump + WAL shipping for PITR
- Model artifacts: snapshot on model publication and keep for at least 1 year (or as per compliance)
- Use object-store versioning to protect against accidental deletes
- Tag backups with metadata (created_by, commit_sha, model_registry_version)

Verification policy
- Periodically (weekly/monthly):
  - run `scripts/verify_backup.sh` on a sample backup
  - run a dry-restore into a sandbox DB and run smoke tests

Audit & alerts
- Emit Prometheus metrics on backup success/failure and alert if failures > 1
- Send backup run summaries to an ops Slack/email via webhook
