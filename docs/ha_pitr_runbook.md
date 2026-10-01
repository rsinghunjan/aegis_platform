 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
```markdown
  - If restore required: run scripts/pg_restore_walg.sh with appropriate timestamp
  - If restore fails: escalate to DBA and follow failover playbook

Security & secrets
------------------
- Use Vault/ExternalSecrets to inject S3/MinIO credentials into backup jobs; don’t store creds in repo.
- Verify bucket policies and encryption at rest (SSE), and IAM least-privilege for backup agent.

Restore verification checklist
------------------------------
- The restored DB accepts connections on configured port
- The expected tables exist (e.g., users, models, jobs)
- A set of smoke queries (select counts, key row existence) pass
- Optional app-level smoke test: run minimal API calls against restored DB instance if application can be pointed at it

Failure modes & mitigations
---------------------------
- WAL missing for requested timeframe:
  - Mitigation: check object-store for WAL segments; if missing, adjust RPO or increase WAL retention.
- Corrupted backup:
  - Keep multiple recent backups; validate base backups after creation (wal-g verify).
- Permission errors to object-store:
  - Use IAM roles and rotate keys via Vault; test access for the backup agent continuously.

Next steps / automated improvements
----------------------------------
- Add alerting to PagerDuty with runbook links.
- Add daily lightweight backup verification job (validate last N backups exist and wal-g verify).
- Schedule weekly/automated restore CI pipelines (we included one) and enforce passing for promotions to prod.
- If self-hosted: add Patroni Helm chart and automate failover recovery procedure.

Appendix: Quick commands (examples)
-----------------------------------
# Push base backup
export WALG_S3_PREFIX="s3://aegis-backups/postgres"
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
./scripts/pg_backup_walg.sh

# Restore latest base backup into /tmp/restore_pgdata and start postgres:
./scripts/pg_restore_walg.sh --pgdata /tmp/restore_pgdata --listen-port 55432

# Verify restore:
./scripts/verify_restore.sh --host localhost --port 55432 --user postgres

```
docs/ha_pitr_runbook.md
