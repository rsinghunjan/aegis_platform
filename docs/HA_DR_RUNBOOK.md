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
 83
```markdown
# HA / DR Runbook — Postgres multi‑AZ & automated restore test

Purpose
- Describe recommended production architecture for Postgres (multi‑AZ / highly available).
- Provide operational runbook for backups, restores, and a CI smoke test that verifies backups can be restored end‑to‑end.
- Provide scripts and automation to run periodic restore tests to validate backup integrity and recovery procedures.

Goals / Targets
- RTO: target recovery time objective (example) — 15 minutes for small test restores, document expected RTO for production.
- RPO: target recovery point objective — e.g., continuous WAL archiving => near zero (minutes).
- Backup retention: e.g., daily full backup for 14 days, weekly archive for 12 weeks, monthly for 12 months (tune to policy).
- Security: backups encrypted at rest (S3 SSE), access via least‑privilege IAM role, no private keys or credentials checked into repo.

High level options (pick one)
- Managed (recommended for many environments)
  - RDS / Cloud SQL / Azure DB: built-in HA, automated backups, cross‑AZ replicas; use snapshots + PITR.
- Self‑managed in Kubernetes (example artifacts supplied)
  - Patroni / Postgres Operator (recommended): provides leader election, replication, automated failover.
  - Example Helm chart: bitnami/postgresql‑ha or Zalando Postgres Operator.
  - Must run with zone/zoneAffinity, PodTopologySpread and StatefulSet across AZs.

Backup strategies
- Logical backups
  - pg_dump / pg_dumpall for periodic point-in-time logical snapshots. Good for logical migrations, small DBs.
- Physical backups + WAL archiving (recommended)
  - Base backup (pg_basebackup or pgbackrest) + WAL archiving to object store (S3/GCS).
  - Enables point‑in‑time recovery (PITR).
- Hybrid: periodic logical dumps + continuous WAL archive.

Storage & object store
- Use a dedicated bucket for backups, with versioning enabled and server‑side encryption (SSE).
- Bucket lifecycle policy to move older backups to colder storage or delete per retention policy.

Operational runbook — manual restore (physical / logical)
1. Identify backup
   - List backups in bucket (by date / tag).
2. For logical restore (pg_dump)
   - Download dump: aws s3 cp s3://BUCKET/path/to/dump.sql /tmp/dump.sql
   - Restore to new instance: psql -h <host> -U <user> -d postgres -f /tmp/dump.sql
3. For physical + WAL (PITR)
   - Create a new cluster from base backup:
     - prepare data directory with base backup
     - configure recovery.conf or postgresql.conf & recovery settings to point to WAL archive
     - start server in recovery mode and wait until recovery_target_time achieved
4. Verify
   - Run smoke queries (row counts, checksum queries).
   - Check application connectivity and correctness.
5. Promote and update DNS / service:
   - Point application to restored instance only after verification and necessary promotions.

Automated restore test (overview)
- Purpose: validate that backups are usable and that restore procedures work end‑to‑end.
- Approach:
  - A CI job (GitHub Actions) downloads the latest backup from S3, starts a fresh Postgres container, restores the backup, and runs validation queries.
  - Job runs weekly (or on every backup) and fails if restore or validation fails.
  - The CI job uses ephemeral credentials (secrets) and does not modify production systems.

Files included in repo
- helm/values-postgresql-ha.yaml — example Helm values to run a zone‑aware, multi‑AZ Postgres HA cluster (bitnami/postgresql‑ha).
- scripts/pg_logical_backup.sh — simple logical backup script (pg_dump) that uploads to S3.
- scripts/pg_logical_restore.sh — restores a logical dump into a target Postgres instance.
- scripts/pg_restore_test.sh — CI smoke test to download latest backup, restore locally into ephemeral Postgres Docker container and validate data.
- .github/workflows/pg_restore_test.yml — GitHub Actions workflow to run the smoke test on schedule or on demand.

Security & permissions
- Use an IAM role with least privilege for backup/restore S3 operations.
- Store AWS credentials (or OIDC role) as GitHub repo secrets for CI.
- Use S3 SSE (KMS) or bucket encryption to protect backups at rest.
- Ensure backups are not publicly readable and that lifecycle/retention policies are applied.

Monitoring & alerts
- Alert when:
  - scheduled backup fails,
  - object store upload fails,
  - daily restore test fails,
  - disk usage approaches thresholds for backup volume.
- Track backup job metrics (duration, size, upload latency).

Cleanup & rotation
- Implement lifecycle policies in S3 for retention policy enforcement.


docs/HA_DR_RUNBOOK.md
