# Operational HA & Resiliency Runbook — Aegis

Scope
-----
This runbook covers:
- Highly available Postgres (managed RDS/Aurora preferred, Patroni for self-hosted)
- WAL/PITR and logical backups
- Model artifact snapshotting and object-store lifecycle
- Redis HA (managed Redis/ElastiCache preferred; Redis Cluster / Sentinel for self-hosted)
- Celery Beat leader election, worker sizing, autoscaling (CPU/queue-length/GPU)
- Disaster Recovery (DR) testing & verification
- Prometheus alerts for HA signals

Design principles / recommendations
- Prefer managed services (RDS/Aurora, Elasticache/MemoryDB, Cloud Storage) for production to avoid running HA DB yourself.
- If self-hosting Postgres use Patroni + etcd/consul and streaming replication + WAL shipping with wal-g/wal-e for PITR.
- For Redis prefer a managed cluster; if self-hosted use Redis Cluster for scaling or Redis Sentinel for simple HA.
- Use WAL archiving (wal-g) for continuous PITR; take periodic base backups (pg_basebackup) and test restores.
- Use KEDA or Prometheus Adapter for Celery autoscaling based on queue backlog; use node pools + taints for GPU workers.
- Celery Beat should run with leader election (single active beat) — run as a Deployment with leader election or use a dedicated scheduler like "RedBeat" or "APScheduler" with Redis leader lock.
- Test DR plans with scheduled drills; record times to RTO/RPO.

Quick decision matrix
- Production: managed RDS + managed Redis + object store snapshots + managed Prometheus (or self-hosted with HA)
- Staging/Dev: single-instance Postgres + Redis (but run nightly integration/pit restores to detect regressions)
- Self-hosted production: Patroni Postgres + etcd + wal-g + Redis Cluster + Kubernetes + backups to S3 with lifecycle rules

Operational run steps (top-level)
1. Provision HA components (prefer managed):
   - RDS/Aurora with automated backups (daily), enable continuous WAL/PITR (retain for N days).
   - Redis (Elasticache / MemoryDB or a Redis cluster)
   - S3/GCS bucket with versioning and lifecycle policy.
2. Configure DB connection for replicas and read-only endpoints in your app or query routing if required.
3. Configure WAL archiving (wal-g) to S3 for self-hosted Postgres.
4. Add CronJob (or pipeline) to snapshot model registry and upload to object-store.
5. Configure Celery:
   - Run a small number of always-on workers for interactive traffic.
   - Run GPU-capable workers in a separate node pool. Use taints/tolerations and nodeSelectors to schedule them.
   - Use KEDA ScaledObjects to automatically scale workers based on queue length.
   - Run Celery Beat as highly-available with leader lock (e.g., RedBeat or using Kubernetes leader election pattern).
6. Monitoring:
   - Alert on Postgres replication lag, WAL archival failures, Redis replication health, disk pressure, Celery queue backlog, and worker OOMs.
   - Add runbook links in alert annotations for ops.
7. DR test:
   - Periodic restore test (weekly dry-run): restore a backup into staging and run smoke tests and data validation.
   - Chaos test: drain a Postgres primary and verify failover.
8. Practice rotation of backup keys/secrets (Vault OIDC flows).

What to test (examples)
- Restore a database snapshot to a temporary instance and run smoke tests.
- Restore model artifacts from snapshots and verify model registry integrity.
- Simulate Redis master failure and validate HA failover.
- Simulate worker node loss (GPU node) and ensure jobs are rescheduled or retried.

References
- WAL base backups & WAL-G: https://github.com/wal-g/wal-g
- Patroni: https://patroni.readthedocs.io
- KEDA: https://keda.sh
- Elasticache / MemoryDB docs (cloud provider).
