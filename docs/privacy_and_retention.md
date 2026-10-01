# Privacy, Data Retention and Federated Hooks — Aegis

This document describes the added components for privacy-preserving behavior and data governance.

What was added
- api/privacy.py
  - register_retention_policy / enforce_retention_policies
  - anonymize_user
  - tenant_isolation_check
  - federated_aggregation_protect (minimum participants + optional DP noise)
- api/audit.py
  - record_audit + get_audit_events; stores audit events in audit_logs table
- DB extensions:
  - AuditLog and DataRetentionPolicy models (alembic migration provided)
- Tests:
  - tests/test_privacy.py covering retention dry-run, anonymization, federated protection and audit logs

How to use
1. Register policies (example):
   from api.privacy import register_retention_policy
   register_retention_policy("safety_90", table="safety_events", retention_days=90, action="delete")

2. Dry-run enforcement:
   from api.privacy import enforce_retention_policies
   enforce_retention_policies(dry_run=True)

3. Schedule enforcement:
   - Run `enforce_retention_policies(dry_run=False)` from a secure scheduled runner (Celery beat / k8s CronJob).
   - Ensure runner has least-privilege DB credentials.

4. Anonymize a user:
   from api.privacy import anonymize_user
   anonymize_user(user_id, actor="ops@example.com")

5. Federated aggregation protection:
   - Call `federated_aggregation_protect(aggregated_value, participant_count, min_participants=5, dp_epsilon=0.5)`
   - Set dp_epsilon according to privacy budget policy; lower epsilon -> more noise.

Audit & compliance
- All retention actions, anonymizations and federated protections write AuditLog entries.
- Export audit_logs for compliance systems or SIEMs (Prometheus / Grafana can display counts).

Operational recommendations
- Run retention enforcement during low traffic windows.
- Keep previous key material for verifying artifacts during retention/rotation windows.
- For stricter privacy, anonymize before deletion (double-action).
- Review DataRetentionPolicy table and ensure policies are approved by legal/privacy teams.

Caveats
- The anonymize implementation is a best-effort generic approach. For complex schemas, implement table-specific anonymizers to redact or remove PII safely.
- Avoid running destructive deletion without backup snapshots; always test with dry_run first.

Next steps (optional)
- Add a k8s CronJob + GitHub Action to schedule retention enforcement with observability (metrics + alerts on deleted rows).
- Implement tenant-aware retention runs (safely multi-tenant aware).
- Integrate with a data catalog to enumerate PII-containing tables and propose default retention policies.
