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
```markdown
# On‑call Runbook & Game-Day Checklist (Aegis)

This document describes alert routing, owners, and the game-day drills you should run to exercise on-call readiness.

Owners & escalation
- Critical alerts (severity=critical): PagerDuty primary -> SRE on-call
- Warning alerts (severity=warning): Slack #aegis-alerts -> Ops triage
- Safety-specific alerts: Slack #aegis-safety and notify Safety on-call

Top alerts and owners (map in team directory)
- AegisSafetySpike — Safety team primary, SRE secondary
- AegisHighInferenceLatencyP95 — SRE primary, Model owner secondary
- AegisModelNotLoaded — SRE primary, Release owner secondary
- AegisAuditWriteFailures — SRE + Compliance

Immediate playbook for each critical alert
- AegisSafetySpike:
  1. Acknowledge alert in PagerDuty.
  2. Open Safety dashboard (Grafana) and narrow by reason label.
  3. Pull recent audit events: query OpenSearch for recent flagged items (or run logs/s3 export).
  4. If false positives: rollback latest policy/model change. If true positives: scale review queue and notify legal if PII.
  5. Document timeline in incident ticket.

- AegisHighInferenceLatencyP95:
  1. Acknowledge alert.
  2. Check /health endpoint for model loaded and batch queue size.
  3. Inspect pod CPU/GPU metrics and queue depth; if queue depth high, scale HPA or increase replicas.
  4. Consider reducing BATCH_MAX_LATENCY_MS temporarily to reduce tail latency.
  5. Run load_test_inference.py in staging to reproduce and tune.

- AegisModelNotLoaded:
  1. Acknowledge alert.
  2. Check pod logs for registry errors, signature verification errors (AEGIS_MODEL_SIGNATURE_ERRORS).
  3. If signature mismatch: revert to previous signed model and restart pods.
  4. If registry S3 access denied: check IAM role and KMS permissions.

- AegisAuditWriteFailures:
  1. Acknowledge alert.
  2. Check audit writer logs and AEGIS_AUDIT_WRITE_ERRORS metric.
  3. Verify S3 bucket reachability and KMS decrypt permissions.
  4. If S3 unavailable, enable fallback (local buffered store) and schedule migration.

Game-day checklist (run in staging)
1. Pre-game:
   - Ensure Alertmanager config and receivers are configured (Slack webhook, PagerDuty key).
   - Ensure Prometheus alert_rules.yml is loaded.
2. Exercise 1 — Test alert routing:
   - Run ./scripts/trigger_test_alert.sh to send a test alert. Verify it reaches Slack and PagerDuty.
3. Exercise 2 — Safety spike:
   - Ingest synthetic flagged events (increase aegis_safety_flags_total or push audit events).
   - Verify alert fires, team receives notifications, and the Safety runbook is followed.
4. Exercise 3 — Model load fail:
   - Simulate registry failure (change model pubkey to invalid) or set MODEL_SIGNING_REQUIRED=true with no key.
   - Verify AegisModelNotLoaded fires; follow rollback steps.
5. Exercise 4 — Audit backend fail:
   - Temporarily misconfigure AUDIT_S3_BUCKET or revoke KMS decrypt role in staging.
   - Verify alert and run fallback/resume plan.
6. Post-game:
   - Collect timings (acknowledge, mitigation, restore).
   - Update runbooks and owners based on lessons.

Runbook maintenance
- Owners update this file after each drill.
- Each runbook step should include exact kubectl/curl/OpenSearch commands to run (examples in appendix).
```
docs/runbook_oncall.md
