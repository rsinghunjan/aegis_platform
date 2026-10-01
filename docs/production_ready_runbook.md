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
 84
# Production readiness runbook — finalize and validate Aegis
2. Ensure audit table exists (alembic migration applied). After a load attempt, record actor and verification result.
3. Run unit tests: pytest tests/test_orchestrator_verify_enforcement.py

Step F — Helm deploy to staging & acceptance
1. Use staging.env to set helm values (objectStore.bucket, database.url). Then:
   helm upgrade --install aegis ./helm -n aegis --create-namespace -f helm/values.<provider>.yaml --set image.tag=<tag>
2. Validate pods, services and readiness:
   kubectl -n aegis get pods
   kubectl -n aegis logs <orchestrator-pod>
3. Run smoke validations (export_savedmodel.py for saved models and end-to-end integration tests).

Step G — Canary test + validation
1. Ensure Argo Rollouts + Prometheus + Alertmanager are installed.
2. Deploy Rollout manifest (k8s/argo-rollouts/example-rollout.yaml) and alert rules (k8s/prometheus/alert-rules.yaml).
3. Trigger canary test:
   python3 scripts/simulate_canary_load.py --url <canary-url> --qps 20 --duration 120 --error-rate 0.05
4. Observe Prometheus alerts and ensure Argo Rollouts auto-rollback on alert.

Step H — DB backup & restore drill
1. Run CronJob backup once or manually:
   kubectl -n aegis create job --from=cronjob/aegis-db-backup aegis-db-backup-manual
2. Wait until backup uploaded to object store; note URI.
3. Run restore job replacing ARTIFACT_URI and restore into a clean staging DB. Run acceptance tests against restored DB.

Sign-off checklist
- pre-merge scanner returns zero findings
- verifier CI (OIDC role) successfully downloads staging artifact and verifies signature
- cosign pub key present in ExternalSecrets and used by verifier
- orchestrator refuses to load artifact with bad signature (test)
- helm deploy works without chart template changes using terraform outputs
- canary test triggers rollback on injected errors
- DB backup & restore validated and documented
- audit events exist in model_audit table for promotions/approvals

docs/production_ready_runbook.md
