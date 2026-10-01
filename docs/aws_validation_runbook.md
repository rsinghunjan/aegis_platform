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
```markdown
- Trigger build-and-sign workflow in GitHub UI or dispatch with workflow_dispatch; supply model_dir pointing to model_registry/example_multimodal/0.1
- The workflow will produce /tmp/model.tar.gz and model.tar.gz.sig and upload artifact or allow you to upload to S3
- If your workflow doesn’t upload to S3, upload manually:
  aws s3 cp /tmp/model.tar.gz s3://<bucket>/model-archives/example/0.1/model.tar.gz
  aws s3 cp /tmp/model.tar.gz.sig s3://<bucket>/model-archives/example/0.1/model.tar.gz.sig
- Run verifier workflow (or trigger a job that runs scripts/verify_model_signatures.py) while GitHub Actions uses id-token -> assume role to access the bucket
- Trigger helm-deploy-aws.yml (workflow_dispatch) with image_tag=<tag>
Verify:
- Verifier logs: model verified (scripts/verify_model_signatures.py prints "verified")
- helm: pods in aegis namespace are READY (kubectl -n aegis get pods)
- Orchestrator logs: fetch_and_verify_model called and succeeds

Step 9 — Canary + monitoring validation
- Ensure Argo Rollouts controller installed (kubectl apply -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml)
- Apply rollout & alert rules:
  kubectl apply -f k8s/argo-rollouts/example-rollout.yaml
  kubectl apply -f k8s/prometheus/alert-rules.yaml
- Generate traffic:
  python3 scripts/simulate_canary_load.py --url http://<ingress>/infer --qps 20 --duration 180 --error-rate 0.02
Observe:
- kubectl -n aegis get rollouts
- Prometheus alerts for AegisCanaryErrorRateHigh / LatencyHigh
- Rollout should rollback on alert (describe rollout)

Step 10 — Backup & restore drill
- Trigger manual backup job:
  kubectl -n aegis create job --from=cronjob/aegis-db-backup manual-backup-$(date +%s)
- Verify backup exists in S3
- Edit k8s/jobs/restore-db-job.yaml to point ARTIFACT_URI and run:
  kubectl apply -f k8s/jobs/restore-db-job.yaml
- Verify restored DB and run smoke tests

Next steps — production hardening
- Replace toy model training with real pipelines and data storage
- Tighten RDS networking/subnet groups & remove public access
- Add HPA/VPA, resource requests/limits, PDBs and liveness/readiness tweaks
- Implement cosign private key rotation and automated signing service (optional)
```
docs/aws_validation_runbook.md
