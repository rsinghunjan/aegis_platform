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
# Canary rollout automation (Aegis)

Overview
--------
This system wires model registry promotions to automated k8s canary rollouts using Flagger.

Quickstart
1. Install prerequisites (Prometheus, NGINX Ingress, Flagger). See k8s/flagger/install-flagger.sh
2. Deploy the auto-canary controller (use the api/auto_canary_controller.py script as a Deployment).
   - Give it DB credentials (via Kubernetes Secret) and KUBECONFIG or run in-cluster.
   - Grant RBAC permissions to create Deployments and Flaggers' Canary CRs.
3. Promote a model version to canary:
   curl -X POST /v1/registry/<model>/<version>/promote -F "target_stage=canary"
4. The controller will:
   - create a canary Deployment (aegis-canary-...)
   - create a Flagger Canary CR that runs analysis and shifts traffic incrementally
5. Monitor:
   - Flagger logs and Canary status (kubectl get canary -n <ns>)
   - Prometheus metrics (prediction latency and request success rate)
   - If analysis succeeds, Flagger will finalize the rollout and you can then promote to prod.

Tuning
- Adjust Flagger metric queries to include the right job/labels for your Prometheus scrape.
- Use model-specific thresholds for latency and success rates; Flagger supports per-canary metric overrides.
- For GPU models, set nodeSelector/tolerations in the deployment template created by the controller.

Security & RBAC
- Controller requires:
  - create/update/delete on deployments in the namespace
  - get/list/watch/create for Flagger Canary CRD (group flagger.app)
- Run controller with a minimal service account bound to those permissions.

Rollback
- If analysis fails, Flagger will automatically rollback and restore the previous service routing.
- The Flagger Canary status shows rollout, analysis, and rollback history.

Notes
- This example uses a simple pattern (spawn fresh deployment that reads the model at startup). You can instead use sidecar patterns or CDS-based image updates depending on your infra.
docs/canary_rollout.md
