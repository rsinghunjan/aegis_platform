````markdown
# Canary Rollouts with Flagger + Prometheus + NGINX (Aegis)

Overview
--------
This guide shows how to enable automated canary analysis for model versions registered in the Model Registry.

Prerequisites
- Kubernetes cluster (v1.22+)
- NGINX Ingress Controller installed (or supported ingress provider)
- Prometheus installed and scraping your API (aegis_api job label or target)
- Flagger installed and configured to use the nginx provider and Prometheus
- The API image in a registry accessible by the cluster (the controller uses that image to spawn canary deployments)

How it works
1. Operator/admin promotes a model version to "canary" via the registry API:
   POST /v1/registry/{model}/{version}/promote  (target_stage=canary)
2. The auto-canary controller notices the model_version metadata stage == "canary".
3. The controller creates a k8s Deployment for the canary (env MODEL_NAME / MODEL_VERSION).
4. The controller creates a Flagger Canary CR that runs analysis based on:
   - Request success rate (Prometheus query)
   - P95 prediction latency (Prometheus histogram)
5. Flagger starts shifting traffic to the canary incrementally. If SLOs hold -> canary is promoted; otherwise rolled back.

Notes
- The Deployment created uses the same api image but instructs the app to load the model version (your app must support loading model versions at startup by MODEL_NAME/MODEL_VERSION env or similar).
- For GPU-backed models, edit the deployment template to request GPUs and schedule onto GPU node pool (values are documented in the template).
- Tune Flagger analysis (interval, metrics, thresholds) per-model class (large vs small models have different latencies).

Security
- The controller needs permission to read model_versions from the DB and create k8s resources.
- Run the controller as a Kubernetes Deployment with a minimal RBAC rolebound service account.

Next steps
- Add model-specific analysis queries / labels (e.g., quality metrics).
- Connect a canary-notification channel to Slack or PagerDuty for failed canaries.
- For multi-tenant deployments, ensure tenant-scoped promotions create namespace-scoped canaries and enforce RBAC.k
