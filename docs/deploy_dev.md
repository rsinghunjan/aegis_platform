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
# Deploy to Dev — reproducible pipeline

This document describes how the GitHub Actions pipeline deploys a reproducible "dev" environment.

What the workflow does
1. On push to main or dev:
   - Runs tests (pytest)
   - Builds Docker image and pushes to GitHub Container Registry (ghcr.io)
2. Deploys Helm chart to your dev K8s cluster (helm upgrade --install)
3. Runs DB migration Job (k8s job that runs `alembic upgrade head` and `python scripts/seed_db.py`)
4. Waits for rollout, then runs smoke tests against the service.

Secrets required
- CR_PAT — personal access token used to push images to GHCR (or configure authentication differently)
- KUBECONFIG — base64-encoded kubeconfig for the cluster the workflow will deploy to

How to configure
1. Create a small dev Kubernetes cluster (kind / k3s / cloud provider).
2. Ensure helm is installed.
3. Add KUBECONFIG to the repository secrets:
   - Base64-encode your kubeconfig: `base64 -w0 ~/.kube/config`
   - In GitHub repo Settings → Secrets → Actions, add KUBECONFIG with this value.
4. Add CR_PAT secret with permission to push packages (or configure GHCR push via GITHUB_TOKEN if permitted).
5. Optionally update helm/aegis-api/values-dev.yaml to set DB connection (for dev the chart uses plaintext credentials).

Running locally (manual alternative)
- You can run the same steps locally:
  - Build image: `docker build -t ghcr.io/<org>/aegis-api:local .`
  - Push to registry (or load into your local cluster)
  - Helm install: `helm upgrade --install aegis-api helm/aegis-api -f helm/aegis-api/values-dev.yaml`
  - Run migration job: `kubectl apply -f k8s/migration-job.yaml` (replace image placeholder)
  - Run smoke tests: `scripts/deploy_smoke_test.sh http://localhost:8080`

Notes & best practices
- Use a dedicated dev namespace and a short-lived kubeconfig token with minimal privileges for CI.
- For production deployments, extend the workflow with:
  - Infrastructure provisioning (Terraform / Pulumi) that creates the cluster and managed DB
  - Blue/green or canary deployment steps with health & traffic-shift automation
  - Run alembic migrations in a controlled migration pipeline (with backups and prechecks)
- If you prefer Docker Compose for dev, reuse docker/docker-compose.celery.yml with appropriate env overrides.

If you want, next I can:
- Add a Terraform module to provision the dev cluster (EKS/GKE) and RDS/Postgres instance, and wire it into a separate `infra/` workflow.
- Add a deploy-step that runs smoke tests in parallel from multiple regions and collects test artifacts.
- Add an alternative workflow that deploys to a remote VM via SSH (if you prefer not to use k8s).
Which of those would you like me to implement next?
