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
```markdown
# Terraform staging overlay (placeholder)

This directory contains provider-agnostic scaffolding for a staging overlay. Pick a provider
subdirectory (aws / gcp /oci /azure) and copy/modify the example files there, or place your
provider-specific Terraform implementation here.

Suggested workflow:
1. Choose a provider to target (AWS, GCP, OCI, Azure).
2. Copy this placeholder directory to infra/terraform/overlays/<provider>.
3. Implement provider-specific resources:
   - object store (S3/GCS/OCI Object Storage/Azure Blob)
   - managed Postgres/Cloud SQL/RDS/Autonomous DB
   - Kubernetes cluster or cluster access outputs (kubeconfig/cluster_name)
   - IAM role for GitHub OIDC (optional)
4. Ensure outputs include:
   - bucket_name
   - database_url
   - (optional) eks_cluster_name / gke_cluster_name / oci_cluster_id
5. Update helm/values.<provider>.yaml with the outputs and verify helm installs without modifying chart templates.

CI integration:
- I provide workflows that assume terraform outputs are exported to env or a tf_outputs.json file.
- For security, use GitHub OIDC to assume a role for CI rather than long-lived secrets.
```
infra/terraform/overlays/placeholder/README.md
