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
```markdown
  - Fetches kube credentials for the created GKE cluster.
  - Calls scripts/gcp_wi_bind.sh to map the Kubernetes ServiceAccount to the GCP Service Account (Workload Identity).

- Azure:
  - Runs terraform in infra/azure (if present) to create AKS + ACR + GPU node pool (turnkey).
  - Fetches kube credentials via az aks get-credentials (if AZ_RG & AKS_NAME available).
  - Calls scripts/azure_wi_turnkey.sh to create AAD app, federated credential and bind a k8s ServiceAccount (Workload Identity for AKS).

- OCI:
  - Runs terraform in infra/oci (if present) to create OKE cluster + node pool (turnkey).
  - Calls scripts/oci_finalize_setup.sh to create OCI Object Storage bucket and show OCIR guidance.
  - Prints guidance to fetch kubeconfig and bind identities (OCI auth varies by tenancy).

Notes & safety
- The script attempts to be idempotent and will print guidance when a helper is not found.
- Workload Identity / federated credential creation may require Azure AD or GCP org permissions or manual steps depending on tenant policy; the helper scripts attempt REST calls where possible but may require tenant-admin approval.
- Review all created resources (GKE, AKS, OKE, storage buckets, Artifact Registry/ACR/OCIR) for naming, cost and policy compliance before running in production accounts.
- For production: prefer Workload Identity / IRSA / Managed Identity over JSON keys. If you create long-lived keys as a convenience, rotate and delete them immediately.

Troubleshooting
- If terraform command fails due to missing providers or credentials: run `terraform init` and ensure environment variables / auth are set for the target cloud.
- For Azure federated credential creation: the script uses `az rest` to call Microsoft Graph beta endpoints; if you lack permission, create the federated credential manually in the AAD app registration UI.
- For OCI: ensure the OCI CLI config is present (~/.oci/config) or use instance principals.

After running
- Validate kube access:
  kubectl get nodes -o wide
- Validate binding:
  - GCP: `gcloud iam service-accounts get-iam-policy <sa-email> --project <project>` should show the workload identity binding
  - Azure: confirm federated credential exists under AAD app and the k8s serviceaccount annotation was applied
  - OCI: follow OCI docs to map pod identity or use instance principals

If you want, I can:
- Add a GitHub Action wrapper to run this script from CI (with appropriate repo secrets).
- Extend the script to push trainer images to the cloud registries after provisioning.
- Add a safety prompt/confirm step for production projects.
```
```
