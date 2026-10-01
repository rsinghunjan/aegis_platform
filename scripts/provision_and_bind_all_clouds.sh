132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
#
    echo "Running scripts/azure_wi_turnkey.sh to create AAD app, federated credential and bind k8s SA ..."
    AZ_RG="${AZ_RG:-}" AKS_NAME="${AKS_NAME:-}" NAMESPACE="${K8S_NAMESPACE:-aegis}" K8S_SA="${K8S_SA:-aegis-trainer}" STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-}" ./scripts/azure_wi_turnkey.sh
  else
    echo "Warning: scripts/azure_wi_turnkey.sh missing; run Azure Workload Identity mapping manually per docs."
  fi

  echo "=== Azure: done ==="
}

run_oci() {
  echo "=== OCI: starting provisioning and identity guidance ==="

  req_cli oci || true
  req_cli terraform || true
  req_cli kubectl || true

  # Terraform apply for OCI if provided
  if [ -d "infra/oci" ] && ls infra/oci/*.tf >/dev/null 2>&1; then
    echo "Applying Terraform for OCI (infra/oci)..."
    (cd infra/oci && terraform init -input=false && terraform apply -auto-approve)
  else
    echo "No infra/oci terraform files found; ensure OKE is provisioned or provision externally."
  fi

  # Run OCI finalize helper to create bucket and show OCIR guidance
  if [ -x "./scripts/oci_finalize_setup.sh" ]; then
    if [ -z "${OCI_NAMESPACE:-}" ] || [ -z "${OCI_COMPARTMENT_ID:-}" ]; then
      echo "ERROR: For OCI finalize you must set OCI_NAMESPACE and OCI_COMPARTMENT_ID env vars" >&2
      return 2
    fi
    OCI_NAMESPACE="$OCI_NAMESPACE" OCI_COMPARTMENT_ID="$OCI_COMPARTMENT_ID" OCI_BUCKET="${OCI_BUCKET:-aegis-model-bucket}" ./scripts/oci_finalize_setup.sh
  else
    echo "Warning: scripts/oci_finalize_setup.sh missing; please create OCI Object Storage bucket and OCIR manually."
  fi

  # If terraform outputs provide cluster kubeconfig instructions, print guidance
  OKE_CLUSTER_ID="$(terraform -chdir=infra/oci output -raw oke_cluster_id 2>/dev/null || true)"
  if [ -n "${OKE_CLUSTER_ID:-}" ]; then
    echo "OKE cluster created: ${OKE_CLUSTER_ID}. Use OCI CLI or terraform outputs to fetch kubeconfig and configure kubectl."
  fi

  echo "=== OCI: done ==="
}

if [ $# -lt 1 ]; then
  echo "Usage: $0 <clouds...>   where cloud in: gcp azure oci  (order matters if you want sequential runs)"
  echo "Example: PROJECT=... GCS_BUCKET=... AZ_RG=... AKS_NAME=... OCI_NAMESPACE=... OCI_COMPARTMENT_ID=... $0 gcp azure oci"
  exit 1
fi

for cloud in "$@"; do
  case "$cloud" in
    gcp) run_gcp ;;
    azure) run_azure ;;
    oci) run_oci ;;
    *) echo "Unknown cloud: $cloud" >&2; exit 2 ;;
  esac
done

echo "Provisioning & binding sequence completed. Review output and any errors above."
scripts/provision_and_bind_all_clouds.sh
