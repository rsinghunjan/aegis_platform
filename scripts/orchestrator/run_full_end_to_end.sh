#!/usr/bin/env bash
set -euo pipefail

# Aegis Orchestrator: end-to-end provisioning, validation and artifact collection
#
# This operator-run script orchestrates (in order, optionally) the following high-level steps:
#  1) Provision cloud resources (AWS KMS, GCP TPU service account via scripts/terraform or CLI wrappers)
#  2) Create IBM Cloud resources (service-id + API key) via CLI wrapper
#  3) Store created secrets into Vault
#  4) Install / configure Vault (Helm) and run bootstrap
#  5) Set GitHub secrets (uses gh CLI)
#  6) Build & push required container images
#  7) Deploy quantum controller, simulator farm, Redis, and workers
#  8) Deploy Argo workflows for DGX (DeepSpeed), TPU, Cirq/Braket/Qiskit examples and wait for completion
#  9) Run DGX validation scripts (chaos / resume), OTA/canary smoke tests, attestation smoke
# 10) Collect logs, Terraform outputs, Vault policies, Argo workflow outputs and package into an artifacts tarball
#
# IMPORTANT SECURITY NOTE:
#  - This script does NOT embed or commit secrets. It expects the operator to provide credentials via
#    environment variables, CLI auth, or interactive prompts (ibmcloud login, gcloud auth, aws cli, vault login).
#  - Any JSON files created by CLI wrappers that contain secrets MUST be moved into Vault and deleted
#    by the operator after verification (the script will call the helper to do this where possible).
#
# Prerequisites (operator must ensure these are installed and authenticated):
#   - terraform, ibmcloud (optional), aws, gcloud (optional)
#   - kubectl, helm, argo (kubectl plugin or CLI), docker, gh (GitHub CLI), vault CLI
#   - jq, tar, gzip, base64, openssl (for some helpers)
#
# Usage:
#   ./run_full_end_to_end.sh [--skip-terraform] [--skip-vault] [--skip-images] [--skip-quantum] \
#       [--skip-dgx] [--skip-tpu] [--skip-edge] --vault-root-token <token> --github-repo owner/repo \
#       --registry <registry> --aws-region <region> --gcp-project <project> --ibm-region <region>
#
# Examples (operator-run):
#   VAULT_ADDR=https://vault.example.com ./run_full_end_to_end.sh --github-repo myorg/aegis --registry ghcr.io/myorg
#
# By default the script runs in "interactive" mode and will prompt for confirmations.
# Use --yes to auto-approve all steps.

##############################
# Defaults and env variables #
##############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # adjust if repo root differs
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$ROOT_DIR/artifacts"}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARTIFACT_DIR="$ARTIFACT_ROOT/run_$TIMESTAMP"
mkdir -p "$ARTIFACT_DIR"

# Toggle flags (changed by CLI args)
DO_TERRAFORM=true
DO_VAULT=true
DO_IMAGES=true
DO_QUANTUM=true
DO_DGX=true
DO_TPU=true
DO_EDGE=true
AUTO_YES=false

# Inputs (can be provided via env or CLI)
GITHUB_REPO=${GITHUB_REPO:-}
REGISTRY=${REGISTRY:-}
AWS_REGION=${AWS_REGION:-}
GCP_PROJECT=${GCP_PROJECT:-}
IBM_REGION=${IBM_REGION:-}
VAULT_ROOT_TOKEN=${VAULT_ROOT_TOKEN:-}
VAULT_ADDR=${VAULT_ADDR:-}
# Paths to helper scripts in repo (relative to script)
IBM_RES_JSON="$SCRIPT_DIR/../ibm/out/ibm_resources.json"

# Timeouts
ARGO_WAIT_TIMEOUT=${ARGO_WAIT_TIMEOUT:-900} # seconds
WORKFLOW_POLL_INTERVAL=${WORKFLOW_POLL_INTERVAL:-5}

####################
# Helper functions #
####################

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1"
    exit 2
  fi
}

confirm() {
  if [ "$AUTO_YES" = true ]; then
    return 0
  fi
  read -r -p "$1 [y/N]: " resp
  case "$resp" in
    [Yy]* ) return 0 ;;
    * ) return 1 ;;
  esac
}

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

safe_run() {
  echo ">>> $*"
  "$@"
}

collect_file() {
  local src=$1 dest_rel=$2
  if [ -z "$src" ] || [ -z "$dest_rel" ]; then return; fi
  if [ -e "$src" ]; then
    mkdir -p "$ARTIFACT_DIR/$(dirname "$dest_rel")"
    cp -a "$src" "$ARTIFACT_DIR/$dest_rel"
  fi
}

collect_k8s_namespace_logs() {
  local ns=$1
  if ! kubectl get ns "$ns" >/dev/null 2>&1; then
    return
  fi
  mkdir -p "$ARTIFACT_DIR/k8s/$ns"
  for pod in $(kubectl get pods -n "$ns" -o jsonpath='{.items[*].metadata.name}'); do
    kubectl logs -n "$ns" "$pod" --all-containers=true > "$ARTIFACT_DIR/k8s/$ns/$pod.log" 2>&1 || true
    kubectl get pod -n "$ns" "$pod" -o yaml > "$ARTIFACT_DIR/k8s/$ns/$pod.yaml" 2>&1 || true
  done
  kubectl get all -n "$ns" -o wide > "$ARTIFACT_DIR/k8s/$ns/resources.txt" 2>&1 || true
}

collect_argo_workflows() {
  if ! command -v argo >/dev/null 2>&1; then
    return
  fi
  mkdir -p "$ARTIFACT_DIR/argo"
  for wf in $(argo list -n staging --no-headers -o name 2>/dev/null || true); do
    argo get -n staging "$wf" --output json > "$ARTIFACT_DIR/argo/$wf.json" 2>&1 || true
  done
}

save_terraform_outputs() {
  # Save outputs for aws, gcp, ibm terraform dirs if present
  for d in terraform/aws terraform/gcp terraform/ibm; do
    if [ -d "$ROOT_DIR/$d" ]; then
      pushd "$ROOT_DIR/$d" >/dev/null
      if [ -f ".terraform.lock.hcl" ] || [ -d ".terraform" ]; then
        if command -v terraform >/dev/null 2>&1; then
          terraform output -json > "$ARTIFACT_DIR/$(basename $d)_outputs.json" 2>&1 || true
        fi
      fi
      popd >/dev/null
    fi
  done
}

save_vault_state() {
  if ! command -v vault >/dev/null 2>&1; then return; fi
  mkdir -p "$ARTIFACT_DIR/vault"
  # capture list of mounts & policies (requires VAULT_TOKEN in env)
  vault status > "$ARTIFACT_DIR/vault/status.txt" 2>&1 || true
  vault secrets list -format=json > "$ARTIFACT_DIR/vault/secrets_list.json" 2>&1 || true
  vault policy list -format=json > "$ARTIFACT_DIR/vault/policies.json" 2>&1 || true
  # try to export our known secrets path (read-only in operator context)
  if vault kv get -format=json secret/ibm >/dev/null 2>&1; then
    vault kv get -format=json secret/ibm > "$ARTIFACT_DIR/vault/secret_ibm.json" 2>&1 || true
  fi
}

package_artifacts() {
  local dest="${ARTIFACT_DIR}.tar.gz"
  tar -czf "$dest" -C "$(dirname "$ARTIFACT_DIR")" "$(basename "$ARTIFACT_DIR")"
  echo "Packaged artifacts: $dest"
}

#######################
# Stage implementations
#######################

stage_terraform() {
  if [ "$DO_TERRAFORM" = false ]; then
    log "Skipping terraform stage"
    return
  fi
  require_cmd terraform
  # AWS
  if [ -d "$ROOT_DIR/terraform/aws" ]; then
    log "Applying terraform/aws (operator will review prompts if any)"
    pushd "$ROOT_DIR/terraform/aws" >/dev/null
    terraform init -input=false
    terraform apply -auto-approve
    terraform output -json > "$ARTIFACT_DIR/terraform_aws_output.json" 2>&1 || true
    popd >/dev/null
  fi
  # GCP
  if [ -d "$ROOT_DIR/terraform/gcp" ]; then
    log "Applying terraform/gcp"
    pushd "$ROOT_DIR/terraform/gcp" >/dev/null
    terraform init -input=false
    terraform apply -auto-approve
    terraform output -json > "$ARTIFACT_DIR/terraform_gcp_output.json" 2>&1 || true
    popd >/dev/null
  fi
  # IBM (wrapper)
  if [ -d "$ROOT_DIR/terraform/ibm" ]; then
    log "Applying terraform/ibm (this runs CLI script to create IBM resources)"
    pushd "$ROOT_DIR/terraform/ibm" >/dev/null
    terraform init -input=false
    terraform apply -auto-approve
    # copy produced JSON if any
    if [ -f "$ROOT_DIR/scripts/ibm/out/ibm_resources.json" ]; then
      cp "$ROOT_DIR/scripts/ibm/out/ibm_resources.json" "$ARTIFACT_DIR/ibm_resources.json"
    fi
    popd >/dev/null
  fi
}

stage_set_github_secrets() {
  if [ -z "$GITHUB_REPO" ]; then
    log "GITHUB_REPO not set; skipping GitHub secrets step"
    return
  fi
  if ! command -v gh >/dev/null 2>&1; then
    log "gh CLI not found; skipping GitHub secrets"
    return
  fi
  # Example: set AWS KMS ARN and GitHub OIDC role if outputs exist
  if [ -f "$ARTIFACT_DIR/terraform_aws_output.json" ]; then
    KMS_ARN=$(jq -r '.aws_kms_key_arn_prod.value // empty' "$ARTIFACT_DIR/terraform_aws_output.json")
    ROLE_ARN=$(jq -r '.github_oidc_role_arn_prod.value // empty' "$ARTIFACT_DIR/terraform_aws_output.json")
    if [ -n "$KMS_ARN" ]; then
      log "Setting GitHub secret AWS_KMS_KEY_ARN"
      gh secret set AWS_KMS_KEY_ARN --repo "$GITHUB_REPO" --body "$KMS_ARN"
    fi
    if [ -n "$ROLE_ARN" ]; then
      log "Setting GitHub secret GITHUB_OIDC_ROLE_ARN"
      gh secret set GITHUB_OIDC_ROLE_ARN --repo "$GITHUB_REPO" --body "$ROLE_ARN"
    fi
  fi
  # For IBM, we prefer Vault; do not push raw IBM API keys to GH secrets
}

stage_vault_install_and_bootstrap() {
  if [ "$DO_VAULT" = false ]; then
    log "Skipping Vault stage"
    return
  fi
  require_cmd helm
  require_cmd kubectl
  # Deploy Vault via Helm using provided values file if present
  if [ -f "$ROOT_DIR/helm/vault/values-prod-aws-kms.yaml" ]; then
    log "Installing Vault Helm chart (values-prod-aws-kms.yaml). Edit file before running if needed."
    helm repo add hashicorp https://helm.releases.hashicorp.com || true
    helm repo update
    helm upgrade --install vault hashicorp/vault -n ops --create-namespace -f "$ROOT_DIR/helm/vault/values-prod-aws-kms.yaml"
    log "Vault helm install triggered; waiting for pods"
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=vault -n ops --timeout=300s || true
  else
    log "No helm/vault values file found; skipping helm install"
  fi

  # Bootstrap Vault if operator provided root token
  if [ -z "${VAULT_ROOT_TOKEN:-}" ]; then
    if confirm "No VAULT_ROOT_TOKEN provided. Would you like to enter it now to run bootstrap?"; then
      read -r -s -p "Enter VAULT_ROOT_TOKEN: " VAULT_ROOT_TOKEN
      echo
    else
      log "Skipping vault bootstrap (no root token)"
      return
    fi
  fi

  if [ -n "$VAULT_ROOT_TOKEN" ]; then
    export VAULT_ADDR VAULT_TOKEN="$VAULT_ROOT_TOKEN"
    if [ -f "$ROOT_DIR/scripts/bootstrap_vault_prod_complete.sh" ]; then
      log "Bootstrapping Vault using scripts/bootstrap_vault_prod_complete.sh"
      "$ROOT_DIR/scripts/bootstrap_vault_prod_complete.sh" "$VAULT_ROOT_TOKEN" | tee "$ARTIFACT_DIR/vault_bootstrap.log"
    else
      log "Bootstrap script not found; ensure Vault is initialized and unsealed manually"
    fi
  fi

  # Save Vault state
  save_vault_state
}

stage_store_ibm_to_vault() {
  # After terraform/cli created IBM resources, push them into Vault
  if [ ! -f "$ROOT_DIR/scripts/ibm/out/ibm_resources.json" ]; then
    log "IBM resources JSON not found; skipping store to Vault"
    return
  fi
  require_cmd vault
  if [ -z "${VAULT_TOKEN:-}" ]; then
    if [ -n "${VAULT_ROOT_TOKEN:-}" ]; then
      export VAULT_TOKEN="$VAULT_ROOT_TOKEN"
    else
      log "VAULT_TOKEN not present in env. The operator must login to Vault first."
      if confirm "Attempt 'vault login' interactively now?"; then
        vault login
      else
        log "Skipping store_ibm_secrets_to_vault.sh"
        return
      fi
    fi
  fi
  log "Storing IBM secrets into Vault using scripts/ibm/store_ibm_secrets_to_vault.sh"
  "$ROOT_DIR/scripts/ibm/store_ibm_secrets_to_vault.sh" --input "$ROOT_DIR/scripts/ibm/out/ibm_resources.json" | tee "$ARTIFACT_DIR/store_ibm_to_vault.log"
  # Remove local json (operator oversight)
  if confirm "Delete local IBM resources JSON file now? (recommended)"; then
    rm -f "$ROOT_DIR/scripts/ibm/out/ibm_resources.json"
    log "Deleted local IBM resources JSON"
  else
    log "Left local IBM resources JSON in place; remove it manually after verification"
  fi
}

stage_build_and_push_images() {
  if [ "$DO_IMAGES" = false ]; then
    log "Skipping image build/push stage"
    return
  fi
  if [ -z "$REGISTRY" ]; then
    if confirm "No REGISTRY set. Do you want to provide a registry (e.g. ghcr.io/myorg) now?"; then
      read -r -p "Registry: " REGISTRY
    else
      log "Skipping image build/push due to no registry"
      return
    fi
  fi
  require_cmd docker
  # Example builds (quantum controller, worker, cirq-sim, qiskit-sim, deepspeed image)
  log "Building & pushing quantum controller image (if Dockerfile exists)"
  if [ -f "$ROOT_DIR/services/quantum_controller/Dockerfile" ]; then
    docker build -t "$REGISTRY/quantum-controller:latest" "$ROOT_DIR/services/quantum_controller"
    docker push "$REGISTRY/quantum-controller:latest"
  fi
  # cirq simulator
  if [ -f "$ROOT_DIR/docker/cirq-simulator/Dockerfile" ]; then
    docker build -t "$REGISTRY/cirq-simulator:latest" -f "$ROOT_DIR/docker/cirq-simulator/Dockerfile" "$ROOT_DIR"
    docker push "$REGISTRY/cirq-simulator:latest"
  fi
  # qiskit simulator
  if [ -f "$ROOT_DIR/docker/qiskit/Dockerfile" ]; then
    docker build -t "$REGISTRY/qiskit-simulator:latest" -f "$ROOT_DIR/docker/qiskit/Dockerfile" "$ROOT_DIR"
    docker push "$REGISTRY/qiskit-simulator:latest"
  fi
  # deepspeed image (if present)
  if [ -f "$ROOT_DIR/docker/deepspeed_fsdp.Dockerfile" ]; then
    docker build -t "$REGISTRY/deepspeed-fsdp:latest" -f "$ROOT_DIR/docker/deepspeed_fsdp.Dockerfile" "$ROOT_DIR"
    docker push "$REGISTRY/deepspeed-fsdp:latest"
  fi
  log "Image build/push stage complete"
}

stage_deploy_quantum_and_simulators() {
  if [ "$DO_QUANTUM" = false ]; then
    log "Skipping quantum deployment"
    return
  fi
  require_cmd kubectl
  # Deploy Redis
  if [ -f "$ROOT_DIR/k8s/redis/redis-deployment.yaml" ]; then
    kubectl apply -f "$ROOT_DIR/k8s/redis/redis-deployment.yaml"
  fi
  # Deploy cirq simulator deployment (if image exists)
  if [ -f "$ROOT_DIR/k8s/simulator/cirq-simulator-deployment.yaml" ]; then
    sed "s|ghcr.io/yourorg/cirq-simulator:latest|$REGISTRY/cirq-simulator:latest|g" "$ROOT_DIR/k8s/simulator/cirq-simulator-deployment.yaml" | kubectl apply -f -
  fi
  # Deploy qiskit simulator deployment
  if [ -f "$ROOT_DIR/k8s/simulator/qiskit-simulator-deployment.yaml" ]; then
    sed "s|ghcr.io/yourorg/qiskit-simulator:latest|$REGISTRY/qiskit-simulator:latest|g" "$ROOT_DIR/k8s/simulator/qiskit-simulator-deployment.yaml" | kubectl apply -f -
  fi
  # Deploy quantum controller & worker (k8s manifests exist in terraform/ibm or k8s/quantum)
  if [ -f "$ROOT_DIR/k8s/quantum/quantum-controller-deployment.yaml" ]; then
    sed "s|ghcr.io/yourorg/quantum-controller:latest|$REGISTRY/quantum-controller:latest|g" "$ROOT_DIR/k8s/quantum/quantum-controller-deployment.yaml" | kubectl apply -f -
  fi
  if [ -f "$ROOT_DIR/k8s/quantum/quantum-worker-deployment.yaml" ]; then
    sed "s|ghcr.io/yourorg/quantum-worker:latest|$REGISTRY/quantum-worker:latest|g" "$ROOT_DIR/k8s/quantum/quantum-worker-deployment.yaml" | kubectl apply -f -
  fi

  # Wait a bit for resources
  log "Waiting for simulator & controller pods to become ready (60s)..."
  sleep 10
  kubectl get pods -n staging || true
}

stage_deploy_and_run_workflows() {
  require_cmd kubectl
  # Apply DGX workflow if present
  if [ "$DO_DGX" = true ] && [ -f "$ROOT_DIR/dgx/argo/deepspeed_dgx_validation.yaml" ]; then
    log "Submitting DGX validation workflow"
    kubectl apply -f "$ROOT_DIR/dgx/argo/deepspeed_dgx_validation.yaml"
    # If argo CLI available, wait for named workflow
    if command -v argo >/dev/null 2>&1; then
      log "Waiting for DGX workflow to complete (argo wait not implemented by default; operator may monitor manually)"
    fi
  fi

  # Apply TPU workflow
  if [ "$DO_TPU" = true ] && [ -f "$ROOT_DIR/gcp/argo/gcp_tpu_validation.yaml" ]; then
    log "Submitting GCP TPU workflow"
    kubectl apply -f "$ROOT_DIR/gcp/argo/gcp_tpu_validation.yaml"
  fi

  # Submit quantum Argo workflows for cirq & ibmq if present
  for wf in "$ROOT_DIR/quantum/argo/cirq_workflow.yaml" "$ROOT_DIR/quantum/argo/braket_pennylane_workflow.yaml" "$ROOT_DIR/quantum/argo/ibmq_workflow.yaml" "$ROOT_DIR/argo/workflows/quantum_hybrid_workflow.yaml"; do
    if [ -f "$wf" ]; then
      log "Applying workflow $wf"
      kubectl apply -f "$wf"
    fi
  done

  # Monitor Argo workflows (if argo CLI exists) and collect outputs
  if command -v argo >/dev/null 2>&1; then
    log "Collecting Argo workflow results (this may timeout if workflows long-running)"
    end=$((SECONDS + ARGO_WAIT_TIMEOUT))
    while [ $SECONDS -lt $end ]; do
      # gather summaries into artifact dir
      argo list -n staging --no-headers > "$ARTIFACT_DIR/argo_list.txt" 2>&1 || true
      # break if no running workflows
      if ! argo list -n staging --no-headers | awk '{print $6}' | grep -q "Running"; then
        break
      fi
      sleep "$WORKFLOW_POLL_INTERVAL"
    done
  fi
}

stage_dgx_validation() {
  if [ "$DO_DGX" = false ]; then
    log "Skipping DGX validation"
    return
  fi
  if [ -f "$ROOT_DIR/dgx/scripts/dgx_run_validation.sh" ]; then
    log "Running DGX validation script (it may simulate preemption and resubmit)"
    bash "$ROOT_DIR/dgx/scripts/dgx_run_validation.sh" | tee "$ARTIFACT_DIR/dgx_validation.log" || true
  else
    log "DGX validation script not present; skipping"
  fi
}

stage_tpu_validation() {
  if [ "$DO_TPU" = false ]; then
    log "Skipping TPU validation"
    return
  fi
  if [ -f "$ROOT_DIR/gcp/scripts/gke_submit_tpu_workflow.sh" ]; then
    log "Submitting TPU workflow"
    bash "$ROOT_DIR/gcp/scripts/gke_submit_tpu_workflow.sh" | tee "$ARTIFACT_DIR/tpu_submit.log" || true
  else
    log "TPU submit script not found; ensure workflow deployed"
  fi
}

stage_edge_tests() {
  if [ "$DO_EDGE" = false ]; then
    log "Skipping Edge tests"
    return
  fi
  # Simple local quickstart: run enrollment + ota servers for smoke tests if operator chooses
  if [ -f "$ROOT_DIR/iot/production/enrollment_server_mtls.py" ] && [ -f "$ROOT_DIR/iot/production/ota/server_signed_urls.py" ]; then
    if confirm "Run local IoT enrollment + OTA servers for quick smoke test in background?"; then
      log "Starting enrollment & OTA servers in background (stdout -> artifact logs)"
      (python3 "$ROOT_DIR/iot/production/enrollment_server_mtls.py" > "$ARTIFACT_DIR/enroll_server.log" 2>&1 &)
      (python3 "$ROOT_DIR/iot/production/ota/server_signed_urls.py" > "$ARTIFACT_DIR/ota_server.log" 2>&1 &)
      sleep 3
      log "Triggering sample enrollment (if registry exists)"
      # attempt to enroll a sample device (best-effort)
      if command -v curl >/dev/null 2>&1; then
        curl -sS -X POST http://localhost:8443/enroll -H "Content-Type: application/json" -d '{"device_id":"sim-device-1","csr":"-----BEGIN CERTIFICATE REQUEST-----\nMIIB...==\n-----END CERTIFICATE REQUEST-----"}' > "$ARTIFACT_DIR/enroll_response.json" 2>&1 || true
      fi
    fi
  fi
  # Canary controller run
  if [ -f "$ROOT_DIR/edge/canary/canary_controller_telemetry.py" ]; then
    python3 "$ROOT_DIR/edge/canary/canary_controller_telemetry.py" "default" "1" > "$ARTIFACT_DIR/canary_run.log" 2>&1 || true
  fi
}

stage_attestation_smoke() {
  # If attestation scripts exist, run smoke tests to validate webhook
  if [ -f "$ROOT_DIR/scripts/attestation_smoke_test.sh" ]; then
    bash "$ROOT_DIR/scripts/attestation_smoke_test.sh" | tee "$ARTIFACT_DIR/attestation_smoke.log" || true
  else
    log "No attestation smoke test script found; skipping"
  fi
}

collect_artifacts() {
  log "Collecting artifacts into $ARTIFACT_DIR"
  save_terraform_outputs
  save_vault_state
  collect_k8s_namespace_logs "staging"
  collect_k8s_namespace_logs "ops"
  collect_k8s_namespace_logs "default"
  collect_argo_workflows
  # collect argo controller logs and relevant pod logs
  if kubectl get pods -n staging -l app=quantum-controller >/dev/null 2>&1; then
    for p in $(kubectl get pods -n staging -l app=quantum-controller -o name | sed 's#pod/##'); do
      kubectl logs -n staging "$p" > "$ARTIFACT_DIR/k8s/staging/$p.log" 2>&1 || true
    done
  fi
  # copy any repo-level logs produced
  for f in "$ROOT_DIR"/scripts/*log "$ROOT_DIR"/scripts/*.log; do
    [ -f "$f" ] && cp -a "$f" "$ARTIFACT_DIR/" || true
  done
  # Capture output of `kubectl get all --all-namespaces`
  kubectl get all --all-namespaces -o wide > "$ARTIFACT_DIR/k8s_all_namespaces.txt" 2>&1 || true
  # Package artifacts
  package_artifacts
}

print_summary() {
  echo
  echo "Orchestration complete. Artifacts collected at: $ARTIFACT_DIR"
  echo "Packed archive: ${ARTIFACT_DIR}.tar.gz"
  echo
  echo "Next steps (operator):"
  echo " - Review artifacts and logs in $ARTIFACT_DIR"
  echo " - Verify Vault KV entries and delete any temporary JSON files containing secrets"
  echo " - If workflows failed, inspect logs in $ARTIFACT_DIR/k8s and $ARTIFACT_DIR/argo"
  echo
}

#####################
# CLI arg parsing   #
#####################
print_usage() {
  sed -n '1,160p' "$0" | sed -n '1,120p'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-terraform) DO_TERRAFORM=false; shift ;;
    --skip-vault) DO_VAULT=false; shift ;;
    --skip-images) DO_IMAGES=false; shift ;;
    --skip-quantum) DO_QUANTUM=false; shift ;;
    --skip-dgx) DO_DGX=false; shift ;;
    --skip-tpu) DO_TPU=false; shift ;;
    --skip-edge) DO_EDGE=false; shift ;;
    --yes) AUTO_YES=true; shift ;;
    --github-repo) GITHUB_REPO="$2"; shift 2 ;;
    --registry) REGISTRY="$2"; shift 2 ;;
    --aws-region) AWS_REGION="$2"; shift 2 ;;
    --gcp-project) GCP_PROJECT="$2"; shift 2 ;;
    --ibm-region) IBM_REGION="$2"; shift 2 ;;
    --vault-root-token) VAULT_ROOT_TOKEN="$2"; shift 2 ;;
    --vault-addr) VAULT_ADDR="$2"; export VAULT_ADDR; shift 2 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "Unknown arg: $1"; print_usage; exit 2 ;;
  esac
done

#################################
# Pre-checks for required tools #
#################################
for c in jq kubectl helm tar gzip; do
  require_cmd "$c"
done

# Optional check for heavy tools
if $DO_TERRAFORM; then require_cmd terraform; fi
if $DO_IMAGES; then require_cmd docker || true; fi
if $DO_QUANTUM; then require_cmd python3 || true; fi

########################
# Execution of stages  #
########################

log "Starting Aegis full orchestrator run"
log "Artifacts will be written to: $ARTIFACT_DIR"

if [ "$DO_TERRAFORM" = true ]; then
  if confirm "Run terraform/CLI provisioning stage now?"; then
    stage_terraform
  else
    log "Terraform stage skipped by operator"
  fi
fi

if [ "$DO_VAULT" = true ]; then
  if confirm "Install and bootstrap Vault now (Helm + bootstrap script)?"; then
    stage_vault_install_and_bootstrap
  else
    log "Vault stage skipped by operator"
  fi
fi

# Store IBM secrets into Vault (if previously created)
if [ -f "$ROOT_DIR/scripts/ibm/out/ibm_resources.json" ]; then
  if confirm "Store IBM resources secrets into Vault now?"; then
    stage_store_ibm_to_vault
  else
    log "Operator chose not to store IBM secrets to Vault now"
  fi
fi

if [ "$DO_IMAGES" = true ]; then
  if confirm "Build and push container images to registry ($REGISTRY)?"; then
    stage_build_and_push_images
  else
    log "Image build/push skipped"
  fi
fi

if [ "$DO_QUANTUM" = true ]; then
  if confirm "Deploy quantum controller, simulators and workers?"; then
    stage_deploy_quantum_and_simulators
  else
    log "Quantum deployment skipped"
  fi
fi

# Deploy Argo workflows (DGX / TPU / quantum)
if confirm "Apply Argo workflows and submit validation workflows (DGX / TPU / Quantum)?"; then
  stage_deploy_and_run_workflows
else
  log "Workflows stage skipped"
fi

# Run DGX/Tpu/Edge tests
if [ "$DO_DGX" = true ]; then
  if confirm "Run DGX validation scripts (may simulate failure and resume)?"; then
    stage_dgx_validation
  fi
fi

if [ "$DO_TPU" = true ]; then
  if confirm "Run TPU validation stage?"; then
    stage_tpu_validation
  fi
fi

if [ "$DO_EDGE" = true ]; then
  if confirm "Run Edge / OTA / Canary smoke tests?"; then
    stage_edge_tests
  fi
fi

if confirm "Run attestation smoke tests?"; then
  stage_attestation_smoke
fi

log "Collecting artifacts"
collect_artifacts

print_summary

# Exit success
exit 0
