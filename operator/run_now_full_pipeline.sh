#!/usr/bin/env bash
# Run-now operator script: sequences remaining operational steps to reach full production readiness.
# PURPOSE
#  - Vendor HSM provisioning & validation
#  - Credentialed QPU pilots (Braket & IBM)
#  - Broker production cutover (managed RDS, cert-manager, mTLS, JWT rotation, HPA)
#  - CI / auditor certification checks (Rekor enforcement, evidence bundle)
#  - Observability & billing integration test (CUR ingestion, cost spike -> fallback)
#  - Produce auditor evidence bundle
#
# USAGE (example)
#  ./run_now_full_pipeline.sh \
#    --hsm-tfvars ./cloud/hsm/terraform.tfvars \
#    --rds-tfvars ./broker/terraform/prod.tfvars \
#    --vault-path secret/data/quantum/providers \
#    --braket-device arn:aws:braket:... \
#    --ibm-token "QISKIT_IBM_TOKEN_VALUE" \
#    --s3-bucket my-staging-bucket \
#    --hsm-audit-bucket my-hsm-audit-bucket \
#    --cur-s3-path s3://billing-cur-dest/cur-2025-12-24/ \
#    --mlflow-url http://mlflow:5000 \
#    --skip-confirm   # to run non-interactively
#
# NOTE: This script orchestrates invocation of many existing scripts in this repo.
#       It assumes operator credentials (aws, kubectl, vault, helm, terraform) are configured.
#       Many steps are vendor / environment dependent; review prompts and logs carefully.
set -euo pipefail

# -------- Configurable defaults --------
WORKDIR="$(pwd)"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
OUTDIR="/tmp/aegis_runnow_${TIMESTAMP}"
mkdir -p "${OUTDIR}"

# default empty
HSM_TF_VARS=""
RDS_TF_VARS=""
VAULT_PATH=""
BRAKET_DEVICE=""
IBM_TOKEN=""
PROGRAM=""
S3_BUCKET=""
HSM_AUDIT_BUCKET=""
CUR_S3_PATH=""
MLFLOW_URL=""
SKIP_CONFIRM=false
NAMESPACE="aegis"
HELM_RELEASE="aegis-quantum-broker"

# -------- Helpers --------
log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
check_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"
}
confirm_or_die() {
  if [ "${SKIP_CONFIRM}" = true ]; then return 0; fi
  read -p "$1 (y/N): " ans
  if [[ ! "$ans" =~ ^[Yy]$ ]]; then die "Operator aborted"; fi
}

# -------- Parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hsm-tfvars) HSM_TF_VARS="$2"; shift 2;;
    --rds-tfvars) RDS_TF_VARS="$2"; shift 2;;
    --vault-path) VAULT_PATH="$2"; shift 2;;
    --braket-device) BRAKET_DEVICE="$2"; shift 2;;
    --ibm-token) IBM_TOKEN="$2"; shift 2;;
    --program) PROGRAM="$2"; shift 2;;
    --s3-bucket) S3_BUCKET="$2"; shift 2;;
    --hsm-audit-bucket) HSM_AUDIT_BUCKET="$2"; shift 2;;
    --cur-s3-path) CUR_S3_PATH="$2"; shift 2;;
    --mlflow-url) MLFLOW_URL="$2"; shift 2;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --helm-release) HELM_RELEASE="$2"; shift 2;;
    --skip-confirm) SKIP_CONFIRM=true; shift 1;;
    --help) sed -n '1,200p' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

# -------- Prereqs check --------
log "Checking prerequisites..."
REQUIRED_CMDS=(aws vault kubectl helm terraform jq python3 curl)
for c in "${REQUIRED_CMDS[@]}"; do
  if ! command -v "${c}" >/dev/null 2>&1; then
    log "WARNING: command '${c}' not found. Some steps may fail or must be run manually."
  fi
done

log "Outputs & logs will be written to: ${OUTDIR}"

# -------- Step 1: Vendor HSM provisioning & validation --------
log "STEP 1: Vendor HSM provisioning & validation"

if [ -n "${HSM_TF_VARS}" ]; then
  confirm_or_die "About to run Terraform to provision Cloud/Vendor HSM using ${HSM_TF_VARS}. Continue?"
  if [ -d "cloud/hsm" ]; then
    log "Running cloud/hsm/provision_cloudhsm.sh with --tf-vars ${HSM_TF_VARS}"
    bash cloud/hsm/provision_cloudhsm.sh --tf-vars "${HSM_TF_VARS}" | tee "${OUTDIR}/provision_cloudhsm.log"
  else
    log "cloud/hsm not found in repo; skipping terraform provisioning step; assume vendor HSM is provisioned separately"
  fi
else
  log "No HSM_TF_VARS given; skipping HSM terraform provisioning. Expect vendor HSM already provisioned."
fi

# Install PKCS#11 module on signing hosts (Ansible)
log "Installing PKCS#11 module or SoftHSM on signing hosts (if Ansible inventory available)"
if [ -f "quantum/hsm/ansible/install_pkcs11.yml" ]; then
  confirm_or_die "Run Ansible install_pkcs11.yml against inventory to install PKCS#11 module? (inventory must be configured)"
  if command -v ansible-playbook >/dev/null 2>&1; then
    ansible-playbook -i inventory quantum/hsm/ansible/install_pkcs11.yml | tee "${OUTDIR}/ansible_install_pkcs11.log"
  else
    log "ansible-playbook not available; skipped. Please run the playbook manually."
  fi
else
  log "No Ansible playbook found; skip install step."
fi

# Validate HSM end-to-end: requires operator to know pkcs11 module path/slot/pin/keylabel
log ""
log "HSM end-to-end validation requires PKCS#11 parameters and at least a test artifact."
confirm_or_die "Run HSM end-to-end validation now? (this will prompt for PKCS#11 params interactively)"
if true; then
  read -p "PKCS#11 module path (e.g. /opt/vendor/lib/pkcs11.so): " PKCS11_LIB
  read -p "PKCS#11 slot (e.g. 0): " PKCS11_SLOT
  read -p "PKCS#11 user PIN: " -s PKCS11_PIN
  echo
  read -p "PKCS#11 key label (e.g. pqkey): " PKCS11_KEYLABEL
  read -p "Path to artifact to sign (or press Enter to create /tmp/aegis_hsm_test.bin): " ART
  if [ -z "${ART}" ]; then
    ART="/tmp/aegis_hsm_test.bin"
    echo "aegis hsm test" > "${ART}"
  fi
  HSM_VALIDATE_CMD=(bash quantum/hsm/validate_hsm_end_to_end.sh --artifact "${ART}" --pkcs11-lib "${PKCS11_LIB}" --pkcs11-slot "${PKCS11_SLOT}" --pkcs11-pin "${PKCS11_PIN}" --pkcs11-keylabel "${PKCS11_KEYLABEL}")
  if [ -n "${HSM_AUDIT_BUCKET}" ]; then
    HSM_VALIDATE_CMD+=("--s3-bucket" "${HSM_AUDIT_BUCKET}")
  fi
  "${HSM_VALIDATE_CMD[@]}" | tee "${OUTDIR}/hsm_validate.log" || log "HSM validation script returned non-zero; inspect ${OUTDIR}/hsm_validate.log"
fi

# Run rotation test (operator must create new key in HSM using vendor CLI before running)
confirm_or_die "Run HSM rotation test? (operator must have created a new key labeled like 'pqkey-v2' in HSM beforehand)"
read -p "New key label to test (e.g. pqkey-v2): " NEW_LABEL
read -p "Path to new public key PEM (exported by vendor CLI): " NEW_PUBKEY
bash quantum/hsm/vendor_rotation_test.sh --vault-path "secret/data/hsm/config" --new-label "${NEW_LABEL}" --pubkey "${NEW_PUBKEY}" --pkcs11-lib "${PKCS11_LIB}" --slot "${PKCS11_SLOT}" --pin "${PKCS11_PIN}" --artifact "${ART}" --s3-bucket "${HSM_AUDIT_BUCKET:-}" | tee "${OUTDIR}/hsm_rotation_test.log" || log "Rotation test failed; see ${OUTDIR}/hsm_rotation_test.log"

log "STEP 1 complete; collect ${OUTDIR}/hsm_validate.log and ${OUTDIR}/hsm_rotation_test.log for compliance."

# -------- Step 2: Credentialed QPU pilots --------
log "STEP 2: Credentialed QPU pilots (Braket & IBM)"

if [ -z "${VAULT_PATH}" ]; then
  read -p "Vault path to write provider creds (e.g. secret/data/quantum/providers): " VAULT_PATH
fi

if [ -z "${PROGRAM}" ]; then
  read -p "Path to QASM program for pilots (e.g. demo.qasm) [will create a small placeholder if blank]: " PROGRAM
  if [ -z "${PROGRAM}" ]; then
    PROGRAM="/tmp/aegis_demo.qasm"
    cat > "${PROGRAM}" <<'EOF'
    // placeholder OPENQASM - simple demo (not necessarily runnable on hardware)
    OPENQASM 2.0;
    qreg q[1];
    h q[0];
    measure q[0] -> c[0];
EOF
  fi
fi

# Braket
if [ -z "${BRAKET_DEVICE}" ]; then
  read -p "Braket device ARN (leave empty to skip Braket pilot): " BRAKET_DEVICE
fi
if [ -n "${BRAKET_DEVICE}" ]; then
  log "Publishing Braket device ARN to Vault and running pilot"
  ./quantum/pilot/store_creds_and_run.sh --vault-path "${VAULT_PATH}" --provider braket --key braket_device --value "${BRAKET_DEVICE}" --program "${PROGRAM}" --s3-bucket "${S3_BUCKET}" | tee "${OUTDIR}/braket_pilot.log" || log "Braket pilot invocation failed (see log)."
else
  log "No Braket device provided; skipping Braket pilot."
fi

# IBM
if [ -z "${IBM_TOKEN}" ]; then
  read -p "IBM Qiskit token (leave empty to skip IBM pilot): " IBM_TOKEN
fi
if [ -n "${IBM_TOKEN}" ]; then
  log "Publishing IBM token to Vault and running pilot"
  ./quantum/pilot/store_creds_and_run.sh --vault-path "${VAULT_PATH}" --provider ibm --key ibm_token --value "${IBM_TOKEN}" --program "${PROGRAM}" --s3-bucket "${S3_BUCKET}" --ibm-backend "${IBM_BACKEND:-}" | tee "${OUTDIR}/ibm_pilot.log" || log "IBM pilot invocation failed (see log)."
else
  log "No IBM token provided; skipping IBM pilot."
fi

log "Waiting for pilots to produce artifacts & MLflow runs (sleep 60s then check)..."
sleep 60
if [ -n "${MLFLOW_URL}" ]; then
  log "Verifying MLflow Rekor tags for recent runs..."
  python3 quantum/rekor/check_mlflow_rekor.py --mlflow-url "${MLFLOW_URL}" --experiment quantum-pilots --threshold 10 | tee "${OUTDIR}/mlflow_rekor_check.log" || log "MLflow Rekor check failed or found missing tags."
else
  log "MLFLOW_URL not set; skipping MLflow Rekor verification."
fi

# Attempt playback for any artifacts found
log "Attempting to verify playback against MLflow artifacts (if any)"
python3 quantum/pilot/verify_mlflow_and_playback.py --mlflow-url "${MLFLOW_URL:-http://mlflow:5000}" --experiment quantum-pilots --s3-bucket "${S3_BUCKET:-}" --qasm-file "${PROGRAM}" --max-runs 10 | tee "${OUTDIR}/pilot_playback.log" || log "Playback verification encountered issues."

log "STEP 2 complete; check ${OUTDIR}/braket_pilot.log ${OUTDIR}/ibm_pilot.log ${OUTDIR}/mlflow_rekor_check.log ${OUTDIR}/pilot_playback.log"

# -------- Step 3: Broker production cutover --------
log "STEP 3: Broker production cutover (RDS, cert-manager, Helm deploy, JWT rotation, HPA)"

if [ -n "${RDS_TF_VARS}" ]; then
  confirm_or_die "About to run Terraform to provision managed Postgres using ${RDS_TF_VARS}. Continue?"
  bash broker/scripts/deploy_broker_prod.sh --tf-vars "${RDS_TF_VARS}" --namespace "${NAMESPACE}" --helm-release "${HELM_RELEASE}" 2>&1 | tee "${OUTDIR}/deploy_broker_prod.log"
else
  log "No RDS TF vars provided; skipping Terraform RDS provisioning. Ensure RDS already exists and JOB_DATABASE_URL secret is present."
  echo "If RDS is already provisioned, ensure SECRET aegis-db-secret exists in namespace ${NAMESPACE}."
fi

confirm_or_die "Proceed to install cert-manager & apply issuer manifest (requires cluster-admin)?"
kubectl apply -f broker/k8s/cert-manager-issuer.yaml || log "Applying cert-manager issuer manifest may have failed."

log "Deploying Helm chart (production values); ensure aegis-db-secret & aegis-broker-secret are populated in cluster"
helm upgrade --install "${HELM_RELEASE}" broker/helm -n "${NAMESPACE}" --values broker/helm/values-production.yaml 2>&1 | tee "${OUTDIR}/helm_deploy.log" || log "Helm deploy may have issues; inspect ${OUTDIR}/helm_deploy.log"

log "Deploy worker autoscale manifests"
kubectl apply -f broker/k8s/worker-deployment-autoscale.yaml -n "${NAMESPACE}" 2>&1 | tee "${OUTDIR}/worker_deploy.log" || log "Worker deployment may have failed."

log "Rotate JWT secret and verify broker"
./broker/scripts/jwt_rotate_and_verify.sh "${NAMESPACE}" "aegis-broker-secret" | tee "${OUTDIR}/jwt_rotate_verify.log" || log "JWT rotation verification encountered issues."

log "Run HPA load test to exercise autoscaling (requires valid JWT)."
read -p "Provide a valid broker JWT to run load test (or press Enter to skip): " BROKER_JWT
if [ -n "${BROKER_JWT}" ]; then
  ./broker/runbooks/load_test_hpa.sh "http://aegis-quantum-broker.aegis.svc.cluster.local/submit" "${BROKER_JWT}" 100 | tee "${OUTDIR}/hpa_load_test.log" || log "HPA load test run; inspect logs."
else
  log "No JWT provided; skipping HPA load test."
fi

log "STEP 3 complete; check ${OUTDIR}/deploy_broker_prod.log ${OUTDIR}/helm_deploy.log ${OUTDIR}/worker_deploy.log ${OUTDIR}/jwt_rotate_verify.log"

# -------- Step 4: CI / Auditor certification checks --------
log "STEP 4: CI & Auditor certification"

confirm_or_die "Run local MLflow Rekor enforcement check now (requires MLFLOW_URL)?"
if [ -n "${MLFLOW_URL}" ]; then
  python3 quantum/rekor/check_mlflow_rekor.py --mlflow-url "${MLFLOW_URL}" --experiment quantum-pilots --threshold 10 | tee "${OUTDIR}/ci_rekor_check.log" || log "CI Rekor check failed (some runs missing Rekor tags)"
else
  log "MLFLOW_URL not set; skip CI Rekor check"
fi

log "Produce auditor evidence bundle (packager)"
python3 compliance/packager.py 2>&1 | tee "${OUTDIR}/auditor_packager.log" || log "packager.py may have issues; operator may run compliance/audit_packager_enhanced.sh manually."

log "STEP 4 complete; check ${OUTDIR}/ci_rekor_check.log ${OUTDIR}/auditor_packager.log"

# -------- Step 5: Observability & billing integration --------
log "STEP 5: Observability & billing integration"

if [ -n "${CUR_S3_PATH}" ]; then
  confirm_or_die "About to ingest CUR files from ${CUR_S3_PATH} into billing (placeholder transform). Continue?"
  bash observability/cur/ingest_cur.sh "${CUR_S3_PATH}" "/tmp/aegis_cur_${TIMESTAMP}" default | tee "${OUTDIR}/cur_ingest.log" || log "CUR ingest encountered issues."
else
  log "No CUR path provided; skip CUR ingestion. Operator should configure CUR -> Lambda / ingest pipeline."
fi

log "Simulate a cost spike to test fallback automation (will insert fake billing row)."
confirm_or_die "Inject fake billing row to trigger cost_enforcer fallback? (this will add a row to billing DB)"
bash observability/test_cost_spike.sh "/tmp/aegis_fake_billing.csv" default | tee "${OUTDIR}/cost_spike.log" || log "Cost spike injection may have failed."

log "Wait 60s and query broker admin fallback state..."
sleep 60
FALLBACK_STATE=$(curl -s "http://aegis-quantum-broker.${NAMESPACE}.svc.cluster.local/admin/get-fallback" || echo "{}")
echo "${FALLBACK_STATE}" > "${OUTDIR}/fallback_state.json"
log "Fallback state: $(cat ${OUTDIR}/fallback_state.json)"

log "STEP 5 complete; review ${OUTDIR}/cur_ingest.log ${OUTDIR}/cost_spike.log ${OUTDIR}/fallback_state.json"

# -------- Step 6: Legal & compliance sign-offs --------
log "STEP 6: Legal & compliance sign-offs (manual + aided by scripts)"

log "Produced auditor manifest template at compliance/auditor/evidence_manifest_template.yaml"
log "Operator: send compliance/legal/contract_signoff_template.md to legal and use compliance/auditor/deterministic_signing_README.md for auditor handoff."

confirm_or_die "Run final enhanced audit packager to produce evidence tar.gz now?"
bash compliance/audit_packager_enhanced.sh "${OUTDIR}/final_evidence" | tee "${OUTDIR}/final_evidence_packager.log" || log "Final evidence packager may have had issues."

log "FINAL: All orchestration steps invoked. Output logs and artifacts in ${OUTDIR}."
log "Please review the logs, gather evidence, and follow up with vendor/legal teams for signoffs."

# Summary
echo
echo "Run summary (files in ${OUTDIR}):"
ls -lh "${OUTDIR}" || true
echo
echo "Next manual actions (high priority):"
echo " - Verify HSM vendor audit logs and vendor-supplied evidence"
echo " - Legal to countersign vendor contracts (use compliance/legal/contract_signoff_template.md)"
echo " - Run SOC2 audit engagement using ${OUTDIR}/final_evidence.tar.gz"
echo
echo "DONE."
