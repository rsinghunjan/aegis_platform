#!/usr/bin/env bash
set -euo pipefail
#
# apply_azure_identity_crs.sh
#
# Render and apply AzureIdentity / AzureIdentityBinding (aad-pod-identity v1)
# or AzureAssignedIdentity / AzureIdentityBinding (v2-style) CRs for the backup UAMI and annotate the ServiceAccount.
#
# Usage:
#   ./scripts/apply_azure_identity_crs.sh \
#     --variant aad-pod-identity-v1 \
#     --name milvus-backup-identity \
#     --resource-id /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<uami> \
#     --client-id <uami-client-id> \
#     --namespace aegis \
#     --ksa milvus-backup-sa \
#     --selector milvus-backup-selector
#
set -u

VARIANT="aad-pod-identity-v1"
NAME=""
RESOURCE_ID=""
CLIENT_ID=""
NAMESPACE="aegis"
KSA="milvus-backup-sa"
SELECTOR=""
TEMPLATE_FILE="infra/azure/templates/azureidentity-template.yaml"
TMP_RENDER="/tmp/aegis_azure_identity_rendered.yaml"
KUBECTL=${KUBECTL:-kubectl}

usage() {
  cat <<EOF
Usage: $0 --variant <aad-pod-identity-v1|pod-identity-v2> --name NAME --resource-id RESOURCE_ID --client-id CLIENT_ID --namespace NS --ksa KSA --selector SELECTOR
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --resource-id) RESOURCE_ID="$2"; shift 2;;
    --client-id) CLIENT_ID="$2"; shift 2;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --ksa) KSA="$2"; shift 2;;
    --selector) SELECTOR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg $1"; usage; exit 2;;
  esac
done

if [ -z "$NAME" ] || [ -z "$RESOURCE_ID" ] || [ -z "$CLIENT_ID" ] || [ -z "$SELECTOR" ]; then
  usage
  exit 2
fi

if [ ! -f "$TEMPLATE_FILE" ]; then
  echo "Template not found: $TEMPLATE_FILE"
  exit 1
fi

# Render template using simple sed replacements (keeps dependencies minimal)
cp "$TEMPLATE_FILE" "$TMP_RENDER"
sed -i "s|{{ .name }}|${NAME}|g" "$TMP_RENDER"
sed -i "s|{{ .resourceID }}|${RESOURCE_ID}|g" "$TMP_RENDER"
sed -i "s|{{ .clientID }}|${CLIENT_ID}|g" "$TMP_RENDER"
sed -i "s|{{ .bindingName }}|${NAME}-binding|g" "$TMP_RENDER"
sed -i "s|{{ .selector }}|${SELECTOR}|g" "$TMP_RENDER"

# Choose variant-specific CRs to apply
if [ "$VARIANT" = "aad-pod-identity-v1" ]; then
  echo "[azure] Applying aad-pod-identity v1 CRs"
  $KUBECTL -n "$NAMESPACE" apply -f "$TMP_RENDER"
  echo "[azure] Annotating ServiceAccount ${KSA} with aadpodidbinding=${SELECTOR}"
  $KUBECTL -n "$NAMESPACE" annotate sa "$KSA" "aadpodidbinding=${SELECTOR}" --overwrite || true
else
  echo "[azure] Applying pod-identity v2-style CRs (operator-specific). Applying rendered CRs — adapt if needed."
  $KUBECTL -n "$NAMESPACE" apply -f "$TMP_RENDER"
  echo "[azure] Annotating ServiceAccount ${KSA} for v2-style operator (attempt best-effort)"
  $KUBECTL -n "$NAMESPACE" annotate sa "$KSA" "azure.workload.identity/${SELECTOR}=true" --overwrite || true
fi

echo "[azure] Applied CRs and annotations. Check CR status (kubectl get AzureIdentity, AzureAssignedIdentity, AzureIdentityBinding, AzureAssignedIdentity) to confirm."
