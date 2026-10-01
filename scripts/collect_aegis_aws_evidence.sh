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
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
#!/usr/bin/env bash
# Collect read-only evidence for Aegis AWS validation.
# Usage: ./scripts/collect_aegis_aws_evidence.sh [OUTDIR]
# Requires: aws, kubectl, jq, curl, (optional) gh, VAULT_ADDR+VAULT_TOKEN env vars, PROM_URL optional.
set -euo pipefail

OUTDIR=${1:-aegis-aws-evidence-$(date +%Y%m%dT%H%M%S)}
mkdir -p "$OUTDIR"

echo "Collecting evidence into: $OUTDIR"

# 1) Terraform outputs
TF_PATH="infra/terraform/overlays/aws/tf_outputs.json"
echo "1) Terraform outputs..."
if [ -f "$TF_PATH" ]; then
  cp "$TF_PATH" "$OUTDIR"/tf_outputs.json
  if command -v jq >/dev/null 2>&1; then
    jq '.' "$TF_PATH" > "$OUTDIR"/tf_outputs.pretty.json || cp "$TF_PATH" "$OUTDIR"/tf_outputs.pretty.json
  fi
else
  echo "WARN: $TF_PATH not found" > "$OUTDIR"/tf_outputs.missing
fi

# Parse bucket and role if available
BUCKET=""
ROLE_ARN=""
if [ -f "$OUTDIR/tf_outputs.json" ] && command -v jq >/dev/null 2>&1; then
  BUCKET=$(jq -r '.bucket_name.value // ""' "$OUTDIR"/tf_outputs.json || true)
  ROLE_ARN=$(jq -r '.github_oidc_role_arn.value // ""' "$OUTDIR"/tf_outputs.json || true)
fi
echo "bucket=${BUCKET}" > "$OUTDIR"/summary.txt
echo "github_oidc_role_arn=${ROLE_ARN}" >> "$OUTDIR"/summary.txt

# 2) S3 bucket listing
echo "2) S3 bucket listing..."
if [ -n "$BUCKET" ] && command -v aws >/dev/null 2>&1; then
  aws s3 ls "s3://${BUCKET}" > "$OUTDIR"/s3_root.txt 2>&1 || true
  aws s3 ls "s3://${BUCKET}/model-archives/" > "$OUTDIR"/s3_model_archives.txt 2>&1 || true
  aws s3 ls "s3://${BUCKET}/db-backups/" > "$OUTDIR"/s3_db_backups.txt 2>&1 || true
else
  echo "No bucket detected or aws CLI missing; skipped S3 checks" > "$OUTDIR"/s3_missing.txt
fi

# 3) GitHub Actions recent runs (optional)
echo "3) GitHub Actions recent runs..."
if command -v gh >/dev/null 2>&1; then
  REPO="${GITHUB_REPO:-}"
  if [ -z "$REPO" ]; then
    # try to infer from git remote
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      ORIGIN=$(git remote get-url origin 2>/dev/null || true)
      if [ -n "$ORIGIN" ]; then
        # normalize SSH/HTTPS to owner/repo
        REPO=$(echo "$ORIGIN" | sed -E 's#.*[:/](.+/.+)(\.git)?$#\1#')
      fi
    fi
  fi
  if [ -n "$REPO" ]; then
    gh run list --repo "$REPO" --limit 20 > "$OUTDIR"/gh_runs.txt 2>&1 || true
  else
    echo "GH repo not set and cannot infer; skipping GH run listing" > "$OUTDIR"/gh_runs.missing
  fi
else
  echo "gh CLI not found; skipping GH run listing" > "$OUTDIR"/gh_runs.missing
fi

# 4) Vault cosign public key check (optional; requires VAULT_ADDR+VAULT_TOKEN env)
echo "4) Vault cosign public key check..."
if [ -n "${VAULT_ADDR:-}" ] && [ -n "${VAULT_TOKEN:-}" ]; then
  VAULT_PATH=${VAULT_COSIGN_PATH:-secret/data/aegis/cosign}
  set +e
  curl -sS -H "X-Vault-Token: ${VAULT_TOKEN}" "${VAULT_ADDR%/}/v1/${VAULT_PATH}" > "$OUTDIR"/vault_cosign.json 2>&1
  CURL_RC=$?
  set -e
  if [ "$CURL_RC" -ne 0 ]; then
    echo "Vault query failed; see $OUTDIR/vault_cosign.json" >&2
  fi
else
  echo "VAULT_ADDR/VAULT_TOKEN not set; skipping Vault check" > "$OUTDIR"/vault_missing.txt
fi

# 5) Kubernetes checks
echo "5) Kubernetes checks (kubectl required)..."
if command -v kubectl >/dev/null 2>&1; then
  kubectl -n aegis get pods -o wide > "$OUTDIR"/k8s_pods.txt 2>&1 || true
  kubectl -n aegis get secret cosign-public-key -o yaml > "$OUTDIR"/k8s_cosign_secret.yaml 2>&1 || true
  kubectl -n aegis get rollouts -o yaml > "$OUTDIR"/k8s_rollouts.yaml 2>&1 || true || true
  # capture the first deployment name as a candidate for logs
  ORCH_DEPLOY=$(kubectl -n aegis get deploy -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  if [ -n "$ORCH_DEPLOY" ]; then
    kubectl -n aegis logs deploy/"$ORCH_DEPLOY" --tail=500 > "$OUTDIR"/orchestrator_logs.txt 2>&1 || true
  else
    echo "No deployment found in aegis namespace; skipped orchestrator logs" > "$OUTDIR"/orchestrator_missing.txt
  fi
else
  echo "kubectl not found; skipping k8s checks" > "$OUTDIR"/k8s_missing.txt
fi

# 6) Prometheus alerts (optional, needs PROM_URL)
echo "6) Prometheus alerts (optional)..."
if [ -n "${PROM_URL:-}" ]; then
  curl -sS "${PROM_URL%/}/api/v1/alerts" > "$OUTDIR"/prometheus_alerts.json 2>&1 || true
else
  echo "PROM_URL not set; skipping Prometheus check" > "$OUTDIR"/prometheus_missing.txt
fi

# 7) Verify verifier workflow log reference (optional manual step)
echo "7) Verifier workflow hint..."
echo "If you want verifier workflow logs, provide the GitHub Actions run id(s) or check: gh run list --repo <owner/repo>"

echo
echo "Evidence collection complete."
echo "Files saved to: $OUTDIR"
echo "Summary:"
cat "$OUTDIR"/summary.txt || true
echo
echo "Next: tar czf ${OUTDIR}.tgz $OUTDIR  (optional) and share selected files (tf_outputs.pretty.json, s3_model_archives.txt, k8s_pods.txt, orchestrator_logs.txt, vault_cosign.json, gh_runs.txt) for verification."

exit 0
scripts/collect_aegis_aws_evidence.sh
