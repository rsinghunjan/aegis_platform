#!/usr/bin/env bash
set -euo pipefail
#
# aws_tenant_commands.sh
#
# Purpose:
# - Final, copy/paste-ready commands and artifacts to finalize AWS tenant wiring for Aegis.
# - Creates (or instructs creation of) OIDC trust role for GitHub Actions, applies a least-privilege inline policy
#   for KMS sign + S3 access, optionally creates an asymmetric KMS signing key, and prints GitHub secret creation commands.
#
# Safety:
# - This script only prints commands by default. To apply, review each command and run them in your operator shell.
# - Replace placeholders (ACCOUNT_ID, ORG, REPO, BUCKET, REGION, GH_REPO) before running.
#
# Usage (recommended):
# 1. Edit variables below with your values.
# 2. Run: ./aws_tenant_commands.sh
#
# If you want the script to actually call AWS, set APPLY=true (not recommended until you review).

APPLY=false         # Set to true to run AWS CLI commands automatically from this script (review first)
CREATE_KMS=false    # Set to true to create an asymmetric KMS key for signing
GH_AUTO=false       # If true and GH_TOKEN env var is present, will call gh secret set to create secrets (use locally)

# --- CONFIGURE THESE VALUES BEFORE RUNNING ---
ACCOUNT_ID="${ACCOUNT_ID:-123456789012}"        # your AWS account id
ORG="${ORG:-your-gh-org}"                       # GitHub org
REPO="${REPO:-your-gh-repo}"                    # GitHub repo
GH_REPO="${GH_REPO:-${ORG}/${REPO}}"            # gh repo string
REGION="${REGION:-us-east-1}"                   # AWS region
S3_BUCKET="${S3_BUCKET:-aegis-model-repo}"      # bucket used for artifacts / lakeFS staging
ROLE_NAME="${ROLE_NAME:-aegis-github-oidc-${ORG}-${REPO}}"
# ------------------------------------------------

OIDC_PROVIDER="token.actions.githubusercontent.com"

echo "=== Aegis AWS tenant finalization helper ==="
echo "Review the variables at the top of the script before running."
echo

# 1) OIDC provider check
echo "1) Verify OIDC provider exists in AWS account (should be token.actions.githubusercontent.com):"
echo "   aws iam list-open-id-connect-providers --output text"
echo
if $APPLY; then
  aws iam list-open-id-connect-providers --output text || true
fi

# 2) Create trust policy JSON for GitHub Actions OIDC
TRUST_JSON="/tmp/aegis-trust-${ROLE_NAME}.json"
cat > "$TRUST_JSON" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}" },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:${ORG}/${REPO}:ref:refs/heads/main"
        }
      }
    }
  ]
}
EOF

echo "Trust policy written to: $TRUST_JSON"
echo "Contents (review):"
cat "$TRUST_JSON"
echo

# 3) Commands to create the IAM role
echo "2) IAM role creation commands (review then run):"
echo "   aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document file://${TRUST_JSON}"
echo
if $APPLY; then
  aws iam create-role --role-name "${ROLE_NAME}" --assume-role-policy-document "file://${TRUST_JSON}" || true
fi

# 4) Inline policy granting least-privilege actions (example; tighten ARNs)
INLINE_POLICY="/tmp/aegis-inline-policy-${ROLE_NAME}.json"
cat > "$INLINE_POLICY" <<EOF
{
  "Version":"2012-10-17",
  "Statement":[
    {
      "Effect":"Allow",
      "Action":[ "kms:Sign", "kms:GetPublicKey", "kms:DescribeKey" ],
      "Resource":"*"
    },
    {
      "Effect":"Allow",
      "Action":[ "s3:GetObject", "s3:PutObject", "s3:ListBucket" ],
      "Resource":[ "arn:aws:s3:::${S3_BUCKET}", "arn:aws:s3:::${S3_BUCKET}/*" ]
    }
  ]
}
EOF

echo "Inline policy written to: $INLINE_POLICY"
echo "Attach inline policy with:"
echo "   aws iam put-role-policy --role-name ${ROLE_NAME} --policy-name AegisOIDCPolicy --policy-document file://${INLINE_POLICY}"
echo
if $APPLY; then
  aws iam put-role-policy --role-name "${ROLE_NAME}" --policy-name "AegisOIDCPolicy" --policy-document "file://${INLINE_POLICY}"
fi

# 5) (Optional) Create asymmetric KMS key for signing
if $CREATE_KMS; then
  echo "3) Create asymmetric KMS key (ECC_NIST_P384) for signing (recommended for cosign verify with public key)"
  echo "   aws kms create-key --customer-master-key-spec ECC_NIST_P384 --key-usage SIGN_VERIFY --description 'Aegis signing key'"
  if $APPLY; then
    KMS_KEY_ID=$(aws kms create-key --customer-master-key-spec ECC_NIST_P384 --key-usage SIGN_VERIFY --description "Aegis signing key" --query KeyMetadata.KeyId -o text)
    echo "Created KMS KeyId: ${KMS_KEY_ID}"
    KMS_KEY_ARN=$(aws kms describe-key --key-id "$KMS_KEY_ID" --query KeyMetadata.Arn -o text)
    echo "KMS ARN: $KMS_KEY_ARN"
  fi
else
  echo "Skipping KMS key creation. If you want creation, re-run with CREATE_KMS=true."
fi
echo

# 6) Print GitHub secrets to set
echo "4) GitHub repository secrets to set (copy & create as repository secrets):"
ROLE_ARN_CMD="aws iam get-role --role-name ${ROLE_NAME} --query Role.Arn --output text"
if $APPLY; then
  ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn -o text)
else
  ROLE_ARN="<REPLACE_WITH_ROLE_ARN>"
fi
KMS_ARN="${KMS_KEY_ARN:-<REPLACE_WITH_KMS_ARN>}"

echo "Secrets:"
echo "  AWS_ROLE_ARN=${ROLE_ARN}"
echo "  AWS_REGION=${REGION}"
echo "  KMS_KEY_ARN=${KMS_ARN}"
echo "  S3_BUCKET=${S3_BUCKET}"
echo

echo "To create these using gh CLI locally:"
echo "  gh secret set AWS_ROLE_ARN --repo ${GH_REPO} --body \"${ROLE_ARN}\""
echo "  gh secret set AWS_REGION --repo ${GH_REPO} --body \"${REGION}\""
echo "  gh secret set KMS_KEY_ARN --repo ${GH_REPO} --body \"${KMS_ARN}\""
echo "  gh secret set S3_BUCKET --repo ${GH_REPO} --body \"${S3_BUCKET}\""
echo

if $GH_AUTO && [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "Creating GH secrets via gh CLI (using GITHUB_TOKEN in environment)..."
  gh secret set AWS_ROLE_ARN --repo "${GH_REPO}" --body "${ROLE_ARN}"
  gh secret set AWS_REGION --repo "${GH_REPO}" --body "${REGION}"
  gh secret set KMS_KEY_ARN --repo "${GH_REPO}" --body "${KMS_ARN}"
  gh secret set S3_BUCKET --repo "${GH_REPO}" --body "${S3_BUCKET}"
  echo "GitHub secrets created."
fi

# 7) Validation: simulate IAM permissions and test OIDC exchange (recommended)
echo "5) Validation steps (manual):"
echo "  a) Use AWS policy simulator to validate permissions for the role:"
echo "     aws iam simulate-principal-policy --policy-source-arn ${ROLE_ARN} --action-names kms:Sign s3:PutObject"
echo "  b) Create a small GitHub Actions workflow in the repo that uses OIDC to assume the role and runs a simple aws sts get-caller-identity"
cat > /tmp/aegis-oidc-test.yml <<'YAML'
name: OIDC assume-role test
on:
  workflow_dispatch:
jobs:
  assume:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Assume role via OIDC
        id: assume
        run: |
          ROLE_ARN="<ROLE_ARN>"
          AWS_REGION="<REGION>"
          # exchange OIDC token for temp creds
          echo "Testing STS assume-role-with-web-identity..."
          # Note: this sample uses the aws cli default behavior in GitHub Actions
          # Replace ROLE_ARN before use
          aws sts assume-role-with-web-identity --role-arn $ROLE_ARN --role-session-name gha-oidc-test --web-identity-token ${{ steps.get_token.outputs.id_token }} --duration-seconds 900
YAML
echo "    (saved sample workflow to /tmp/aegis-oidc-test.yml — replace placeholders and add to .github/workflows for testing)"
echo

echo "== Done: Review printed commands, replace placeholders, and run them locally. =="
