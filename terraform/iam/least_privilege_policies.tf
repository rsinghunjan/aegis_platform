# Terraform snippets: least-privilege IAM policies for GitHub Actions OIDC and service roles.
# Usage: integrate these into your terraform/irsa module or call via aws_iam_policy / aws_iam_role_attachment.
#
# NOTE: Replace placeholders and scope ARNs before applying. Do NOT commit sensitive values.

# Policy: allow ECR push/pull (narrow to specific registry)
data "aws_caller_identity" "current" {}

variable "ecr_registry_account" {
  type    = string
  default = data.aws_caller_identity.current.account_id
}

data "aws_iam_policy_document" "github_actions_oidc_trust" {
  statement {
    effect = "Allow"
    principals {
      type        = "Federated"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"]
    }
    actions = ["sts:AssumeRoleWithWebIdentity"]
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

# ECR push policy (narrow)
data "aws_iam_policy_document" "ecr_push_policy" {
  statement {
    effect = "Allow"
    actions = [
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:GetAuthorizationToken"
    ]
    resources = ["arn:aws:ecr:${var.region}:${var.ecr_registry_account}:repository/aegis/*"]
  }
}

# KMS Sign policy (for cosign)
data "aws_iam_policy_document" "kms_sign_policy" {
  statement {
    effect = "Allow"
    actions = [
      "kms:Sign",
      "kms:Verify",
      "kms:DescribeKey",
      "kms:GetPublicKey",
    ]
    resources = ["arn:aws:kms:${var.region}:${var.ecr_registry_account}:key/REPLACE_COSIGN_KEY_ID"]
  }
}

# SecretsManager read for ExternalSecrets operator role (narrow)
data "aws_iam_policy_document" "secretsmanager_read_policy" {
  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
      "secretsmanager:ListSecretVersionIds"
    ]
    resources = [
      "arn:aws:secretsmanager:${var.region}:${var.ecr_registry_account}:secret:aegis/*"
    ]
  }
}

# S3 access for evidence bucket (narrow)
data "aws_iam_policy_document" "s3_evidence_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:DeleteObject"
    ]
    resources = [
      "arn:aws:s3:::REPLACE_EVIDENCE_BUCKET",
      "arn:aws:s3:::REPLACE_EVIDENCE_BUCKET/*"
    ]
  }
}

output "docs" {
  value = "Use these policy documents to create roles with least privilege. Replace placeholders before applying."
}
