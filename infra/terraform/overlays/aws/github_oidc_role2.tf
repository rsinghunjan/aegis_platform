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
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Variables (see variables.tf in same overlay)
# var.github_org, var.github_repo, var.branch_filter (optional)

data "aws_iam_policy_document" "github_oidc_assume" {
  statement {
    effect = "Allow"

    principals {
      type        = "Federated"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"]
    }

    actions = ["sts:AssumeRoleWithWebIdentity"]

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_org}/${var.github_repo}:ref:refs/heads/${var.branch_filter}"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_iam_role" "github_oidc_role" {
  name               = "${var.project_prefix}-gha-verifier"
  assume_role_policy = data.aws_iam_policy_document.github_oidc_assume.json
  description        = "Role assumed by GitHub Actions OIDC for Aegis verifier & terraform (minimal perms)"
  tags = {
    Project = var.project_prefix
    Env     = var.env
  }
}

# Minimal policy for S3 read access to the staging bucket(s)
data "aws_iam_policy_document" "s3_read" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [
      "arn:aws:s3:::${var.staging_bucket}",
      "arn:aws:s3:::${var.staging_bucket}/*",
    ]
  }
}

resource "aws_iam_policy" "s3_read_policy" {
  name        = "${var.project_prefix}-gha-s3-read"
  description = "Read-only access to staging model bucket for verifier"
  policy      = data.aws_iam_policy_document.s3_read.json
}

resource "aws_iam_role_policy_attachment" "attach_s3_read" {
  role       = aws_iam_role.github_oidc_role.name
  policy_arn = aws_iam_policy.s3_read_policy.arn
}

output "github_oidc_role_arn" {
  value = aws_iam_role.github_oidc_role.arn
}
infra/terraform/overlays/aws/github_oidc_role.tf
