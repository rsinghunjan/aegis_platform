#!/usr/bin/env python3
"""
Validate required Secrets/KMS/ IAM bindings for Aegis.

Purpose:
 - Check required SecretsManager/SSM parameters exist
 - Validate KMS key ARN format and key is enabled
 - Validate CI IAM role has expected KMS permissions (best-effort)
 - Print human-actionable remediation steps when checks fail

Usage:
  export AWS_PROFILE=...
  python ops/validate_secrets_and_iam.py --secret-prefix aegis --ci-role-arn arn:aws:iam::123456789012:role/aegis-ci-role
"""
import argparse
import boto3
import botocore
import re
import sys

REQUIRED_SSM = [
    "/{prefix}/mlflow/tracking_uri",
    "/{prefix}/s3/evidence_bucket"
]
REQUIRED_SECRETS = [
    "{prefix}/cosign",
    "{prefix}/rekor"
]
KMS_ARN_RE = re.compile(r"^arn:aws:kms:[a-z0-9-]+:\d{12}:key/[a-f0-9-]+$")

def check_ssm(prefix, region):
    ssm = boto3.client("ssm", region_name=region)
    missing = []
    for path in REQUIRED_SSM:
        key = path.format(prefix=prefix)
        try:
            ssm.get_parameter(Name=key)
        except botocore.exceptions.ClientError as e:
            missing.append(key)
    return missing

def check_secretsmanager(prefix, region):
    sm = boto3.client("secretsmanager", region_name=region)
    missing = []
    for name in REQUIRED_SECRETS:
        sname = name.format(prefix=prefix)
        try:
            sm.describe_secret(SecretId=sname)
        except botocore.exceptions.ClientError:
            missing.append(sname)
    return missing

def check_kms(kms_arn, region):
    if not kms_arn:
        return False, "KMS ARN not provided"
    if not KMS_ARN_RE.match(kms_arn):
        return False, f"KMS ARN doesn't match expected format: {kms_arn}"
    kms = boto3.client("kms", region_name=region)
    try:
        resp = kms.describe_key(KeyId=kms_arn)
        state = resp["KeyMetadata"]["KeyState"]
        return state == "Enabled", f"KMS key state: {state}"
    except botocore.exceptions.ClientError as e:
        return False, f"KMS describe_key failed: {e}"

def simulate_ci_role_permissions(role_arn, kms_arn, region):
    """
    Best-effort check: we attempt to simulate sts:GetCallerIdentity by assuming role (if possible)
    and call kms:DescribeKey / kms:Sign (DescribeKey is used here as proxy).
    Operator must ensure role is assumable by this caller.
    """
    try:
        sts = boto3.client("sts", region_name=region)
        caller = sts.get_caller_identity()
    except Exception:
        caller = None

    # Try assume role
    try:
        parts = role_arn.split(":")
        if len(parts) < 6:
            return False, "Invalid role ARN"
        arn_parts = role_arn.split("/")
        role_session_name = "aegis-ci-check"
        sts = boto3.client("sts", region_name=region)
        assumed = sts.assume_role(RoleArn=role_arn, RoleSessionName=role_session_name)
        creds = assumed["Credentials"]
        kms = boto3.client("kms", aws_access_key_id=creds["AccessKeyId"],
                           aws_secret_access_key=creds["SecretAccessKey"],
                           aws_session_token=creds["SessionToken"],
                           region_name=region)
        try:
            kms.describe_key(KeyId=kms_arn)
            return True, "CI role can describe KMS key"
        except Exception as e:
            return False, f"CI role cannot access KMS key: {e}"
    except Exception as e:
        return False, f"AssumeRole failed: {e}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secret-prefix", default="aegis")
    p.add_argument("--kms-arn", default="")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--ci-role-arn", default="")
    args = p.parse_args()

    print("[1/4] Checking SSM parameters...")
    missing_ssm = check_ssm(args.secret_prefix, args.region)
    if missing_ssm:
        print("MISSING SSM PARAMETERS:")
        for m in missing_ssm:
            print(" -", m)
    else:
        print("SSM parameters OK")

    print("[2/4] Checking SecretsManager secrets...")
    missing_secrets = check_secretsmanager(args.secret_prefix, args.region)
    if missing_secrets:
        print("MISSING SECRETS:")
        for m in missing_secrets:
            print(" -", m)
    else:
        print("SecretsManager secrets OK")

    print("[3/4] Checking KMS key...")
    ok, msg = check_kms(args.kms_arn, args.region)
    print(" KMS check:", ok, msg)

    if args.ci_role_arn:
        print("[4/4] Simulating CI role permissions (best-effort)...")
        ok2, msg2 = simulate_ci_role_permissions(args.ci_role_arn, args.kms_arn, args.region)
        print(" CI role check:", ok2, msg2)
    else:
        print("No CI role provided; skip CI role permission check.")

    if missing_ssm or missing_secrets or not ok:
        print("\nACTION REQUIRED: fix missing secrets/KMS before deploying Aegis.")
        sys.exit(2)
    print("\nValidation completed successfully.")

if __name__ == "__main__":
    main()
