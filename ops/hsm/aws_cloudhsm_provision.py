#!/usr/bin/env python3
"""
Provision AWS CloudHSM cluster and create a KMS Customer Managed Key backed by CloudHSM
(Operator-run script; requires awscli boto3 credentials with necessary permissions).

High level steps (this script attempts them but operator must validate networking & HSM appliance setup):
  1. create CloudHSM cluster
  2. wait until cluster is ACTIVE
  3. create a cluster membership (requires manual user certificate upload step - not automated here)
  4. create a KMS Custom Key Store associated with CloudHSM (may require on-console steps)
  5. create a KMS asymmetric key (SIGN_VERIFY) with Origin=AWS_KMS using the custom key store

This script is best-effort and emits instructions for manual steps auditors will want to observe.
"""
import os
import time
import json
import boto3
from datetime import datetime, timezone

HSM = boto3.client("cloudhsmv2")
KMS = boto3.client("kms")
CT = boto3.client("cloudtrail")

REGION = os.environ.get("AWS_REGION", "us-east-1")
TAG = os.environ.get("HSM_TAG", "aegis-cloudhsm")

def create_cluster(subnet_id=None):
    kwargs = {"TagList": [{"Key": "Name", "Value": TAG}], "HsmType":"hsm1.medium"}
    if subnet_id:
        kwargs["SubnetIds"] = [subnet_id]
    resp = HSM.create_cluster(**kwargs)
    return resp["Cluster"]

def wait_for_cluster(cluster_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        resp = HSM.describe_clusters(Filters=[{"Name":"clusterIds","Values":[cluster_id]}])
        c = resp.get("Clusters", [])[0]
        status = c.get("State")
        print("cluster", cluster_id, "state", status)
        if status == "ACTIVE":
            return c
        time.sleep(20)
    raise RuntimeError("timeout waiting for cluster active")

def create_custom_key_store(cluster_arn, key_store_alias):
    # Create a KMS custom key store mapping to CloudHSM cluster
    resp = KMS.create_custom_key_store(
        CustomKeyStoreName=key_store_alias,
        CloudHsmClusterId=cluster_arn,
        TrustAnchorCertificate="",
        XksProxyUriEndpoint=""
    )
    return resp

def create_kms_key(alias, key_usage="SIGN_VERIFY", origin="AWS_KMS"):
    resp = KMS.create_key(Description=f"Aegis PQC key {alias}", KeyUsage=key_usage, Origin=origin)
    kid = resp["KeyMetadata"]["KeyId"]
    try:
        KMS.create_alias(AliasName=f"alias/{alias}", TargetKeyId=kid)
    except Exception:
        pass
    return resp["KeyMetadata"]

def export_cloudtrail_events(resource_arn, lookback_minutes=60):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    events = []
    try:
        resp = CT.lookup_events(
            LookupAttributes=[{"AttributeKey":"ResourceName","AttributeValue":resource_arn}],
            StartTime=start,
            EndTime=end,
            MaxResults=50
        )
        events = resp.get("Events", [])
    except Exception:
        events = []
    return events

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--subnet", help="subnet id for HSM cluster (recommended)")
    p.add_argument("--alias", default="aegis-pqc-kms")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()

    print("Creating CloudHSM cluster (this may take ~15-25 minutes)...")
    cluster = create_cluster(args.subnet)
    cluster_id = cluster["ClusterId"]
    print("cluster created id:", cluster_id)
    if args.wait:
        print("waiting for cluster ACTIVE...")
        active = wait_for_cluster(cluster_id)
        print("cluster active:", active["ClusterId"])
        print("IMPORTANT: You must complete the CloudHSM initialization step (initialize HSMs and add HSM users).")
        print("See AWS CloudHSM docs and provide the HSMs to auditors for verification.")
    print("Creating KMS key (best-effort) - operator may need to create a CustomKeyStore in console if prerequisites missing")
    try:
        kms_meta = create_kms_key(args.alias)
        print("KMS key created:", kms_meta.get("KeyId"))
        print("Cosign KMS URI: awskms:///alias/{}".format(args.alias))
    except Exception as e:
        print("KMS key creation failed (operator action required):", e)
        print("If using CloudHSM-backed KMS, follow AWS steps to create a custom key store in KMS and create key referencing it.")

if __name__ == "__main__":
    main()
