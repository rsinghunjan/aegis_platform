#!/usr/bin/env python3
"""
Run an external auditor session for CloudHSM/KMS.

- Uses ops/hsm/audit_automation.py to produce evidence bundles
- Runs live signing checks via the in-cluster HSM signer
- Collects operator checklist answers and produces an auditor_report.json
- Uploads artifacts to S3 (EVIDENCE_BUCKET) when configured
"""
import os
import json
import time
from datetime import datetime
import subprocess
from ops.hsm.audit_automation import produce_bundle as produce_hsm_bundle  # reuses earlier artifact
import boto3

S3 = boto3.client("s3") if os.environ.get("EVIDENCE_BUCKET") else None
EVIDENCE_BUCKET = os.environ.get("EVIDENCE_BUCKET","")
HSM_SIGNER_URL = os.environ.get("HSM_SIGNER_URL","http://aegis-hsm-signer.aegis-system.svc.cluster.local:8085")

def live_sign_check(sample_path, key_label=None):
    # Call the HSM signer directly (HTTP)
    import requests, base64
    blob_b64 = base64.b64encode(open(sample_path,"rb").read()).decode()
    payload = {"blob_b64": blob_b64, "actor": "auditor_session"}
    if key_label:
        payload["key_label"] = key_label
    r = requests.post(HSM_SIGNER_URL + "/v1/sign", json=payload, timeout=60)
    return r.json()

def run_session(sample_blob, kms_key_id=None, cloudhsm_cluster_id=None, operator_notes=None):
    start = datetime.utcnow().isoformat()+"Z"
    bundle_info = produce_hsm_bundle(sample_blob, key_id=kms_key_id, cloudhsm_cluster_id=cloudhsm_cluster_id)
    live_sign = live_sign_check(sample_blob)
    report = {
        "start_ts": start,
        "bundle": bundle_info,
        "live_sign_result": live_sign,
        "operator_notes": operator_notes or "",
        "end_ts": datetime.utcnow().isoformat()+"Z"
    }
    out = "/tmp/auditor_report.json"
    with open(out,"w") as fh:
        json.dump(report, fh, indent=2, default=str)
    if S3:
        key = f"auditor_sessions/{os.path.basename(out)}"
        S3.upload_file(out, EVIDENCE_BUCKET, key)
        report["s3_report"] = f"s3://{EVIDENCE_BUCKET}/{key}"
    return report

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    p.add_argument("--kms-key-id")
    p.add_argument("--cloudhsm-cluster-id")
    p.add_argument("--notes")
    args = p.parse_args()
    r = run_session(args.sample, args.kms_key_id, args.cloudhsm_cluster_id, args.notes)
    print(json.dumps(r, indent=2))
