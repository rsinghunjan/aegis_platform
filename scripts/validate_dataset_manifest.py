#!/usr/bin/env python
"""
Validate dataset manifest(s).

Checks:
- manifest JSON has required fields: version, s3_path or local_path, created_at, checksum (optional), provenance
- if s3_path is present and MINIO/AWS creds are present, check object exists
- if local_path present, check file exists relative to repo root
Usage:
  python scripts/validate_dataset_manifest.py --path data/dataset_manifests --require-latest
Returns non-zero on validation failure.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/dataset_manifests", help="Path containing dataset manifests (dir or file)")
    p.add_argument("--require-latest", action="store_true", help="Require a 'latest' manifest for each dataset")
    p.add_argument("--s3-endpoint", default=os.environ.get("MLFLOW_S3_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT"))
    return p.parse_args()

def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"ERROR: failed to parse JSON {p}: {e}")
        return None

def check_s3_object(s3_uri: str, endpoint: Optional[str]):
    # minimal presence check using boto3 if available
    try:
        import boto3
    except Exception:
        print("NOTICE: boto3 not installed; skipping S3 existence check")
        return True
    from urllib.parse import urlparse
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        print(f"WARNING: s3_path does not look like s3:// URI: {s3_uri}")
        return False
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3", endpoint_url=endpoint,
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        print(f"ERROR: s3 object not accessible: s3://{bucket}/{key} -> {e}")
        return False

def validate_manifest_file(p: Path, args) -> bool:
    j = load_json(p)
    if j is None:
        return False
    required = ["version", "created_at", "provenance"]
    ok = True
    for r in required:
        if r not in j:
            print(f"ERROR: manifest {p} missing required field: {r}")
            ok = False
    # require either s3_path or local_path
    if "s3_path" not in j and "local_path" not in j:
        print(f"ERROR: manifest {p} must include s3_path or local_path")
        ok = False
    if "s3_path" in j:
        if not check_s3_object(j["s3_path"], args.s3_endpoint):
            ok = False
    if "local_path" in j:
        local = Path(j["local_path"])
        if not local.exists():
            print(f"ERROR: manifest {p} references missing local_path: {local}")
            ok = False
    # optional: validate created_at parsable
    try:
        datetime.datetime.fromisoformat(j["created_at"].replace("Z", "+00:00"))
    except Exception:
        print(f"WARNING: manifest {p} has unparseable created_at: {j.get('created_at')}")
    return ok

def main():
    args = parse_args()
    p = Path(args.path)
    if p.is_file():
        ok = validate_manifest_file(p, args)
        sys.exit(0 if ok else 2)
    if not p.exists():
        print(f"ERROR: manifests dir not found: {p}")
        sys.exit(2)
    manifests = list(p.rglob("manifest*.json"))
    if not manifests:
        print(f"ERROR: no manifest JSON files found under {p}")
        sys.exit(2)
    overall = True
    dataset_dirs = set([m.parent for m in manifests])
    for m in manifests:
        print(f"Validating {m}")
        ok = validate_manifest_file(m, args)
        overall = overall and ok
    if args.require_latest:
        for d in dataset_dirs:
            latest = d / "latest_manifest.json"
            if not latest.exists():
                print(f"ERROR: dataset {d} missing latest_manifest.json")
                overall = False
    sys.exit(0 if overall else 3)

if __name__ == "__main__":
    main()
