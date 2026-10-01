#!/usr/bin/env python3
"""
Simple dataset snapshot & metadata writer.

- Downloads/collects dataset (or copies from a source s3 URI).
- Computes checksum and writes a small JSON manifest with git commit + metadata.
- Uploads dataset + manifest to staging S3 path: s3://<bucket>/<prefix>/<snapshot-id>/

Usage:
  export OBJECT_STORE_BUCKET=REPLACE_WITH_BUCKET
  python3 scripts/data_ingest.py --src s3://my-source-bucket/datasets/cifar --prefix datasets/cifar
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
import shutil
import uuid
import datetime
import boto3

S3 = boto3.client("s3")

def compute_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def download_s3_prefix(s3_uri: str, dst: Path):
    # naive: assumes s3://bucket/prefix/ and downloads objects under that prefix
    parts = s3_uri.replace("s3://","").split("/",1)
    bucket = parts[0]; prefix = parts[1] if len(parts)>1 else ""
    paginator = S3.get_paginator("list_objects_v2")
    dst.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents",[]):
            key = obj["Key"]
            rel = Path(key).relative_to(prefix)
            local = dst / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            S3.download_file(bucket, key, str(local))
    return dst

def git_commit_sha():
    try:
        out = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
        return out
    except Exception:
        return ""

def upload_dir_to_s3(local_dir: Path, bucket: str, s3_prefix: str):
    for p in local_dir.rglob("*"):
        if p.is_file():
            key = f"{s3_prefix}/{p.relative_to(local_dir)}"
            S3.upload_file(str(p), bucket, key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="s3://... or local path to dataset root")
    ap.add_argument("--prefix", required=True, help="destination prefix in OBJECT_STORE_BUCKET (e.g. datasets/cifar)")
    args = ap.parse_args()
    bucket = os.environ.get("OBJECT_STORE_BUCKET")
    if not bucket:
        raise SystemExit("OBJECT_STORE_BUCKET env var not set")

    snapshot_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        local_src = td / "data"
        if args.src.startswith("s3://"):
            print("Downloading from S3 source:", args.src)
            download_s3_prefix(args.src, local_src)
        else:
            shutil.copytree(args.src, local_src)
        # compute checksum of concatenated file hashes (simple deterministic manifest)
        hashes = {}
        for f in sorted([p for p in local_src.rglob("*") if p.is_file()]):
            hashes[str(f.relative_to(local_src))] = compute_sha256(f)
        manifest = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit_sha(),
            "file_hashes": hashes
        }
        manifest_path = td + "/manifest.json"
        with open(manifest_path,"w") as fh:
            json.dump(manifest, fh, indent=2)
        s3_prefix = f"{args.prefix}/{snapshot_id}"
        print("Uploading snapshot to s3://%s/%s" % (bucket, s3_prefix))
        upload_dir_to_s3(local_src, bucket, s3_prefix)
        S3.upload_file(manifest_path, bucket, f"{s3_prefix}/manifest.json")
        print("Snapshot uploaded:", s3_prefix)
        print("Manifest:", manifest_path)

if __name__ == "__main__":
    main()
