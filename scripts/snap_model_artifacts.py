#!/usr/bin/env python3
"""
Snapshot the model_registry/ artifacts and upload tar.gz to S3.

Usage:
  export MODEL_REGISTRY_DIR=model_registry
  export BACKUP_S3_BUCKET=s3://my-bucket/aegis-backups
  python scripts/snap_model_artifacts.py
"""
import os
import tarfile
import tempfile
import hashlib
import boto3
from datetime import datetime

MODEL_DIR = os.environ.get("MODEL_REGISTRY_DIR", "model_registry")
S3_BUCKET = os.environ.get("BACKUP_S3_BUCKET", "")
TS = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
ARCHIVE_NAME = f"aegis-models-{TS}.tar.gz"

if not S3_BUCKET:
    raise SystemExit("BACKUP_S3_BUCKET not set")

tmp = tempfile.mkdtemp()
archive_path = os.path.join(tmp, ARCHIVE_NAME)

with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(MODEL_DIR, arcname=os.path.basename(MODEL_DIR))

# compute sha256
h = hashlib.sha256()
with open(archive_path, "rb") as fh:
    for chunk in iter(lambda: fh.read(8192), b""):
        h.update(chunk)
checksum = h.hexdigest()

# Upload via boto3
s3 = boto3.client("s3")
# support s3://bucket/prefix
if S3_BUCKET.startswith("s3://"):
    _, _, path = S3_BUCKET.partition("s3://")
    bucket, _, prefix = path.partition("/")
else:
    raise SystemExit("S3 bucket must be s3://bucket[/prefix]")

key = f"{prefix.rstrip('/')}/models/{ARCHIVE_NAME}" if prefix else f"models/{ARCHIVE_NAME}"
print("Uploading", archive_path, "->", f"s3://{bucket}/{key}")
s3.upload_file(archive_path, bucket, key, ExtraArgs={"StorageClass": "STANDARD_IA"})
# upload checksum
s3.put_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/models/{ARCHIVE_NAME}.sha256" if prefix else f"models/{ARCHIVE_NAME}.sha256", Body=checksum)

print("Uploaded. checksum:", checksum)
