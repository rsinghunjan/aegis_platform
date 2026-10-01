"""
Minimal feature extraction & store (parquet files) example.

- Loads raw images or CSVs, computes simple features, writes parquet to s3://<bucket>/<prefix>/features/<snapshot>/
- Records metadata JSON with feature schema & source snapshot id.

Usage:
  OBJECT_STORE_BUCKET=... python3 features/feature_store.py --input-s3 s3://.../datasets/cifar/<snapshot> --out-prefix features/cifar
"""
import argparse, os, json, tempfile
from pathlib import Path
import pandas as pd
import boto3
import pyarrow as pa, pyarrow.parquet as pq
import numpy as np

S3 = boto3.client("s3")

def list_s3_keys(s3_uri):
    bucket, prefix = s3_uri.replace("s3://","").split("/",1)
    paginator = S3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return bucket, prefix, keys

def download_to_dir(s3_uri, dst: Path):
    bucket, prefix = s3_uri.replace("s3://","").split("/",1)
    dst.mkdir(parents=True, exist_ok=True)
    for key in list_s3_keys(s3_uri)[2]:
        rel = Path(key).relative_to(prefix)
        local = dst/rel
        local.parent.mkdir(parents=True, exist_ok=True)
        S3.download_file(bucket, key, str(local))
    return dst

def compute_simple_features(df: pd.DataFrame):
    # Assume df has numeric columns; compute mean/std per row or simple augmentation
    feats = pd.DataFrame({
        "row_mean": df.mean(axis=1),
        "row_std": df.std(axis=1),
    })
    return feats

def upload_parquet(df: pd.DataFrame, bucket: str, key: str):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)/"out.parquet"
        df.to_parquet(p, index=False)
        S3.upload_file(str(p), bucket, key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-s3", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--snapshot-id", required=True)
    args = ap.parse_args()
    bucket = os.environ.get("OBJECT_STORE_BUCKET")
    if not bucket:
        raise SystemExit("Set OBJECT_STORE_BUCKET")
    tmp = Path("/tmp/features")
    if args.input_s3.startswith("s3://"):
        download_to_dir(args.input_s3, tmp)
        # Simplify: look for CSV files and concatenate
        dfs = []
        for f in tmp.rglob("*.csv"):
            dfs.append(pd.read_csv(f))
        if not dfs:
            raise SystemExit("No CSVs found under input")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(args.input_s3)

    feats = compute_simple_features(df)
    key = f"{args.out_prefix}/{args.snapshot_id}/features.parquet"
    print("Uploading features to s3://%s/%s" % (bucket, key))
    upload_parquet(feats, bucket, key)
    meta = {
        "snapshot_id": args.snapshot_id,
        "feature_cols": list(feats.columns),
        "rows": len(feats)
    }
    S3.put_object(Bucket=bucket, Key=f"{args.out_prefix}/{args.snapshot_id}/features.meta.json", Body=json.dumps(meta).encode())
    print("Feature extraction complete.")
    
if __name__ == "__main__":
    main()
