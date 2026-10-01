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
#!/usr/bin/env python
"""
Create synthetic events parquet, upload to S3 (MinIO) with a versioned path,
write a small manifest.json with version metadata, apply Feast definitions and materialize.

Usage (local dev):
  # start minio + redis:
  docker compose -f docker/docker-compose-feast.yml up -d

  # run ingestion
  python feast/feature_repo/ingest_to_s3.py --bucket feast-offline --prefix aegis --versioning latest

Notes:
- Requires boto3 and s3fs (feast offline uses s3 URIs).
- The script uploads to s3://{bucket}/{prefix}/datasets/user_events/{version}/user_events.parquet
- It also writes s3://{bucket}/{prefix}/datasets/user_events/{version}/manifest.json
- When versioning=='latest' the script copies the dataset to a 'latest/' location for easy Feast FileSource referencing.
"""
import os
import argparse
import json
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import boto3
import botocore
from feast import FeatureStore

ROOT = os.path.dirname(os.path.abspath(__file__))

def create_synthetic_events(n_users=200, days=7):
    rows = []
    now = datetime.utcnow()
    for user_id in range(1, n_users + 1):
        for _ in range(np.random.randint(1, 8)):
            event_ts = now - timedelta(days=np.random.rand() * days)
            created_ts = event_ts + timedelta(seconds=np.random.randint(0, 300))
            avg_session = float(np.random.rand() * 30.0)
            last_purchase = float(np.random.choice([0.0, np.random.rand() * 200.0]))
            purchase_count_30d = int(np.random.poisson(1.0))
            rows.append({
                "user_id": int(user_id),
                "event_ts": event_ts,
                "created_ts": created_ts,
                "avg_session_length": avg_session,
                "last_purchase_amount": last_purchase,
                "purchase_count_30d": purchase_count_30d,
            })
    df = pd.DataFrame(rows)
    return df

def upload_file_to_s3(local_path: str, bucket: str, s3_key: str, endpoint_url: str = None):
    s3 = boto3.client('s3',
                      endpoint_url=endpoint_url,
                      aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID")),
