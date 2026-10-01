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
 84
 85
 86
 87
 88
 89
#!/usr/bin/env python
"""
Model quality & drift monitor.

Overview
- Loads baseline feature histograms (local JSON or s3:// URI)
- Loads a recent sample of live data (local CSV or s3:// URI). Live data should include features used
  for drift checks and optionally a 'label' column for scoring.
- Computes PSI (Population Stability Index) per numeric feature and a simple aggregate drift_score.
- Computes accuracy if label and prediction columns are present.
- Pushes metrics to Prometheus Pushgateway (PUSHGATEWAY_URL) so Prometheus can scrape/alert.
- Emits a JSON report to OUTPUT_PATH for auditing.

Configuration (env / CLI)
- BASELINE_STATS: local path or s3:// bucket/key to baseline_stats.json (generated with generate_baseline_stats.py)
- LIVE_SAMPLES: local CSV or s3:// path with recent samples (prediction-time window)
- FEATURES: comma-separated list of features to compute PSI for (defaults to all in baseline)
- PREDICTION_COL: name of predicted label column (optional)
- LABEL_COL: name of ground-truth label column (optional)
- PUSHGATEWAY_URL: e.g. http://pushgateway:9091 (if omitted metrics are not pushed)
- JOB_NAME: job name used when pushing metrics (defaults to aegis-model-monitor)
- MODEL_NAME: model identifier (tagged on metrics)
- OUTPUT_PATH: where to write the JSON report (default: ./drift_report.json)
- BUCKET_ENDPOINT_URL: optional S3 endpoint URL for MinIO in dev (e.g. http://localhost:9001)
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("aegis.monitor_drift")
logging.basicConfig(level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=os.environ.get("BASELINE_STATS"))
    p.add_argument("--live", default=os.environ.get("LIVE_SAMPLES"))
    p.add_argument("--features", default=os.environ.get("FEATURES", ""))
    p.add_argument("--prediction-col", default=os.environ.get("PREDICTION_COL", "prediction"))
    p.add_argument("--label-col", default=os.environ.get("LABEL_COL", "label"))
    p.add_argument("--pushgateway", default=os.environ.get("PUSHGATEWAY_URL"))
    p.add_argument("--job-name", default=os.environ.get("JOB_NAME", "aegis-model-monitor"))
    p.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "unknown-model"))
    p.add_argument("--output", default=os.environ.get("OUTPUT_PATH", "drift_report.json"))
    p.add_argument("--s3-endpoint", default=os.environ.get("BUCKET_ENDPOINT_URL"))
    return p.parse_args()

# ==== Utilities: S3 helpers (minimal) ====
def is_s3_path(p: str) -> bool:
    return p is not None and p.startswith("s3://")

def s3_read_json(s3_uri: str, endpoint_url: Optional[str] = None) -> Dict:
    import boto3
    from urllib.parse import urlparse
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

def s3_read_csv(s3_uri: str, endpoint_url: Optional[str] = None) -> pd.DataFrame:
    import boto3
    from urllib.parse import urlparse
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])

# ==== Drift metric helpers ====
def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    expected / actual are frequency arrays for the same bins (sum to the same total or normalized).
    PSI = sum((expected

scripts/monitor_drift.py
