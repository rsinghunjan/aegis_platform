#!/usr/bin/env python
"""
Check that a promoted MLflow run has required audit tags.

Usage:
  python scripts/check_promotion_audit.py --run-id <RUN_ID> --mlflow-uri http://localhost:5000

Required tags:
 - aegis.registry.registered
 - aegis.registry.model_name
 - aegis.registry.model_version
 - aegis.dataset.version  (optional but recommended)

Exit codes:
 0 = OK
 2 = missing tag(s)
 3 = mlflow client error
"""
import argparse
import sys
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return p.parse_args()

def main():
    args = parse_args()
    try:
        from mlflow.tracking import MlflowClient
    except Exception as e:
        print("ERROR: mlflow not installed:", e)
        return 3
    client = MlflowClient(tracking_uri=args.mlflow_uri)
    try:
        run = client.get_run(args.run_id)
    except Exception as e:
        print("ERROR: failed to fetch run:", e)
        return 3
    tags = run.data.tags
    required = ["aegis.registry.registered", "aegis.registry.model_name", "aegis.registry.model_version"]
    missing = [t for t in required if t not in tags or not tags[t]]
    if missing:
        print("ERROR: Missing required promotion audit tags:", missing)
        print("Available tags:", list(tags.keys()))
        return 2
    print("OK: required audit tags present.")
    # optional dataset tag
    if "aegis.dataset.version" not in tags:
        print("NOTICE: aegis.dataset.version tag missing (recommended to record dataset version used for training).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
