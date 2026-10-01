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
#!/usr/bin/env python
"""
Ingestion script for Feast demo:

- Creates synthetic user_events parquet file in feast/feature_repo/data/
- Applies the feature definitions (feast apply)
- Materializes features into the online store (Redis) for a recent time window

Run:
  python feast/feature_repo/ingest.py
"""
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from feast import FeatureStore

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PARQUET_PATH = os.path.join(DATA_DIR, "user_events.parquet")

def create_synthetic_events(n_users=50, days=7):
    rows = []
    now = datetime.utcnow()
    for user_id in range(1, n_users + 1):
        # generate a few events per user
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
    # Feast expects event_ts as timestamp column (arrow friendly)
    return df

def main():
    print("Creating synthetic events parquet:", PARQUET_PATH)
    df = create_synthetic_events(n_users=200, days=14)
    df.to_parquet(PARQUET_PATH, index=False)
    print("Wrote sample events:", df.shape)

    # Initialize FeatureStore pointing to this repo (feature_store.yaml in same folder)
    fs = FeatureStore(repo_path=os.path.dirname(__file__))

    print("Applying feature repo (this will register feature definitions)...")
    fs.apply([os.path.join(os.path.dirname(__file__), "features.py")])

    # Materialize recent features to online store
    end_ts = datetime.utcnow()
    start_ts = end_ts - timedelta(hours=24)
    print(f"Materializing features to online store from {start_ts} to {end_ts} ...")
    fs.materialize(start_date=start_ts, end_date=end_ts)
    print("Materialization completed. You can now query online features (Feast online store).")

if __name__ == "__main__":
    main()
