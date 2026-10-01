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
#!/usr/bin/env python
"""
Training example that demonstrates retrieving historical features from Feast
for offline training, then training a scikit-learn model.

Usage:
  python examples/train_with_features.py

Prereqs:
  - Run: python feast/feature_repo/ingest.py (creates sample data and materializes to online store)
  - Ensure MLFLOW or other tracking not required for this demo
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from feast import FeatureStore
except Exception as exc:
    raise RuntimeError("Feast not installed - pip install -r feast/requirements-feast.txt") from exc

def main():
    repo_path = os.path.join("feast", "feature_repo")
    fs = FeatureStore(repo_path=repo_path)

    # Build an entity dataframe for historical retrieval.
    # For demo, create an entities dataframe with user_id and event_timestamp for each
    import datetime
    now = datetime.datetime.utcnow()
    users = pd.DataFrame({
        "user_id": np.arange(1, 201),  # match ingest.py user ids
        "event_timestamp": [now - datetime.timedelta(hours=1)] * 200
    })

    # Request historical features (joins features from offline store to entity dataframe)
    feature_refs = [
        "user_features:avg_session_length",
        "user_features:last_purchase_amount",
        "user_features:purchase_count_30d",
    ]
    print("Requesting historical features:", feature_refs)
    training_df = fs.get_historical_features(entity_df=users, features=feature_refs).to_df()
    # The result includes event_timestamp and entity columns
    print("Retrieved training dataframe shape:", training_df.shape)
    # Pivot/prepare dataset: simple example treat purchase_count_30d > 0 as label
    training_df["label"] = (training_df["user_features__purchase_count_30d"] > 0).astype(int)
    X = training_df[[
        "user_features__avg_session_length",
        "user_features__last_purchase_amount",
        "user_features__purchase_count_30d"
    ]].fillna(0.0).values
    y = training_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Validation accuracy:", acc)

    # Save model artifact locally
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "sk_model.joblib")
    joblib.dump(model, model_path)
    print("Saved model to", model_path)

if __name__ == "__main__":
    main()
