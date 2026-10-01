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
#!/usr/bin/env python
"""
Example training script instrumented with MLflow.

- Trains a tiny sklearn classifier on iris dataset
- Logs params, metrics, and a model artifact to MLflow
- Prints the MLflow run_id for downstream registration

Usage:
  MLFLOW_TRACKING_URI=http://localhost:5000 python examples/train_with_mlflow.py --experiment demo --run-name trial1
"""
import os
import argparse
import time
import json
from datetime import datetime

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    mlflow_tracking = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        # params
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("script", "examples/train_with_mlflow.py")

        # load data & train
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(C=args.C, max_iter=args.max_iter)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        mlflow.log_metric("accuracy", acc)

        # save model artifact
        artifact_dir = "artifact"
        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # write a small provenance file
        prov = {
            "run_id": run_id,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "params": {"C": args.C, "max_iter": args.max_iter}
        }
        with open(os.path.join(artifact_dir, "provenance.json"), "w") as fh:
            json.dump(prov, fh)
        mlflow.log_artifact(os.path.join(artifact_dir, "provenance.json"), artifact_path="model")

        print("MLflow run_id:", run_id)
        # keep run context alive for a tiny bit to ensure artifacts flushed
        time.sleep(1)

if __name__ == "__main__":
    main()
