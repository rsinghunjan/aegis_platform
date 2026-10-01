"""
Example: log a model artifact and metadata to MLflow.

Usage:
  # set MLflow tracking URI, e.g.
  export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
  python examples/log_model_mlflow.py
"""
import json
import os
import time
import mlflow

def main():
    mlflow.set_experiment("aegis-model-registry-examples")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Example params/metrics
        mlflow.log_param("model_name", "example-model")
        mlflow.log_metric("val_accuracy", 0.92)
        # Log a dummy artifact file
        os.makedirs("tmp_artifacts", exist_ok=True)
        model_path = "tmp_artifacts/example_model.bin"
        with open(model_path, "wb") as fh:
            fh.write(b"dummy-model-bytes")
        mlflow.log_artifact(model_path, artifact_path="model")
        # Save metadata (example)
        metadata = {
            "name": "example-model",
            "version": "0.1.0",
            "source": {"git_url": "https://github.com/your/repo", "git_commit": "abcdef"},
            "dataset_manifest": "data/dataset-manifest.yaml",
            "training_hash": "sha256:deadbeef",
            "license": "Apache-2.0",
            "artifact": "model/example_model.bin",
            "metrics": {"val_accuracy": 0.92},
            "created_by": "your-team",
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "notes": "Example model logged via MLflow"
        }
        with open("tmp_artifacts/metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)
        mlflow.log_artifact("tmp_artifacts/metadata.json", artifact_path="model")
        print("Logged model artifacts and metadata to MLflow. run_id:", run_id)

if __name__ == "__main__":
    main()
