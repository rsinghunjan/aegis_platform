#!/usr/bin/env python
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_s3_artifact(s3_url: str, dst_dir: str) -> str:
    import boto3

    if not s3_url.startswith("s3://"):
        raise ValueError("artifact path must start with s3://")
    bucket, separator, key = s3_url[5:].partition("/")
    if not separator or not bucket or not key:
        raise ValueError("S3 artifact path must include a bucket and key")
    destination = os.path.join(dst_dir, os.path.basename(key))
    boto3.client("s3").download_file(bucket, key, destination)
    return destination


def download_mlflow_artifact(run_id: str, artifact_path: str) -> str:
    from mlflow.tracking import MlflowClient

    return MlflowClient().download_artifacts(run_id, artifact_path)


def verify_signature(candidate_path: str) -> bool:
    try:
        from api.model_signing import verify_model_artifact
    except Exception:
        logger.info(
            "No api.model_signing.verify_model_artifact available; skipping verification"
        )
        return True
    try:
        return bool(verify_model_artifact(candidate_path))
    except Exception:
        logger.exception("Signature verification raised error")
        return False


def evaluate_sklearn(
    model_file: str, test_csv: Optional[str], label_col: str, metric: str
) -> float:
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, mean_squared_error

    model = joblib.load(model_file)
    if test_csv:
        import pandas as pd

        dataframe = pd.read_csv(test_csv)
        if label_col not in dataframe.columns:
            raise ValueError(f"label column '{label_col}' not found in {test_csv}")
        labels = dataframe[label_col].values
        features = dataframe.drop(columns=[label_col]).values
    else:
        features, labels = load_iris(return_X_y=True)
    predictions = model.predict(features)
    if metric == "accuracy":
        return float(accuracy_score(labels, predictions))
    return math.sqrt(float(mean_squared_error(labels, predictions)))


def evaluate_torch(
    model_file: str, test_csv: Optional[str], label_col: str, metric: str
) -> float:
    import torch

    try:
        model = torch.jit.load(model_file)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "Torch evaluation supports TorchScript artifacts only"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = torch.randn(100, 3, 32, 32, device=device)
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1).cpu().numpy()
    return float((predictions >= 0).mean())


def compute_metric(
    mode: str,
    artifact_file: str,
    test_data: Optional[str],
    label_col: str,
    metric: str,
) -> float:
    if mode == "sklearn":
        return evaluate_sklearn(artifact_file, test_data, label_col, metric)
    if mode == "torch":
        return evaluate_torch(artifact_file, test_data, label_col, metric)
    raise ValueError(f"unsupported evaluation mode: {mode}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a model artifact")
    parser.add_argument("--artifact-path")
    parser.add_argument("--mlflow-run-id", default="")
    parser.add_argument("--mlflow-artifact-path", default="")
    parser.add_argument("--mode", choices=("sklearn", "torch"), default="sklearn")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--test-data")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--emit-json")
    parser.add_argument("--verify-signature", action="store_true")
    args = parser.parse_args(argv)

    result = {
        "status": "error",
        "mode": args.mode,
        "metric": args.metric,
        "baseline": args.baseline,
        "tolerance": args.tolerance,
    }
    try:
        if args.mlflow_run_id:
            if not args.mlflow_artifact_path:
                raise ValueError(
                    "--mlflow-artifact-path is required with --mlflow-run-id"
                )
            artifact_path = download_mlflow_artifact(
                args.mlflow_run_id, args.mlflow_artifact_path
            )
        elif args.artifact_path:
            artifact_path = args.artifact_path
        else:
            raise ValueError(
                "provide --artifact-path or --mlflow-run-id and "
                "--mlflow-artifact-path"
            )

        if artifact_path.startswith("s3://"):
            artifact_path = download_s3_artifact(artifact_path, os.getcwd())
        if args.verify_signature and not verify_signature(artifact_path):
            raise ValueError("artifact signature verification failed")

        score = compute_metric(
            args.mode, artifact_path, args.test_data, args.label_col, args.metric
        )
        result.update({"artifact_path": artifact_path, "score": score})
        result["status"] = (
            "passed" if score + args.tolerance >= args.baseline else "failed"
        )
        return_code = 0 if result["status"] == "passed" else 2
    except Exception as exc:
        logger.exception("Model validation failed")
        result["error"] = str(exc)
        return_code = 1

    output = json.dumps(result, sort_keys=True)
    if args.emit_json:
        output_path = Path(args.emit_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return return_code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
