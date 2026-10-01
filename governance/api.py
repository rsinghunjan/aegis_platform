#!/usr/bin/env python3
"""Governance API for inspecting MLflow runs and promoting model versions."""

import os

from flask import Flask, jsonify, request

if __package__:
    from .promotion import PromotionError, promote_run, resolve_promoted_model
else:
    from promotion import PromotionError, promote_run, resolve_promoted_model

app = Flask(__name__)
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow.aegis.svc.cluster.local:5000"
)
client = None


def get_mlflow_client():
    global client
    if client is None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_URI)
    return client


@app.route("/runs/<experiment_name>")
def list_runs(experiment_name):
    mlflow_client = get_mlflow_client()
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if not experiment:
        return jsonify({"error": "experiment not found"}), 404
    runs = mlflow_client.search_runs([experiment.experiment_id], max_results=50)
    output = [
        {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]
    return jsonify(output)


@app.route("/promote", methods=["POST"])
def promote():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    run_id = payload.get("run_id")
    if not run_id:
        return jsonify({"error": "run_id required"}), 400
    try:
        result = promote_run(
            str(run_id),
            client=get_mlflow_client(),
            model_name=str(payload.get("model_name") or "") or None,
            version=str(payload.get("version") or "1"),
            tenant_id=str(payload.get("tenant_id") or "default"),
            tenant_name=str(payload.get("tenant_name") or ""),
            actor=str(payload.get("user") or "system"),
            notes=str(payload.get("notes") or ""),
        )
    except PromotionError as exc:
        app.logger.info("Promotion request rejected: %s", exc)
        return (
            jsonify({"error": "run failed promotion validation", "run_id": run_id}),
            400,
        )
    except Exception:
        app.logger.exception("Model promotion failed")
        return jsonify({"error": "promotion could not be persisted"}), 500
    return jsonify({"ok": True, "run_id": result["run_id"]})


@app.route("/models/<model_name>/promoted")
def promoted_model(model_name):
    model = resolve_promoted_model(
        model_name, tenant_id=request.args.get("tenant_id") or None
    )
    if model is None:
        return jsonify({"error": "promoted model not found"}), 404
    return jsonify(model)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
