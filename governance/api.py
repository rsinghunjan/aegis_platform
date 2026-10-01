#!/usr/bin/env python3
"""Governance API for inspecting MLflow runs and promoting model versions."""

import os
import re

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


def authorize(action: str, tenant_id: str) -> tuple[dict | None, tuple]:
    """Require a trusted host callback for identity, policy, and approval evidence."""
    callback = app.config.get("AEGIS_GOVERNANCE_AUTHORIZER")
    if not callable(callback):
        return None, (jsonify({"error": "governance authorization is not configured"}), 503)
    grant = callback(request, action, tenant_id)
    if not grant:
        return None, (jsonify({"error": "governance access denied"}), 403)
    if not isinstance(grant, dict) or not grant.get("actor_id"):
        return None, (jsonify({"error": "trusted actor identity is unavailable"}), 403)
    if grant.get("action") != action or grant.get("tenant_id") != tenant_id:
        return None, (jsonify({"error": "governance grant scope mismatch"}), 403)
    return grant, ()


def _promotion_evidence_is_complete(grant: dict) -> bool:
    required = (
        "approval_id",
        "policy_version",
        "decision_evidence_sha256",
        "artifact_sha256",
    )
    valid_hashes = all(
        isinstance(grant.get(key), str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", grant[key]) is not None
        for key in ("decision_evidence_sha256", "artifact_sha256")
    )
    return (
        all(isinstance(grant.get(key), str) and grant[key] for key in required)
        and grant.get("signature_verified") is True
        and valid_hashes
    )


def get_mlflow_client():
    global client
    if client is None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_URI)
    return client


@app.route("/runs/<experiment_name>")
def list_runs(experiment_name):
    grant, error = authorize("model.read", request.args.get("tenant_id", "default"))
    if error:
        return error
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
    tenant_id = str(payload.get("tenant_id") or "default")
    grant, error = authorize("model.promote", tenant_id)
    if error:
        return error
    if not _promotion_evidence_is_complete(grant):
        return jsonify({"error": "promotion requires verified artifact and approval evidence"}), 403
    governance_evidence = {
        key: grant[key]
        for key in (
            "approval_id",
            "policy_version",
            "decision_evidence_sha256",
            "artifact_sha256",
            "signature_verified",
        )
    }
    try:
        result = promote_run(
            str(run_id),
            client=get_mlflow_client(),
            model_name=str(payload.get("model_name") or "") or None,
            version=str(payload.get("version") or "1"),
            tenant_id=tenant_id,
            tenant_name=str(payload.get("tenant_name") or ""),
            actor=grant["actor_id"],
            notes=str(payload.get("notes") or ""),
            governance_evidence=governance_evidence,
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
    tenant_id = request.args.get("tenant_id") or "default"
    _grant, error = authorize("model.read", tenant_id)
    if error:
        return error
    model = resolve_promoted_model(
        model_name, tenant_id=tenant_id
    )
    if model is None:
        return jsonify({"error": "promoted model not found"}), 404
    return jsonify(model)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
