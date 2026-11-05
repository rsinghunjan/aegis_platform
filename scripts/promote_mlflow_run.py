#!/usr/bin/env python
"""
CLI to promote an MLflow run artifact into the Aegis ModelRegistry.

Examples:
  python scripts/promote_mlflow_run.py --run-id <RUN_ID> --artifact-path model/model.joblib --model-name my-model --sign-key aegis-model-sign

Notes:
 - Requires MLFLOW_TRACKING_URI set or default MLflow client configuration.
 - Requires Vault env (VAULT_ADDR & VAULT_TOKEN) if sign_key is used and signing relies on Vault Transit.
 - Registry integration is best-effort; adapt imports if your registry API differs.
"""
import os
import sys
import argparse
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger("aegis.promote_mlflow")
logging.basicConfig(level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--artifact-path", required=True, help="artifact path inside MLflow run (e.g., model/model.joblib)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--version", default=None)
    p.add_argument("--sign-key", default=None)
    p.add_argument("--manifest", default=None, help="approved retraining manifest JSON to record as provenance")
    p.add_argument("--approved-manifest-sha256", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    try:
        from api.mlflow_registry import promote_run_to_registry
    except Exception:
        logger.exception("api.mlflow_registry helper not found; ensure this repo contains api/mlflow_registry.py")
        raise

    manifest_sha = None
    if args.manifest:
        from services.feedback_loop import verify_retraining_manifest

        with open(args.manifest, encoding="utf-8") as handle:
            manifest_sha = verify_retraining_manifest(json.load(handle), args.approved_manifest_sha256 or "")
    res = promote_run_to_registry(
        run_id=args.run_id,
        artifact_path=args.artifact_path,
        model_name=args.model_name,
        version=args.version,
        sign_key=args.sign_key,
    )

    if manifest_sha:
        res["retraining_manifest_sha256"] = manifest_sha
        try:
            from mlflow.tracking import MlflowClient

            MlflowClient().set_tag(args.run_id, "aegis.retraining.manifest_sha256", manifest_sha)
        except Exception:
            logger.exception("Could not tag MLflow run with manifest hash")
            raise
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
