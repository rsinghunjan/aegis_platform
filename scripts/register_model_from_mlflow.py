#!/usr/bin/env python
"""
Download an artifact from MLflow run, sign it with Vault Transit, and register in Aegis ModelRegistry (if available).

Usage:
  python scripts/register_model_from_mlflow.py --run-id <RUN_ID> --artifact-path model/model.joblib --model-name my-model --version v1 --sign-key aegis-model-sign

Notes:
- Requires MLflow_TRACKING_URI env or pass --tracking-uri.
- Requires VAULT_ADDR and VAULT_TOKEN (or VAULT_AGENT-injected token) if sign_key is used.
- Registry integration is best-effort: the script tries to import api.registry and ModelConfig from api.model_runner.
  Adapt to your codebase if registry API differs.
"""
import os
import argparse
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger("aegis.register_model_from_mlflow")
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-path", required=True, help="relative path inside MLflow run artifacts (e.g. model/model.joblib)")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--sign-key", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # download artifact to tempdir
    td = tempfile.mkdtemp(prefix="aegis_mlflow_art_")
    logger.info("Downloading artifact %s from run %s to %s", args.artifact_path, args.run_id, td)
    local_path = client.download_artifacts(run_id=args.run_id, path=args.artifact_path, dst_path=td)
    logger.info("Downloaded artifact to %s", local_path)

    # if sign_key provided, attempt to sign local file(s)
    signature_files = []
    if args.sign_key:
        try:
            # prefer to sign the specific file if artifact_path points to a single file
            from api.model_signing import sign_model_artifact
            # If local_path is a directory, find the file matching basename
            target = Path(local_path)
            if target.is_dir():
                # if artifact_path ends with a filename, use that basename
                basename = Path(args.artifact_path).name
                candidates = list(target.rglob(basename))
                if candidates:
                    target_file = str(candidates[0])
                else:
                    # fallback: sign all files in dir (choose first)
                    files = [p for p in target.rglob("*") if p.is_file()]
                    target_file = str(files[0]) if files else None
            else:
                target_file = str(target)
            if target_file:
                sig = sign_model_artifact(target_file, args.sign_key)
                signature_files.append(sig)
                logger.info("Signed artifact %s -> %s", target_file, sig)
            else:
                logger.warning("No file found to sign in %s", local_path)
        except Exception:
            logger.exception("sign_model_artifact failed (ensure VAULT_ADDR & VAULT_TOKEN/Agent are configured)")

    # attempt to register in registry (best-effort)
    try:
        from api import registry as aegis_registry
        try:
            from api.model_runner import ModelConfig
        except Exception:
            ModelConfig = None

        model_path = local_path  # local path; prefer uploading to storage and passing s3:// URL
        ver = args.version or f"mlflow-{args.run_id}"
        if ModelConfig:
            mc = ModelConfig(model_path=model_path)
            aegis_registry.register(args.model_name, ver, mc)
            logger.info("Registered model %s:%s via registry.register()", args.model_name, ver)
        else:
            # fallback: if registry exposes simple register_artifact
            if hasattr(aegis_registry, "register_artifact"):
                aegis_registry.register_artifact(args.model_name, ver, model_path)
                logger.info("Registered via registry.register_artifact()")
            else:
                logger.warning("Registry API not compatible; please adapt register_model_from_mlflow.py to your registry API")
    except Exception:
        logger.exception("Registry registration step failed (non-fatal)")

    print("artifact_local_path:", local_path)
    if signature_files:
        print("signature_files:", signature_files)

if __name__ == "__main__":
    main()
