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
import argparse
import json
import logging

logger = logging.getLogger("aegis.promote_mlflow")
logging.basicConfig(level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--artifact-path", required=True, help="artifact path inside MLflow run (e.g., model/model.joblib)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--version", default=None)
    p.add_argument("--sign-key", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    try:
        from api.mlflow_registry import promote_run_to_registry
    except Exception:
        logger.exception("api.mlflow_registry helper not found; ensure this repo contains api/mlflow_registry.py")
        raise

    res = promote_run_to_registry(
        run_id=args.run_id,
        artifact_path=args.artifact_path,
        model_name=args.model_name,
        version=args.version,
        sign_key=args.sign_key,
    )

    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
