# Model registry policy & workflow

This document explains the model metadata policy and recommended artifact management using MLflow + DVC.

Goal
- Require metadata for every model push to model_registry/, including:
  - source (git repo + commit)
  - dataset_manifest (path or URL)
  - training_hash (commit or deterministic model artifact hash)
  - license (SPDX identifier or text)
  - artifact path (relative path to model artifact in the model dir)
- Require a human-readable model card (MODEL_CARD.md) in each model directory.
- Use DVC for large artifact storage and MLflow for experiment tracking & model registration.

Recommended layout (under model_registry/)
- model_registry/<model-name>-<version>/
  - metadata.yaml (or metadata.json)  # validated by CI
  - MODEL_CARD.md                     # human-readable model card
  - model.pth / model.pkl / model.zip # artifact (can be dvc-tracked)
  - dataset-manifest.yaml             # (or external URL referenced in metadata)
  - metrics.json (optional)
  - dvc.lock / dvc.yaml (if using DVC)

Local workflow (pip-compile / pip-tools environment recommended)
1. Add or update model metadata and card under model_registry/<model>:
   - Fill `metadata.yaml` according to the schema (see MODEL_METADATA_SCHEMA.json).
   - Add `MODEL_CARD.md` (use the template in docs/model_card_template.md).
   - If model artifact is large, add it via DVC:
     - dvc add model_registry/<model>/model.pth
     - git add model_registry/<model>/model.pth.dvc
     - dvc push
2. Validate locally:
   - python3 -m pip install jsonschema pyyaml
   - python3 scripts/validate_model_metadata.py model_registry/<model>
3. Commit and open a PR.

CI enforcement
- A workflow `.github/workflows/model-registry.yml` will run on pushes to model_registry/** and:
  - Validate metadata against the schema
  - Assert MODEL_CARD.md exists
  - Optionally fail if the artifact is not present locally (for small models) or if DVC indicates missing artifacts
- Configure branch protection to require this workflow before merging changes to model_registry.

MLflow & DVC guidance (short)
- Use MLflow for experiment tracking: log params, hyperparameters, metrics and the training run reference.
  - At training time, log the run and record the MLflow run id in metadata.yaml (optional).
  - Example to register artifacts with MLflow is in `examples/log_model_mlflow.py`.
- Use DVC to manage large artifacts:
  - dvc init
  - dvc add model_registry/<model>/model.pth
  - Commit .dvc files and push artifacts to remote (S3/GCS).
  - In metadata.yaml, reference the DVC-tracked path and note the DVC remote.

Notes and suggestions
- Enforce human readable MODEL_CARD.md with at least: model description, intended use, limitations, dataset sources, evaluation metrics, license, and contact.
- The CI will ensure metadata is present & valid, but not that the artifact content is correct — you may add additional checks (hash verification, simple inference smoke test) later.

References
- MLflow: https://mlflow.org
- DVC: https://dvc.org
