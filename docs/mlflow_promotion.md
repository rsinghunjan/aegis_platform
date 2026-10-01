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
```markdown
MLflow promotion -> Aegis ModelRegistry (quick guide)

Purpose
- Provide a reproducible, auditable way to promote MLflow runs into the Aegis ModelRegistry:
  - Download artifact from MLflow run
  - Optionally sign artifact with Vault Transit
  - Register artifact in the ModelRegistry (ModelConfig or register_artifact)
  - Annotate the MLflow run with registry provenance tags

Files added
- api/mlflow_registry.py : core logic to download, sign, register and annotate MLflow runs
- scripts/promote_mlflow_run.py : CLI wrapper for operators/CI to promote a run
- docs/mlflow_promotion.md : usage and operational guidance

How to use (manual)
1. After a training job completes, obtain the MLflow run_id and the artifact path
   (e.g., model/model.joblib).

2. Promote the run:
   MLFLOW_TRACKING_URI=http://mlflow:5000 python scripts/promote_mlflow_run.py \
     --run-id <RUN_ID> \
     --artifact-path model/model.joblib \
     --model-name my-product-model \
     --sign-key aegis-model-sign

3. Verify:
   - Check ModelRegistry for the new model version (or artifact registration).
   - Check MLflow run tags for aegis.registry.* keys to confirm provenance.

CI integration
- In CI (post-training), call the promote script with the run_id as an input.
- Gate deployment on successful registration (script returns metadata with registered=True).

Operational notes
- Signing requires api.model_signing.sign_model_artifact to be implemented and reachable (Vault configured).
- Registry integration tries multiple common registration APIs; adapt registry wrapper if your registry exposes different functions.
- Temporary download directories are preserved by default for debugging. You can enable cleanup in api/mlflow_registry.py.
- If you want to upload artifacts to shared object storage before registering, extend promote_run_to_registry to upload to S3/MinIO and pass an s3:// URL into registry.

Security & audit
- Promoting a run should be done by trusted CI or an operator account.
- Ensure Vault policies for the signing key are appropriately scoped.
- MLflow run tags capture provenance so it's easy to audit model origin.

```

```python name=tests/test_promote_mlflow_run.py
"""
Unit tests for api.mlflow_registry.promote_run_to_registry.
