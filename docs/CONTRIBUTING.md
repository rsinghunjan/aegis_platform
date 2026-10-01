```markdown
Contributing & CI standards (Aegis repo)

Quick rules
- Keep file names canonical and consistent. Use docker/docker-compose-mlflow.yml (preferred) or docker/docker-compose.mlflow.yml (legacy) — but do not keep both.
- Add dataset manifests for any production dataset under data/dataset_manifests/<dataset>/manifest.json
- All model promotion steps must include MLflow tags documenting registry provenance:
  - aegis.registry.registered
  - aegis.registry.model_name
  - aegis.registry.model_version
  - aegis.dataset.version (recommended)
- Use the provided workflows for training, promotion, validation and optimization. Avoid ad-hoc scripts without CI hooks.

CI / PR checks
- The repo includes lightweight CI checks:
  - .github/workflows/ci-lint.yml runs basic linting, filename checks, and manifest validation.
  - .github/workflows/check-dataset-manifests.yml validates dataset manifests under data/dataset_manifests.
  - .github/workflows/verify-promotion-audit.yml can be used by operators to assert promotions set audit tags.

How to add a dataset manifest
- Create directory: data/dataset_manifests/<name>/
- Add manifest.json with required fields:
  - version (string)
  - created_at (ISO8601)
  - provenance (object with source job id, git sha)
  - s3_path OR local_path
- Run: python scripts/validate_dataset_manifest.py --path data/dataset_manifests --require-latest

Local development
- Before opening a PR, run:
  - python scripts/check_filenames.py
  - python scripts/validate_dataset_manifest.py --path data/dataset_manifests
- Keep tests and examples small and deterministic for CI smoke tests.

Thanks — following these rules makes governance, reproducibility and CI more reliable.
