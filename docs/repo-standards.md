```markdown
Repository standards & hygiene (Aegis)

Overview
This document captures minimal repo conventions used across the Aegis repo to improve discoverability and automation.

1) Directory layout
- docker/: Dockerfiles & compose snippets (canonical names)
- examples/: runnable examples (training, inference, hpo)
- scripts/: small CLI helpers used in CI & ops
- data/: sample data & dataset manifests (not large datasets)
- feast/, mlflow/, docs/: integration-specific guidance

2) Filenames & canonical choices
- Use docker/docker-compose-mlflow.yml as the canonical mlflow compose filename. If using the dotted variant (legacy), document it in README.
- Dataset manifests: data/dataset_manifests/<name>/manifest.json
- Training scripts emit the MLflow run_id line: "MLflow run_id: <id>" or (preferred) a JSON line for CI parsing.

3) CI behavior expectations
- All PRs must pass:
  - basic lint & filename checks (.github/workflows/ci-lint.yml)
  - dataset manifest check if manifests were modified
  - unit tests present under tests/
- Promotion workflows must annotate MLflow runs with aegis.* tags for audit.

4) Dataset manifest schema (example)
{
  "version": "20251201T120000Z",
  "created_at": "2025-12-01T12:00:00Z",
  "provenance": {"job_id":"ingest-123","git_sha":"abcd..."},
  "s3_path": "s3://feast-offline/aegis/datasets/user_events/20251201T120000Z/user_events.parquet",
  "checksum": "sha256:..."
}

5) Adding a new integration
- Provide docs/quickstart.md for the integration.
- Provide a small CI smoke job that runs locally by default.
