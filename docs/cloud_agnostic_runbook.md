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
```markdown
# Cloud-agnostic runbook (scan, refactor, CI verification, DB, Helm, presign)

This runbook shows the commands and intent for finishing the cloud-agnostic work you asked for.

1) Find remaining direct SDK usages
   python3 scripts/scan_cloud_sdk_usage.py
   - Review the printed file list and refactor each usage:
     - For artifact reads: replace direct SDK calls with aegis_multimodal_ai_system.model_registry.loader.download_artifact(uri)
     - For general object-store usage: replace with create_storage_client(bucket).upload/download/list/get_presigned_url

2) Ensure DB is provider neutral
   - Set DATABASE_URL env var in your env and in CI
   - Use alembic for migrations. In alembic/env.py set sqlalchemy.url to this same DATABASE_URL
   - Run migrations:
     alembic upgrade head

3) CI artifact verification using provider secrets
   - Add provider secrets to GitHub (see docs)
   - The workflow .github/workflows/verify-model-signatures-remote.yml will pick up secrets and run scripts/verify_model_signatures.py to download and verify remote artifacts.
   - For security, prefer read-only credentials or short-lived access.

4) Helm validation & provider overlays
   - Use scripts/helm_validate_values.sh to lint and template manifests:
     ./scripts/helm_validate_values.sh --values helm/values.aws.yaml
   - Run Kind + Helm acceptance workflow to validate the chart generically (we added a workflow earlier). For provider overlays, test in a staging cluster in the target cloud.

5) Presigned / PAR validation
   - Provide provider credentials to CI, then run tests/presign/test_presign_urls.py
   - Note: OCI returns PARs that have semantics different than AWS presigned URLs; ensure your callers support PAR URLs (they may include tokens in the path).

6) Incremental PR policy for model_registry
   - Require model_signature.json in PR additions to model_registry. The verifier job will attempt remote verification; if remote artifact verification is not possible, provide a signed signature file in the PR or use CI-provided credentials.

7) Migration & testing
   - Run a staged deploy (infra overlay apply), upload one or two artifacts to the target provider bucket, and run the verify workflow in staging to confirm CI can download & verify.

Notes & caveats
- S3 vs GCS vs Azure vs OCI differences:
  - signed URL semantics differ; PARs (OCI) are similar but not identical
  - metadata headers and content-type handling differ; test uploads and downloads end-to-end
  - GCS and Azure SDKs may require different auth patterns in CI (JSON key vs connection string vs service principal)
- Always prefer read-only credentials for CI verifier and restrict scope & lifetime.

```
docs/cloud_agnostic_runbook.md
