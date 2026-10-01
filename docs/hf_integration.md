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
```markdown
Hugging Face integration for Aegis — quick guide

Goals
- Provide a safe, auditable way to import models from the Hugging Face Hub into Aegis:
  - download model repo snapshot
  - package artifact (tar.gz)
  - sign artifact via Vault Transit (no raw private keys in Aegis)
  - optionally upload artifact to S3 (for serving) and register in ModelRegistry

Provided components
- api/hf_importer.py : core logic to download, package, sign, and register
- scripts/hf_import.py : CLI wrapper to run import from local machine or CI
- api/requirements-hf.txt : extra deps for HF integrations (huggingface-hub, transformers, boto3)
- tests/test_hf_importer.py : unit tests (mocked, no network)

How to use (manual)
1) Install deps (prefer a venv):
   pip install -r api/requirements-hf.txt

2) Basic import to local file:
   python scripts/hf_import.py --repo-id google/flan-t5-small --model-name flan-t5 --version hf-20251203

3) Sign in CI using Vault OIDC:
   - Ensure Vault OIDC role for GitHub Actions is configured (docs/vault_oidc_approle.md)
   - In GH Actions, authenticate and set VAULT_TOKEN in env; run script with --sign-key aegis-model-sign

4) Upload to S3 and register in registry:
   python scripts/hf_import.py --repo-id ... --model-name ... --upload-s3 --s3-bucket my-bucket --sign-key aegis-model-sign --register

Important notes & operational guidance
- Artifact size: HF models can be large. When packaging and uploading, ensure you have adequate disk space and network bandwidth.
- Conversion: converting HF checkpoints to TorchScript/ONNX is model-dependent; conversion is optional and best-effort in the importer.
- Signing: we sign the local tar.gz artifact with Vault Transit; ensure CI or operator has access to short-lived VAULT_TOKEN via OIDC/AppRole.
- Storage: prefer storing artifacts in object storage (S3/MinIO) and using s3:// URLs as model_path in ModelRegistry. Ensure your model loader can fetch/load from s3 (or MinIO) at runtime.
- Model cards & metadata: The importer preserves HF repo files (README, model card). Consider extracting model card metadata and storing as ModelRegistry metadata (future work).

Acceptance criteria (when this integration is "done")
- You can import a HF model in CI (OICD->Vault) and produce a signed artifact.
- Artifact signature verifies at runtime using api.model_signing.verify_model_artifact().
- ModelRegistry.register() succeeds with model_path pointing to s3://... or a locally served artifact.
- Tests: tests/test_hf_importer.py passes in CI, and a small end-to-end test exists in staging that downloads the artifact and attempts a model load (or at least verifies signature+tarfile integrity).

Next enhancements you may want
- Automatic extraction of model card metadata into registry fields (author, license, tags).
- Add an HF model catalog UI in Aegis that lists available HF imports and provenance (who imported, when, signature).
- Convert to ONNX/TorchScript during import for faster runtime performance (requires per-model conversion scripts).
- Add streaming large-file uploads to S3 with retries and multipart uploads.

