"""
Hugging Face Hub importer for Aegis.

Features:
- Download a model repo from Hugging Face Hub (snapshot_download).
- Package the repo into a single tar.gz artifact suitable for registry ingestion.
- Optionally upload artifact to S3 (AWS) if S3_BUCKET env is configured.
- Sign the artifact using api.model_signing.sign_model_artifact (Vault Transit).
- Optionally register the artifact in the ModelRegistry via registry.register() using ModelConfig.

Usage:
  from api.hf_importer import import_from_hf
  import_from_hf(repo_id="google/flan-t5-small", model_name="flan-t5-small", version="hf-20251203",
                 sign_key="aegis-model-sign", upload_s3=True, register=True, registry=registry)

Notes:
- This script does not attempt to convert model weights to a particular runtime (ONNX/TorchScript).
  For conversion, set convert_to_torchscript=True and ensure torch is installed. Conversion is best-effort
  and may require custom model classes / example scripts.
- If you upload to S3, provide AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY and optionally AWS_REGION.
- The ModelRegistry.register() call requires ModelConfig and a storage-accessible model_path (s3://... or file path).
"""

import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    from api.model_signing import sign_model_artifact
except ImportError:
    sign_model_artifact = None


def import_from_hf(
    repo_id,
    model_name,
    version=None,
    sign_key=None,
    upload_s3=False,
    s3_bucket=None,
    s3_prefix=None,
    registry=None,
    register=False,
    convert_to_torchscript=False,
):
    if upload_s3:
        s3_bucket = s3_bucket or os.environ.get("HF_S3_BUCKET") or os.environ.get("S3_BUCKET")
        if not s3_bucket:
            raise ValueError("An S3 bucket is required when upload_s3 is enabled")

    if snapshot_download is None:
        raise ImportError("huggingface_hub is required to import models from Hugging Face")

    safe_model_name = Path(model_name).name
    if safe_model_name in {"", ".", ".."}:
        raise ValueError("model_name must include a valid filename")
    version = version or datetime.now(timezone.utc).strftime("hf-%Y%m%d%H%M%S")
    safe_version = Path(str(version)).name
    if safe_version in {"", ".", ".."}:
        raise ValueError("version must include a valid filename")

    cache_dir = tempfile.mkdtemp(prefix="aegis_hf_cache_")
    repo_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        resume_download=True,
    )

    artifact_dir = tempfile.mkdtemp(prefix="aegis_hf_artifact_")
    artifact_path = str(Path(artifact_dir) / f"{safe_model_name}-{safe_version}.tar.gz")
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(repo_path, arcname=safe_model_name)

    signature_path = None
    if sign_key:
        if sign_model_artifact is None:
            raise ImportError("Model signing support is unavailable")
        signature_path = sign_model_artifact(artifact_path, sign_key)

    s3_uri = None
    if upload_s3:
        import boto3

        key = "/".join(part.strip("/") for part in (s3_prefix, Path(artifact_path).name) if part)
        boto3.client("s3").upload_file(artifact_path, s3_bucket, key)
        s3_uri = f"s3://{s3_bucket}/{key}"

    registered = False
    if register:
        if registry is None:
            raise ValueError("A registry instance is required when register is enabled")
        from api.model_runner import ModelConfig

        registry.register(model_name, version, ModelConfig(model_path=s3_uri or artifact_path))
        registered = True

    return {
        "repo_id": repo_id,
        "model_name": model_name,
        "version": version,
        "artifact_path": artifact_path,
        "signature_path": signature_path,
        "s3_uri": s3_uri,
        "registered": registered,
    }
