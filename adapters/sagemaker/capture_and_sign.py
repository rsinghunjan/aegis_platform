#!/usr/bin/env python3
"""
SageMaker adapter for model artifact capture and Vault transit signing.

This script is designed to run as a post-training hook in SageMaker or
as a standalone script to capture model artifacts from a training job
and sign them using HashiCorp Vault transit encryption.

Usage:
  python adapters/sagemaker/capture_and_sign.py \
    --model-dir /opt/ml/model \
    --output-bucket s3://my-bucket/model-artifacts \
    --vault-addr https://vault.example.com \
    --vault-key aegis-cosign

Environment variables:
  VAULT_ADDR: Vault server address (alternative to --vault-addr)
  VAULT_TOKEN: Vault authentication token
  VAULT_AUDIENCE: JWT audience for OIDC auth (default: VAULT_AUDIENCE placeholder)
  AWS_DEFAULT_REGION: AWS region for S3 operations

Requires:
  - Vault transit engine enabled with signing key configured
  - AWS credentials with S3 write permissions
  - boto3 and requests libraries
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_sha256_base64(file_path: str) -> str:
    """Compute SHA256 hash of a file and return base64 encoded."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return base64.b64encode(sha256_hash.digest()).decode("utf-8")


def create_deterministic_archive(source_dir: str, output_path: str) -> str:
    """
    Create a deterministic tar.gz archive of a directory.

    Uses fixed mtime and uid/gid for reproducibility.

    Args:
        source_dir: Directory to archive
        output_path: Output tar.gz file path

    Returns:
        SHA256 hash of the created archive
    """
    source_path = Path(source_dir).resolve()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fixed_mtime = 0  # Unix epoch for determinism

    with tarfile.open(str(output_file), "w:gz") as tar:
        for file_path in sorted(source_path.rglob("*")):
            if file_path.is_dir():
                continue
            arcname = str(file_path.relative_to(source_path))
            tarinfo = tar.gettarinfo(str(file_path), arcname=arcname)
            tarinfo.mtime = fixed_mtime
            tarinfo.uid = 0
            tarinfo.gid = 0
            tarinfo.uname = ""
            tarinfo.gname = ""
            with open(file_path, "rb") as f:
                tar.addfile(tarinfo, f)

    return compute_sha256(str(output_file))


def sign_with_vault_transit(
    vault_addr: str,
    vault_token: str,
    key_name: str,
    digest_base64: str,
) -> Dict[str, Any]:
    """
    Sign a digest using Vault transit engine.

    Args:
        vault_addr: Vault server address
        vault_token: Vault authentication token
        key_name: Name of the transit signing key
        digest_base64: Base64-encoded SHA256 digest

    Returns:
        Dictionary containing signature response from Vault
    """
    import requests

    url = f"{vault_addr.rstrip('/')}/v1/transit/sign/{key_name}"
    headers = {"X-Vault-Token": vault_token}
    payload = {"input": digest_base64}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    return response.json()


def upload_to_s3(local_path: str, s3_uri: str) -> None:
    """
    Upload a file to S3 using boto3.

    Args:
        local_path: Local file path
        s3_uri: S3 URI (s3://bucket/key)
    """
    import boto3

    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} to {s3_uri}")


def capture_and_sign(
    model_dir: str,
    output_bucket: str,
    vault_addr: str,
    vault_key: str,
    job_name: Optional[str] = None,
    vault_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function to capture model artifacts, sign them, and upload to S3.

    Args:
        model_dir: Directory containing model artifacts
        output_bucket: S3 bucket URI for output
        vault_addr: Vault server address
        vault_key: Vault transit key name
        job_name: Optional job name for artifact naming
        vault_token: Optional Vault token (defaults to VAULT_TOKEN env)

    Returns:
        Dictionary with operation results
    """
    # Generate job name if not provided
    if not job_name:
        job_name = f"sagemaker-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    # Get Vault token
    token = vault_token or os.environ.get("VAULT_TOKEN")
    if not token:
        raise ValueError("VAULT_TOKEN environment variable not set")

    # Create temporary directory for artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_name = f"{job_name}.tar.gz"
        archive_path = os.path.join(tmpdir, archive_name)

        # Create deterministic archive
        print(f"Creating deterministic archive from {model_dir}...")
        sha256_hex = create_deterministic_archive(model_dir, archive_path)
        print(f"Archive created: {archive_path} (SHA256: {sha256_hex})")

        # Compute base64 digest for Vault signing
        digest_b64 = compute_sha256_base64(archive_path)

        # Sign with Vault
        print(f"Signing artifact with Vault transit key: {vault_key}...")
        signature_response = sign_with_vault_transit(
            vault_addr=vault_addr,
            vault_token=token,
            key_name=vault_key,
            digest_base64=digest_b64,
        )

        # Save signature
        sig_path = f"{archive_path}.sig.json"
        with open(sig_path, "w") as f:
            json.dump(signature_response, f, indent=2)
        print(f"Signature saved: {sig_path}")

        # Create metadata file
        metadata = {
            "job_name": job_name,
            "model_dir": model_dir,
            "archive_name": archive_name,
            "sha256": sha256_hex,
            "vault_key": vault_key,
            "vault_addr": vault_addr,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "signature": signature_response,
        }
        metadata_path = os.path.join(tmpdir, f"{job_name}.metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload to S3
        if output_bucket:
            base_uri = output_bucket.rstrip("/")
            upload_to_s3(archive_path, f"{base_uri}/{archive_name}")
            upload_to_s3(sig_path, f"{base_uri}/{archive_name}.sig.json")
            upload_to_s3(metadata_path, f"{base_uri}/{job_name}.metadata.json")

            metadata["s3_artifact_uri"] = f"{base_uri}/{archive_name}"
            metadata["s3_signature_uri"] = f"{base_uri}/{archive_name}.sig.json"
            metadata["s3_metadata_uri"] = f"{base_uri}/{job_name}.metadata.json"

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Capture and sign SageMaker model artifacts with Vault"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model artifacts (e.g., /opt/ml/model)",
    )
    parser.add_argument(
        "--output-bucket",
        required=True,
        help="S3 URI for output (e.g., s3://bucket/prefix)",
    )
    parser.add_argument(
        "--vault-addr",
        default=os.environ.get("VAULT_ADDR"),
        help="Vault server address (required, or set VAULT_ADDR env var)",
    )
    parser.add_argument(
        "--vault-key",
        default="aegis-cosign",
        help="Vault transit key name (default: aegis-cosign)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Job name for artifact naming (optional)",
    )
    args = parser.parse_args()

    # Validate vault_addr is provided
    if not args.vault_addr:
        print("Error: --vault-addr is required or set VAULT_ADDR environment variable",
              file=sys.stderr)
        sys.exit(1)

    try:
        result = capture_and_sign(
            model_dir=args.model_dir,
            output_bucket=args.output_bucket,
            vault_addr=args.vault_addr,
            vault_key=args.vault_key,
            job_name=args.job_name,
        )

        print("\nCapture and sign completed successfully!")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
