 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
#!/usr/bin/env python3
"""
Cosign verification helper that attempts:
- a SHA256 hash check if model_signature.json contains a sha256 entry
- a cosign verify-blob check if cosign public key is available and signature object exists

Uses create_storage_client to download remote signature files when needed.
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .signature import compute_sha256, load_model_signature
from aegis_multimodal_ai_system.storage.factory import create_storage_client

def _cosign_available() -> bool:
    try:
        subprocess.run(["cosign", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def verify_with_cosign_blob(local_path: Path, sig_blob_path: Path, pubkey_path: Optional[Path] = None) -> bool:
    """
    Verify a local blob with cosign verify-blob using a public key.
    """
    if not _cosign_available():
        return False
    cmd = ["cosign", "verify-blob"]
    if pubkey_path:
        cmd += ["--key", str(pubkey_path)]
    cmd += ["--signature", str(sig_blob_path), str(local_path)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False

def verify_artifact_remote_with_cosign(artifact_uri: str, model_dir: Path, cosign_pubkey_secret_path: Optional[str] = None) -> bool:
    """
    artifact_uri: e.g. s3://bucket/path/model.tar.gz or file://...
    model_dir: path to model dir in repo which contains model_signature.json with signature info
    cosign_pubkey_secret_path: optional path in external secrets or mounted file where pubkey is present
    """
    sig = load_model_signature(model_dir)
    if not sig:
        return False
    # If sha256 exists, prefer hash check
    tmpdir = Path(tempfile.mkdtemp())
    try:
        # download artifact
        # naive parsing: support s3:// and file:// and model_registry/ local paths
        local_arc = None
        if artifact_uri.startswith("s3://"):
            parts = artifact_uri[len("s3://"):].split("/", 1)
            bucket = parts[0]; key = parts[1]
            client = create_storage_client(bucket=bucket)
            local_arc = tmpdir / Path(key).name
            client.download(key, local_arc)
        elif artifact_uri.startswith("file://") or not ("://" in artifact_uri):
            path = artifact_uri.replace("file://", "")
