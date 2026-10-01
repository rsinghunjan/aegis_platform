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
#!/usr/bin/env python3
"""
Cosign verification helper.

- Supports a sha256 check if model_signature.json contains "sha256"
- Optionally runs cosign verify-blob with a provided public key
- Uses StorageClient to download artifacts when needed
"""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from aegis_multimodal_ai_system.storage.factory import create_storage_client
from aegis_multimodal_ai_system.model_registry.signature import compute_sha256, load_model_signature

def _cosign_available() -> bool:
    try:
        subprocess.run(["cosign", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def verify_blob_with_cosign(local_blob: Path, sig_blob: Path, pubkey_path: Optional[Path] = None) -> bool:
    if not _cosign_available():
        return False
    cmd = ["cosign", "verify-blob", "--signature", str(sig_blob), str(local_blob)]
    if pubkey_path:
        cmd = ["cosign", "verify-blob", "--key", str(pubkey_path), "--signature", str(sig_blob), str(local_blob)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False

def verify_artifact(artifact_uri: str, model_dir: Path, cosign_pubkey_path: Optional[str] = None) -> bool:
    """
    Verify artifact by either sha256 or cosign signature. Returns True on success.
    """
    sig = load_model_signature(model_dir)
    if not sig:
        return False

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # download artifact to tmp
        local_arc = tmpdir / "artifact"
        if artifact_uri.startswith("s3://"):
            parts = artifact_uri[len("s3://"):].split("/", 1)
            bucket = parts[0]; key = parts[1]
            client = create_storage_client(bucket=bucket)
            client.download(key, local_arc)
        elif artifact_uri.startswith("file://") or "://" not in artifact_uri:
            local_arc = Path(artifact_uri.replace("file://", ""))
        else:
            # best-effort: try model_registry loader
            from . import loader as registry_loader
            local_arc = Path(registry_loader.download_artifact(artifact_uri))

        if not local_arc.exists():
            return False

        want = sig.get("sha256")
        if want:
            got = compute_sha256(local_arc)
            if got == want:
                return True

        # try cosign signature (sig file expected at same path + .sig or specified in signature metadata)
        sigfile = model_dir / sig.get("signature_file", "artifact.sig")
        if not sigfile.exists() and artifact_uri.startswith("s3://"):
            maybe_sig_key = key + ".sig"
            sigpath = tmpdir / "artifact.sig"
            try:
                client.download(maybe_sig_key, sigpath)
                sigfile = sigpath
            except Exception:
                pass

        pubkey = Path(cosign_pubkey_path) if cosign_pubkey_path else None
        if sigfile.exists() and _cosign_available():
            return verify_blob_with_cosign(local_arc, sigfile, pubkey)
    finally:
        pass
    return False
aegis_multimodal_ai_system/model_registry/cosign_verify.pymodel_re
