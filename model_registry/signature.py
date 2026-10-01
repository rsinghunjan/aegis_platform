 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
#!/usr/bin/env python3
"""
Model artifact signature & hash verification helpers.

- compute_sha256(path) -> hex str
- load_model_signature(model_dir) -> dict or None (reads model_signature.json)
- verify_artifact_hash(local_path, signature_dict) -> bool
- verify_artifact_with_cosign(local_path_or_uri, sig_key_envvar="COSIGN_VERIFY_KEY") -> bool (best-effort,
  requires `cosign` on PATH and a signature entry in signature.json; optional)

Notes:
- This module performs offline sha256 verification (always available).
- Optionally attempts cosign verification if COSIGN_VERIFY env var is set and cosign is available.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_signature(model_dir: Path) -> Optional[Dict[str, Any]]:
    sig_file = model_dir / "model_signature.json"
    if not sig_file.exists():
        return None
    try:
        return json.loads(sig_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def verify_artifact_hash(local_path: Path, signature: Dict[str, Any]) -> bool:
    """
    Verify the sha256 in signature matches the computed sha256 of local_path.
    Returns True on match, False otherwise.
    """
    want = signature.get("sha256")
    if not want:
        return False
    got = compute_sha256(local_path)
    return got == want


def _cosign_available() -> bool:
    try:
        subprocess.run(["cosign", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def verify_with_cosign(subject: str, sig_entry: Dict[str, Any], key_envvar: str = "COSIGN_VERIFY_KEY") -> bool:
    """
    Attempt cosign verification. `subject` may be a file path or a URI (container/image or file).
    Requires COSIGN_VERIFY=1 (or env present) and COSIGN_VERIFY_KEY env var to point to a key or public key file.
    Returns True if cosign verifies (exit code 0), False otherwise.
    """
    if not os.environ.get("COSIGN_VERIFY"):
        # not enabled
        return False
    cosign_key = os.environ.get(key_envvar)
    if not cosign_key:
        return False
    if not _cosign_available():
        return False
    # Try verifying; if subject is a local file, cosign supports verifying signed attachments with `cosign verify-blob`
    try:
        if Path(subject).exists():
            # verify-blob requires the signature file; many cosign flows use remote sigs; best-effort:
            # if signature contains "signed_by" with a signature file path, try verify-blob
            sigfile = sig_entry.get("signature_file")
            if sigfile and Path(sigfile).exists():
                subprocess.run(["cosign", "verify-blob", "--key", cosign_key, "--signature", sigfile, subject], check=True)
                return True
            # otherwise attempt `cosign verify` (may only work for images)
            subprocess.run(["cosign", "verify", "--key", cosign_key, subject], check=True)
            return True
        else:
            subprocess.run(["cosign", "verify", "--key", cosign_key, subject], check=True)
            return True
    except Exception:
        return False


def verify_artifact_by_signature_or_hash(local_path: Path, model_dir: Path) -> bool:
    """
    Given a local artifact file and the model dir containing model_signature.json,
    verify either:
    - sha256 matches; OR
    - cosign verification (if configured)
    Returns True if verification passes.
    """
    sig = load_model_signature(model_dir)
    if not sig:
        return False
    # Try sha256 check
    try:
        if verify_artifact_hash(local_path, sig):
            return True
    except Exception:
        pass
    # Try cosign if available / enabled
    try:
        if verify_with_cosign(str(local_path), sig):
            return True
    except Exception:
        pass
