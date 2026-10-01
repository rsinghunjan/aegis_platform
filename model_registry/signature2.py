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
#!/usr/bin/env python3
"""
Model signature helper used by verifier: compute_sha256, load_model_signature.

This mirrors the earlier references in cosign_verify and verify scripts.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_model_signature(model_dir: Path) -> Optional[Dict[str, Any]]:
    sig_path = Path(model_dir) / "model_signature.json"
    if not sig_path.exists():
        return None
    try:
        return json.loads(sig_path.read_text(encoding="utf-8"))
    except Exception:
        return None
