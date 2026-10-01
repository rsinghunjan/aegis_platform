#!/usr/bin/env python3
"""
High-level helper: download artifact for a model and verify its signature/hash.

API:
- fetch_and_verify_model(model_name, artifact_name="model.onnx", dest_dir=None) -> Path (local artifact path)
  - downloads artifact (via model_registry.loader.download_artifact)
  - verifies signature/hash using model_registry.signature.verify_artifact_by_signature_or_hash
  - raises RuntimeError on failure

This is intended to be the canonical path orchestrator and loaders use before loading a model into memory.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

def fetch_and_verify_model(model_name: str, artifact_name: str = "model.onnx", dest_dir: Optional[Path] = None) -> Path:
    """
    Download model artifact and verify signature/hash. Returns local Path on success.
    Raises RuntimeError on verification failure.
    """
    base = Path.cwd() / "model_registry" / model_name
    if not base.exists():
        raise RuntimeError(f"Model directory not found: {base}")
    from . import signature as sig
    from . import loader as registry_loader

    # 1) If artifact exists locally under model_registry/<model>/<artifact_name>, use it.
    cand = base / artifact_name
    if cand.exists():
        local = registry_loader.download_artifact(str(cand), dest_dir=dest_dir)
    else:
        # 2) If metadata.yaml points to a remote artifact, download it
        local = registry_loader.load_model_from_registry(model_name, artifact_name)

    # Verify signature or hash
    ok = sig.verify_artifact_by_signature_or_hash(Path(local), base)
    if not ok:
        raise RuntimeError(f"Signature/hash verification failed for {local} (model {model_name})")
    return Path(local)
