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
#!/usr/bin/env python3
"""
CI helper: verify model signatures for model_registry entries.

Usage (CI example):
  # compute changed model dirs in CI (git diff --name-only origin/main...HEAD)
  python3 scripts/verify_model_signatures.py modelA modelB

If no model names are passed, verifies all folders under model_registry/.
Policy:
 - Fails if a model directory lacks model_signature.json
 - If signature exists and artifact is locally available (file:// or local path), verifies sha256
 - If artifact is remote and MODEL_REGISTRY_API_URL is configured, attempts to download via model_registry.loader.download_artifact
   (requires storage credentials to be present in CI env)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from aegis_multimodal_ai_system.model_registry import signature as sig
from aegis_multimodal_ai_system.model_registry import loader as registry_loader

def _verify_one(model_name: str) -> bool:
    base = Path.cwd() / "model_registry" / model_name
    print("Verifying model:", model_name, "at", base)
    if not base.exists() or not base.is_dir():
        print(f"  SKIP: model dir not found: {base}")
        return True
    s = sig.load_model_signature(base)
    if not s:
        print(f"  ERROR: model_signature.json missing for {model_name}")
        return False
    # Determine artifact URI from metadata if present
    meta_file = base / "metadata.yaml"
    artifact_uri = None
    if meta_file.exists():
        try:
            import yaml
            meta = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
            art = meta.get("artifact")
            if isinstance(art, dict):
                artifact_uri = art.get("uri") or art.get("path")
            else:
                artifact_uri = art
        except Exception:
            artifact_uri = None
    # fallback to local file attempt
    if not artifact_uri:
        # try common filenames
        for name in ("model.onnx", "model.pt", "model.pth", "model.zip"):
            p = base / name
            if p.exists():
                local = p
                break
        else:
            print(f"  ERROR: no artifact path found in metadata and no local artifact for {model_name}")
            return False
    else:
        # if artifact_uri is file:// or local path, resolve
        try:
            local = registry_loader.download_artifact(artifact_uri)
        except Exception as e:
            print(f"  WARN: could not download artifact for {model_name}: {e}")
            print("  SKIP: unable to verify remote artifact without credentials")
            # Conservative: treat as failure unless CI is configured with storage credentials.
            return False

    # Now verify sha
    ok = sig.verify_artifact_by_signature_or_hash(Path(local), base)
    if not ok:
        print(f"  ERROR: verification failed for {model_name}")
        return False
    print(f"  OK: verified {model_name}")
    return True

def main(names: Optional[List[str]] = None) -> int:
    if not names:
        # check all dirs
        names = [p.name for p in (Path.cwd() / "model_registry").iterdir() if p.is_dir()]
    all_ok = True
    for n in names:
        try:
            ok = _verify_one(n)
        except Exception as e:
            print("  ERROR during verify:", e)
            ok = False
        all_ok = all_ok and ok
    return 0 if all_ok else 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="List of model names under model_registry to verify")
