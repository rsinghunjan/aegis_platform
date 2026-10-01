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
 50
#!/usr/bin/env python3
"""
High-level verifier script to be used in CI.

Usage:
  python3 scripts/verify_model_signatures.py [model1 model2 ...]

If no models specified, it scans model_registry/* directories and verifies each model_signature.json.
Requires OBJECT_STORE_TYPE and provider credentials in the environment (or CI OIDC configured).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from aegis_multimodal_ai_system.model_registry.cosign_verify import verify_artifact
from aegis_multimodal_ai_system.model_registry.signature import load_model_signature

def find_models(root: Path = Path("model_registry")):
    for d in sorted(root.iterdir()):
        if d.is_dir():
            yield d

def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else [str(p) for p in find_models()]
    failures = []
    for m in models:
        model_dir = Path(m)
        sig = load_model_signature(model_dir)
        if not sig:
            print(f"{m}: model_signature.json missing or invalid")
            failures.append(m)
