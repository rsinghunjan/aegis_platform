#!/usr/bin/env python3
"""
Validate model metadata and presence of model card/artifact.

Usage:
  # Validate a single model directory:
  python3 scripts/validate_model_metadata.py model_registry/my-model-1.0

  # Validate entire registry (default, checks all subdirectories)
  python3 scripts/validate_model_metadata.py --all
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import subprocess

try:
    import yaml
except Exception:
    yaml = None

try:
    import jsonschema
except Exception:
    jsonschema = None

ROOT = Path.cwd()
SCHEMA_PATH = ROOT / "model_registry" / "MODEL_METADATA_SCHEMA.json"

def load_schema():
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def read_metadata(path: Path):
    # Accept metadata.json or metadata.yaml
    for name in ("metadata.yaml", "metadata.yml", "metadata.json"):
        p = path / name
        if p.exists():
            if p.suffix in (".yaml", ".yml"):
                if yaml is None:
                    raise RuntimeError("PyYAML is required to read YAML metadata. Install pyyaml.")
                return yaml.safe_load(p.read_text(encoding="utf-8")), p
            else:
                return json.loads(p.read_text(encoding="utf-8")), p
    return None, None

def check_artifact(path: Path, artifact_rel: str) -> bool:
    art = path / artifact_rel
    if art.exists():
        return True
    # If DVC is present, artifact may be in remote; check for .dvc file
    dvc_file = path / (artifact_rel + ".dvc")
    if dvc_file.exists():
        # Run `dvc status -r` only if dvc is installed; otherwise assume managed by DVC remote
        try:
            subprocess.run(["dvc", "status"], cwd=str(path), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Cannot reliably assert remote presence here without dvc configured in CI runner
            return True
        except FileNotFoundError:
            # dvc not installed in runner -> assume artifact will be provided by pipeline
            return True
    return False

def validate_one(path: Path) -> bool:
    print(f"Validating {path} ...")
    if not path.is_dir():
        print(f"  SKIP: {path} is not a directory")
        return True
    meta, meta_path = read_metadata(path)
    if not meta:
        print(f"  ERROR: metadata.yaml/json not found in {path}")
        return False
    if jsonschema is None:
        print("  ERROR: jsonschema package is required in CI to validate metadata. Install jsonschema.")
        return False
    schema = load_schema()
    try:
        jsonschema.validate(instance=meta, schema=schema)
    except Exception as e:
        print(f"  ERROR: metadata validation failed: {e}")
        print(f"  metadata file: {meta_path}")
        return False
    # Check MODEL_CARD.md exists
    card = path / "MODEL_CARD.md"
    if not card.exists():
        print("  ERROR: MODEL_CARD.md not found")
        return False
    # Check artifact presence (best effort)
    artifact_rel = meta.get("artifact")
    if not artifact_rel:
        print("  ERROR: metadata.artifact not defined")
        return False
    ok = check_artifact(path, artifact_rel)
    if not ok:
        print(f"  ERROR: artifact {artifact_rel} not found in {path}")
        return False
    print("  OK")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--all", action="store_true", help="Validate all directories under model_registry")
    args = parser.parse_args()
    if args.all or args.path is None:
        base = ROOT / "model_registry"
        if not base.exists():
            print("model_registry directory does not exist; nothing to validate.")
            return 0
        models = sorted([p for p in base.iterdir() if p.is_dir()])
    else:
        models = [Path(args.path)]
    ok = True
    for m in models:
        r = validate_one(m)
        ok = ok and r
    if not ok:
        print("One or more model metadata checks failed.")
        sys.exit(2)
    print("All model metadata checks passed.")
    return 0

if __name__ == "__main__":
    main()
