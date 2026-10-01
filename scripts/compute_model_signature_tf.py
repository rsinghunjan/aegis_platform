  1
  2
  3
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
#!/usr/bin/env python3
"""
Compute model_signature.json for a TF SavedModel by creating a deterministic archive
and writing a small signature file alongside the model directory.

Usage:
  python3 scripts/compute_model_signature_tf.py model_registry/<model>/<version>/saved_model

Output:
  - model_registry/<model>/<version>/model.tar.gz
  - model_registry/<model>/<version>/model_signature.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import hashlib
import time
import subprocess

from typing import Dict

def run_make_archive(saved_model_dir: Path, out_archive: Path) -> Dict[str, str]:
    # call the make_deterministic_archive script for consistent behavior
    import subprocess
    cmd = ["python3", "scripts/make_deterministic_archive.py", str(saved_model_dir), str(out_archive)]
    subprocess.check_call(cmd)
    # compute sha256 and size locally as double-check
    h = hashlib.sha256()
    size = 0
    with out_archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            size += len(chunk)
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size_bytes": size, "archive": str(out_archive)}

def write_signature(model_dir: Path, sig: Dict):
    sig_path = model_dir / "model_signature.json"
    sig["framework"] = "tensorflow"
    sig["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sig_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print("Wrote signature to", sig_path)

def main():
    if len(sys.argv) < 2:
        print("Usage: compute_model_signature_tf.py <saved_model_dir>", file=sys.stderr)
        sys.exit(2)
    saved_model_dir = Path(sys.argv[1]).resolve()
    if not saved_model_dir.exists():
        print("SavedModel not found:", saved_model_dir, file=sys.stderr)
        sys.exit(2)
    parent = saved_model_dir.parent
    out_archive = parent / "model.tar.gz"
    sig = run_make_archive(saved_model_dir, out_archive)
    write_signature(parent, sig)
    print("Signature:", sig)

if __name__ == "__main__":
    main()
scripts/compute_model_signature_tf.py
