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
#!/usr/bin/env python3
"""
Create a model_signature.json containing artifact_uri and sha256.

Usage:
  python3 scripts/create_model_signature.py <artifact_path> <out_model_dir/model_signature.json> --artifact-uri <uri>
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def compute_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_path", type=Path)
    ap.add_argument("out_signature_path", type=Path)
    ap.add_argument("--artifact-uri", required=True)
    args = ap.parse_args()

    sha = compute_sha256(args.artifact_path)
    sig = {
        "artifact_uri": args.artifact_uri,
        "sha256": sha,
        "signature_file": None
    }
    out = args.out_signature_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print("Wrote model_signature.json to", out)

if __name__ == "__main__":
    main()
s
