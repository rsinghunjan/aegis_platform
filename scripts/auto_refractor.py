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
#!/usr/bin/env python3
"""
Safe auto-refactor helper to create .refactored.py candidate files.

It will search for common patterns (boto3.client('s3'), direct boto3/oci/google imports)
and produce annotated .refactored.py files for review. It does not replace files in-place.

Run:
  python3 scripts/auto_refactor.py --root . --out-dir refactors
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

RULES = [
    (re.compile(r"\bboto3\.client\s*\(\s*['\"]s3['\"]\s*\)"), "# REPLACE with create_storage_client(bucket=...)"),
    (re.compile(r"\bboto3\.resource\s*\(\s*['\"]s3['\"]\s*\)"), "# REPLACE with create_storage_client(bucket=...)"),
    (re.compile(r"(^\s*import\s+boto3\s*$)", re.MULTILINE), "# REPLACE: remove direct boto3 import in runtime modules"),
    (re.compile(r"(^\s*from\s+boto3\s+import\s+.+$)", re.MULTILINE), "# REPLACE: remove direct boto3 import in runtime modules"),
    (re.compile(r"(^\s*import\s+google\.cloud.+$)", re.MULTILINE), "# REPLACE: use storage.factory or model_registry.loader"),
    (re.compile(r"(^\s*import\s+azure.+$)", re.MULTILINE), "# REPLACE: use storage.factory or adapter"),
    (re.compile(r"(^\s*import\s+oci.+$)", re.MULTILINE), "# REPLACE: use storage.factory or oci-adapter"),
]

EXCLUDE_DIRS = {".git", "venv", ".venv", "build", "dist", "__pycache__", "tests", "scripts"}

def scan_and_refactor(root: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    reports = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        text = p.read_text(encoding="utf-8")
        new_text = text
        changed = False
        for pat, comment in RULES:
            if pat.search(new_text):
                new_text = pat.sub(lambda m: f"{m.group(0)}  {comment}", new_text)
                changed = True
        if changed:
            rel = p.relative_to(root)
            outp = outdir / (str(rel).replace("/", "_") + ".refactored.py")
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(new_text, encoding="utf-8")
            reports.append((p, outp))
    return reports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="refactors")
    args = ap.parse_args()
    reports = scan_and_refactor(Path(args.root), Path(args.out_dir))
    if not reports:
        print("No candidate refactors found.")
        return 0
    print("Wrote candidate refactors:")
    for orig, outp in reports:
        print(f"  {orig} -> {outp}")
    print("\nReview the .refactored.py files, run tests, then replace originals as appropriate.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
