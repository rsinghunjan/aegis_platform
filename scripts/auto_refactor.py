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
#!/usr/bin/env python3
"""
Auto-refactor helper (safe mode by default).

- Scans for common direct provider SDK usage patterns (boto3, botocore, google.cloud, azure, oci).
- Prints suggested replacement snippets (download_artifact / create_storage_client).
- If --apply is provided, writes <filename>.refactored.py with the suggested replacements applied
  (human-review step required before replacing original files).

Usage:
  python3 scripts/auto_refactor.py                # dry-run, print suggestions
  python3 scripts/auto_refactor.py --apply       # produce .refactored.py files for review
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple

# Patterns to match and suggested replacements (simple heuristics)
RULES: List[Tuple[re.Pattern, str]] = [
    # boto3 client s3 -> use create_storage_client(bucket) or registry_loader.download_artifact
    (re.compile(r"boto3\.client\(['\"]s3['\"]\)"), "create_storage_client(bucket=...)  # replace boto3.client('s3')"),
    (re.compile(r"boto3\.resource\(['\"]s3['\"]\)"), "create_storage_client(bucket=...)  # replace boto3.resource('s3')"),
    (re.compile(r"(\w+)\.download_file\(\s*([A-Za-z0-9_\"'\./:-]+)\s*,\s*([A-Za-z0-9_\"'\./:-]+)\s*,\s*([A-Za-z0-9_\"'\./:-]+)\s*\)"),
     "# REPLACE: use registry_loader.download_artifact or create_storage_client().download(key, local_path)"),
    (re.compile(r"import\s+boto3\b"), "# REPLACE: remove direct boto3 import; use storage.factory/create_storage_client or model_registry.loader"),
    (re.compile(r"import\s+botocore\b"), "# REPLACE: remove direct botocore usage in runtime"),
    (re.compile(r"import\s+google\.cloud\b"), "# REPLACE: remove google.cloud import in runtime; use storage.factory or adapter"),
    (re.compile(r"import\s+azure\b"), "# REPLACE: remove azure import in runtime; use storage.factory or adapter"),
    (re.compile(r"import\s+oci\b"), "# REPLACE: remove oci import in runtime; use storage.factory or adapter"),
]

EXCLUDE_DIRS = {".git", "venv", ".venv", "build", "dist", "__pycache__", "tests", "scripts"}

def scan_file(path: Path) -> List[Tuple[int, str, str]]:
    """Return list of (lineno, matched_text, suggestion)."""
    hits = []
    text = path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), start=1):
        for pat, suggestion in RULES:
            if pat.search(line):
                hits.append((i, line.rstrip(), suggestion))
    return hits

def apply_replacements(text: str) -> str:
    out = text
    for pat, suggestion in RULES:
        out = pat.sub(lambda m: f"{m.group(0)}  # {suggestion}", out)
    return out

def main(root: str, apply: bool):
    rootp = Path(root)
    findings = {}
    for p in rootp.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        hits = scan_file(p)
        if hits:
            findings[p] = hits

    if not findings:
        print("No direct SDK usage patterns found (in scanned runtime locations).")
        return 0

    for p, hits in sorted(findings.items()):
        print(f"\nFile: {p}")
        for ln, txt, suggestion in hits:
            print(f"  {ln:4d}: {txt}")
            print(f"       -> suggestion: {suggestion}")
        if apply:
            # create a .refactored.py file for human review
            new_text = apply_replacements(p.read_text(encoding="utf-8"))
            outp = p.with_name(p.name + ".refactored.py")
            outp.write_text(new_text, encoding="utf-8")
            print(f"  Wrote suggested refactor to {outp}")

    print("\nReview .refactored.py files, test them, then replace originals as appropriate.")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true", help="Write .refactored.py files for review")
    args = ap.parse_args()
    raise SystemExit(main(args.root, args.apply))
scripts/auto_refactor.py
