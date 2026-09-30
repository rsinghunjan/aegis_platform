#!/usr/bin/env python3
"""
Simple repo linter that fails if sensitive signing keys or patterns are added.
Exit code 0 => no issues, non-zero => found issues.

Usage:
  python3 scripts/repo_secret_lint.py [path ...]
"""
import sys
import re
from pathlib import Path

PATTERNS = [
    re.compile(r"COSIGN_" + r"PRIVATE_" + r"KEY_B64"),
    re.compile(r"COSIGN_" + r"PASS" + r"WORD"),
    re.compile(r"-----BEGIN .*" + r"PRIVATE" + r" KEY-----"),
    re.compile(r"PRIVATE_" + r"KEY_B64"),
]

def scan_file(path: Path):
    issues = []
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return issues
    for p in PATTERNS:
        if p.search(text):
            issues.append((str(path), p.pattern))
    return issues

def main(paths):
    if not paths:
        paths = ["."]
    found = []
    for p in paths:
        base = Path(p)
        files = [base] if base.is_file() else base.rglob("*")
        for f in files:
            if (
                f.is_file()
                and f.suffix not in {".png", ".jpg", ".jpeg", ".gif", ".so", ".bin"}
            ):
                found += scan_file(f)
    if found:
        print("Potential secret/key patterns found:")
        for fn, pattern in found:
            print(f" - {fn}: matches {pattern}")
        return 2
    print("No secret signing patterns found.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
