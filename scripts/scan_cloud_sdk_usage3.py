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
#!/usr/bin/env python3
"""
Strict scanner used by pre-commit & CI to ensure runtime code does not import cloud SDKs.
(If you already have a copy, ensure CI uses --strict.)

Usage:
  python3 scripts/scan_cloud_sdk_usage.py --strict
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

DISALLOWED_PATTERNS = [
    re.compile(r'^\s*import\s+boto3\b'),
    re.compile(r'^\s*from\s+boto3\b'),
    re.compile(r'^\s*import\s+botocore\b'),
    re.compile(r'^\s*import\s+google\.cloud\b'),
    re.compile(r'^\s*from\s+google\.cloud\b'),
    re.compile(r'^\s*import\s+azure\b'),
    re.compile(r'^\s*from\s+azure\b'),
    re.compile(r'^\s*import\s+oci\b'),
    re.compile(r'^\s*from\s+oci\b'),
]

EXCLUDE_DIRS = {"tests", "scripts", "docker", "infra", ".venv", "venv", "data", ".git"}

def scan(root: Path):
    findings = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in DISALLOWED_PATTERNS:
                if pat.search(line):
                    findings.append((p, i, line.strip()))
    return findings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    findings = scan(Path(args.root))
    if not findings:
        print("No disallowed cloud SDK imports found.")
        return 0
    print("Disallowed direct cloud SDK imports found (must refactor to use StorageClient/loader):")
    for p, ln, txt in findings:
        print(f"{p}:{ln}: {txt}")
    if args.strict:
        print("\nStrict mode enabled - exiting non-zero to fail CI/pre-commit.")
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
scripts/scan_cloud_sdk_usage.py
