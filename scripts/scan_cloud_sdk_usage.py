#!/usr/bin/env python3
"""
Scan the repository for direct cloud SDK imports and surface files/lines.

Searches for:
 - import boto3
 - from boto3 import ...
 - import google.cloud
 - import azure
 - import oci

Usage:
  python3 scripts/scan_cloud_sdk_usage.py [--root /path/to/repo] [--strict]

This script does NOT rewrite files automatically (safe). It prints suggestions and a quick
replacement template you can use to refactor to model_registry.loader.download_artifact or
aegis_multimodal_ai_system.storage.factory.create_storage_client().
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

PATTERNS = [
    (re.compile(r'^\s*import\s+boto3\b'), "boto3"),
    (re.compile(r'^\s*from\s+boto3\b'), "boto3"),
    (re.compile(r'^\s*import\s+botocore\b'), "botocore"),
    (re.compile(r'^\s*import\s+google\.cloud\b'), "google-cloud"),
    (re.compile(r'^\s*from\s+google\.cloud\b'), "google-cloud"),
    (re.compile(r'^\s*import\s+azure\b'), "azure-sdk"),
    (re.compile(r'^\s*from\s+azure\b'), "azure-sdk"),
    (re.compile(r'^\s*import\s+oci\b'), "oci"),
    (re.compile(r'^\s*from\s+oci\b'), "oci"),
]

EXCLUDE_DIRS = {".git", "venv", ".venv", "node_modules", "build", "dist", "__pycache__"}

def scan(root: Path, files: List[str] | None = None) -> List[Tuple[Path, int, str]]:
    findings = []
    paths = (
        [Path(file) if Path(file).is_absolute() else root / file for file in files]
        if files is not None
        else root.rglob("*.py")
    )
    for p in paths:
        if p.suffix != ".py":
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for pat, name in PATTERNS:
                if pat.search(line):
                    findings.append((p, i, line.strip()))
    return findings

def print_suggestions(findings: List[Tuple[Path,int,str]]):
    if not findings:
        print("No direct cloud SDK imports found.")
        return
    print("Found direct cloud SDK imports. Suggested action: replace direct SDK calls with:")
    print("  - model_registry.loader.download_artifact(uri)  (for artifacts)")
    print("  - aegis_multimodal_ai_system.storage.factory.create_storage_client() (for generic object store usage)")
    print()
    by_file = {}
    for p, ln, txt in findings:
        by_file.setdefault(p, []).append((ln, txt))
    for p, entries in sorted(by_file.items()):
        print(f"\nFile: {p}")
        for ln, txt in entries:
            print(f"  {ln:4d}: {txt}")
        print("  Quick refactor example (artifact download):")
        print("    # BEFORE: uses boto3 client directly")
        print("    s3 = boto3.client('s3')\n    s3.download_file(bucket, key, local_path)")
        print("    # AFTER: use registry loader")
        print("    from aegis_multimodal_ai_system.model_registry import loader as registry_loader")
        print("    local_path = registry_loader.download_artifact('s3://{bucket}/{key}')")
        print("  Or use create_storage_client():")
        print("    from aegis_multimodal_ai_system.storage.factory import create_storage_client")
        print("    client = create_storage_client(bucket='...')\n    client.download(key, local_path)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root to scan")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when direct SDK imports are found")
    ap.add_argument("--files", nargs="*", help="Scan only the supplied files")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print(f"Scanning {root} for direct cloud SDK imports...")
    findings = scan(root, args.files)
    print_suggestions(findings)
    if args.strict and findings:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
