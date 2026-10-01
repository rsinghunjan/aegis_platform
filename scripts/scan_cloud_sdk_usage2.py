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
Scan repository for direct cloud SDK imports (boto3, google.cloud, azure, oci, botocore).
Exits with non-zero if any findings are present so CI can fail PRs that include direct SDK usage.

This is a stricter variant of the earlier scanner that enforces the "no direct provider SDKs"
policy for runtime code. Tests/tools/scripts may still import providers (the scanner excludes tests/ and scripts/ by flag).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

# Patterns of imports to disallow in runtime directories
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

EXCLUDE_DIRS = {"tests", "scripts", "docker", "infra"sc
