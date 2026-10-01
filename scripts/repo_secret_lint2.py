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
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
#!/usr/bin/env python3
"""
Repo secret linter for key/leak patterns.

Exit codes:
  0 => no issues
  2 => matches found

Usage:
  python3 scripts/repo_secret_lint.py [path ...]

Notes:
  - Skips common binary / vendored directories.
  - Designed to be conservative (look for key patterns, not exact secrets).
  - Add patterns to PATTERNS as needed.
"""
import sys
import re
from pathlib import Path

# Patterns to detect likely private-key material or cosign private-key env usage.
PATTERNS = [
    re.compile(r"COSIGN_PRIVATE_KEY_B64"),
    re.compile(r"COSIGN_PASSWORD"),
    re.compile(r"PRIVATE_KEY_B64"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"-----BEGIN RSA PRIVATE KEY-----"),
    re.compile(r"-----BEGIN EC PRIVATE KEY-----"),
    re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
    re.compile(r"(?i)secretkey"),
    re.compile(r"(?i)aws_secret_access_key"),
]

# Files/dirs to skip scanning
SKIP_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    "__pycache__",
    ".venv",
    "venv",
    ".terraform",
    ".gradle",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "bin",
    "obj",
}

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".so", ".dll", ".zip", ".tar.gz", ".tgz", ".db", ".sqlite3", ".bin"}

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

def should_skip(path: Path):
    parts = set(p.name for p in path.resolve().parts)
    if parts & SKIP_DIRS:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False

def main(argv):
    if len(argv) <= 1:
        paths = ["."]
    else:
        paths = argv[1:]
    found = []
    for p in paths:
        base = Path(p)
        if base.is_file():
            if not should_skip(base):
                found += scan_file(base)
            continue
        for f in base.rglob("*"):
            if f.is_file() and not should_skip(f):
                found += scan_file(f)
    if found:
        print("Potential secret/key patterns found:")
        for fn, pattern in found:
            print(f" - {fn}: matches {pattern}")
        return 2
    print("No secret signing patterns found.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
scripts/repo_secret_lint.py
