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
Check for major version bumps between origin/main and the PR branch for pinned dependencies.

Behavior:
- For each requirements file at repo root (requirements*.txt, requirements.in), compares versions pinned with '=='
- If package present in both base and head and the MAJOR version increased (e.g., 1.x.x -> 2.x.x),
  the script writes a JSON report to /tmp/major_bumps.json and exits with code 1.

Requirements:
- Run in a GitHub Actions job that checked out full history (fetch-depth: 0)
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
from packaging.version import parse as parse_version

ROOT = Path.cwd()
PATTERNS = ["requirements.txt", "requirements.in"]
# also consider any other requirements*.txt
PATTERNS += [p.name for p in ROOT.glob("requirements*.txt") if p.name not in PATTERNS]

REQ_REGEX = re.compile(r"^\s*([A-Za-z0-9_.+-]+)\s*==\s*([0-9a-zA-Z.\-+]+)")

def read_versions_from_text(text: str) -> dict:
    result = {}
    for ln in text.splitlines():
        m = REQ_REGEX.match(ln)
        if m:
            pkg = m.group(1).lower()
            ver = m.group(2)
            try:
                parsed = parse_version(ver)
            except Exception:
                parsed = None
            result[pkg] = {"raw": ver, "parsed": parsed}
    return result

def git_show(ref: str, path: str) -> str | None:
    try:
        out = subprocess.check_output(["git", "show", f"{ref}:{path}"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8")
    except subprocess.CalledProcessError:
        return None

def main():
    base_ref = "origin/main"
    major_bumps = []
    for fname in sorted(set(PATTERNS)):
        fpath = ROOT / fname
        # get base content
        base_content = git_show(base_ref, fname)
        head_content = None
        if fpath.exists():
            head_content = fpath.read_text(encoding="utf-8")
        if not base_content or not head_content:
            continue
        base_vers = read_versions_from_text(base_content)
        head_vers = read_versions_from_text(head_content)
        for pkg, headinfo in head_vers.items():
            if pkg in base_vers:
                baseinfo = base_vers[pkg]
                try:
                    base_parsed = baseinfo["parsed"]
                    head_parsed = headinfo["parsed"]
                except Exception:
                    base_parsed = None
                    head_parsed = None
                if base_parsed and head_parsed:
                    base_rel = getattr(base_parsed, "release", ())
                    head_rel = getattr(head_parsed, "release", ())
                    base_major = base_rel[0] if base_rel else 0
                    head_major = head_rel[0] if head_rel else 0
                    if head_major > base_major:
                        major_bumps.append({
                            "file": fname,
                            "package": pkg,
                            "old_version": baseinfo["raw"],
                            "new_version": headinfo["raw"],
                            "old_major": base_major,
                            "new_major": head_major
                        })
    if major_bumps:
        with open("/tmp/major_bumps.json","w",encoding="utf-8") as fh:
            json.dump(major_bumps, fh, indent=2)
        print("Major bumps detected:")
        print(json.dumps(major_bumps, indent=2))
        # exit with non-zero so callers can detect easily
        raise SystemExit(1)
    else:
        print("No major dependency bumps detected.")
        raise SystemExit(0)

if __name__ == "__main__":
    main()
