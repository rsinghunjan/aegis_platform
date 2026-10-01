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
#!/usr/bin/env python3
"""
Create a deterministic tar.gz archive of a directory.

Usage:
  python3 scripts/make_deterministic_archive.py <source_dir> <out_tar_gz>

Deterministic behavior:
 - Sort files by path
 - Set mtime to a fixed timestamp (Jan 1 2020)
 - Use reproducible gzip level and deterministic owner/group bits
"""
from __future__ import annotations
import argparse
import tarfile
import time
from pathlib import Path

FIXED_MTIME = int(time.mktime((2020,1,1,0,0,0,0,0,0)))

def make_deterministic_tar(source_dir: Path, out_path: Path):
    with tarfile.open(out_path, "w:gz", format=tarfile.PAX_FORMAT, compresslevel=6) as tar:
        for p in sorted(source_dir.rglob("*")):
            if p.is_dir():
                continue
            arcname = str(p.relative_to(source_dir))
            ti = tar.gettarinfo(str(p), arcname=arcname)
            ti.mtime = FIXED_MTIME
            ti.uname = "root"
            ti.gname = "root"
            ti.uid = 0
            ti.gid = 0
            # addfile requires a fileobj for regular files
            with open(p, "rb") as f:
                tar.addfile(ti, fileobj=f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("out_tar", type=Path)
    args = ap.parse_args()
    make_deterministic_tar(args.source_dir, args.out_tar)
    print("Wrote deterministic archive:", args.out_tar)

if __name__ == "__main__":
    main()
