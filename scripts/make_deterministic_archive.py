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
#!/usr/bin/env python3
"""
Create a deterministic tar.gz of a SavedModel directory suitable for stable hashing.

- Sort files by path
- Use a fixed mtime (EPOCH) for every TarInfo entry
- Set uid/gid to 0 and uname/gname to empty strings to reduce variability
- Write gzip compressed tar and print sha256 + size to stdout

Usage:
  python3 scripts/make_deterministic_archive.py /path/to/saved_model /tmp/output-model.tar.gz
"""
from __future__ import annotations
import sys
import tarfile
import hashlib
import time
from pathlib import Path

FIXED_MTIME = 0  # unix epoch

def add_dir_to_tar(tar: tarfile.TarFile, base: Path, root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        arcname = str(p.relative_to(base))
        ti = tar.gettarinfo(str(p), arcname=arcname)
        ti.mtime = FIXED_MTIME
        ti.uid = 0
        ti.gid = 0
        ti.uname = ""
        ti.gname = ""
        with p.open("rb") as fh:
            tar.addfile(ti, fh)

def make_archive(src_dir: Path, out_path: Path) -> None:
    src_dir = src_dir.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # create gzipped tar
    with tarfile.open(str(out_path), "w:gz", format=tarfile.gnu) as tar:
        add_dir_to_tar(tar, src_dir, src_dir)
    # compute sha256 and size
    h = hashlib.sha256()
    size = 0
    with out_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            size += len(chunk)
            h.update(chunk)
    print(out_path, flush=True)
    print("sha256:", h.hexdigest(), flush=True)
    print("size_bytes:", size, flush=True)

def main():
    if len(sys.argv) < 3:
        print("Usage: make_deterministic_archive.py <saved_model_dir> <out_tar_gz>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not src.exists():
        print("Source not found:", src, file=sys.stderr)
        sys.exit(2)
    make_archive(src, out)

if __name__ == "__main__":
    main()
scripts/make_deterministic_archive.py
