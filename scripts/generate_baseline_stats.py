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
#!/usr/bin/env python
Generate baseline histograms for numeric features and save as JSON.

Usage:
  python scripts/generate_baseline_stats.py --input path/to/historical.csv --output baseline_stats.json --features feat1,feat2 --bins 10

The output format:
{
  "feat1": {
    "bins": [b0, b1, ..., bN],
    "counts": [c0, c1, ..., cN-1]
  },
  ...
}
"""
from __future__ import annotations
import argparse
import json
import os
import pandas as pd
import numpy as np
from typing import List

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--features", required=False, default="")
    p.add_argument("--bins", type=int, default=10)
    return p.parse_args()

def compute_hist(df: pd.DataFrame, col: str, bins: int):
    arr = df[col].dropna().values
    if len(arr) == 0:
        return None
    hist, bin_edges = np.histogram(arr, bins=bins)
    return {"bins": bin_edges.tolist(), "counts": hist.astype(int).tolist()}

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    feature_list = [f.strip() for f in args.features.split(",")] if args.features else list(df.select_dtypes(include=[np.number]).columns)
    out = {}
    for f in feature_list:
        if f not in df.columns:
            continue
        res = compute_hist(df, f, args.bins)
        if res:
            out[f] = res
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote baseline stats to", args.output)

if __name__ == "__main__":
    main()
