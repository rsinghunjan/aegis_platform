#!/usr/bin/env python
"""Merge operator-reviewed production failures into the regression golden set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.regression_sets import merge_cases  # noqa: E402

DEFAULT_THRESHOLDS = {
    "expected_fact_recall": 1.0,
    "groundedness": 0.7,
    "citation_coverage": 1.0,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed", type=Path, help="JSON array of reviewed cases")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/golden_datasets/regression.json"
    )
    args = parser.parse_args(argv)
    dataset = (
        json.loads(args.dataset.read_text(encoding="utf-8"))
        if args.dataset.exists()
        else {"thresholds": DEFAULT_THRESHOLDS, "cases": []}
    )
    reviewed = json.loads(args.reviewed.read_text(encoding="utf-8"))
    if not isinstance(reviewed, list):
        print("reviewed file must be a JSON array", file=sys.stderr)
        return 1
    dataset["cases"], rejected = merge_cases(dataset["cases"], reviewed)
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    args.dataset.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(dataset["cases"]), "rejected": rejected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
