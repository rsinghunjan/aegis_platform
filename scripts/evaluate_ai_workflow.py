#!/usr/bin/env python
"""Run the configured AI workflow against a versioned golden dataset."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.ai_evaluation import evaluate_dataset  # noqa: E402
from services.ai_storage import AIDataStore  # noqa: E402
from services.ai_workflow import AIWorkflow  # noqa: E402


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
        raise ValueError("dataset must contain a cases array")
    for case in dataset["cases"]:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not isinstance(case.get("query"), str)
            or not isinstance(case.get("context"), str)
            or not isinstance(case.get("expected_facts"), list)
            or not case["expected_facts"]
            or any(not isinstance(fact, str) or not fact for fact in case["expected_facts"])
        ):
            raise ValueError("dataset contains an invalid golden case")
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/golden_datasets/ai_workflow.json",
    )
    parser.add_argument("--emit-json", type=Path)
    parser.add_argument(
        "--feedback-rating", type=int, action="append", default=[],
        help="Include a 1-5 agent feedback rating in the tuning review.",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
        with tempfile.TemporaryDirectory(prefix="aegis-evaluation-") as directory:
            workflow = AIWorkflow(
                data_store=AIDataStore(f"sqlite:///{Path(directory) / 'evaluation.db'}")
            )

            def answer_case(case: dict[str, Any]) -> dict[str, Any]:
                tenant_id = f"golden-{case['id']}"
                workflow.ingest(tenant_id, case["context"])
                return workflow.answer(tenant_id, case["query"])

            result = evaluate_dataset(
                dataset["cases"],
                answer_case,
                dataset["thresholds"],
                feedback_ratings=args.feedback_rating,
            )
        output = json.dumps(result, indent=2, sort_keys=True)
        if args.emit_json:
            args.emit_json.parent.mkdir(parents=True, exist_ok=True)
            args.emit_json.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0 if result["status"] == "passed" else 2
    except Exception as exc:
        print(f"AI evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
