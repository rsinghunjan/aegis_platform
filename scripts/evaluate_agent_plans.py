#!/usr/bin/env python
"""Evaluate the configured LLM planner against a golden planning dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic.planner import OpenAICompatiblePlanner  # noqa: E402
from services.inference import InferenceRequest  # noqa: E402
from services.inference.gateway import build_inference_router  # noqa: E402
from services.plan_evaluation import evaluate_plans  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/golden_datasets/agent_plans.json"
    )
    parser.add_argument("--emit-json", type=Path)
    args = parser.parse_args(argv)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        router = build_inference_router()
        planner = OpenAICompatiblePlanner("https://planner.invalid/v1", "unused")
        tools = set(dataset["tools"])

        def generate(prompt: str) -> str:
            return router.generate(
                InferenceRequest(prompt=prompt, max_tokens=1024)
            ).text

        def plan_case(case):
            plan = planner.plan_with_inference(case["goal"], tools, generate)
            return plan, planner.last_source

        result = evaluate_plans(dataset["cases"], plan_case, dataset["thresholds"])
        output = json.dumps(result, indent=2, sort_keys=True)
        if args.emit_json:
            args.emit_json.parent.mkdir(parents=True, exist_ok=True)
            args.emit_json.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0 if result["status"] == "passed" else 2
    except Exception as exc:
        print(f"Plan evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
