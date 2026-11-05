"""Golden-dataset evaluator used to gate prompt promotion."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from services.ai_evaluation import evaluate_dataset
from services.ai_storage import AIDataStore

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data/golden_datasets/ai_workflow.json"


def build_prompt_evaluator(
    workflow: Any, dataset_path: Path = DEFAULT_DATASET
) -> Callable[[str], dict[str, Any]]:
    """Return ``evaluate(template) -> report`` using an isolated scratch store."""

    def evaluate(template: str) -> dict[str, Any]:
        from services.ai_workflow import AIWorkflow

        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="aegis-prompt-eval-") as directory:
            scratch = AIWorkflow(
                inference_router=workflow.inference_router,
                embedding_provider=workflow.embedding_provider,
                data_store=AIDataStore(f"sqlite:///{Path(directory) / 'eval.db'}"),
            )

            def answer_case(case: dict[str, Any]) -> dict[str, Any]:
                tenant = f"golden-{case['id']}"
                scratch.ingest(tenant, case["context"])
                return scratch.answer(tenant, case["query"], prompt_template=template)

            return evaluate_dataset(dataset["cases"], answer_case, dataset["thresholds"])

    return evaluate
