import json
from pathlib import Path

import pytest

from services.ai_evaluation import evaluate_dataset, score_answer
from scripts.evaluate_ai_workflow import load_dataset


def test_score_answer_measures_facts_grounding_and_citations():
    score = score_answer(
        "Aegis combines retrieval with governed AI workflows.",
        "Aegis combines retrieval with governed AI workflows.",
        ["Aegis combines retrieval", "governed AI workflows"],
        [{"document_id": "doc-1"}],
    )

    assert score == {
        "expected_fact_recall": 1.0,
        "groundedness": 1.0,
        "citation_coverage": 1.0,
    }


def test_evaluation_fails_regressions_and_recommends_approval_gated_tuning():
    cases = [
        {
            "id": "case-1",
            "context": "Aegis combines retrieval with governed AI workflows.",
            "expected_facts": ["governed AI workflows"],
        }
    ]
    result = evaluate_dataset(
        cases,
        lambda _case: {
            "answer": "Aegis provides fast cryptocurrency trading.",
            "citations": [],
        },
        {
            "expected_fact_recall": 1.0,
            "groundedness": 0.8,
            "citation_coverage": 1.0,
        },
        feedback_ratings=[2, 3],
    )

    assert result["status"] == "failed"
    assert set(result["regressions"]) == {
        "expected_fact_recall",
        "groundedness",
        "citation_coverage",
    }
    assert result["feedback"] == {"rating_count": 2, "average_rating": 2.5}
    assert {item["area"] for item in result["tuning_recommendations"]} == {
        "retrieval",
        "prompt",
        "routing",
    }
    assert all(
        item["requires_human_approval"]
        for item in result["tuning_recommendations"]
    )


def test_evaluation_rejects_invalid_feedback_rating():
    with pytest.raises(ValueError, match="feedback ratings"):
        evaluate_dataset(
            [{"id": "case", "context": "source", "expected_facts": ["source"]}],
            lambda _case: {"answer": "source", "citations": [1]},
            {
                "expected_fact_recall": 0,
                "groundedness": 0,
                "citation_coverage": 0,
            },
            feedback_ratings=[6],
        )


def test_golden_dataset_has_cases_and_valid_thresholds():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/golden_datasets/ai_workflow.json"
    )
    dataset = load_dataset(path)
    assert dataset["cases"]
    assert set(dataset["thresholds"]) == {
        "expected_fact_recall",
        "groundedness",
        "citation_coverage",
    }


def test_dataset_rejects_malformed_case(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps({"cases": [{"id": "bad"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid golden case"):
        load_dataset(path)
