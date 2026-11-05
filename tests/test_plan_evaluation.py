import json

import pytest

from agentic.planner import OpenAICompatiblePlanner
from agentic.runtime import DeterministicJSONPlanner
from services.plan_evaluation import evaluate_plans

TOOLS = {"search_documents", "summarize"}
THRESHOLDS = {"llm_plan_rate": 1.0, "tool_sequence_accuracy": 1.0, "input_key_recall": 1.0}
CASES = [
    {
        "id": "a",
        "goal": "find and summarize",
        "expected_tools": ["search_documents", "summarize"],
        "expected_input_keys": ["query"],
    }
]
GOOD = json.dumps(
    {
        "steps": [
            {"tool": "search_documents", "input": {"query": "x"}},
            {"tool": "summarize", "input": {}},
        ]
    }
)


def _run(response):
    planner = OpenAICompatiblePlanner("https://p.example/v1", "m", "k")

    def plan_case(case):
        plan = planner.plan_with_inference(case["goal"], TOOLS, lambda _p: response)
        return plan, planner.last_source

    return evaluate_plans(CASES, plan_case, THRESHOLDS)


def test_good_llm_plan_passes():
    assert _run(GOOD)["status"] == "passed"


def test_fallback_plan_is_flagged_as_regression():
    planner = OpenAICompatiblePlanner(
        "https://p.example/v1", "m", "k", fallback=DeterministicJSONPlanner()
    )

    def plan_case(case):
        plan = planner.plan_with_inference(
            '{"tool":"summarize","input":{}}', TOOLS, lambda _p: "not json"
        )
        return plan, planner.last_source

    result = evaluate_plans(CASES, plan_case, THRESHOLDS)
    assert result["status"] == "failed"
    assert "llm_plan_rate" in result["regressions"]


def test_wrong_tool_sequence_fails():
    wrong = json.dumps({"steps": [{"tool": "summarize", "input": {}}]})
    assert "tool_sequence_accuracy" in _run(wrong)["regressions"]


def test_planner_enabled_by_default(monkeypatch):
    monkeypatch.delenv("AEGIS_ALLOW_DEV_PROVIDERS", raising=False)
    monkeypatch.delenv("AEGIS_LLM_PLANNER_ENABLED", raising=False)
    monkeypatch.delenv("AEGIS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AEGIS_LLM_API_KEY"):
        OpenAICompatiblePlanner.from_environment()
    monkeypatch.setenv("AEGIS_LLM_API_KEY", "k")
    assert isinstance(OpenAICompatiblePlanner.from_environment(), OpenAICompatiblePlanner)
    monkeypatch.setenv("AEGIS_LLM_PLANNER_ENABLED", "false")
    assert isinstance(OpenAICompatiblePlanner.from_environment(), DeterministicJSONPlanner)
