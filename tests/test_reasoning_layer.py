"""Tests for agentic/reasoning: decomposition, prompting, uncertainty,
refinement, bandits, verification."""
import pytest

from agentic.reasoning import (
    decompose_goal,
    build_chain_of_thought_prompt,
    build_tree_of_thought_prompt,
    score_confidence,
    PlanRefiner,
    EpsilonGreedyBandit,
    UCB1Bandit,
    verify_plan,
)
from agentic.reasoning.verification import backtrack_to_last_valid_step


def test_decompose_goal_sequential_delimiters():
    subgoals = decompose_goal("Fetch the data; then clean it and then summarize it")
    texts = [s.text for s in subgoals]
    assert texts == ["Fetch the data", "clean it", "summarize it"]
    assert subgoals[0].depends_on == ()
    assert subgoals[1].depends_on == (0,)


def test_decompose_goal_single_sentence():
    subgoals = decompose_goal("Do one thing")
    assert len(subgoals) == 1
    assert subgoals[0].text == "Do one thing"


def test_decompose_goal_custom_decomposer():
    subgoals = decompose_goal("ignored", decomposer_fn=lambda g: ["a", "b", "c"])
    assert [s.text for s in subgoals] == ["a", "b", "c"]


def test_chain_of_thought_prompt_contains_goal():
    prompt = build_chain_of_thought_prompt("Summarize the report", context="Report: ...")
    assert "Summarize the report" in prompt
    assert "step by step" in prompt
    assert "Report: ..." in prompt


def test_tree_of_thought_prompt_has_branches():
    prompt = build_tree_of_thought_prompt("Pick a strategy", num_branches=2)
    assert "Branch 1" in prompt
    assert "Branch 2" in prompt
    assert "Branch 3" not in prompt


def test_score_confidence_penalizes_hedging_and_short_answers():
    confident = score_confidence("The answer is 42.")
    hedging = score_confidence("I think it might possibly be 42, I'm not sure.")
    assert confident.value > hedging.value
    assert confident.is_confident

    empty = score_confidence("")
    assert not empty.is_confident


def test_score_confidence_uses_self_consistency():
    consistent = score_confidence("42", samples=["42", "42", "42"])
    inconsistent = score_confidence("42", samples=["42", "7", "99"])
    assert consistent.value > inconsistent.value


def test_plan_refiner_converges():
    calls = {"count": 0}

    def critique(plan):
        calls["count"] += 1
        return None if "final" in plan else "needs more detail"

    def revise(plan, critique_msg):
        return plan + " final"

    refiner = PlanRefiner(critique, revise, max_iterations=5)
    result = refiner.refine("draft plan")
    assert result.converged
    assert "final" in result.final_plan
    assert result.iterations == 1


def test_plan_refiner_stops_at_max_iterations():
    refiner = PlanRefiner(
        critique_fn=lambda plan: "always needs work",
        revise_fn=lambda plan, msg: plan + "+",
        max_iterations=3,
    )
    result = refiner.refine("start")
    assert not result.converged
    assert result.iterations == 3
    assert result.final_plan == "start+++"


def test_epsilon_greedy_bandit_exploits_best_arm():
    bandit = EpsilonGreedyBandit(["a", "b"], epsilon=0.0, seed=42)
    bandit.update("a", 1.0)
    bandit.update("b", 0.0)
    assert bandit.select() == "a"


def test_ucb1_bandit_tries_all_arms_first():
    bandit = UCB1Bandit(["a", "b", "c"])
    selections = set()
    for _ in range(3):
        arm = bandit.select()
        selections.add(arm)
        bandit.update(arm, 1.0)
    assert selections == {"a", "b", "c"}


def test_verify_plan_flags_missing_tool_and_disallowed_tool():
    steps = [{"tool": ""}, {"tool": "not_allowed"}, {"tool": "fetch", "input": {}}]
    issues = verify_plan(steps, allowed_tools=["fetch"])
    assert any(i.step_index == 0 for i in issues)
    assert any(i.step_index == 1 for i in issues)
    assert not any(i.step_index == 2 for i in issues)


def test_verify_plan_flags_duplicate_steps_as_warning():
    steps = [{"tool": "fetch", "input": {}}, {"tool": "fetch", "input": {}}]
    issues = verify_plan(steps)
    assert issues[0].severity == "warning"


def test_backtrack_to_last_valid_step_truncates_at_first_error():
    steps = [{"tool": "a"}, {"tool": "b"}, {"tool": ""}, {"tool": "d"}]
    issues = verify_plan(steps)
    truncated = backtrack_to_last_valid_step(steps, issues)
    assert truncated == [{"tool": "a"}, {"tool": "b"}]
