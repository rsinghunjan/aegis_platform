"""Chain-of-thought and tree-of-thought prompt builders.

These helpers build prompt strings for an LLM planner/reasoner; they do
not call any model themselves, keeping this module dependency-free and
easy to unit test.
"""
from __future__ import annotations

from typing import List, Optional


def build_chain_of_thought_prompt(goal: str, context: Optional[str] = None) -> str:
    context_block = f"Context:\n{context}\n\n" if context else ""
    return (
        f"{context_block}"
        f"Goal: {goal}\n\n"
        "Think through this step by step before answering. Enumerate the "
        "reasoning steps required, then provide a final concise answer "
        "prefixed with 'Final Answer:'."
    )


def build_tree_of_thought_prompt(
    goal: str, num_branches: int = 3, context: Optional[str] = None
) -> str:
    context_block = f"Context:\n{context}\n\n" if context else ""
    branch_lines = "\n".join(
        f"Branch {i + 1}: propose one distinct approach and evaluate its viability (1-10)."
        for i in range(num_branches)
    )
    return (
        f"{context_block}"
        f"Goal: {goal}\n\n"
        f"Explore {num_branches} independent reasoning branches before committing to a plan:\n"
        f"{branch_lines}\n\n"
        "After evaluating all branches, select the highest-viability branch "
        "and provide a final concise answer prefixed with 'Final Answer:'."
    )
