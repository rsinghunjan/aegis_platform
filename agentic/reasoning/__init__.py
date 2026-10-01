"""Advanced reasoning: decomposition, prompting strategies, uncertainty,
refinement, tool-selection bandits, and plan verification/backtracking.
"""

from .decomposition import Subgoal, decompose_goal
from .prompting import build_chain_of_thought_prompt, build_tree_of_thought_prompt
from .uncertainty import ConfidenceScore, score_confidence
from .refinement import PlanRefiner, RefinementResult
from .bandit import EpsilonGreedyBandit, UCB1Bandit
from .verification import VerificationIssue, verify_plan

__all__ = [
    "Subgoal",
    "decompose_goal",
    "build_chain_of_thought_prompt",
    "build_tree_of_thought_prompt",
    "ConfidenceScore",
    "score_confidence",
    "PlanRefiner",
    "RefinementResult",
    "EpsilonGreedyBandit",
    "UCB1Bandit",
    "VerificationIssue",
    "verify_plan",
]
