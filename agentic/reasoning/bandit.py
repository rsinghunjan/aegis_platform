"""Multi-armed bandit strategies for tool selection optimization."""
from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


class EpsilonGreedyBandit:
    """Epsilon-greedy bandit: explores randomly with probability ``epsilon``,
    otherwise exploits the best-known arm (tool).
    """

    def __init__(self, arms: List[str], epsilon: float = 0.1, seed: Optional[int] = None):
        if not arms:
            raise ValueError("bandit requires at least one arm")
        self.arms = list(arms)
        self.epsilon = epsilon
        self._stats: Dict[str, ArmStats] = {arm: ArmStats() for arm in arms}
        self._lock = threading.Lock()
        self._random = random.Random(seed)

    def select(self) -> str:
        with self._lock:
            if self._random.random() < self.epsilon:
                return self._random.choice(self.arms)
            return max(self.arms, key=lambda arm: self._stats[arm].average_reward)

    def update(self, arm: str, reward: float) -> None:
        if arm not in self._stats:
            raise KeyError(arm)
        with self._lock:
            stats = self._stats[arm]
            stats.pulls += 1
            stats.total_reward += reward

    def stats(self) -> Dict[str, ArmStats]:
        with self._lock:
            return dict(self._stats)


class UCB1Bandit:
    """Upper Confidence Bound (UCB1) bandit for tool selection.

    Balances exploration/exploitation without a tunable epsilon by
    preferring arms with high uncertainty (few pulls) early on.
    """

    def __init__(self, arms: List[str]):
        if not arms:
            raise ValueError("bandit requires at least one arm")
        self.arms = list(arms)
        self._stats: Dict[str, ArmStats] = {arm: ArmStats() for arm in arms}
        self._total_pulls = 0
        self._lock = threading.Lock()

    def select(self) -> str:
        with self._lock:
            # Ensure every arm is tried at least once first.
            for arm in self.arms:
                if self._stats[arm].pulls == 0:
                    return arm

            def ucb_score(arm: str) -> float:
                stats = self._stats[arm]
                bonus = math.sqrt(2 * math.log(self._total_pulls) / stats.pulls)
                return stats.average_reward + bonus

            return max(self.arms, key=ucb_score)

    def update(self, arm: str, reward: float) -> None:
        if arm not in self._stats:
            raise KeyError(arm)
        with self._lock:
            stats = self._stats[arm]
            stats.pulls += 1
            stats.total_reward += reward
            self._total_pulls += 1

    def stats(self) -> Dict[str, ArmStats]:
        with self._lock:
            return dict(self._stats)
