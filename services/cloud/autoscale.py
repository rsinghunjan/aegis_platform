"""Simple rule-based auto-scaling policies."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScalingDecision(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


@dataclass
class AutoScalingPolicy:
    """Threshold-based horizontal autoscaler.

    Mirrors the shape of a Kubernetes HPA: scale up when utilization
    exceeds ``target_utilization`` by more than ``tolerance``, scale down
    when it falls well below, otherwise hold steady. Bounded by
    ``min_replicas``/``max_replicas``.
    """

    min_replicas: int = 1
    max_replicas: int = 10
    target_utilization: float = 0.7
    tolerance: float = 0.1

    def decide(self, current_replicas: int, current_utilization: float) -> ScalingDecision:
        if current_utilization > self.target_utilization + self.tolerance:
            return ScalingDecision.SCALE_UP
        if current_utilization < self.target_utilization - self.tolerance:
            return ScalingDecision.SCALE_DOWN
        return ScalingDecision.HOLD

    def next_replica_count(self, current_replicas: int, current_utilization: float) -> int:
        decision = self.decide(current_replicas, current_utilization)
        if decision is ScalingDecision.SCALE_UP:
            desired = max(current_replicas + 1, round(current_replicas * current_utilization / self.target_utilization))
        elif decision is ScalingDecision.SCALE_DOWN:
            desired = min(current_replicas - 1, round(current_replicas * current_utilization / self.target_utilization) or current_replicas - 1)
        else:
            desired = current_replicas
        return max(self.min_replicas, min(self.max_replicas, desired))
