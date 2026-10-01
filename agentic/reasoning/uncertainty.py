"""Uncertainty quantification and confidence scoring.

No model-internal logprobs are assumed to be available (many providers do
not expose them); instead this module combines lightweight, observable
signals into a bounded [0, 1] confidence score:

* self-consistency across repeated samples (if provided)
* response length / specificity heuristics
* presence of hedging language ("might", "possibly", "I'm not sure")
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_HEDGE_WORDS = (
    "might",
    "may",
    "possibly",
    "perhaps",
    "i'm not sure",
    "i am not sure",
    "unclear",
    "uncertain",
    "i think",
    "probably",
)


@dataclass
class ConfidenceScore:
    value: float
    rationale: List[str]

    @property
    def is_confident(self) -> bool:
        return self.value >= 0.6


def _self_consistency(samples: List[str]) -> float:
    if len(samples) < 2:
        return 1.0
    normalized = [re.sub(r"\s+", " ", s.strip().lower()) for s in samples]
    most_common = max(set(normalized), key=normalized.count)
    agreement = normalized.count(most_common) / len(normalized)
    return agreement


def _hedge_penalty(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for word in _HEDGE_WORDS if word in lowered)
    return min(0.5, hits * 0.15)


def score_confidence(response: str, samples: Optional[List[str]] = None) -> ConfidenceScore:
    rationale: List[str] = []
    base = 0.8
    rationale.append(f"base_score={base}")

    if samples:
        consistency = _self_consistency(samples)
        base = base * 0.5 + consistency * 0.5
        rationale.append(f"self_consistency={consistency:.2f}")

    penalty = _hedge_penalty(response)
    if penalty:
        base -= penalty
        rationale.append(f"hedge_penalty=-{penalty:.2f}")

    if len(response.strip()) < 3:
        base -= 0.3
        rationale.append("penalty=-0.30 (response too short)")

    value = max(0.0, min(1.0, base))
    return ConfidenceScore(value=value, rationale=rationale)
