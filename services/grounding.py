"""Sentence-level grounding check of an answer against retrieved context."""
from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "has", "have", "its", "their", "into", "not", "but", "can", "will", "also",
}
_ABSTAIN = re.compile(r"\b(?:i )?(?:do not|don't|cannot|can't) know\b", re.IGNORECASE)


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def check_grounding(
    answer: str, context: str, min_support: float = 0.6
) -> dict[str, Any]:
    """Fraction of answer sentences whose content words are mostly in the context."""
    if _ABSTAIN.search(answer):
        return {"score": 1.0, "grounded": True, "unsupported": [], "abstained": True}
    context_words = _content(context)
    sentences = [s.strip() for s in _SENTENCE.split(answer) if s.strip()]
    unsupported, supported = [], 0
    for sentence in sentences:
        words = _content(sentence)
        if not words or len(words & context_words) / len(words) >= min_support:
            supported += 1
        else:
            unsupported.append(sentence[:200])
    score = supported / len(sentences) if sentences else 0.0
    return {"score": score, "grounded": score >= 0.999, "unsupported": unsupported, "abstained": False}
