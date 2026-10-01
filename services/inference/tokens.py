"""Token counting and cost estimation utilities.

Uses ``tiktoken`` when available for accurate counts, otherwise falls back
to a deterministic whitespace/character heuristic (~4 chars per token,
the commonly cited approximation for English text).
"""
from __future__ import annotations

from typing import Optional


def count_tokens(text: str, model: str = "default") -> int:
    if not text:
        return 0
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Heuristic fallback: ~4 characters per token, minimum 1 per word.
        words = text.split()
        char_estimate = max(1, len(text) // 4)
        return max(len(words), char_estimate)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input_tokens: float,
    cost_per_1k_output_tokens: Optional[float] = None,
) -> float:
    """Estimate USD cost given per-1K-token pricing.

    If ``cost_per_1k_output_tokens`` is omitted, the input rate is reused.
    """
    output_rate = (
        cost_per_1k_output_tokens
        if cost_per_1k_output_tokens is not None
        else cost_per_1k_input_tokens
    )
    return (input_tokens / 1000.0) * cost_per_1k_input_tokens + (
        output_tokens / 1000.0
    ) * output_rate
