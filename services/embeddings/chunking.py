"""Semantic chunking for documents."""
from __future__ import annotations

import re
from typing import List


def semantic_chunk(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> List[str]:
    """Split ``text`` into overlapping chunks aligned to sentence boundaries.

    This is a lightweight, dependency-free approximation of "semantic"
    chunking: it splits on sentence boundaries first, then greedily packs
    sentences into chunks up to ``max_chars``, carrying a trailing overlap
    into the next chunk to preserve context across chunk boundaries.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")

    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
        else:
            # Single sentence longer than max_chars: hard-split it.
            for start in range(0, len(sentence), max_chars - overlap_chars):
                chunks.append(sentence[start : start + max_chars])
            current = ""

        if len(current) > max_chars:
            # Still too long after merging overlap tail; flush immediately.
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap_chars :]

    if current:
        chunks.append(current)

    return chunks
