import logging
import re
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BLOCKED_KEYWORDS = [
    # Simple illustrative keyword list; replace with production-safe policies or model
    "kill",
    "murder",
    "suicide",
    "bomb",
    "porn",
    "child sexual",
    "ssn",
    "social security",
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+"
        r"(?:(?:all|any|the|previous|prior|above|earlier)\s+)*"
        r"(?:system\s+)?instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|print|repeat|show)\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:output|reveal|show)\s+(?:your\s+)?"
        r"(?:private|secret|api|ssh)\s+keys?\b",
        re.IGNORECASE,
    ),
]


class SafetyChecker:
    """
    Lightweight safety checker with:
    - Optional injected model callable: model(text)->float (score 0.0..1.0, higher means more unsafe)
    - Keyword-based heuristics
    - PII regex detection (SSN as an example)
    """

    def __init__(
        self,
        blocked_keywords: Optional[Iterable[str]] = None,
        pii_regexes: Optional[List[re.Pattern]] = None,
        model_callable: Optional[Callable[[str], float]] = None,
    ):
        self.blocked_keywords = [
            k.lower() for k in (blocked_keywords or DEFAULT_BLOCKED_KEYWORDS)
        ]
        self.model_callable = model_callable
        # Basic PII regexes (example: US SSN); expand for production
        self.pii_regexes = pii_regexes or [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN 123-45-6789
            re.compile(r"\b\d{9}\b"),  # 9 digit straight
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
            re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # payment-card-like numbers
            re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b"),  # phone-like numbers
        ]
        logger.debug("SafetyChecker initialized with %d keywords", len(self.blocked_keywords))

    def is_unsafe(self, text: str, threshold: float = 0.7) -> bool:
        """
        Returns True if text is considered unsafe.
        Heuristic checks run before the optional model callable so detected PII
        is not passed to an external safety model.
        """
        if not text:
            logger.debug("is_unsafe called with empty text -> safe")
            return False

        text_lower = text.lower()

        if any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS):
            logger.info("Text flagged unsafe by prompt-injection pattern")
            return True

        for kw in self.blocked_keywords:
            if kw and re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text_lower):
                logger.info("Text flagged unsafe by keyword match: %s", kw)
                return True

        for regex in self.pii_regexes:
            if regex.search(text):
                logger.info("Text flagged unsafe by PII regex: %s", regex.pattern)
                return True

        if self.model_callable is not None:
            try:
                score = float(self.model_callable(text))
                logger.debug("Model returned safety score %s", score)
                if score >= threshold:
                    logger.info("Text flagged unsafe by model (score=%s)", score)
                    return True
            except Exception as e:
                logger.exception("Safety model callable raised an exception: %s", e)

        logger.debug("Text considered safe by heuristics")
        return False  
 
 
 



      

        
