  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..safety_checker import SafetyChecker

DEFAULT_RULE_VERSION = "ruleset-v1"

class Decision:
    """
    Decision returned by PolicyEngine:
    action: 'allow' | 'block' | 'review'
    reason: short reason string
    details: dict with extra info (model_score, matched_keyword, pii_regex, etc.)
    rule_version, model_version, decision_id, timestamp
    """
    def __init__(
        self,
        action: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        rule_version: str = DEFAULT_RULE_VERSION,
        model_version: Optional[str] = None,
    ):
        self.action = action
        self.reason = reason or ""
        self.details = details or {}
        self.rule_version = rule_version
        self.model_version = model_version or "unknown"
        self.decision_id = str(uuid.uuid4())
        self.timestamp = time.time()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "reason": self.reason,
            "details": self.details,
            "rule_version": self.rule_version,
            "model_version": self.model_version,
        }


class PolicyEngine:
    """
    Compose model + heuristic rules into a single policy decision.

    Behavior:
    - If model_callable is supplied and returns score >= block_threshold => block
    - Else if model score >= review_threshold => review
    - Else check heuristics (keywords/pii) => block
    - Else allow.

    You can extend to include per-customer rules, dynamic rule loading, or remote policy APIs.
    """
    def __init__(
        self,
        model_callable: Optional[Any] = None,
        model_version: Optional[str] = None,
        blocked_keywords: Optional[List[str]] = None,
        review_threshold: float = 0.5,
        block_threshold: float = 0.8,
    ):
        # reuse SafetyChecker heuristics but we want higher-level decisions
        self.safety = SafetyChecker(blocked_keywords=blocked_keywords, model_callable=model_callable, model_version=model_version)
        self.model_callable = model_callable
        self.model_version = model_version or "model-unknown"
        self.review_threshold = review_threshold
        self.block_threshold = block_threshold

    def decide(self, text: str) -> Decision:
        # 1) model-based decision if available
        if self.model_callable is not None:
            try:
                score = float(self.model_callable(text))
            except Exception as e:
                # Treat model errors as non-fatal: fall back to heuristics and mark details
                score = 0.0
                details = {"model_error": str(e)}
            else:
                details = {"model_score": score}
            if score >= self.block_threshold:
                return Decision(action="block", reason="model_block", details=details, model_version=self.model_version)
            if score >= self.review_threshold:
                return Decision(action="review", reason="model_review", details=details, model_version=self.model_version)

        # 2) heuristics via SafetyChecker (keywords/pii)
        # We call is_unsafe but want to know which heuristic triggered; reuse it but we also inspect
        # We'll run a simple check for keywords/pii
        text_lower = (text or "").lower()
        for kw in self.safety.blocked_keywords:
            if kw and kw in text_lower:
                return Decision(action="block", reason="keyword", details={"keyword": kw}, model_version=self.model_version)

        for regex in self.safety.pii_regexes:
            if regex.search(text):
                return Decision(action="block", reason="pii", details={"pii_regex": regex.pattern}, model_version=self.model_version)

        # default allow
        return Decision(action="allow", reason="none", details={}, model_version=self.model_version)
