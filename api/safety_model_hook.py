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
"""
Register the model-based safety hook with api.safety.register_hook so the model classifier is used
in addition to rule-based heuristics.

Policy:
 - If model classifier score > MODEL_FLAG_THRESHOLD -> FLAG
 - If score > MODEL_BLOCK_THRESHOLD -> BLOCK (configurable)
 - Save classifier result into findings via the hook for audit.

This file imports at app startup (e.g., api/api_server.py should import this module or it's imported via package init).
"""
import os
import logging
from api.safety import register_hook, Decision
from api.safety_model import classify

logger = logging.getLogger("aegis.safety_hook")

MODEL_FLAG_THRESHOLD = float(os.environ.get("MODEL_FLAG_THRESHOLD", "0.6"))
MODEL_BLOCK_THRESHOLD = float(os.environ.get("MODEL_BLOCK_THRESHOLD", "0.9"))


@register_hook
def model_safety_hook(payload, user, model_name, version):
    try:
        res = classify(payload)
        score = float(res.get("score", 0.0))
        labels = res.get("labels", {})
        reason = f"model_score={score:.2f}"
        # embed classifier details into reason; also persist classifier meta in SafetyEvent via saving hook results (done by safety.save)
        if score >= MODEL_BLOCK_THRESHOLD:
            return (Decision.BLOCK, reason + f";labels={labels}")
        if score >= MODEL_FLAG_THRESHOLD:
            return (Decision.FLAG, reason + f";labels={labels}")
        # no opinion
        return None
    except Exception:
        logger.exception("model_safety_hook failed")
        return None
