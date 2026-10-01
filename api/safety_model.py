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
"""
Model-based safety classifier wrapper (pluggable).

Behavior:
 - Attempts to call an external safety service (API endpoint) if SAFETY_SERVICE_URL env var is set:
     POST {SAFETY_SERVICE_URL}/classify  with JSON { "text": "...", "metadata": {...} }
   Expects response: { "labels": {"toxicity":0.85, "sexual":0.1, ...}, "score": 0.85 }
 - If no external service, uses a lightweight local heuristic fallback (fast).
 - Provides `classify(payload)` returning a dict of labels->score and top_label/confidence.

This module is intentionally small and avoids heavy ML deps. Swap in an onnx/torch classifier later or call a hosted classifier.
"""
import os
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger("aegis.safety_model")

SAFETY_SERVICE_URL = os.environ.get("SAFETY_SERVICE_URL")  # optional external classifier


def _local_fallback_classify(text: str) -> Dict[str, float]:
    # Very simple heuristic: check profanity + PII presence, return coarse scores.
    text_l = (text or "").lower()
    labels = {}
    # profanity-ish
    profane_words = {"fuck", "shit", "damn"}
    labels["toxicity"] = 1.0 if any(w in text_l for w in profane_words) else 0.0
    labels["pii"] = 1.0 if "@" in text_l or "ssn" in text_l or "credit card" in text_l else 0.0
    labels["prompt_injection"] = 1.0 if any(k in text_l for k in ["ignore previous", "disregard previous", "forget"]) else 0.0
    # aggregate score
    score = max(labels.values()) if labels else 0.0
    return {"labels": labels, "score": score}


def classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: may contain "text", "parameters", "image_url", etc.
    returns: { "labels": {name:score}, "score": float, "model": "external|fallback" }
    """
    text = ""
    if isinstance(payload.get("text"), str):
        text = payload.get("text", "")
    elif isinstance(payload.get("parameters"), dict):
        # sometimes prompts in parameters
        text = text + " " + str(payload.get("parameters"))

    if SAFETY_SERVICE_URL:
        try:
            resp = requests.post(f"{SAFETY_SERVICE_URL.rstrip('/')}/classify", json={"text": text, "metadata": payload}, timeout=3.0)
            if resp.status_code == 200:
                body = resp.json()
                # normalize shape
                labels = body.get("labels", {})
                score = body.get("score", max(labels.values()) if labels else 0.0)
                return {"labels": labels, "score": score, "model": "external"}
            else:
                logger.warning("safety service returned status %s, falling back", resp.status_code)
        except Exception:
            logger.exception("safety service call failed; falling back to local classifier")
    # fallback
    out = _local_fallback_classify(text)
    out["model"] = "fallback"
    return out
api/safety_model.py
