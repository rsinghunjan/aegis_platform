"""
Input sanitization, policy enforcement and audit logging for Aegis.

Features:
- Pre-inference checks: PII (emails, phone, CCN), profanity, prompt-injection patterns, URL domain blacklist.
- Pluggable policy hooks: register custom_check(func) to add extra validations.
- Decisions: ALLOW, FLAG, BLOCK. FLAG logs but allows continuation; BLOCK prevents inference.
- Audit log: writes SafetyEvent rows to the DB (api.models.SafetyEvent).
- Truncates/stores input snapshot for audit (configurable max length).

Usage:
- Call check_and_log(request_id, user, model_name, version, input_payload)
  before running inference. It returns (decision, reasons).
- Or call enforce_and_maybe_block(...) to raise HTTPException on BLOCK.
"""

import re
import json
import logging
from enum import Enum
from typing import Dict, Any, List, Callable, Tuple, Optional

from datetime import datetime

from api.db import SessionLocal
from api.models import SafetyEvent, Model, ModelVersion, User as DBUser

logger = logging.getLogger("aegis.safety")

# Configurable
MAX_SNAPSHOT_CHARS = 4096
PROFANITY_LIST = {"damn", "shit", "fuck"}  # extend for your policy
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (previous|above|all)", re.I),
    re.compile(r"disregard (previous|above|all)", re.I),
    re.compile(r"forget (previous|above|all)", re.I),
    re.compile(r"rewrite the following", re.I),
]
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "credit_card": re.compile(r"(?:\d[ -]*?){13,16}"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\+?\d[\d \-\(\)]{7,}\d"),
}
URL_PATTERN = re.compile(r"https?://[^\s]+")

# Simple blacklist; replace with company-managed list
BLACKLISTED_DOMAINS = {"malicious.example", "bad-domain.test"}

# Pluggable hooks: each hook receives (payload:dict, user:dict, model_name, version)
# and returns Optional[Tuple[Decision, str]] where str is reason. If returns None -> no opinion.
Hook = Callable[[Dict[str, Any], Optional[Dict[str, Any]], str, str], Optional[Tuple["Decision", str]]]
_registered_hooks: List[Hook] = []


class Decision(str, Enum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


def register_hook(fn: Hook):
    """Register a custom policy hook (e.g., remote classifier, enterprise rule)."""
    _registered_hooks.append(fn)
    return fn


def _contains_profanity(text: str) -> bool:
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words.intersection(PROFANITY_LIST))


def _detect_pii(text: str) -> List[str]:
    found = []
    for name, pat in PII_PATTERNS.items():
        if pat.search(text):
            found.append(name)
    return found


def _detect_prompt_injection(text: str) -> List[str]:
    matches = []
    for pat in PROMPT_INJECTION_PATTERNS:
        if pat.search(text):
            matches.append(pat.pattern)
    return matches


def _extract_urls(text: str) -> List[str]:
    return URL_PATTERN.findall(text or "")


def _url_host(url: str) -> Optional[str]:
    try:
        # quick parse
        host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0].lower()
        return host
    except Exception:
        return None


def _check_blacklisted_domains(urls: List[str]) -> List[str]:
    bad = []
    for u in urls:
        host = _url_host(u)
        if host and any(host.endswith(d) for d in BLACKLISTED_DOMAINS):
            bad.append(host)
    return bad


def _truncated_snapshot(payload: Dict[str, Any]) -> str:
    txt = json.dumps(payload, default=str, ensure_ascii=False)
    if len(txt) <= MAX_SNAPSHOT_CHARS:
        return txt
    return txt[:MAX_SNAPSHOT_CHARS] + "...(truncated)"


def check_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run core checks on the payload and return structured findings:
    {
      "pii": [...],
      "profanity": bool,
      "prompt_injection": [...],
      "blacklisted_domains": [...],
      "hooks": [ { "name": "<hook>", "decision": "FLAG", "reason": "..." }, ... ]
    }
    """
    text = ""
    if isinstance(payload.get("text"), str):
        text += " " + payload["text"]
    if isinstance(payload.get("parameters"), dict):
        # include parameter values for PII scanning (sometimes users embed secrets)
        text += " " + json.dumps(payload.get("parameters", {}), default=str)
    # also check image_url/image_base64/audio_base64 as strings
    text += " " + str(payload.get("image_url") or "")
    text += " " + str(payload.get("image_base64") or "")
    text += " " + str(payload.get("audio_base64") or "")

    findings = {
        "pii": _detect_pii(text),
        "profanity": _contains_profanity(text),
        "prompt_injection": _detect_prompt_injection(text),
        "blacklisted_domains": _check_blacklisted_domains(_extract_urls(text)),
        "hooks": [],
    }

    # run custom hooks
    for h in _registered_hooks:
        try:
            # pass None for user (caller can include identity if desired)
            result = h(payload, None, payload.get("model_name", ""), payload.get("version", ""))
            if result:
                decision, reason = result
                findings["hooks"].append({"decision": decision.value if isinstance(decision, Decision) else str(decision), "reason": reason, "hook": getattr(h, "__name__", str(h))})
        except Exception as exc:
            logger.exception("safety hook failed: %s", exc)
            findings["hooks"].append({"hook": getattr(h, "__name__", str(h)), "error": str(exc)})

    return findings


def assess_decision(findings: Dict[str, Any]) -> Tuple[Decision, List[str]]:
    """
    Convert findings into a final Decision and a list of reasons.
    Policy:
      - If blacklisted domains found -> BLOCK
      - If prompt injection patterns match -> FLAG (escalate to BLOCK if configured)
      - If PII found -> FLAG
      - If profanity only -> FLAG
      - If any hook returns BLOCK -> BLOCK; if hook returns FLAG -> FLAG
      - Default -> ALLOW
    """
    reasons: List[str] = []
    # check hooks for decisive opinions
    for h in findings.get("hooks", []):
        dec = h.get("decision")
        reason = h.get("reason", "")
        if dec == Decision.BLOCK.value:
            return Decision.BLOCK, [f"hook:{h.get('hook')}:{reason or 'blocked by hook'}"]
        if dec == Decision.FLAG.value:
            reasons.append(f"hook:{h.get('hook')}:{reason or 'flagged by hook'}")

    if findings.get("blacklisted_domains"):
        reasons.append(f"blacklisted_domains:{','.join(findings['blacklisted_domains'])}")
        return Decision.BLOCK, reasons

    if findings.get("prompt_injection"):
        reasons.append(f"prompt_injection:{','.join(findings['prompt_injection'])}")
        # default to FLAG for prompt-injection; ops may choose to escalate to BLOCK via hook
        return Decision.FLAG, reasons

    if findings.get("pii"):
        reasons.append(f"pii:{','.join(findings['pii'])}")
        return Decision.FLAG, reasons

    if findings.get("profanity"):
        reasons.append("profanity_detected")
        return Decision.FLAG, reasons

    return Decision.ALLOW, reasons


def save_safety_event(request_id: str, user_obj: Optional[Dict[str, Any]], model_name: str, version: str,
                      payload: Dict[str, Any], decision: Decision, reasons: List[str]) -> None:
    """Persist SafetyEvent row for audit."""
    session = SessionLocal()
    try:
        # try to resolve DB user/model ids when possible
        user_id = None
        if user_obj and user_obj.get("username"):
            # look up user in DB (best effort)
            try:
                db_user = session.query(DBUser).filter_by(username=user_obj["username"]).one_or_none()
                if db_user:
                    user_id = db_user.id
            except Exception:
                user_id = None

        model_version_id = None
        try:
            m = session.query(Model).filter_by(name=model_name).one_or_none()
            if m:
                mv = session.query(ModelVersion).filter_by(model_id=m.id, version=version).one_or_none()
                if mv:
                    model_version_id = mv.id
        except Exception:
            model_version_id = None

        ev = SafetyEvent(
            request_id=request_id,
            user_id=user_id,
            model_version_id=model_version_id,
            decision=decision.value,
            reasons=reasons,
            input_snapshot=_truncated_snapshot(payload),
            created_at=datetime.utcnow()
        )
        session.add(ev)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to save safety event to DB")
    finally:
        session.close()


def check_and_log(request_id: str, user_obj: Optional[Dict[str, Any]], model_name: str, version: str,
                  payload: Dict[str, Any]) -> Tuple[Decision, List[str]]:
    """
    Run checks, determine decision, save audit row, and return decision+reasons.
    """
    findings = check_payload(payload)
    decision, reasons = assess_decision(findings)
    save_safety_event(request_id=request_id, user_obj=user_obj, model_name=model_name, version=version,
                      payload=payload, decision=decision, reasons=reasons)
    return decision, reasons


def enforce_and_maybe_block(request_id: str, user_obj: Optional[Dict[str, Any]], model_name: str, version: str,
                            payload: Dict[str, Any]):
    """
    Helper for request handlers: runs checks and raises an exception on BLOCK.
    Returns (decision, reasons) if not blocked.
    """
    decision, reasons = check_and_log(request_id, user_obj, model_name, version, payload)
    if decision == Decision.BLOCK:
        # raise a standardized exception (FastAPI HTTPException expected in handlers)
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail={"message": "Request blocked by safety policy", "reasons": reasons})
    return decision, reasons
