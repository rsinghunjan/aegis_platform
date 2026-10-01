#!/usr/bin/env python3
"""
Safety middleware for Aegis.

- Provides a decorator `@enforce_safety` and helper `check_and_execute` to ensure
  safety checks run before any action that performs external calls or side-effects.
- Integrates with existing `safety_checker` module if present (tries many common function names).
- Falls back to simple text/image heuristics if no safety_checker API is available.
- Raises SafetyBlocked on blocked content.
"""

from __future__ import annotations
import functools
import inspect
import re
from typing import Any, Callable, Tuple, Optional

# Try to import repo's safety_checker if present
try:
    import safety_checker as sc  # type: ignore
except Exception:
    sc = None

class SafetyBlocked(Exception):
    """Raised when safety checks block the action."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason

def _call_safety_checker_for_text(text: str) -> Tuple[bool, Optional[str]]:
    """
    Try multiple likely function names in repo's safety_checker module to evaluate text safety.
    Returns (is_safe, reason_if_blocked)
    """
    if sc is None:
        return _heuristic_text_check(text)
    # Common function names to try
    for fn in ("is_safe_text", "check_text", "filter_text", "safety_check_text", "is_text_safe", "check_content"):
        if hasattr(sc, fn):
            try:
                res = getattr(sc, fn)(text)
                # If function returns tuple or dict-like: interpret it
                if isinstance(res, tuple):
                    ok = bool(res[0])
                    reason = None if ok else (res[1] if len(res) > 1 else "blocked by safety_checker")
                    return ok, reason
                if isinstance(res, dict):
                    ok = bool(res.get("safe", True))
                    reason = res.get("reason")
                    return ok, reason
                # If boolean
                if isinstance(res, bool):
                    return res, None if res else "blocked by safety_checker"
                # String means blocked reason
                if isinstance(res, str):
                    # treat non-empty string as blocked reason
                    return False, res
            except Exception:
                # ignore and fallback
                continue
    # No matching API or all failed -> fallback to heuristic
    return _heuristic_text_check(text)

def _call_safety_checker_for_image(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Try likely functions in safety_checker module to evaluate image safety.
    Returns (is_safe, reason_if_blocked)
    """
    if sc is None:
        return _heuristic_image_check(image_bytes)
    for fn in ("is_safe_image", "check_image", "filter_image", "safety_check_image", "is_image_safe"):
        if hasattr(sc, fn):
            try:
                res = getattr(sc, fn)(image_bytes)
                if isinstance(res, tuple):
                    ok = bool(res[0])
                    reason = None if ok else (res[1] if len(res) > 1 else "blocked by safety_checker")
                    return ok, reason
                if isinstance(res, dict):
                    ok = bool(res.get("safe", True))
                    reason = res.get("reason")
                    return ok, reason
                if isinstance(res, bool):
                    return res, None if res else "blocked by safety_checker"
                if isinstance(res, str):
                    return False, res
            except Exception:
                continue
    return _heuristic_image_check(image_bytes)

# Simple heuristic detectors as fallback
_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (previous|all) instructions\b",
    r"\bdisregard (previous|all) instructions\b",
    r"\boutput your (?:(?:ssh|api|private) keys|private key|secret)\b",
    r"\bprovide your (password|credentials|private key)\b",
    r"\bexecute the following code\b",
    r"\bdo not mention the policies\b",
]
_PROMPT_INJECTION_RE = re.compile("|".join(_PROMPT_INJECTION_PATTERNS), flags=re.IGNORECASE)

def _heuristic_text_check(text: str) -> Tuple[bool, Optional[str]]:
    if not text or not isinstance(text, str):
        return True, None
    if _PROMPT_INJECTION_RE.search(text):
        return False, "heuristic: prompt-injection pattern matched"
    # over-length suspicious prompt?
    if len(text) > 10000:
        return False, "heuristic: unusually long text"
    return True, None

def _heuristic_image_check(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    # Very simple heuristics: look for suspicious literal markers in image bytes that might indicate embedded secrets
    if not image_bytes:
        return True, None
    try:
        # If the image payload (decoded) contains ascii markers for private key-like content — block
        sample = image_bytes[:4096]
        try:
            txt = sample.decode("utf-8", errors="ignore")
        except Exception:
            txt = ""
        if "-----BEGIN PRIVATE KEY-----" in txt or "PRIVATE_KEY" in txt or "SECRET_KEY" in txt:
            return False, "heuristic: private-key like content found in image bytes"
        # If it's extremely small but declared as image (too small to be a real image)
        if len(image_bytes) < 128:
            return False, "heuristic: image payload too small (possible malicious placeholder)"
    except Exception:
        # on doubt, allow (but in production you may want to fail-open/closed per policy)
        return True, None
    return True, None

def enforce_safety(action_type: str = "text"):
    """
    Decorator to enforce safety checks before running the wrapped function.
    - action_type: "text" (default) or "image"
    The wrapped function's first argument is assumed to be the content to check for 'text' actions,
    or accepts a parameter named 'image_bytes' or first positional bytes for 'image' actions.
    The decorated function will be executed only if checks pass; otherwise SafetyBlocked is raised.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Determine input to check
            if action_type == "text":
                # find first str argument or keyword 'text'/'prompt'/'input'
                content = None
                for a in args:
                    if isinstance(a, str):
                        content = a
                        break
                if content is None:
                    for k in ("text", "prompt", "input", "message"):
                        if k in kwargs and isinstance(kwargs[k], str):
                            content = kwargs[k]
                            break
                content = content or ""
                ok, reason = _call_safety_checker_for_text(content)
                if not ok:
                    raise SafetyBlocked(reason or "text blocked by safety policy")
                return fn(*args, **kwargs)
            elif action_type == "image":
                # find first bytes-like arg or kw 'image_bytes' or 'image'
                img = None
                for a in args:
                    if isinstance(a, (bytes, bytearray)):
                        img = bytes(a)
                        break
                if img is None:
                    for k in ("image_bytes", "image", "img"):
                        if k in kwargs and isinstance(kwargs[k], (bytes, bytearray)):
                            img = bytes(kwargs[k])
                            break
                img = img or b""
                ok, reason = _call_safety_checker_for_image(img)
                if not ok:
                    raise SafetyBlocked(reason or "image blocked by safety policy")
                return fn(*args, **kwargs)
            else:
                # For other action types, default to text check of serialized args
                summary = " ".join([str(a) for a in args]) + " " + " ".join([f"{k}={v}" for k, v in kwargs.items()])
                ok, reason = _call_safety_checker_for_text(summary)
                if not ok:
                    raise SafetyBlocked(reason or "blocked by safety policy")
                return fn(*args, **kwargs)
        return wrapper
    return decorator

def check_and_execute(func: Callable, action_type: str = "text", *args, **kwargs) -> Any:
    """
    Convenience helper: run safety checks for the given action_type, then execute func.
    Raises SafetyBlocked if blocked.
    """
    # Use the decorator wrapper
    wrapped = enforce_safety(action_type)(func)
    return wrapped(*args, **kwargs)

# Small helper to assert enforcement at runtime for frameworks that call middleware chains
def assert_safety_gate_for_callable(fn: Callable) -> None:
    """
    Inspect a callable and warn (raise) if it's not wrapped by our middleware.
    This is a best-effort check: it looks for the presence of 'enforce_safety' in the wrapper chain.
    Use this in orchestrator startup to fail fast if critical paths are not protected.
    """
    # walk wrapper chain
    current = fn
    seen = set()
    while hasattr(current, "__wrapped__"):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = getattr(current, "__wrapped__", None) or current
        # check closure for our decorator function presence
        closure = getattr(current, "__closure__", None)
        if closure:
            for cell in closure:
                val = cell.cell_contents
                if val is enforce_safety:
                    return
    # not found -> raise to alert integrator
    raise RuntimeError(f"Callable {fn.__name__} does not appear to be wrapped by enforce_safety; please apply safety middleware.")
