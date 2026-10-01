# Safety middleware integration (make safety checks mandatory)

This document explains how to make the safety middleware mandatory across your orchestrator / action execution paths, and how to run the tests locally and in CI.

Files added
- aegis_multimodal_ai_system/middleware/safety_middleware.py
- tests/test_safety_middleware.py

1) How to enforce middleware across action/external call boundaries
- The middleware exposes:
  - `@enforce_safety(action_type="text"|"image")` decorator — use this to wrap functions that perform external calls or side effects (model calls, plugin actions, API calls).
  - `check_and_execute(func, action_type, *args, **kwargs)` — helper to dynamically perform checks and then run `func`.
  - `SafetyBlocked` exception raised when content is blocked.

2) Integration examples
- If your orchestrator uses an `execute_action` function, wrap it:

```python
# inside orchestrator (example)
from aegis_multimodal_ai_system.middleware.safety_middleware import enforce_safety, SafetyBlocked

@enforce_safety(action_type="text")
def execute_action(payload: str, *args, **kwargs):
    # existing logic that sends payload to model or external system
    return run_model_inference(payload)
```

- Or call dynamically before performing an action:

```python
from aegis_multimodal_ai_system.middleware.safety_middleware import check_and_execute, SafetyBlocked

def orchestrator_handle(payload):
    def action():
        # external call / side-effect here
        return run_model_inference(payload)
    try:
        # checks the payload as text before executing action
        return check_and_execute(action, action_type="text")
    except SafetyBlocked as e:
        # handle blocked content: log, return safe error, or take other remediation
        return {"error": "blocked_by_safety", "reason": e.reason}
```

3) Where to apply
- Mandatory safety gates should be placed immediately before:
  - Model inference calls
  - Plugin action execution (RAG, docQA, audio processing that may call external APIs)
  - Any call that posts to external APIs, writes to storage, executes system commands, or triggers agentic workflows

4) CI & testing
- The included tests `tests/test_safety_middleware.py` are PyTest tests that simulate malicious prompts and images.
- Add these tests to your CI test matrix (unit/smoke/integration as you prefer). Example: the existing CI workflow that runs unit tests should include this test file.
- Run locally:
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt  # or install pytest
  - pip install pytest
  - pytest -q tests/test_safety_middleware.py

5) Notes and next steps
- The middleware first tries to use your repo's `safety_checker` module (if available) and tries a number of plausible function names (e.g., `check_text`, `is_safe_text`, `check_image`, etc.). If your `safety_checker` uses different function names or a different API, either:
  - Rename/alias the functions in `safety_checker` to one of the tried names, or
  - Update `_call_safety_checker_for_text` / `_call_safety_checker_for_image` to call your API.
- The fallback heuristics are intentionally conservative examples. For production you should:
  - Use a vetted safety-checker implementation with clear policies and deterministic behavior.
  - Decide whether to fail-open or fail-closed on errors (this module fails-closed for the heuristics that match).
- Consider logging blocked attempts (audit) and surfacing them to monitoring/alerts.

If you'd like, I can:
- Open a PR adding these files to your repo and enabling the tests in CI (I can modify your CI workflow to include these tests),
- Or adapt the middleware to call the exact functions in your existing safety_checker module if you tell me the function names and expected return types.

Which would you like next?

