```markdown
# Aegis Safety Policy & Enforcement Hooks

This document describes the runtime safety checks and how to extend them.

Core checks (implemented by api/safety.py)
- PII detection: emails, phone numbers, credit-card-like sequences, SSN-like patterns.
- Prompt-injection heuristics: patterns that attempt to override prior instructions.
- Profanity: basic profanity list (policy team should expand or swap to a proper classifier).
- URL detection + domain blacklist: blocks requests that reference disallowed domains.
- Pluggable hooks: enterprise site can add additional checks (e.g., model-specific safety ML classifier).

Decision model
- BLOCK: immediate refusal to process, logged to safety_events and caller receives 403 with reasons.
- FLAG: allow processing but record safety_event and reasons; ops or downstream human review can inspect flagged items.
- ALLOW: no issues detected.

Integration points
- Call `api.safety.enforce_and_maybe_block(...)` in your predict endpoint before invoking model inference.
- Alternatively call `api.safety.check_and_log(...)` to log and get the decision without auto-blocking.

Audit
- Safety events are persisted to `safety_events` table via SQLAlchemy model `SafetyEvent`.
- Stored fields include: request_id, user_id, model_version_id, decision, reasons, truncated input snapshot and timestamp.
- Retention & access: set a DB retention policy (e.g., 90 days) and restrict access via DB RBAC for security/privacy.

Extending policies
- Register custom hooks with `api.safety.register_hook(func)` where func(payload, user, model, version) -> Optional[(Decision, reason)].
- Hooks can be used to call remote safety services (model-based toxicity detection), check tenant policies, or consult blocklists.

Operational playbook
- Flagged items should create a triage queue that ops/security reviews (optionally via the Jobs queue).
- For BLOCK events, notify the client with user-friendly message and log details to SRE incident channel.
- When PII is flagged, ensure privacy team handles data minimization; consider redaction for stored snapshots.

Testing safety
- Add unit tests that call safety.check_payload with crafted payloads (emails, CCNs, prompt-injection strings).
- Add e2e tests that ensure predict endpoints reject blocked payloads and return proper error payload & status.

Privacy note
- The input_snapshot is truncated and stored for audit. If sensitive data should never be logged (per tenant policy), enable redaction before saving.
- Consider hashing or storing only a fingerprint of the input where required by privacy policies.
```
```
