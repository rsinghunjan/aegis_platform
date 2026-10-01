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
```markdown
# Safety Review Runbook (Aegis)

Purpose
- Provide operators and reviewers with a concise guide to triaging safety-flagged items, retention rules, and escalation paths.

Components
- PolicyEngine: emits decisions (allow, review, block) including rule_version and model_version.
- Audit backend: stores audit events (masked snippets and decision metadata).
- Review queue: SQLite db at logs/review_queue.db containing pending items for human review.
- Review API: /pending, /item/{id}, /item/{id}/review protected by AUTH.

Reviewer workflow
1. Notification & access
   - When a safety spike or individual flagged item occurs, the alert contains request_id and review_id (if queued).
   - Reviewer opens the review UI or calls GET /pending to list items.

2. Review item
   - Fetch item details: GET /item/{id}.
   - Examine text_snippet, reason, metadata and model_version.
   - Do NOT attempt to reconstruct full PII; if more context is required, contact data-owner and follow escalation policy.

3. Decision
   - Mark item reviewed via POST /item/{id}/review with verdict=allow|dismiss|block and add notes.
   - If blocking, follow escalation: create ticket, notify legal/security per on-call rota.

Retention & retention pruning
- Default retention: AUDIT_RETENTION_DAYS (90 days).
- Review queue retention: keep reviewed records for at least retention period + 30 days for audits.
- Use audit.prune or S3 lifecycle for automated deletion.

Escalation
- If the item is a suspected data-exfiltration or legal breach:
  - Immediately notify Security on-call (Pager) and create incident ticket.
  - Preserve audit log snapshot (export and put in secure bucket) and lock access.
  - Follow incident playbook in docs/incidents.md.

Operational notes
- Only authorized reviewers may access review API. Ensure AUTH_MODE and credentials are configured.
- Rotate reviewer credentials periodically.
- Monitor review queue backlog; set alerts if > X pending items older than Y hours.

Developer notes
- PolicyEngine decisions must be recorded in audit logs; include decision_id and review_id (when queued) for traceability.
- For high-throughput systems, consider a message queue (SQS/Kafka) and a dedicated review UI rather than SQLite.
```
