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
Hosted inference proxy (Hugging Face / managed endpoints) — Aegis integration guide

Overview
- Purpose: let Aegis route model requests to hosted inference providers (Hugging Face Inference Endpoints, Vertex AI, custom hosted model endpoints) for tenants or models you choose.
- Benefits: reduce ops cost for certain tenants/workloads; get turnkey scaling; quick onboarding of new models.
- Tradeoffs: outbound provider cost, potential data exposure to third-party provider, higher latency vs local GPUs for some workloads.

Key features implemented
- HfProxyRuntime: forwards requests to HF model endpoints with retries, timeouts and error handling.
- Billing hook: billing_meter_hf_call records usage events for invoicing and quota enforcement.
- Test harness: unit tests to validate forwarding logic and retries.

Security & privacy
- Do NOT store HF API tokens in source. Inject via Vault (recommended) or Kubernetes Secret with Agent.
- Avoid sending PII/raw sensitive payloads to third-party providers. Preprocess/anonymize inputs before forwarding.
- Consider per-tenant HF accounts to isolate tenant data (store tenant HF API tokens in Vault).
- If routing only public or anonymized embeddings, document data-flow and get legal signoff.

Billing & quotas
- Meter every proxied call: duration, request/response bytes, model id, tenant_id.
- Add per-tenant budget/soft-limit policy (e.g., warn then throttle) to avoid runaway costs.
- Tag provider charges in invoices (provider=hf, model=owner/model) and reconcile with provider billing.

Operational & reliability
- Add provider failure handling:
  - Retries with exponential backoff for transient 502/503/429.
  - Circuit breaker: if provider has sustained errors, mark provider as degraded and fallback to local model or return friendly error.
- Monitor provider latency and error rates, alert on increases.
- For low-latency critical models keep a local fallback (CPU/GPU) or cache outputs.

Acceptance criteria (done)
- Aegis can register a model with runtime set to "hf_proxy" and model_endpoint pointing to HF (or tenant config routes requests to proxy).
- HF API token injected via Vault; no tokens in repo or images.
- Billing events are created per call and stored for daily invoice runs.
- Privacy policy updated and stakeholders approve sending tenant data to HF (if applicable).
- Integration tests exist in CI that mock provider calls and exercise billing logic; periodic smoke test to a real HF endpoint in staging (with test data) is scheduled.

Next steps & suggestions
- Wire HfProxyRuntime into ModelRegistry/model_runner as a runtime option ("hf_proxy" or "remote_http").
- Add circuit-breaker and provider health state persisted in Redis so orchestrator can avoid using degraded providers.
- Implement per-tenant provider token lookup from Vault (secret path per-tenant) for data isolation.
- Add a reconciliation job to compare billing_usage events with provider invoices (daily).
- Add UI/UX in Admin to allow selecting hosted provider routing per-model or per-tenant.
