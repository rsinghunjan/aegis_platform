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
```markdown
# Runbook: Enforce Vault-backed secrets and OIDC/mTLS auth end-to-end

Goal
- Ensure runtime secrets come from a central secret manager (Vault or AWS Secrets Manager).
- Enforce either OIDC JWT verification or mTLS for all inbound requests to critical endpoints (inference, enrollment, review).
- Fail fast if critical secrets are not available.

1) Backends & provisioning
- Vault:
  - Provision Vault (HA) and enable a KV v2 mount (e.g., secret/).
  - Create a path for runtime secrets: `secret/data/aegis/runtime`
    - Fields: `oidc_jwks_uri`, `model_signing_pubkey` (PEM), `clients/allowed` (JSON list or newline list)
  - Configure Kubernetes auth role & policy for the service account (aegis-service-account).
  - Option A: use Vault Agent sidecar or Vault Injector to write secrets into pod filesystem or populate K8s Secret.
  - Option B: use Vault CSI driver to mount secrets as files.

- AWS Secrets Manager:
  - Create secrets similarly and ensure the pod/VM has IAM role to read secrets.

2) Application wiring (what to change)
- At service startup (in your FastAPI startup event):
  - Instantiate RuntimeSecrets() and call ensure_required with required keys:
    - Example required list:
       ["aegis/runtime:oidc_jwks_uri", "aegis/runtime:model_signing_pubkey"]
    - For Vault: store at `secret/data/aegis/runtime` and fetch via SecretsManager abstraction.
- Configure AUTH_MODE:
  - For OIDC: set AUTH_MODE=oidc. Ensure OIDC_JWKS_URI is available (via env or Vault).
  - For mTLS: set AUTH_MODE=mtls. Ensure ingress/service mesh verifies client cert and forwards CN in MTLS_ID_HEADER.

3) Ingress / service mesh (mTLS)
- If using mTLS, configure the ingress/mesh (e.g., NGINX, Envoy, Linkerd) to:
  - Terminate TLS and require client certificates.
  - Validate client certificate chain against your trusted CA.
  - Forward verified client identity via a header (e.g., `x-ssl-client-cn`) to the backend.
  - Block traffic to backend if client certificate verification fails (do not rely on a header alone without verification).

4) Testing in staging
- OIDC flow:
  - Use a valid test token issued by your IdP with matching JWKS.
  - Call /predict with Authorization: Bearer <token>; expect 200.
  - Call with expired/invalid token; expect 401.

- mTLS flow:
  - Use generated client certs (scripts/generate_certs.sh) and configure ingress to use the CA.
  - Attempt call without client cert; expect 401.
  - Call with client cert but not enrolled; expect 403 (if enrolled list configured).

5) Fail-fast & monitoring
- Use RuntimeSecrets.ensure_required at startup to prevent running without required secrets.
- Monitor logs for errors retrieving secrets and set an alert when app exits due to missing secrets.
- Expose a metric for auth failures (e.g., aegis_auth_failures_total) and alert on unusual spikes.

6) Onboarding / rotation
- For per-client credentials, prefer:
  - mTLS client certs issued via Vault PKI (short-lived) OR
  - Per-client OIDC client credentials minted by your IdP.
- Rotation:
  - For API keys: rotate and update Vault secret; notify client out-of-band.
  - For certs: automate re-issuance using Vault PKI and use short-lived certs.

7) Rollback & rollback plan
- If enabling AUTH_MODE caused widespread failures:
  1. Check that required secrets are present in Vault/Secrets Manager.
  2. Verify service account has correct permissions to read secrets.
  3. Verify ingress/service mesh TLS/mTLS configuration.
  4. As last resort, scale to a debug image with AUTH_MODE=none to access the cluster and remediate secrets, then re-enable.

8) Minimal checklist before enabling in production
- RuntimeSecrets.ensure_required passes in staging.
- OIDC tokens validate against JWKS in staging.
- mTLS enforced at ingress/mesh and identity header forwarded.
- Enrollment list present in secret backend for mtls mode.
- Logging & alerting configured for auth failures and missing secrets.

docs/runbook_enforce_secrets_identity.mddo
