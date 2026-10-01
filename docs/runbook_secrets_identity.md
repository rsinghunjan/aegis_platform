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
# Runbook: Secrets & Identity for Aegis
  - Document the enrollment process and provide secure channels for distributing credentials.

6) Rotation & revocation
- API keys:
  - Rotate keys periodically (e.g., 90 days) using scripts/rotate_api_keys.py
  - When rotating, create the new key in Vault and notify the client OOB (secure channel).
  - Keep a short overlap window if necessary; revoke the old key after confirmation.
- Client certs:
  - Use short-lived certs and automate re-issuance (e.g., using Vault PKI).
  - Revoke compromised certs by updating CRL/OCSP (or revocation via Vault if using Vault PKI).
- Tokens:
  - Use short-lived tokens and require refresh using client credentials.

7) Audit & monitoring
- Alert on repeated failed auth attempts (Alertmanager rule).
- Alert on Vault read/write failures.
- Log enrollment events (but avoid logging secrets).

8) Quick recovery for leaked secrets
- Immediately rotate leaked secret via Vault or Secrets Manager.
- Revoke old tokens/certs.
- Run repo secret scan (detect-secrets) to ensure no other leaks.
- Follow incident response playbook (notify security/compliance).

9) Example commands
- Generate dev certs:
  - ./scripts/generate_certs.sh certs
- Run inference with mTLS (dev):
  - python scripts/run_uvicorn_mtls.py aegis_multimodal_ai_system.inference.server:app --cert certs/server.cert.pem --key certs/server.key.pem --ca certs/ca.cert.pem --port 9000
- Rotate client key (Vault):
  - python scripts/rotate_api_keys.py --client-id client123 --backend vault --path clients/client123

10) Checklist before pilot
- Vault or cloud secret manager configured and accessible by runtime (via k8s auth or IAM).
- All production secrets removed from repo and environment variables.
- Enforce at least one strong auth method (mTLS for internal; OIDC for external).
- Per-client credentials created for federated clients and documented enrollment procedure.
- Rotation procedures tested and runbook available to SRE/security.

Security notes
- Do not use static VAULT_TOKEN in production images; use k8s auth or instance roles.
- mTLS provides strong mutual authentication; OIDC is useful when clients cannot present certs (web clients).
- For federated learning with sensitive data, require mTLS and enforce client attestation prior to participation.

Contact & escalation
- If Vault or PKI problems occur, escalate to Security/Platform owner and follow incident runbook.
