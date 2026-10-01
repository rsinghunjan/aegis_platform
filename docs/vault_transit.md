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
```markdown
Vault Transit signing (Aegis) — doc & integration notes

Why Transit?
- Transit performs signing and verification inside Vault. The private key never leaves Vault.
- CI and runtime can obtain short-lived Vault tokens (via OIDC/AppRole or Vault Agent) and call the transit sign/verify endpoints.
- This is much safer than storing raw key material in KV and avoids accidental exposure.

Files added
- api/vault_transit.py : sign/verify helpers (byte and file helpers)
- api/model_signing.py : convenience helpers to sign/verify model artifact files and write .sig.json metadata
- scripts/vault_transit_setup.sh : operator CLI to create transit key and policy
- vault/policies/aegis-transit.hcl : minimal policy

CI integration (GitHub Actions)
- Use GitHub OIDC to login to Vault and obtain a short-lived VAULT_TOKEN (already shown in earlier workflows).
  Example step (hashicorp/vault-action):
    - uses: hashicorp/vault-action@v2
      with:
        url: ${{ secrets.VAULT_ADDR }}
        role: ${{ secrets.VAULT_OIDC_ROLE }}
        method: oidc
  The action exposes a VAULT_TOKEN in the job for subsequent steps.
- After login, call the signing script:
    - name: Sign model artifact in CI
      run: |
        python -c "from api.model_signing import sign_model_artifact; print(sign_model_artifact('path/to/artifact.pt','aegis-model-sign'))"
      env:
        VAULT_ADDR: ${{ secrets.VAULT_ADDR }}
        # vault-action sets VAULT_TOKEN in env

Runtime integration (Aegis service)
- Configure Vault access for your service:
  - Use Vault Agent Injector or AppRole/OIDC for pods; never bake VAULT_TOKEN into images or manifests.
  - Pod should have an ephemeral token mounted at /var/run/secrets/vault-token or present in env VAULT_TOKEN.
- Before loading a model artifact, call api.model_signing.verify_model_artifact(path, key_name).
  - If verify fails, refuse to load artifact and raise an alert.

Migration notes (from KV HMAC to Transit)
- Previously you may have stored model signing keys in KV. To migrate:
  1. Create a new Transit key (scripts/vault_transit_setup.sh).
  2. Re-sign current stable artifacts with Transit (CI job or operator script).
  3. Update model registry metadata to reference Transit-signed artifacts (or store .sig.json alongside artifacts).
  4. Update runtime to call verify_model_artifact() before loading.

Operational & security recommendations
- Use OIDC/AppRole for CI and Service Accounts (no long-lived VAULT_TOKEN).
- Use Transit key rotation periodically; Vault can generate new key versions and verify previous signatures.
- Log signing events in Vault audit logs and record an audit entry in your application (api.audit.record_audit).
- For high-volume signing (many artifacts), consider rotating keys less often and maintain a key-version mapping in registry metadata.

Next steps I can implement
- Patch model registry upload/download flows to call sign_model_artifact() in CI and verify_model_artifact() in runtime (I can open a PR).
- Replace existing KV-based rotation scripts with Transit-based rotation + key_versioned signing and an automated CI verification.
- Add integration tests that run signing/verify steps against a local dev Vault (dev server) in the integration docker-compose.

Which of the next steps above should I implement for you now?
