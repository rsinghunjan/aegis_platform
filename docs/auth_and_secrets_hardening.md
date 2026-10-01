
```markdown
   - Refresh tokens are persisted, support a revoke endpoint and admin UI for revocation.
7. Audit:
   - Record token issuance, refresh and revoke events in `audit_logs` for compliance.

Example GitHub Actions (recommended)
- Use the official HashiCorp action to login via GitHub OIDC and retrieve Vault token during CI runs instead of VAULT_TOKEN.
- Store only minimal secrets in GitHub (e.g., role mapping), rely on ephemeral Vault tokens.

Next work I can implement for you
- Wire OIDC/AppRole automation in `.github/workflows` (example step using the HashiCorp Vault GitHub Action).
- Replace the token password grant in the frontend with Authorization Code PKCE examples against Keycloak.
- Add admin endpoints (or UI) for user management, refresh token revocation, and tenant assignment.
- Add unit/integration tests for auth flows (DB-backed) and a migration to backfill legacy users (if needed).

If you'd like, I'll:
- Patch api/api_server.py to mount the new auth router and update imports (swap old auth module).
- Implement a small admin CLI `scripts/create_user.py` that creates hashed users and optionally prints a one-time API key for service accounts.
Which would you like me to do next?
```
