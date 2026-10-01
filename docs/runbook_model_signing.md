```markdown
# Runbook: Model Artifact Signing & Runtime Enforcement

Goal
- Ensure model artifacts are signed by a trusted key in the release pipeline and verified at runtime before loading.
- Prevent tampered or unsigned models from being used in production.

Principles
- Sign artifacts in CI by a trusted signing process (cosign/sigstore recommended).
- Keep the public verification key in a secrets manager (Vault/KMS) and provide it to runtime securely (not in repo).
- Require signature verification at runtime for production (set MODEL_SIGNING_REQUIRED=true).
- Fail-fast: if signature verification fails, the pod should refuse to become ready and exit.

CI: Sign & verify in release pipeline
- Signing options:
  - cosign (recommended): sign with cosign using ephemeral keys or KMS-backed keys:
    cosign sign --key <private-key> <artifact>
  - OpenSSL (legacy): use scripts/sign_model.sh to produce artifact.sig (RSA PKCS1v15 SHA256).
- Verification in CI:
  - Add a workflow (enforce-model-signature.yml) that downloads the artifact and signature (or uses cosign) and verifies using the public key.
  - Fail the release/promotion step if verification fails.

Runtime: Enforce in ModelRegistry
- Set runtime env:
  - MODEL_SIGNING_PUBKEY: PEM public key (or path) to verify signatures (or store in Vault and fetch at startup).
  - MODEL_SIGNING_REQUIRED=true to make signature mandatory.
- Behavior:
  - registry.fetch_from_s3 will download both artifact and signature (if configured) and verify signature before registering the model.
  - registry.register_local will verify signature for local artifacts when provided; if required and missing the signature, it will fail.

Deployment checklist
1. Add signing step to model build/release pipeline (cosign recommended).
2. Publish public key to Vault or Secrets Manager; avoid embedding in images.
3. Set MODEL_SIGNING_PUBKEY (or use RuntimeSecrets to fetch it at startup).
4. Set MODEL_SIGNING_REQUIRED=true in staging first, test load & rollout.
5. Enable in production only after staging verification passes and rollback plan is ready.

Operational notes
- Rotate signing key pair periodically; publish new public key to Vault and plan a rollout that accepts both old and new keys during rotation window (or sign with multiple keys).
- Monitor AEGIS_MODEL_SIGNATURE_ERRORS metric and alert on non-zero occurrences.
- For high assurance, use cosign + Sigstore transparency logs rather than homegrown RSA signatures.

Troubleshooting
- "Signature verification failed":
  - Ensure signature corresponds to the exact artifact build (deterministic packaging).
  - Confirm the public key used by runtime is the correct one for the signer.
  - Check for corruption during upload/download (compare checksums).
- "Signature missing":
  - Verify that the release pipeline uploaded the signature artifact to the same location, or that the registry.fetch_from_s3 was given sig_s3_key.
  - If MODEL_SIGNING_REQUIRED is true and signatures are absent, the model will not load.

Example: CI signing & runtime verification (cosign)
1. CI builds artifact model.tar
2. CI runs: cosign sign --key cosign.key model.tar
3. CI uploads model.tar and model.tar.cosign (or record in Release assets)
4. Release workflow runs enforce-model-signature.yml (use cosign verify)
5. Runtime registries download artifact and verify with cosign or verify via MODEL_SIGNING_PUBKEY.

Security note
- Storing private signing keys in CI must be done with strong controls: use KMS/HSM (e.g., KMS-backed cosign), restrict who can trigger signing, and audit signing events.
