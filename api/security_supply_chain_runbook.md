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
```markdown
Security & Supply-Chain Runbook (summary)

Goals
- Automate SCA (Software Composition Analysis) on PRs
- Produce SBOMs for images and model artifacts
- Enforce scanning gates (fail PRs on HIGH/CRITICAL)
- Sign model artifacts and container images and verify signatures in deploy
- Harden container images and build with reproducible builders and trusted builders

Key components added
1. Dependabot (auto upgrade deps)
2. CI workflow that builds image, generates SBOM (syft), scans with trivy/grype and runs CodeQL
3. Hardened Dockerfile example with non-root user and multi-stage build
4. Scripts to sign/verify model artifacts (GPG) and examples for cosign keyless signing for containers
5. Guidance: use SLSA-friendly CI (reproducible builds, builders with provenance), use cosign for image signatures and Sigstore/fulcio/rekor for audit.

Artifact signing (recommended)
- Container images: use cosign to sign and upload signatures to the registry (keyless or KMS-backed keys). Verify in deploy pipeline with `cosign verify`.
- Model artifacts: sign using a private signing key stored in Vault (or use Vault Transit to sign hash); store .sig next to artifact and verify on download.

Vulnerability triage
- Block PRs on CRITICAL/HIGH severities by default.
- Route medium/low to a triage queue; auto-open an issue for recurring findings.
- Pin/patch vulnerable deps; backport security updates for production branches.

Secrets & key handling
- Use ephemeral credentials in CI (OIDC to Vault / cloud KMS) — no long lived VAULT_TOKEN in repo.
- Store cosign private key in KMS (Cloud KMS / HashiCorp Vault) and use `cosign sign --key` inside secure runner; prefer keyless cosign with fulcio for convenience.
- Audit all signing events in Vault logs and GitHub Actions logs.

Testing & verification
- Add CI job that verifies image signature before deploy (`cosign verify`).
- Add runtime check when loading model artifacts: verify model signature or validate via Vault Transit (signature verification).

Penetration & SCA program
