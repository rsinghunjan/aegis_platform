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
```markdown
Vault OIDC (GitHub) + AppRole integration — notes & acceptance criteria

Overview
- Goal: CI and runtime obtain ephemeral Vault tokens (no long-lived VAULT_TOKEN) to use Transit signing.
- Approach:
  - CI (GitHub Actions): use OIDC. Configure a Vault OIDC role that accepts tokens from GitHub Actions for a specific repo.
  - Runtime (Kubernetes): either use Vault Agent Injector or Kubernetes auth (ServiceAccount -> Vault role) to obtain short-lived tokens.
  - For legacy/non-supported environments: AppRole can be created and used with wrapped secret_id delivery (least preferred).

Steps (operator)
1) Create transit key (if not present):
   vault secrets enable -path=transit transit
   vault write -f transit/keys/aegis-model-sign

2) Create minimal policy (aegis-transit-policy) that allows sign/verify on the transit key.
   (script: scripts/vault_oidc_github_setup.sh writes a policy if missing)

3) Configure Vault OIDC/JWT auth:
   - Use scripts/vault_oidc_github_setup.sh <policy> <role> <org/repo>
   - Validate OIDC role: simulate OIDC token exchange or run GitHub Action sample (see .github workflow)

4) Configure K8s auth or Vault Agent for pods:
   - If using Kubernetes auth:
     - enable auth/kubernetes and create a role bound to the ServiceAccount namespaced identity and policy
     - operator: vault write auth/kubernetes/role/aegis-k8s-role bound_service_account_names="aegis-app" bound_service_account_namespaces="default" policies="aegis-transit-policy" ttl="1h"
   - If using Vault Agent Injector:
     - install injector and annotate pods (see k8s/vault/serviceaccount-and-deployment-example.yaml)

5) CI workflow:
   - Use the provided .github/workflows/sign-model-with-vault.yml as a template.
   - The workflow uses hashicorp/vault-action@v2 with method: oidc and role set to the Vault OIDC role created.

Acceptance criteria (Done)
- GitHub Actions job can obtain an ephemeral VAULT_TOKEN via OIDC (no static tokens in secrets).
- CI step calls sign_model_artifact() and Transit returns a signature; the signature file is created in workspace artifacts.
- Aegis runtime pods obtain short-lived tokens (via Vault Agent or K8s auth) and can call transit/verify before loading a model; verification passes.
- No VAULT_TOKEN or raw private keys are present in repo, container images, or long-lived k8s secrets.
- Vault audit logs show sign/verify operations originating from CI and pods with short TTL tokens.

Security notes
- Limit policy to only transit/sign and transit/verify for the specific key.
- Rotate transit keys as desired; use key_version param in sign calls if needed.
- For AppRole, use response-wrapping to deliver secret_id out-of-band and rotate periodically.

Next steps I can take for you
- A) Open a PR that adds the .github workflow for signing and wires the CI into your existing build jobs.
- B) Add Kubernetes manifests and a Helm values fragment to install Vault Agent Injector and annotation examples for your Chart.
- C) Add a runtime check in model loader to call api.model_signing.verify_model_artifact() and reject unsigned artifacts (I can open a PR for this).
Which one should I implement next?
docs/vault_oidc_approle.md
