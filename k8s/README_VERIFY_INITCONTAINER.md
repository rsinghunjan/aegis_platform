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
```markdown
KServe / Triton initContainer verification and Gatekeeper integration

What this provides
- A KServe InferenceService and a plain Kubernetes Deployment example that use an initContainer named "verify-and-fetch".
- An initContainer script (scripts/verify_and_fetch.sh) that:
  - downloads an artifact and its .sig.json from S3,
  - verifies the signature either with a mounted cosign public key or via Vault transit/verify (if VAULT_ADDR + VAULT_TOKEN available),
  - extracts the artifact into /models for the main server to serve.
- The Gatekeeper ConstraintTemplate you deployed earlier looks for an initContainer with a name matching /verify/ and a secret named cosign-public-key mounted into the Pod. These manifests conform to that policy.

Steps to use

1) Prepare the cosign public key (preferred for initContainer local verify)
   - If you have a cosign keypair:
     cosign generate-key-pair
     # outputs cosign.key and cosign.pub
     kubectl create secret generic cosign-public-key --from-file=pubkey.pem=./cosign.pub -n aegis-canary

   - OR if you use Vault transit only and want pods to use k8s auth:
     - Ensure the model-fetcher serviceAccount is bound to a Vault role that can obtain short-lived tokens via k8s auth.
     - Configure the initContainer to request and set VAULT_TOKEN (or mount via projected serviceAccount token + Vault agent). The provided script will use VAULT_ADDR + VAULT_TOKEN to call transit/verify.

2) Ensure the initContainer image contains the required tools:
   - awscli (or s3-compatible client) to download artifacts
   - jq, openssl, tar
   - cosign binary if you prefer public-key verification
   The example uses ghcr.io/rsinghunjan/aegis-tools:latest — ensure that image includes these binaries.

3) Deploy into a canary namespace
   kubectl create ns aegis-canary
   kubectl apply -f k8s/kserve/inference_service_canary.yaml
   # or for the plain Triton Deployment
   kubectl apply -f k8s/triton/deployment_canary.yaml

4) Verification expectations
   - The pod should be admitted (Gatekeeper is in audit/dryrun) but Gatekeeper will record a violation if the pod lacks the initContainer or the cosign-public-key secret. Use this to iterate.
   - The initContainer will fail pod startup if verification fails, preventing the main container from starting.
   - Logs:
     kubectl logs -n aegis-canary <pod> -c verify-and-fetch

Notes & security considerations
- Prefer mounting a cosign public key in a read-only secret for faster, local verification without contacting Vault from every pod.
- For stronger control, prefer Vault + k8s auth for initContainers, ensuring serviceAccount -> Vault role mapping is narrowly scoped.
- Ensure the initContainer has minimal privileges (no persistent credentials in image). Use IRSA (AWS) or Workload Identity to give S3 read access without static creds.
- Gatekeeper constraint currently looks for:
  - initContainer name matching /verify/
  - a volume secret named cosign-public-key mounted in the pod
  Adjust the Rego if you want to allow Vault-only verification paths or different secret names.

Example troubleshooting
- If the pod stays in CrashLoopBackOff, fetch initContainer logs:
  kubectl logs -n aegis-canary <pod> -c verify-and-fetch
- If S3 download fails, ensure the pod/serviceAccount has access to S3 (IRSA or mounted AWS credentials).
- If cosign verification fails, validate the signature and public key pair:
  cosign verify-blob --key pubkey.pem --signature artifact.sig artifact.tar.gz

Acceptance criteria for canary
- Pod with verify-and-fetch + cosign-public-key mounts starts and the main container serves model artifacts.
- Pod without either initContainer or secret is reported by Gatekeeper (dryrun) and appears in violations listing.
- On verification failure, initContainer exits non-zero and pod does not become Ready.
```
k8s/README_VERIFY_INITCONTAINER.md
