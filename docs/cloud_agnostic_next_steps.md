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
# Next steps we implemented and how to use them

What I implemented for you:
- Finished a canonical "fetch + verify" flow:
  - aegis_multimodal_ai_system/model_registry/verify_and_download.py
  - aegis_multimodal_ai_system/model_registry/signature.py
  - scripts/verify_model_signatures.py (CI helper)
  - .github/workflows/verify-model-signatures.yml (runs on PRs that touch model_registry/**)
- CI-level signature/hash verification:
  - The workflow computes changed model dirs and runs the verifier against them.
  - Policy (default): PRs adding models must include model_signature.json and an artifact whose hash matches the signature (or CI must have credentials to download remote artifact to verify).
- Terraform skeleton and provider overlays:
  - infra/terraform/modules/* and infra/terraform/overlays/* provide a starting point for provider-agnostic modules.
  - Use overlays to implement provider specifics (you or infra team fill in provider resources).

How to wire orchestrator to use verified artifacts:
- Use fetch_and_verify_model(model_name) from `aegis_multimodal_ai_system.model_registry.verify_and_download`
  before loading a model. This ensures only verified artifacts are used.

Example (orchestrator pseudo-code):
```py
from aegis_multimodal_ai_system.model_registry.verify_and_download import fetch_and_verify_model

local_model = fetch_and_verify_model("example-model-0.1")
# load local_model with runtime (onnxruntime / torch / tpu loader)
