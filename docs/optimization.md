```markdown
Model optimization pipelines — ONNX export, quantization, and registry integration

Goal
- Provide a repeatable pipeline to convert model artifacts to optimized runtimes (ONNX), apply quantization, and register optimized artifacts into Aegis ModelRegistry.
- Support deploy-time verification and sign the optimized artifact.

What I added
- scripts/optimize_model.py: CLI to convert TorchScript/ONNX/sklearn -> ONNX, quantize (dynamic), sign and register.
- api/optimizer.py: helper functions for conversion, quantization, upload and registration.
- k8s/job-optimize.yaml: example K8s job that runs optimization in-cluster, using PVC or mounted artifact.
- .github/workflows/optimize-pipeline.yml: CI smoke workflow that builds a tiny TorchScript model and runs the optimizer to validate dependencies and flow.
- docs/optimization.md: guidance on usage, limitations and recommended practices.

Design notes & limitations
- Exporting arbitrary PyTorch state_dict artifacts to ONNX requires model source code to reconstruct the model. Prefer saving TorchScript artifacts (torch.jit.trace/save) when you plan to perform automated ONNX export.
- Quantization in this pipeline uses ONNX Runtime dynamic quantization (weight quantization). For better results (static quant) you need calibration data and a more involved flow; that can be added later.
- The registration step is best-effort: api.optimizer.register_optimized_model() will call into repo's api.registry if available. If your registry expects artifacts to be at s3:// URLs, enable --upload-s3 and provide --s3-bucket.

Recommended production flow
1. During training, produce a TorchScript artifact or saved ONNX artifact and upload it to MLflow/artifact store.
2. Trigger the optimize pipeline (k8s Job or dedicated optimization service) which:
   - Downloads the artifact
   - Exports to ONNX (if needed), applies quantization, validates via a small test input
   - Signs the optimized artifact (api.model_signing)
   - Uploads the optimized artifact to object storage (s3://)
   - Registers the optimized artifact in ModelRegistry (linking original model version -> optimized version)
3. CI or runtime verifies signatures before deployment.

CI smoke and gating
- The included optimize-pipeline.yml demonstrates a minimal CI job that validates the optimizer dependencies by generating a tiny TorchScript model and running the optimizer.
- Extend this job to run as part of your release pipeline (e.g., run optimization on promoted model versions and store optimized artifact versions in registry).

Acceptance criteria
- Optimization job converts valid TorchScript/ONNX/sklearn artifacts to ONNX and writes quantized outputs.
- Signed and registered optimized artifact is visible in registry (or registry logs) and the MLflow provenance is annotated (if promote pipeline is used).
- CI smoke workflow passes on main branch and provides traceable artifacts under optimized/.

Next steps and enhancements (prioritized)
- Add static quantization flow (calibration dataset + onnxruntime.quantization.quantize_static).
- Add functional validation step: run a small set of functional tests (unit test) against optimized artifact to ensure parity.
- Add automated promotion of optimized artifact versions in the registry (link optimized -> original) and reflect cost/latency metadata.
- Add a containerized optimization service (K8s Deployment + Queue) to offload heavy conversions from CI.
```
