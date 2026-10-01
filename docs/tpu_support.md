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
```markdown
TPU support for Aegis — overview and instructions

Summary
-------
This document describes how to run Aegis model servers on TPUs. The repo includes a TPU model wrapper (api/model_loader_tpu.py)
which uses PyTorch/XLA (torch_xla) to run PyTorch scripted models on TPU devices.

Two common deployment patterns:
1. TPU VM (recommended for many workloads)
   - Create a TPU VM on the cloud provider (e.g., Google Cloud TPU VM).
   - Build or use a runtime image that contains matching versions of torch and torch_xla for your TPU type (v2/v3/v4).
   - Run your model server on the TPU VM (systemd/container) and let it load model artifacts locally or from GCS/S3.
   - Benefits: direct access to TPU device(s), predictable performance.

2. Kubernetes nodes labelled for TPU (less common / complex)
   - Some clusters (GKE) support adding TPU-enabled node pools; schedule pods onto those nodes via nodeSelector/affinity/tolerations.
   - Important: ensure the container runtime and kernel modules are compatible; often it's easier to run TPU workloads on TPU VMs.

Installation notes (PyTorch + XLA)
- The exact torch_xla installation is platform and TPU-version-specific. For GCP TPU VMs consult:
  https://github.com/pytorch/xla/blob/master/README.md and cloud provider docs.
- Example (TPU VM with Python wheel host):
  - SSH into TPU VM image or build a container with matching wheels:
    pip install torch==2.1.0
    pip install --upgrade pip
    # the torch_xla wheel location is provided by the cloud vendor or pytorch/xla releases:
    pip install --find-links https://storage.googleapis.com/pytorch-tpu-releases/wheels.html torch_xla

Model artifact guidance
- For TPU, prefer serialized TorchScript modules (.pt) or models saved in a format that can be scripted — this avoids needing arbitrary Python classes to reconstruct models on TPU.
- Avoid saving arbitrary pickled Python model objects; scripted modules are more portable/robust.

Integration with ModelRegistry / ModelRunner
- Register a TPU model by setting runtime="tpu" (or device="tpu") in ModelConfig:
  from api.model_runner import registry, ModelConfig
  registry.register(
    "heavy_model",
    "v1",
    ModelConfig(model_path="/models/heavy_model.pt", runtime="tpu", device="tpu", max_concurrency=2, max_batch_size=64)
  )
- The registry must import api.model_loader_tpu.TPUModelWrapper when creating the wrapper for runtime "tpu".
  If your registry implementation currently supports runtime "torch" and "onnx" add a branch for "tpu" that constructs TPUModelWrapper.

Batching & compilation
- TPUs often compile computations on first execution; warmup with representative batch shapes (use warmup_iters > 1).
- Larger batch sizes yield better TPU utilization, but tune for latency vs throughput.

Limitations & caveats
- torch_xla is sensitive to version mismatches; ensure matching torch/torch_xla/jaxlib versions for your TPU hardware.
- JIT/compilation can incur high first-request latency — ensure warmups at deploy time.
- Not all PyTorch operators may be supported on XLA; test model ops on TPU ahead of time.
- On Kubernetes it can be difficult to provide the exact TPU runtime support; TPU VMs are often simpler.

Alternatives
- If your models are in JAX, prefer JAX/Flax and XLA-native compilation; implement a JAX wrapper instead.
- For large production fleets, consider dedicated TPU VMs per model or managed Vertex AI TPUs (cloud vendor product) rather than running TPUs directly in k8s.

Operational checklist
- Build & test a small TPU-stage (staging TPU VM) and run CI warmup tests there.
- Instrument with Prometheus metrics (prediction latency, batch size, xla compilation time).
- Add a node-pool / VM group for TPU workloads and ensure proper IAM and network access to model buckets (GCS/S3).
- Add a CPU fallback: if TPU is busy/down, route small requests to CPU/GPU instances so interactive latency is preserved.

If you want, next I can:
- Patch api/model_runner.py to automatically select TPUModelWrapper when config.runtime == "tpu" (I can open a PR).
- Add a small TPU test model (scripted TorchScript tiny model) and an integration test that runs on a TPU VM (if you can provide credentials/runner with TPU access).
- Add Helm templates / a k8s Job example for running the model server on a TPU-enabled nodepool.
```
docs/tpu_support.md
