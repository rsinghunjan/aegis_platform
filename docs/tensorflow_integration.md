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
```markdown
# TensorFlow integration for Aegis

Overview
- Use SavedModel as canonical artifact format.
- Export signatures (serving_default) to make serving and loading deterministic.
- Use model_registry.loader.download_artifact and model_registry.verify_and_download.fetch_and_verify_model before loading artifacts.

Training:
- Use training/train_tf.py as a template.
- For distributed training:
  - GPUs: tf.distribute.MirroredStrategy()
  - Multi-node: tf.distribute.MultiWorkerMirroredStrategy()
  - TPUs: tf.distribute.TPUStrategy() (require TPU setup)

Exporting & Registry:
- Save models with tf.saved_model.save(..., signatures=...)
- Add metadata.yaml + model_signature.json and run scripts/export_savedmodel.py to create validation_report.json
- CI should require validation_report.json and model_signature.json (or remote verification via workflow).

Serving:
- TensorFlow Serving:
  - Model repository layout: model_repo/<model_name>/<version>/saved_model.pb
  - Start TF Serving with docker image: tensorflow/serving and mount model_repo
- Triton Inference Server:
  - Triton supports SavedModel platform: create model config for triton model repository

Optimization:
- TF‑TRT: use tf.experimental.tensorrt.Converter for NVIDIA GPUs (lower latency).
- XLA: enable JIT where appropriate for comp graphs.
- TFLite: convert & quantize for mobile/embedded targets.

CI smoke test (example):
- On PRs touching model_registry/** run:
  - scripts/export_savedmodel.py <path-to-saved_model>
  - scripts/smoke_inference.py or small TF loader that runs one inference

Security & verification:
- Use your verify_and_download.fetch_and_verify_model to ensure model signatures match before loading
- Cosign can be used to sign container images and model artifacts; store verification key in CI secrets.

Next steps:
- Add TF‑TRT conversion pipeline in CI for GPU production artifacts.
- Add TFLite pipeline for edge-targeted models.
- Add nightly benchmarks to monitor model latency on target hardware.
```
