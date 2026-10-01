
```markdown
# Example TF Model v0.1.0

Description:
- Tiny demo TensorFlow model for CI smoke tests and demo inference.

Intended use:
- CI smoke/load tests and local development. NOT production-grade.

Artifact:
- SavedModel at model_registry/example-tf-model/0.1/saved_model

Validation:
- See validation_report.json produced by scripts/export_savedmodel.py

Notes:
- For production use: convert to TF-TRT on NVIDIA GPUs or to TFLite for edge devices.
```
