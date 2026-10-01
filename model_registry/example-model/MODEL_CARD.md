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
```markdown
# Example Model — v0.1.0

Description:
- Lightweight vision classifier fine-tuned on a curated subset.

Intended use:
- Low-latency on-device image classification (non-safety-critical).

Limitations:
- Not trained on diverse lighting — may underperform in low light.

Training data:
- See dataset manifest: gs://aegis-datasets/imagenet-subset/manifest.yaml

Evaluation:
- Accuracy: 0.912 on evaluation split.

Artifact:
- ONNX file (quantized int8) at s3://aegis-models/example-model/0.1.0/model.onnx

Contact:
- ml-team@example.com

Changelog:
- v0.1.0 initial release
```
model_registry/example-model/MODEL_CARD.md
