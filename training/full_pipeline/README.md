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
```markdown
Full training pipeline for a real-ish multimodal toy model (CIFAR10 + synthetic captions)

This pipeline:
- downloads CIFAR10,
- builds simple captions (label words),
- trains a TinyMultiModal model (based on the example model),
- exports to ONNX,
- packages a deterministic tar.gz, and
- creates model_signature.json.

Usage (local dev)
1. Create venv and install requirements:
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r training/full_pipeline/requirements.txt

2. Train & export:
   python training/full_pipeline/train_multimodal_cifar.py --out-dir ./model_registry/demo-models/cifar_demo/0.1 --epochs 3

3. Package and sign (local):
   python3 scripts/make_deterministic_archive.py ./model_registry/demo-models/cifar_demo/0.1 ./artifacts/cifar-demo-0.1.tar.gz
   python3 scripts/create_model_signature.py ./artifacts/cifar-demo-0.1.tar.gz ./model_registry/demo-models/cifar_demo/0.1/model_signature.json --artifact-uri file:///$(pwd)/artifacts/cifar-demo-0.1.tar.gz

CI will run an analogous flow and sign using cosign.
```
