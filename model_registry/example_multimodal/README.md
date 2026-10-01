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
```markdown
# Example multimodal model (toy) — training, export, archive, signature, benchmark

This example shows an end-to-end minimal pipeline:
1. Train a tiny multimodal model (image + text) with synthetic data.
2. Export the model to ONNX.
3. Create a deterministic archive (tar.gz) of the exported artifact.
4. Compute sha256 and create model_signature.json (metadata).
5. Optionally sign the archive with `cosign` (CI usually signs).
6. Run local benchmark/inference with ONNX Runtime.

Quick run (local dev)
- Create a venv and install deps:
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r model_registry/example_multimodal/requirements.txt

- Train & export:
  python3 model_registry/example_multimodal/train.py --out-dir ./model_registry/example-multimodal/0.1

- Make deterministic archive:
  python3 scripts/make_deterministic_archive.py ./model_registry/example-multimodal/0.1 ./artifacts/example-model-0.1.tar.gz

- Create signature metadata:
  python3 scripts/create_model_signature.py ./artifacts/example-model-0.1.tar.gz ./model_registry/example-multimodal/0.1/model_signature.json --artifact-uri file:///$(pwd)/artifacts/example-model-0.1.tar.gz

- Benchmark locally:
  python3 scripts/benchmark_local.py --model-onnx ./model_registry/example-multimodal/0.1/model.onnx

Notes
- This is a toy model for functional testing of the pipeline and verifier. It is not production-quality ML.
- The same flow can be used in CI: build deterministic archive -> cosign sign -> upload to object store -> run verifier workflow.
```
model_registry/example_multimodal/README.md
