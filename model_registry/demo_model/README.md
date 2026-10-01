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
```markdown
# Demo multimodal model (licensed-for-demo) & Gradio demo

This folder contains a lightweight demo model flow and a Gradio demo that is prewired to the Aegis orchestrator or runs locally.

Contents
- serve_gradio.py — Gradio app that sends inference requests to the orchestrator endpoint (ORCH_URL) or runs local ONNX model if ORCH_URL not set.
- sample_inputs/ — example images/text (optional).
- README (this file)

Quick start (local, dev)
1. Create a venv and install deps:
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r model_registry/demo_model/requirements.txt

2. If you have a running orchestrator (deployed via Aegis), set:
   export ORCH_URL="https://aegis.your.domain/infer"

   Then start Gradio:
   python model_registry/demo_model/serve_gradio.py

   The Gradio UI will call ORCH_URL for inference.

3. If ORCH_URL is not set, the demo attempts to load a local ONNX model from:
   model_registry/example_multimodal/0.1/model.onnx

   You can create that model by running the example train/export flow:
   python model_registry/example_multimodal/train.py --out-dir model_registry/example-multimodal/0.1

Notes
- This demo uses only permissively licensed components (torchvision ResNet-based image encoder + a tiny text-embedding layer) or the repo's existing TinyMultiModal toy model.
- The Gradio demo is appropriate for internal demos and customer pilots; replace with production models for real workloads.
```
model_registry/demo_model/README.md
