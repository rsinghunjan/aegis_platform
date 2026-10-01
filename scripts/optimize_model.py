#!/usr/bin/env python
"""
Optimize model artifacts: ONNX export + quantization + packaging + optional registry registration/signing.

Supported input artifact types (best-effort):
 - TorchScript .pt/.pth (torch.jit.load) -> ONNX export (requires example input shape)
 - PyTorch state_dict (requires a user-provided model_loader callable; not automated)
 - scikit-learn joblib (.joblib/.pkl) -> skl2onnx conversion (if skl2onnx present)
 - ONNX (.onnx) -> quantization only

Notes and limitations:
 - Exporting arbitrary PyTorch state_dict requires model source code to instantiate the model.
   For simple pipelines, prefer saving a TorchScript artifact or providing an example input factory.
 - Quantization uses onnxruntime's dynamic quantization (works for many ops).
 - The script tries to be resilient: if optional packages (torch, onnxruntime, skl2onnx) are missing,
   it will report and exit non-zero.

Usage examples:
  # Optimize a TorchScript model; provide example input shape for ONNX export
  python scripts/optimize_model.py --input model.pt --input-type torchscript --example-input-shape 1,3,32,32 --model-name my-model --version v1 --quantize dynamic

  # Optimize an ONNX model (quantize only)
  python scripts/optimize_model.py --input model.onnx --input-type onnx --quantize dynamic --model-name my-model --version v1

  # Register the optimized artifact into registry and sign it with Vault key via api.model_signing
  python scripts/optimize_mo
