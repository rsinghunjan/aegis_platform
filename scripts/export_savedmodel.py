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
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
#!/usr/bin/env python3
"""
Helper to validate a SavedModel and produce a small validation_report.json
- Loads SavedModel with tf.saved_model.load
- Runs a single synthetic inference and records shapes/latency
- Writes validation_report.json and returns non-zero on failure
Usage:
  python3 scripts/export_savedmodel.py model_registry/<model>/saved_model
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import tensorflow as tf
import numpy as np

def validate_saved_model(saved_model_dir: Path, n=3):
    try:
        model = tf.saved_model.load(str(saved_model_dir))
    except Exception as e:
        print("Failed to load SavedModel:", e)
        return {"loaded": False, "error": str(e)}

    # find serving_default signature if available
    try:
        infer = model.signatures.get("serving_default", None)
        if infer is None:
            # fallback: try calling model directly if callable
            infer = lambda x: model(x)
    except Exception:
        infer = None

    # build random input that matches expected input shape if possible
    # attempt to read input spec from signature
    input_info = None
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        sig = model.signatures["serving_default"]
        input_info = list(sig.structured_input_signature[1].items())[0]  # (name, TensorSpec)
        name, spec = input_info
        shape = [d if d is not None else 1 for d in spec.shape]
        sample = np.random.randn(*shape).astype(spec.dtype.as_numpy_dtype())
        run_infer = lambda: sig(tf.constant(sample))
    else:
        # generic fallback: assume image [1,28,28,1]
        sample = np.random.randn(1,28,28,1).astype("float32")
        run_infer = lambda: model(tf.constant(sample))

    times = []
    out_shapes = None
    for _ in range(n):
        t0 = time.time()
        try:
            out = run_infer()
        except Exception as e:
            return {"loaded": True, "infer_ok": False, "error": str(e)}
        dt = (time.time() - t0) * 1000.0
        times.append(dt)
        # infer may return dict or Tensor
        if isinstance(out, dict):
            out_shapes = {k: list(v.shape) for k,v in out.items()}
        else:
            try:
                out_shapes = list(out.shape)
            except Exception:
                out_shapes = str(type(out))

    return {"loaded": True, "infer_ok": True, "n": n, "mean_ms": sum(times)/len(times), "p95_ms": sorted(times)[int(0.95*len(times))-1] if len(times)>1 else max(times), "out_shapes": out_shapes}

def main():
    if len(sys.argv) < 2:
        print("Usage: export_savedmodel.py <saved_model_dir>")
        sys.exit(2)
    saved_model_dir = Path(sys.argv[1])
    if not saved_model_dir.exists():
        print("Path not found:", saved_model_dir)
        sys.exit(2)
    report = validate_saved_model(saved_model_dir)
    (saved_model_dir.parent / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Validation report written:", saved_model_dir.parent / "validation_report.json")
    if not report.get("loaded") or not report.get("infer_ok"):
        sys.exit(3)

if __name__ == "__main__":
    main()
scripts/export_savedmodel.py
