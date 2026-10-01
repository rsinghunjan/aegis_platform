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
#!/usr/bin/env python3
"""
Convert a SavedModel to a TF-TRT optimized SavedModel (requires TF GPU build + TensorRT present).

Usage:
  python3 scripts/convert_to_tftrt.py --saved-model model_registry/.../saved_model --out model_registry/.../saved_model_trt --precision FP16

Notes:
- Only works on systems with TensorRT installed and TF built with GPU support.
- The script will attempt an eager-safe conversion and save the converted model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", choices=("FP32","FP16","INT8"), default="FP32")
    args = ap.parse_args()

    saved_model_dir = Path(args.saved_model)
    out_dir = Path(args.out)
    if not saved_model_dir.exists():
        print("SavedModel not found:", saved_model_dir)
        sys.exit(2)

    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
    except Exception as e:
        print("TF-TRT not available in this environment:", e)
        sys.exit(2)

    precision_map = {"FP32": trt.TrtPrecisionMode.FP32, "FP16": trt.TrtPrecisionMode.FP16, "INT8": trt.TrtPrecisionMode.INT8}
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=(1<<30),
        precision_mode=precision_map[args.precision],
        maximum_cached_engines=100
    )
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=str(saved_model_dir), conversion_params=params)
    print("Converting with TF-TRT precision:", args.precision)
    converter.convert()
    # Optionally build TRT engines with representative data (omitted; user should provide calibration for INT8)
    converter.save(str(out_dir))
    print("Saved TF-TRT optimized SavedModel to", out_dir)

if __name__ == "__main__":
    main()
