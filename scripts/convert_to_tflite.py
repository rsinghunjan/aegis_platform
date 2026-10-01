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
#!/usr/bin/env python3
"""
Convert a SavedModel to a TFLite flatbuffer. Optionally apply post-training quantization.

Usage:
  python3 scripts/convert_to_tflite.py --saved-model model_registry/.../saved_model --out model.tflite --quantize dynamic
Options for --quantize: none | dynamic | full_integer
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", choices=("none","dynamic","full_integer"), default="none")
    args = ap.parse_args()

    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow required:", e)
        sys.exit(2)

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    if args.quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quantize == "full_integer":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # You must provide representative dataset generator for full integer quantization
        def rep_gen():
            for _ in range(100):
                yield [np.random.randn(1,28,28,1).astype("float32")]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    outp = Path(args.out)
    outp.write_bytes(tflite_model)
    print("Wrote TFLite model to", outp)

if __name__ == "__main__":
    main()
scripts/convert_to_tflite.py
