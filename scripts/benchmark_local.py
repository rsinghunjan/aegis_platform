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
#!/usr/bin/env python3
"""
Run a small local benchmark using ONNX Runtime against a model.onnx.

Usage:
  python3 scripts/benchmark_local.py --model-onnx ./model_registry/example-multimodal/0.1/model.onnx --iters 200
"""
from __future__ import annotations
import argparse
import numpy as np
import time
import onnxruntime as ort

def run_inference(session, batch_size=8):
    # build random inputs consistent with model signature
    image = np.random.rand(batch_size,1,28,28).astype("float32")
    text = np.random.randint(0,100, size=(batch_size,6)).astype("int64")
    inputs = {"image": image, "text_tokens": text}
    out = session.run(None, inputs)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-onnx", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model_onnx, providers=["CPUExecutionProvider"])
    # warmup
    for _ in range(5):
        run_inference(sess, args.batch_size)
    t0 = time.time()
    for _ in range(args.iters):
        run_inference(sess, args.batch_size)
    t1 = time.time()
    total = args.iters
    print(f"Ran {total} iterations in {t1-t0:.2f}s, avg latency {(t1-t0)/total*1000:.2f} ms")

if __name__ == "__main__":
    main()
scripts/benchmark_local.py
