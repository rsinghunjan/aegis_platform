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
#!/usr/bin/env python3
"""
Reproducible TF training example with explicit seeds and deterministic archive/signature generation.

Usage:
  python training/train_tf_repro.py --output-dir model_registry/example-tf-model/0.1 --epochs 3

This trains a tiny model, saves a SavedModel, computes deterministic archive + model_signature.json
and produces a MODEL_CARD.md and metadata.yaml similar to repo conventions.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

# ensure reproducibility seeds
PYTHON_SEED = 42
np.random.seed(PYTHON_SEED)
random.seed(PYTHON_SEED)
tf.random.set_seed(PYTHON_SEED)

def build_model(input_shape=(28,28,1), num_classes=10):
    inp = keras.Input(shape=input_shape, name="image")
    x = keras.layers.Rescaling(1./255)(inp)
    x = keras.layers.Conv2D(16, 3, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_export(output_dir: Path, epochs=3):
    model = build_model()
    X = np.random.rand(1024, 28, 28, 1).astype("float32")
    y = (np.random.rand(1024) * 10).astype("int32")
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=1)

    saved_model_dir = output_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    # export with a concrete signature for serving_default
    concrete_func = model.call.get_concrete_function(tf.TensorSpec([None,28,28,1], tf.float32, name="image"))
    tf.saved_model.save(model, str(saved_model_dir), signatures=concrete_func)
    print("Saved SavedModel to", saved_model_dir)

    # metadata & model card
    metadata = {
        "name": output_dir.name,
        "version": "0.1.0",
        "framework": "tensorflow",
        "artifact": {"path": "saved_model"},
        "input_spec": {"dtype": "float32", "shape": [1,28,28,1], "name": "image"},
        "created_by": os.environ.get("USER","dev"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    (output_dir / "MODEL_CARD.md").write_text("# Example TF model\n\nThis is a reproducible demo model.", encoding="utf-8")

    # compute deterministic archive + signature
    # this calls scripts/make_deterministic_archive.py and compute_model_signature_tf.py
    import subprocess
    subprocess.check_call(["python3", "scripts/compute_model_signature_tf.py", str(saved_model_dir)])
    print("Computed deterministic archive & signature for", output_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_and_export(out, epochs=args.epochs)

if __name__ == "__main__":
    main()
training/train_tf_repro.py
