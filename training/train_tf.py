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
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
#!/usr/bin/env python3
"""
Minimal TensorFlow training example that:
- trains a tiny Keras model on synthetic data
- saves a SavedModel in model_registry/<model_name>/<version>/<model_dir>
- logs artifacts to MLflow (optional)
- writes metadata.yaml and model_signature.json (via export helper)

Usage:
  python training/train_tf.py --output-dir model_registry/example-tf-model/0.1
Env:
  MLFLOW_TRACKING_URI (optional)
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import hashlib

import tensorflow as tf
from tensorflow import keras
import yaml

try:
    import mlflow
    import mlflow.tensorflow
except Exception:
    mlflow = None

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

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
    # synthetic data
    import numpy as np
    X = np.random.rand(1024, 28, 28, 1).astype("float32")
    y = (np.random.rand(1024) * 10).astype("int32")
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=1)

    # Save SavedModel (serving_default signature)
    saved_model_dir = output_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(saved_model_dir), signatures=model.call.get_concrete_function(tf.TensorSpec([None,28,28,1], tf.float32, name="image")))
    print("Saved SavedModel to", saved_model_dir)

    # create a tiny example input & expected shape for smoke tests
    example_input = X[:1].tolist()
    (output_dir / "example_input.json").write_text(json.dumps(example_input), encoding="utf-8")
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump({
        "name": output_dir.name,
        "version": "0.1.0",
        "framework": "tensorflow",
        "artifact": {"path": "saved_model"},
        "input_spec": {"dtype": "float32", "shape": [1,28,28,1], "name": "image"},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }), encoding="utf-8")
    # compute a model artifact hash (zip the saved_model dir for hash or hash a representative file)
    # For simplicity compute hash of variables/variables.data-00000-of-00001 if present
    varfile = saved_model_dir / "variables" / "variables.data-00000-of-00001"
    if varfile.exists():
        h = sha256_file(varfile)
        signature = {"sha256": h, "framework":"tensorflow", "artifact": str(saved_model_dir)}
        (output_dir / "model_signature.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    else:
        (output_dir / "model_signature.json").write_text(json.dumps({"framework":"tensorflow","note":"no-variable-file-for-hash"}), encoding="utf-8")

    # Optionally log to MLflow
    if mlflow is not None:
        mlflow.set_experiment("aegis-tf-example")
        with mlflow.start_run() as r:
            mlflow.log_param("epochs", epochs)
            mlflow.tensorflow.log_model(model, artifact_path="model")
            print("Logged model to MLflow, run_id:", r.info.run_id)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        print("Cleaning", out)
    out.mkdir(parents=True, exist_ok=True)
    train_and_export(out, epochs=args.epochs)
    print("Train & export complete.")

if __name__ == "__main__":
    main()
training/train_tf.py
