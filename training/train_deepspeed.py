#!/usr/bin/env python3
"""
Minimal DeepSpeed-compatible training script with MLflow logging.

- Logs a dummy training run to MLflow and writes a checkpoint artifact.
- Replace model/training loops with your real logic.
"""
import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn


def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 10)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-epochs", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    with mlflow.start_run() as run:
        mlflow.log_param("script", "train_deepspeed.py")
        model = build_model()
        # TODO: Replace this dummy training loop with DeepSpeed training harness:
        # 1. Initialize DeepSpeed: model_engine, optimizer, _, _ = deepspeed.initialize(
        #        model=model, config=ds_config, model_parameters=model.parameters())
        # 2. Use model_engine.train() and model_engine.step() for training
        # 3. Save with model_engine.save_checkpoint()
        # See: https://www.deepspeed.ai/getting-started/
        for epoch in range(args.max_epochs):
            mlflow.log_metric("train/dummy_loss", 1.0 / (epoch + 1), step=epoch)
        ckpt_path = out_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")
        print("Saved checkpoint and logged to MLflow. run_id=", run.info.run_id)


if __name__ == "__main__":
    main()
