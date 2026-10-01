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
#!/usr/bin/env python
"""
Canonical PyTorch training script (DDP-ready).

Features:
- Train a simple CNN on CIFAR10 (small, illustrative).
- Supports single-process (local) and DistributedDataParallel (DDP) via torch.distributed / torchrun.
- Saves checkpoints and logs basic metrics via MLflow (if MLFLOW_TRACKING_URI env set).
- Expects to be run with:
    Single-node: python examples/train_ddp.py --epochs 2 --batch-size 64
    DDP (recommended in k8s or multi-gpu nodes):
      torchrun --nnodes=1 --nproc_per_node=4 examples/train_ddp.py --epochs 5 --batch-size 64 --ddp
"""
import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

# Optional MLflow logging
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def accuracy(output, target):
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ddp", action="store_true", help="run in DDP mode (requires torch.distributed.launch or torchrun)")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
