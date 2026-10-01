 url=https://github.com/rsinghunjan/aegis_multimodal_ai_system/blob/main/training/pytorch_lightning_distributed.py
#!/usr/bin/env python3
"""
Distributed PyTorch Lightning training script with MLflow integration.

- Uses PyTorch Lightning's Trainer (DDP) when run on multi-GPU nodes/cluster.
- Logs metrics and artifacts to MLflow (MLFLOW_TRACKING_URI env var).
- Exports ONNX and writes a simple model_signature.json with mlflow_run_id.
- Usage (local / single node):
  python training/pytorch_lightning_distributed.py --out-dir ./model_registry/demo-models/cifar_demo/0.1 --max-epochs 3
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
import numpy as np
import mlflow
import mlflow.pytorch

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(42)

class TinyPLModule(LightningModule):
    def __init__(self, vocab_size=100, text_emb=32, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten()
        )
        self.text_emb = nn.Embedding(vocab_size, text_emb)
        self.fc = nn.Sequential(
            nn.Linear(16*7*7 + text_emb, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image, text_tokens):
        x_img = self.conv(image)
        emb = self.text_emb(text_tokens).mean(dim=1)
        x = torch.cat([x_img, emb], dim=1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # dummy text tokens for toy example
        tokens = torch.randint(0, 100, (images.size(0), 6), dtype=torch.int64, device=self.device)
        logits = self(images, tokens)
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

def build_dataloaders(batch_size=64):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def 
