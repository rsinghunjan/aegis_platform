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
107
108
109
110
111
112
113
114
#!/usr/bin/env python3
"""
Train a tiny multimodal (image + text) classifier on synthetic data and export to ONNX.

Produces:
 - <out_dir>/model.onnx
 - <out_dir>/vocab.txt
 - <out_dir>/metadata.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Simple multimodal model: image conv -> flatten + small text embedding -> concat -> linear
class TinyMultiModal(nn.Module):
    def __init__(self, vocab_size=100, text_emb=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten()
        )
        self.text_emb = nn.Embedding(vocab_size, text_emb)
        self.fc = nn.Sequential(
            nn.Linear(8*7*7 + text_emb, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, image, text_tokens):
        x_img = self.conv(image)
        # text_tokens: (batch, seq) -> take mean embedding
        emb = self.text_emb(text_tokens).mean(dim=1)
        x = torch.cat([x_img, emb], dim=1)
        return self.fc(x)

def synthetic_batch(batch_size=8, seq_len=6, vocab_size=100):
    # image: (batch, 1, 28, 28)
    images = np.random.rand(batch_size, 1, 28, 28).astype("float32")
    texts = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype("int64")
    labels = np.random.randint(0, 2, size=(batch_size,)).astype("int64")
    return images, texts, labels

def train_one_epoch(model, opt, loss_fn, steps=100):
    model.train()
    for _ in range(steps):
        images, texts, labels = synthetic_batch()
        images_t = torch.from_numpy(images)
        texts_t = torch.from_numpy(texts)
        labels_t = torch.from_numpy(labels)
        logits = model(images_t, texts_t)
        loss = loss_fn(logits, labels_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return

def export_onnx(model, out_path):
    model.eval()
    dummy_img = torch.randn(1,1,28,28)
    dummy_text = torch.randint(0,100,(1,6), dtype=torch.int64)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_img, dummy_text),
        str(out_path),
        input_names=["image", "text_tokens"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={"image": {0: "batch"}, "text_tokens": {0: "batch", 1: "seq"}}
    )

def build_vocab(out_dir: Path, size=100):
    # simple token -> token mapping file
    vocab = [f"token_{i}" for i in range(size)]
    (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")

def write_metadata(out_dir: Path):
    meta = {"model": "tiny-multimodal", "framework": "pytorch-onnx", "input_image": [1,28,28], "input_text_seq_len": 6}
    (out_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TinyMultiModal(vocab_size=100)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training (toy synthetic data)...")
    for _ in range(args.epochs):
        train_one_epoch(model, opt, loss_fn, steps=200)
    print("Exporting to ONNX...")
    export_onnx(model, out_dir / "model.onnx")
    build_vocab(out_dir, size=100)
    write_metadata(out_dir)
    print("Wrote ONNX model and metadata to", out_dir)

if __name__ == "__main__":
    main()
model_registry/example_multimodal/train.py
