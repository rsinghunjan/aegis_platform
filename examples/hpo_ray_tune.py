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
# Ray Tune HPO example (single-node)
# Usage: pip install ray[tune] && python examples/hpo_ray_tune.py
from functools import partial
import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Simple model reused from train_ddp (slim variant)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            correct += (out.argmax(dim=1) == yb).float().sum().item()
            total += yb.size(0)
    return correct / total

def train_fn(config, checkpoint_dir=None):
    batch_size = int(config["batch_size"])
    lr = config["lr"]
    epochs = 3

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root="/tmp/cifar", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="/tmp/cifar", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = validate(model, val_loader, device)
        tune.report(loss=train_loss, accuracy=val_acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = ASHAScheduler(metric="accuracy", mode="max")
    analysis = tune.run(train_fn, config=config, resources_per_trial={"cpu": 2, "gpu": 0}, num_samples=8, scheduler=scheduler, local_dir="ray_results")

    print("Best config: ", analysis.get_best_config(metric="accuracy", mode="max"))

if __name__ == "__main__":
    main()
