#!/usr/bin/env python3
"""
Distributed PyTorch trainer with checkpoint/resume to S3 (or local FS).
- Supports single-node multi-GPU via torch.distributed.launch style
- Periodically saves checkpoints to S3 (or local path)
- Can resume from the latest checkpoint in S3
Environment variables / args:
- MODEL_DIR: local path to write checkpoints (defaults ./checkpoints)
- S3_CHECKPOINT_URI: s3://bucket/path/prefix (optional)
- RESUME: "true" to attempt resume from S3/latest
- LOCAL_RANK, RANK, WORLD_SIZE handled by launcher (torchrun)
- AWS creds via env or IAM role
- SAVE_INTERVAL: save every N steps (default 100)
- MAX_EPOCHS: total epochs (default 10)
- BATCH_SIZE: local batch size (default 32)
"""

import os
import time
import json
import argparse
from pathlib import Path
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple example model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

def save_checkpoint(epoch, step, model, optimizer, checkpoint_path):
    state = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    torch.save(state, checkpoint_path)

def upload_to_s3(local_path, s3_uri):
    # s3_uri like s3://bucket/prefix/filename
    s3 = boto3.client("s3")
    assert s3_uri.startswith("s3://")
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    s3.upload_file(str(local_path), bucket, key)

def list_s3_prefix(s3_prefix):
    s3 = boto3.client("s3")
    assert s3_prefix.startswith("s3://")
    parts = s3_prefix[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return objs.get("Contents", [])

def download_s3_to_local(s3_uri, local_path):
    s3 = boto3.client("s3")
    assert s3_uri.startswith("s3://")
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def find_latest_checkpoint_in_s3(s3_prefix):
    contents = list_s3_prefix(s3_prefix)
    if not contents:
        return None
    # pick latest by LastModified
    latest = max(contents, key=lambda o: o["LastModified"])
    bucket = s3_prefix[5:].split("/",1)[0]
    key = latest["Key"]
    return f"s3://{bucket}/{key}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=int(os.environ.get("MAX_EPOCHS", 10)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 32)))
    parser.add_argument("--save-interval", type=int, default=int(os.environ.get("SAVE_INTERVAL", 100)))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("MODEL_DIR", "./checkpoints"))
    parser.add_argument("--s3-checkpoint-uri", type=str, default=os.environ.get("S3_CHECKPOINT_URI", ""))
    parser.add_argument("--resume", type=str, default=os.environ.get("RESUME", "true"))
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("WORLD", "1")))
    rank = int(os.environ.get("RANK", "0"))

    # init distributed
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # toy dataset
    input_dim = 10
    X = torch.randn(10000, input_dim)
    y = (torch.randn(10000,1) > 0).float()
    dataset = TensorDataset(X, y)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4)

    model = SimpleModel(input_dim=input_dim).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    start_step = 0

    # resume logic
    os.makedirs(args.model_dir, exist_ok=True)
    if args.resume.lower() == "true" and args.s3_checkpoint_uri:
        try:
            latest = find_latest_checkpoint_in_s3(args.s3_checkpoint_uri.rstrip("/") + "/")
            if latest:
                local_ckpt = os.path.join(args.model_dir, "latest.ckpt")
                print(f"[rank {rank}] Found checkpoint in S3: {latest}, downloading to {local_ckpt}")
                download_s3_to_local(latest, local_ckpt)
                state = torch.load(local_ckpt, map_location=device)
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optim_state"])
                start_epoch = state.get("epoch", 0)
                start_step = state.get("step", 0)
                print(f"[rank {rank}] Resumed from epoch {start_epoch} step {start_step}")
        except Exception as e:
            print("Resume failed:", e)

    global_step = start_step
    for epoch in range(start_epoch, args.max_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        for batch in loader:
            model.train()
            x, target = batch
            x = x.to(device)
            target = target.to(device)
            out = model(x)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % args.save_interval == 0 and (rank == 0):
                ckpt_path = os.path.join(args.model_dir, f"ckpt_epoch{epoch}_step{global_step}.pt")
                print(f"[rank {rank}] Saving checkpoint to {ckpt_path}")
                save_checkpoint(epoch, global_step, model.module if hasattr(model,'module') else model, optimizer, ckpt_path)
                if args.s3_checkpoint_uri:
                    # upload to s3 prefix with timestamp
                    s3_target = args.s3_checkpoint_uri.rstrip("/") + f"/ckpt_epoch{epoch}_step{global_step}.pt"
                    print(f"[rank {rank}] Uploading checkpoint to {s3_target}")
                    upload_to_s3(ckpt_path, s3_target)

        # end epoch: optionally save
        if rank == 0:
            ckpt_path = os.path.join(args.model_dir, f"ckpt_epoch{epoch}_final.pt")
            save_checkpoint(epoch, global_step, model.module if hasattr(model,'module') else model, optimizer, ckpt_path)
            if args.s3_checkpoint_uri:
                s3_target = args.s3_checkpoint_uri.rstrip("/") + f"/ckpt_epoch{epoch}_final.pt"
                upload_to_s3(ckpt_path, s3_target)

    if rank == 0:
        final_path = os.path.join(args.model_dir, "final.pt")
        save_checkpoint(args.max_epochs, global_step, model.module if hasattr(model,'module') else model, optimizer, final_path)
        if args.s3_checkpoint_uri:
            upload_to_s3(final_path, args.s3_checkpoint_uri.rstrip("/") + "/final.pt")

    print("Training complete.")

if __name__ == "__main__":
    main()
