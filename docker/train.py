#!/usr/bin/env python3
"""
Lightweight DDP training script for GPU-cluster acceptance testing.
Generates a tiny synthetic binary classification dataset and trains a small MLP
using PyTorch DistributedDataParallel (DDP).
"""

import os
import sys
import time
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_rank_world_size():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NNODES", 1)))
    return rank, world_size


def get_local_rank():
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_PROCID" in os.environ and "SLURM_GPUS_ON_NODE" in os.environ:
        try:
            proc = int(os.environ["SLURM_PROCID"])
            gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
            return proc % max(1, gpus)
        except Exception:
            return 0
    return 0


class TinyMLP(nn.Module):
    def __init__(self, in_dim=20, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_synthetic_data(n_samples=1024, n_features=20, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w = rng.normal(size=(n_features,)).astype(np.float32)
    logits = X.dot(w) + 0.5 * rng.normal(size=(n_samples,))
    y = (logits > 0).astype(np.float32)
    return X, y


def average_metric_across_workers(tensor, world_size):
    if world_size <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(world_size)
    return tensor


def main():
    start_time = time.time()
    rank, world_size = get_rank_world_size()
    local_rank = get_local_rank()

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    master_addr = os.environ.get("MASTER_ADDR", os.environ.get("SLURM_STEP_NODELIST"))
    master_port = os.environ.get("MASTER_PORT", os.environ.get("MASTER_PORT", "12345"))

    os.environ.setdefault("MASTER_ADDR", str(master_addr))
    os.environ.setdefault("MASTER_PORT", str(master_port))

    print(f"[{datetime.now().isoformat()}] initializing process group: backend={backend}, rank={rank}, \
    world_size={world_size}, local_rank={local_rank}, cuda={use_cuda}", flush=True)

    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Failed to init process group: {e}", file=sys.stderr, flush=True)
        raise

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    seed = 1234 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    total_samples = 2048
    n_features = 20
    X, y = make_synthetic_data(n_samples=total_samples, n_features=n_features, seed=seed)

    split = int(0.8 * total_samples)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    batch_size = 32
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=0)

    val_ds = TensorDataset(X_val_t, y_val_t)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=0)

    model = TinyMLP(in_dim=n_features, hidden=32).to(device)
    if use_cuda:
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.05)

    n_epochs = 3
    for epoch in range(1, n_epochs + 1):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = ddp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        loss_tensor = torch.tensor(running_loss, dtype=torch.float64, device=device)
        total_tensor = torch.tensor(total, dtype=torch.float64, device=device)
        correct_tensor = torch.tensor(correct, dtype=torch.float64, device=device)

        average_metric_across_workers(loss_tensor, world_size)
        average_metric_across_workers(total_tensor, world_size)
        average_metric_across_workers(correct_tensor, world_size)

        avg_loss = (loss_tensor / total_tensor).item() if total_tensor.item() > 0 else float('nan')
        accuracy = (correct_tensor / total_tensor).item() if total_tensor.item() > 0 else float('nan')

        if rank == 0:
            print(f"Epoch {epoch}/{n_epochs} - train_loss={avg_loss:.4f} train_acc={accuracy:.4f}", flush=True)

        ddp_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = ddp_model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.detach().item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss_tensor = torch.tensor(val_loss, dtype=torch.float64, device=device)
        val_total_tensor = torch.tensor(val_total, dtype=torch.float64, device=device)
        val_correct_tensor = torch.tensor(val_correct, dtype=torch.float64, device=device)

        average_metric_across_workers(val_loss_tensor, world_size)
        average_metric_across_workers(val_total_tensor, world_size)
        average_metric_across_workers(val_correct_tensor, world_size)

        val_avg_loss = (val_loss_tensor / val_total_tensor).item() if val_total_tensor.item() > 0 else float('nan')
        val_acc = (val_correct_tensor / val_total_tensor).item() if val_total_tensor.item() > 0 else float('nan')

        if rank == 0:
            print(f"Epoch {epoch}/{n_epochs} - val_loss={val_avg_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"Training complete. elapsed={elapsed:.1f}s", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        print(f"Unhandled exception in train.py: {exc}", file=sys.stderr, flush=True)
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        sys.exit(1)
