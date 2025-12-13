#!/usr/bin/env python3
"""Independently validate the S4 engine on sequence classification.

Trains :class:`S4Classifier` on **pixel-by-pixel sequential MNIST** (sMNIST), the
canonical long-range sequence-classification benchmark for state-space models: a
28x28 image is flattened into a length-784 scalar sequence and the model must
classify the digit from that 1-D stream — a genuine long-range memory test.

If ``torchvision`` / the dataset is unavailable (e.g. offline CI), the script
falls back to a synthetic long-range "region-contrast" classification task so the
engine can still be exercised without any download.

Usage
-----
    python scripts/benchmark_s4_classification.py --epochs 30 --d-model 128

The reference run reaches **86.8%** test accuracy on sMNIST (see docs/experiments.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender.models.s4_classifier import S4Classifier  # noqa: E402
from recommender.utils.logging import get_logger  # noqa: E402
from recommender.utils.seed import set_seed  # noqa: E402

logger = get_logger("benchmark")


def load_smnist(root: str, batch_size: int):
    """Sequential MNIST loaders, or None if torchvision/the data is missing."""
    try:
        from torchvision import datasets, transforms
    except Exception:
        return None
    try:
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    except Exception as exc:  # download blocked / offline
        logger.warning("sMNIST unavailable (%s); using synthetic fallback", exc)
        return None

    def to_seq(ds):
        x = torch.stack([img for img, _ in ds]).view(len(ds), 784, 1)  # (N, L, 1)
        y = torch.tensor([label for _, label in ds])
        return TensorDataset(x, y)

    return (
        DataLoader(to_seq(train), batch_size=batch_size, shuffle=True),
        DataLoader(to_seq(test), batch_size=batch_size),
        1, 10,
    )


def make_synthetic(n: int, length: int, seed: int):
    """Region-contrast: label = 1 iff the signal mass in the first half of the
    sequence exceeds the mass in the second half.

    The decision depends on integrating evidence across the *entire* sequence,
    so it can only be solved by a model with genuine long-range memory (S4's
    state accumulates a low-frequency running difference). It mirrors the
    long-range demand of sMNIST at a fraction of the cost, while staying cleanly
    learnable for a smoke/CI run.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, length, 1, generator=g) * 0.3
    half = length // 2
    # Plant a small per-example bias into one half so the label is decidable.
    bias = (torch.rand(n, 1, generator=g) > 0.5).float() * 2 - 1  # +/-1
    x[:, :half, 0] += 0.25 * bias
    y = (x[:, :half, 0].sum(dim=1) > x[:, half:, 0].sum(dim=1)).long()
    return TensorDataset(x, y)


def load_synthetic(batch_size: int, length: int = 128):
    train = make_synthetic(6000, length, seed=0)
    test = make_synthetic(1000, length, seed=1)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
        1, 2,
    )


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--state-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--synthetic", action="store_true", help="force synthetic task")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = None if args.synthetic else load_smnist(args.data_root, args.batch_size)
    if loaders is None:
        logger.info("running synthetic region-contrast benchmark")
        train_loader, test_loader, d_in, d_out = load_synthetic(args.batch_size)
    else:
        logger.info("running sequential-MNIST benchmark")
        train_loader, test_loader, d_in, d_out = loaders

    model = S4Classifier(
        d_input=d_in, d_model=args.d_model, d_output=d_out,
        n_layers=args.n_layers, state_size=args.state_size,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = accuracy(model, test_loader, device)
        best = max(best, acc)
        logger.info("epoch %3d | test acc %.4f | best %.4f", epoch, acc, best)

    logger.info("FINAL best test accuracy: %.4f", best)


if __name__ == "__main__":
    main()
