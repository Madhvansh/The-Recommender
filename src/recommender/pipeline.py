"""Assemble an end-to-end experiment from a :class:`Config`.

Glue that turns a config + a loaded :class:`Dataset` into ready-to-use data
loaders, a model, and a :class:`Trainer`, so both the CLI and the scripts share
exactly one code path.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from .data.amazon import Dataset, load_amazon, synthetic_interactions
from .data.dataset import EvalSequenceDataset, TrainSequenceDataset
from .data.splits import leave_one_out
from .models import build_model
from .training.trainer import Trainer
from .utils.config import Config


@dataclass
class Experiment:
    cfg: Config
    dataset: Dataset
    model: object
    trainer: Trainer
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader


def load_dataset(cfg: Config) -> Dataset:
    """Resolve the dataset named in ``cfg`` (synthetic or a real Amazon file)."""
    name = cfg.data.name
    if name.startswith("synthetic"):
        return synthetic_interactions(seed=cfg.train.seed, name=name)
    # Expect a preprocessed/raw file at <root>/<name>.* ; let the loader resolve.
    import glob
    import os

    pattern = os.path.join(cfg.data.root, "raw", f"{name}*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no raw file matching {pattern!r}; run scripts/download_data.py "
            f"or use a 'synthetic*' dataset name."
        )
    return load_amazon(matches[0], k_core=cfg.data.min_seq_len, name=name)


def build_experiment(cfg: Config, loss_type: str = "ce") -> Experiment:
    """Build loaders, model and trainer for ``cfg``."""
    dataset = load_dataset(cfg)
    splits = leave_one_out(dataset.interactions, min_seq_len=cfg.data.min_seq_len)

    max_len = cfg.data.max_len
    train_ds = TrainSequenceDataset(
        splits, dataset.num_items, max_len, seed=cfg.train.seed
    )
    valid_ds = EvalSequenceDataset(
        splits, dataset.num_items, max_len, which="valid",
        num_negatives=cfg.data.num_negatives_eval, seed=cfg.train.seed,
    )
    test_ds = EvalSequenceDataset(
        splits, dataset.num_items, max_len, which="test",
        num_negatives=cfg.data.num_negatives_eval, seed=cfg.train.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.train.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size * 2)

    model = build_model(
        cfg.model.name,
        num_items=dataset.num_items,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        state_size=cfg.model.state_size,
        n_heads=cfg.model.n_heads,
        max_len=max_len,
        dropout=cfg.model.dropout,
        discretization=cfg.model.discretization,
    )
    trainer = Trainer(model, cfg, loss_type=loss_type)
    return Experiment(
        cfg=cfg,
        dataset=dataset,
        model=model,
        trainer=trainer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
    )
