"""Minimal YAML-backed experiment configuration.

A small typed config object keeps experiments reproducible without pulling in a
heavy config framework.  Configs are plain YAML files (see ``configs/``) loaded
into nested dataclasses; unknown keys are preserved in ``extra`` so configs stay
forward-compatible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "s4rec"            # "s4rec" | "sasrec"
    d_model: int = 64
    n_layers: int = 2
    state_size: int = 64          # S4 only
    n_heads: int = 1              # SASRec only
    dropout: float = 0.2
    max_len: int = 200
    discretization: str = "bilinear"


@dataclass
class DataConfig:
    name: str = "amazon-beauty"
    root: str = "data"
    min_seq_len: int = 5
    max_len: int = 200
    num_negatives_eval: int = 100  # sampled negatives for ranking metrics


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    patience: int = 20            # early-stopping patience (epochs)
    device: str = "auto"
    seed: int = 42
    eval_every: int = 1
    ks: tuple[int, ...] = (5, 10, 20)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw)
        model = ModelConfig(**raw.pop("model", {}))
        data = DataConfig(**raw.pop("data", {}))
        train_raw = raw.pop("train", {})
        if "ks" in train_raw:
            train_raw["ks"] = tuple(train_raw["ks"])
        train = TrainConfig(**train_raw)
        return cls(model=model, data=data, train=train, extra=raw)

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["train"]["ks"] = list(self.train.ks)
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
