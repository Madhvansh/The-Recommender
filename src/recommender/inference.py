"""Inference helpers: turn a trained model into top-K recommendations.

Wraps a trained :class:`SequentialRecommender` with the left-padding and last-
position read-out used at eval time, exposing a simple ``recommend(history)`` API
that returns the highest-scoring next items (excluding ones already seen).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data.dataset import left_pad
from .models import build_model
from .utils.config import Config


@dataclass
class Recommendation:
    items: list[int]
    scores: list[float]


class Recommender:
    """Load a checkpoint and serve next-item recommendations."""

    def __init__(self, model, max_len: int, pad_idx: int = 0, device: str = "cpu") -> None:
        self.model = model.eval().to(device)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        num_items: int,
        device: str = "cpu",
    ) -> Recommender:
        cfg = Config.from_yaml(config_path)
        model = build_model(
            cfg.model.name,
            num_items=num_items,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            state_size=cfg.model.state_size,
            n_heads=cfg.model.n_heads,
            max_len=cfg.model.max_len,
            dropout=cfg.model.dropout,
            discretization=cfg.model.discretization,
        )
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        return cls(model, cfg.model.max_len, device=device)

    @torch.no_grad()
    def recommend(
        self,
        history: list[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Recommendation:
        """Return the top-``k`` next-item recommendations for a history.

        Parameters
        ----------
        history : chronological list of item ids the user interacted with.
        k : number of recommendations to return.
        exclude_seen : if True, items already in ``history`` are never returned.
        """
        if not history:
            raise ValueError("history must contain at least one item")
        padded = torch.tensor(
            [left_pad(history, self.max_len, self.pad_idx)],
            dtype=torch.long, device=self.device,
        )
        scores = self.model.score_last(padded)[0]  # (num_items+1,)
        scores[self.pad_idx] = float("-inf")        # never recommend padding
        if exclude_seen:
            scores[torch.tensor(history, device=self.device)] = float("-inf")
        top_scores, top_items = torch.topk(scores, k)
        return Recommendation(
            items=top_items.cpu().tolist(),
            scores=top_scores.cpu().tolist(),
        )
