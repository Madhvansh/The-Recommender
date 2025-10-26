"""Base class for sequential recommenders (next-item prediction).

A sequential recommender consumes a user's chronologically-ordered interaction
history ``[i_1, ..., i_{t}]`` and predicts the next item ``i_{t+1}``.  Concrete
models differ only in their *sequence backbone* (S4 vs self-attention); the
embedding table, the causal training objective, and the scoring head live here
so the two models are compared on an identical footing.

Items are 1-indexed; id ``0`` is the padding token.  The item embedding table is
**tied** to the output projection (scores are inner products with item
embeddings), which is standard for sequential recommenders and keeps the two
backbones strictly comparable.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class SequentialRecommender(nn.Module, abc.ABC):
    """Abstract base: embedding + backbone + tied next-item scoring head."""

    def __init__(
        self,
        num_items: int,
        d_model: int,
        max_len: int,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        # +1 for the padding id at index 0.
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=self.d_model**-0.5)
        with torch.no_grad():
            self.item_emb.weight[self.pad_idx].zero_()

    # -- to be provided by subclasses ---------------------------------------
    @abc.abstractmethod
    def encode(self, seq: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Map an embedded, masked sequence to per-position hidden states.

        Parameters
        ----------
        seq      : (B, L, d_model) — embedded input (already normed/dropped).
        pad_mask : (B, L) bool     — True where the position is a real item.

        Returns
        -------
        (B, L, d_model) — contextual representation at every position.
        """
        raise NotImplementedError

    # -- shared machinery ----------------------------------------------------
    def _embed(self, item_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask = item_seq != self.pad_idx  # (B, L)
        x = self.item_emb(item_seq) * (self.d_model**0.5)
        return x, pad_mask

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Return per-position hidden states for ``item_seq`` (B, L)."""
        x, pad_mask = self._embed(item_seq)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        h = self.encode(x, pad_mask)
        # Zero out padded positions so they never leak into pooling/scoring.
        return h * pad_mask.unsqueeze(-1)

    def all_item_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score every catalog item from a hidden vector via the tied table.

        Parameters
        ----------
        hidden : (..., d_model)

        Returns
        -------
        (..., num_items + 1) logits (column 0 is the padding item).
        """
        return hidden @ self.item_emb.weight.t()

    def score_last(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Logits over all items for the position after the last real token."""
        hidden = self.forward(item_seq)  # (B, L, d)
        lengths = (item_seq != self.pad_idx).sum(dim=1)  # (B,)
        last_idx = (lengths - 1).clamp(min=0)
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]  # (B, d)
        return self.all_item_scores(last_hidden)  # (B, num_items+1)

    def sequence_logits(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Logits at *every* position (for the shifted next-item training loss)."""
        hidden = self.forward(item_seq)  # (B, L, d)
        return self.all_item_scores(hidden)  # (B, L, num_items+1)
