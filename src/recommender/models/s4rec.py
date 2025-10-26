"""S4Rec: a state-space sequential recommender.

The sequence backbone is a stack of residual S4 blocks (:class:`S4Block`).
Unlike self-attention, S4 is *intrinsically causal* (the convolution kernel is
one-sided) and scales as O(L log L) rather than O(L^2), so it handles long user
histories cheaply while still capturing long-range dependencies through the
HiPPO-initialized state matrix.

No positional embeddings are needed: the SSM recurrence is inherently
order-aware.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SequentialRecommender
from .s4.s4_layer import S4Block


class S4Rec(SequentialRecommender):
    """State-space sequential recommender.

    Parameters
    ----------
    num_items : int   — catalog size (items are 1-indexed).
    d_model : int     — embedding / hidden width.
    n_layers : int    — number of stacked S4 blocks.
    state_size : int  — SSM state dimension ``N`` per channel.
    max_len : int     — maximum sequence length.
    dropout : float
    discretization : {"bilinear", "zoh"}.
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        n_layers: int = 2,
        state_size: int = 64,
        max_len: int = 200,
        dropout: float = 0.2,
        discretization: str = "bilinear",
    ) -> None:
        super().__init__(num_items, d_model, max_len, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                S4Block(
                    d_model,
                    state_size=state_size,
                    dropout=dropout,
                    discretization=discretization,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def encode(self, seq: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = seq
        mask = pad_mask.unsqueeze(-1)
        for block in self.blocks:
            # Re-mask between blocks so padding can't propagate through the
            # global convolution.
            x = block(x * mask)
        return self.final_norm(x)
