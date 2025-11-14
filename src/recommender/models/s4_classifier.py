"""A small S4 sequence classifier.

This model exists to *independently validate the S4 engine* on a standard
sequence-classification task (e.g. pixel-by-pixel sequential MNIST), separate
from the recommendation setting.  If the same kernel that powers S4Rec also
reaches strong accuracy on sMNIST, we know the long-range memory machinery is
correct end-to-end.

Architecture: linear input encoder -> stack of S4 blocks -> mean-pool over time
-> linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .s4.s4_layer import S4Block


class S4Classifier(nn.Module):
    """S4 backbone with a mean-pooled classification head."""

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int = 4,
        state_size: int = 64,
        dropout: float = 0.1,
        discretization: str = "bilinear",
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
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
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_input) -> logits (B, d_output)."""
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = h.mean(dim=1)  # mean-pool over the sequence
        return self.head(h)
