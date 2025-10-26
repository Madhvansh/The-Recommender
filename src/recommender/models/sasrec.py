"""SASRec: the self-attention sequential recommendation baseline.

A faithful re-implementation of Kang & McAuley, *Self-Attentive Sequential
Recommendation* (ICDM 2018): learned positional embeddings, a stack of causal
multi-head self-attention + point-wise feed-forward blocks, and a tied item
embedding scoring head.  Used as the head-to-head baseline for S4Rec under an
identical embedding table, objective, and evaluation protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SequentialRecommender


class PointWiseFeedForward(nn.Module):
    """The two-layer 1D-conv FFN from the SASRec paper."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d) -> conv over the channel dim.
        y = x.transpose(1, 2)
        y = self.dropout(self.act(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return y.transpose(1, 2)


class SASRecBlock(nn.Module):
    """Pre-norm causal self-attention block + point-wise FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = PointWiseFeedForward(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.attn_norm(x)
        a, _ = self.attn(
            q, q, q,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SASRec(SequentialRecommender):
    """Self-attentive sequential recommender (baseline)."""

    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 1,
        max_len: int = 200,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(num_items, d_model, max_len, dropout=dropout)
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb.weight, std=d_model**-0.5)
        self.blocks = nn.ModuleList(
            [SASRecBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def encode(self, seq: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b, length, _ = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        x = seq + self.pos_emb(positions.clamp(max=self.max_len - 1))

        # Causal mask: position i may only attend to j <= i.
        causal = torch.triu(
            torch.ones(length, length, device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding = ~pad_mask  # True where padded (to be ignored)

        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=key_padding)
        return self.final_norm(x)
