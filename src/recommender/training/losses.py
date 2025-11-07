"""Training objectives for next-item prediction.

Two objectives are supported, both masked to ignore padded positions:

- ``"ce"`` — full-softmax cross-entropy over the whole catalog at every position.
  Strong and simple; the default for S4Rec.
- ``"bce"`` — the SASRec-style binary objective: a logistic loss pushing the
  positive item's score up and one sampled negative's score down per position.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Full-softmax CE over the catalog, averaged over non-pad positions.

    Parameters
    ----------
    logits  : (B, L, V) — per-position scores over all V = num_items+1 items.
    targets : (B, L)    — gold next-item ids.
    mask    : (B, L) bool — True at real (non-pad) positions.
    """
    v = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, v),
        targets.reshape(-1),
        reduction="none",
    )
    loss = loss * mask.reshape(-1).float()
    denom = mask.sum().clamp(min=1)
    return loss.sum() / denom


def bpr_bce_loss(
    model,
    item_seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """SASRec-style binary cross-entropy with one negative per position.

    Computes per-position positive/negative scores by dotting the hidden state
    with the positive and negative item embeddings, then applies a masked
    logistic loss.
    """
    hidden = model.forward(item_seq)             # (B, L, d)
    pos_emb = model.item_emb(pos)                 # (B, L, d)
    neg_emb = model.item_emb(neg)                 # (B, L, d)
    pos_logits = (hidden * pos_emb).sum(dim=-1)   # (B, L)
    neg_logits = (hidden * neg_emb).sum(dim=-1)   # (B, L)

    m = mask.float()
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="none"
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="none"
    )
    loss = ((pos_loss + neg_loss) * m).sum() / m.sum().clamp(min=1)
    return loss
