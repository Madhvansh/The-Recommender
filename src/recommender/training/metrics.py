"""Ranking metrics for top-K recommendation.

Given, for each evaluation example, the scores of a candidate set whose **first**
entry (index 0) is the held-out ground-truth item, we compute the rank of that
true item and aggregate the standard top-K metrics:

- **Hit@K** (a.k.a. Recall@K for a single relevant item): fraction of users whose
  true item appears in the top K.
- **NDCG@K**: normalized discounted cumulative gain; rewards placing the true
  item nearer the top of the list.
- **MRR**: mean reciprocal rank, for reference.

All functions are vectorized over the batch dimension.
"""

from __future__ import annotations

import torch


def ranks_of_target(scores: torch.Tensor, target_col: int = 0) -> torch.Tensor:
    """1-based rank of the target candidate within each row of ``scores``.

    Parameters
    ----------
    scores : (B, C) — candidate scores; column ``target_col`` is the true item.

    Returns
    -------
    (B,) long — rank of the true item (1 = best). Ties are broken pessimistically
    (the target ranks behind any candidate with an equal score), which avoids
    optimistic metric inflation.
    """
    # Defensive: a NaN score would otherwise compare False everywhere and yield
    # a bogus rank of 0 (inflating metrics). Send NaNs to the bottom instead.
    scores = torch.nan_to_num(scores, nan=float("-inf"))
    target = scores[:, target_col].unsqueeze(1)  # (B, 1)
    # Count how many candidates score strictly higher, plus ties (>=) excluding
    # the target itself -> pessimistic rank.
    higher = (scores > target).sum(dim=1)
    ties = (scores == target).sum(dim=1) - 1  # exclude the target itself
    return higher + ties + 1


def hit_at_k(ranks: torch.Tensor, k: int) -> float:
    """Hit@K: fraction of targets ranked within the top ``k``."""
    return (ranks <= k).float().mean().item()


def ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    """NDCG@K for a single relevant item per query."""
    in_top = ranks <= k
    gains = torch.zeros_like(ranks, dtype=torch.float)
    gains[in_top] = 1.0 / torch.log2(ranks[in_top].float() + 1.0)
    return gains.mean().item()


def mrr(ranks: torch.Tensor) -> float:
    """Mean reciprocal rank."""
    return (1.0 / ranks.float()).mean().item()


def compute_metrics(ranks: torch.Tensor, ks: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Aggregate Hit@K / NDCG@K for each ``k`` plus MRR into a flat dict."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranks, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranks, k)
    out["mrr"] = mrr(ranks)
    return out
