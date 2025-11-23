"""Tests for ranking metrics."""

import torch

from recommender.training.metrics import (
    compute_metrics,
    hit_at_k,
    ndcg_at_k,
    ranks_of_target,
)


def test_ranks_target_first():
    # Target (col 0) has the highest score -> rank 1.
    scores = torch.tensor([[5.0, 1.0, 2.0, 0.0]])
    assert ranks_of_target(scores).tolist() == [1]


def test_ranks_target_last():
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert ranks_of_target(scores).tolist() == [4]


def test_ranks_pessimistic_ties():
    # All equal -> target ranks behind the others (pessimistic).
    scores = torch.tensor([[1.0, 1.0, 1.0]])
    assert ranks_of_target(scores).tolist() == [3]


def test_nan_scores_rank_last():
    scores = torch.tensor([[float("nan"), 1.0, 2.0]])
    # Target NaN -> pushed to -inf -> ranks last (3), never a bogus rank 0.
    assert ranks_of_target(scores).tolist() == [3]


def test_hit_and_ndcg():
    ranks = torch.tensor([1, 2, 11])
    assert abs(hit_at_k(ranks, 10) - 2 / 3) < 1e-6
    # NDCG@10: 1/log2(2) + 1/log2(3) + 0, averaged over 3.
    expected = (1.0 / torch.log2(torch.tensor(2.0)) + 1.0 / torch.log2(torch.tensor(3.0))) / 3
    assert abs(ndcg_at_k(ranks, 10) - expected.item()) < 1e-6


def test_compute_metrics_keys():
    ranks = torch.tensor([1, 5, 10, 20])
    m = compute_metrics(ranks, ks=(5, 10))
    assert set(m) == {"hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"}
    assert 0.0 <= m["hit@10"] <= 1.0
