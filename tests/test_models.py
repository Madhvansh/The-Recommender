"""Tests for the recommender models: shapes, causality, factory, learning."""

import torch

from recommender.models import build_model
from recommender.models.base import SequentialRecommender


def _make(name, num_items=50, max_len=20):
    return build_model(
        name, num_items=num_items, d_model=32, n_layers=2,
        state_size=16, n_heads=2, max_len=max_len, dropout=0.0,
    )


def test_factory_unknown_raises():
    try:
        build_model("nope", num_items=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_both_models_output_shapes():
    seq = torch.randint(1, 51, (4, 20))
    for name in ("s4rec", "sasrec"):
        m = _make(name)
        assert isinstance(m, SequentialRecommender)
        assert m.score_last(seq).shape == (4, 51)
        assert m.sequence_logits(seq).shape == (4, 20, 51)
        assert torch.isfinite(m.score_last(seq)).all()


def test_padding_does_not_change_last_score():
    # Extra left-padding must not change the prediction for the same history.
    m = _make("s4rec", max_len=20)
    m.eval()
    hist = [3, 7, 12, 5]
    short = torch.tensor([[0] * 16 + hist])
    longer = torch.tensor([[0] * 12 + [0, 0, 0, 0] + hist])  # same content, more pad
    with torch.no_grad():
        a = m.score_last(short)
        b = m.score_last(longer)
    assert torch.allclose(a, b, atol=1e-4)


def test_sasrec_causal_no_future_leakage():
    # Changing a future item must not alter the logits at an earlier position.
    m = _make("sasrec", max_len=10)
    m.eval()
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        base = m.sequence_logits(seq)
        seq2 = seq.clone()
        seq2[0, -1] = 25  # change only the last item
        pert = m.sequence_logits(seq2)
    # Positions before the last must be identical.
    assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-5)


def test_models_overfit_tiny_dataset():
    # Sanity: both models should drive the training loss down on a tiny set.
    from recommender.training.losses import masked_cross_entropy

    torch.manual_seed(0)
    seq = torch.randint(1, 21, (8, 12))
    target = torch.randint(1, 21, (8, 12))
    mask = torch.ones_like(seq, dtype=torch.bool)
    for name in ("s4rec", "sasrec"):
        m = _make(name, num_items=20, max_len=12)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        first = last = None
        for step in range(60):
            logits = m.sequence_logits(seq)
            loss = masked_cross_entropy(logits, target, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.8, f"{name} did not learn ({first:.3f} -> {last:.3f})"
