"""Tests for SSM discretization rules."""

import torch

from recommender.models.s4.discretization import (
    discretize,
    discretize_bilinear,
    discretize_zoh,
)


def test_bilinear_matches_closed_form():
    lam = torch.tensor([-1.0 + 2.0j, -0.5 + 0.0j])
    step = torch.tensor([0.1, 0.1])
    lam_bar, scale = discretize_bilinear(lam, step)
    half = step * lam / 2
    expected = (1 + half) / (1 - half)
    assert torch.allclose(lam_bar, expected)
    assert torch.allclose(scale, step / (1 - half))


def test_zoh_matches_exp():
    lam = torch.tensor([-1.0 + 0.0j, -2.0 + 1.0j])
    step = torch.tensor([0.05, 0.05])
    lam_bar, _scale = discretize_zoh(lam, step)
    assert torch.allclose(lam_bar, torch.exp(step * lam))


def test_zoh_small_step_no_nan():
    # The (exp-1)/a term must stay finite as a -> 0.
    lam = torch.tensor([1e-9 + 0.0j])
    step = torch.tensor([0.1])
    lam_bar, scale = discretize_zoh(lam, step)
    assert torch.isfinite(lam_bar).all()
    assert torch.isfinite(scale).all()
    # scale -> step when a -> 0.
    assert torch.allclose(scale.real, step, atol=1e-3)


def test_dispatch_unknown_raises():
    try:
        discretize(torch.zeros(2, dtype=torch.cfloat), torch.ones(2), "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")
