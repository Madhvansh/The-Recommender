"""Tests for HiPPO / DPLR initialization."""

import torch

from recommender.models.s4.hippo import make_dplr_hippo, make_hippo, make_nplr_hippo


def test_hippo_shapes_and_lower_triangular():
    n = 16
    a = make_hippo(n)
    assert a.shape == (n, n)
    # Strictly-upper triangle is zero (LegS is lower-triangular).
    assert (torch.as_tensor(a).triu(diagonal=1).abs() < 1e-9).all()


def test_nplr_low_rank_reconstructs_hippo():
    n = 24
    a = torch.as_tensor(make_hippo(n))
    a_normal, p, _b = make_nplr_hippo(n)
    a_normal = torch.as_tensor(a_normal)
    p = torch.as_tensor(p)
    # A = A_normal - P P^T  (the rank-1 correction recovers HiPPO).
    recon = a_normal - torch.outer(p, p)
    assert torch.allclose(recon, a, atol=1e-5)


def test_dplr_eigenvalues_are_stable():
    # Re(Lambda) must be < 0 for a stable continuous-time system.
    lam, p, b, v = make_dplr_hippo(32)
    assert (lam.real < 1e-6).all()
    assert lam.shape == p.shape == b.shape == (32,)


def test_dplr_diagonalizes_normal_part():
    n = 20
    a_normal, _p, _b = make_nplr_hippo(n)
    a_normal = torch.as_tensor(a_normal, dtype=torch.cfloat)
    lam, _p2, _b2, v = make_dplr_hippo(n)
    # V Lambda V^* reconstructs the normal part.
    recon = v @ torch.diag(lam) @ v.conj().t()
    assert torch.allclose(recon, a_normal, atol=1e-4)
