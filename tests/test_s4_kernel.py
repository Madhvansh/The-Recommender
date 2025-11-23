"""Tests for the S4 DPLR kernel, FFT convolution, and recurrent equivalence."""

import torch
import torch.nn.functional as F

from recommender.models.s4.hippo import make_dplr_hippo
from recommender.models.s4.kernel import (
    discretize_dplr_dense,
    fft_conv,
    s4_kernel_dplr,
    s4_recurrent_step,
)
from recommender.models.s4.s4_layer import S4Layer


def _stable_params(n):
    lam, p, b, _ = make_dplr_hippo(n)
    lam = torch.complex(-F.softplus(-lam.real), lam.imag)
    c = (torch.randn(n) + 1j * torch.randn(n)).to(torch.cfloat) * n**-0.5
    return lam, p, b, c


def test_kernel_matches_aliased_dense():
    torch.manual_seed(0)
    n, length = 16, 12
    lam, p, b, c = _stable_params(n)
    step = torch.tensor([0.05])
    k = s4_kernel_dplr(lam[None], p[None], p[None], b[None], c[None], step, length)[0]

    # Build the dense discretized system and sum many periods (aliasing).
    a_bar, b_bar = discretize_dplr_dense(lam[None], p[None], p[None], b[None], step)
    a_bar, b_bar = a_bar[0], b_bar[0]
    periods = 200
    full = []
    x = b_bar.clone()
    for _ in range(periods * length):
        full.append((c * x).sum().real)
        x = a_bar @ x
    full = torch.stack(full).reshape(periods, length).sum(0)
    assert torch.allclose(k, full, atol=1e-4)


def test_fft_conv_causal():
    # A unit impulse at t=0 convolved with a kernel returns the kernel.
    b, h, length = 1, 2, 8
    u = torch.zeros(b, h, length)
    u[:, :, 0] = 1.0
    kernel = torch.randn(h, length)
    y = fft_conv(u, kernel)
    assert torch.allclose(y[0], kernel, atol=1e-5)


def test_fft_conv_no_future_leakage():
    # Output at time t must not depend on inputs after t (causality).
    torch.manual_seed(1)
    h, length = 3, 16
    kernel = torch.randn(h, length)
    u = torch.randn(1, h, length)
    y_full = fft_conv(u, kernel)
    u2 = u.clone()
    u2[:, :, length // 2 :] += 5.0  # perturb the future
    y_pert = fft_conv(u2, kernel)
    assert torch.allclose(y_full[:, :, : length // 2], y_pert[:, :, : length // 2], atol=1e-5)


def test_layer_conv_matches_recurrent():
    torch.manual_seed(0)
    h, n, length, b = 4, 32, 64, 2
    layer = S4Layer(d_model=h, state_size=n)
    with torch.no_grad():
        layer.log_dt.fill_(torch.log(torch.tensor(0.1)))  # decay -> low aliasing
    x = torch.randn(b, length, h)

    y_conv = layer(x)
    state, ys = None, []
    for t in range(length):
        y_t, state = layer.step(x[:, t, :], state)
        ys.append(y_t)
    y_rec = torch.stack(ys, dim=1)
    assert (y_conv - y_rec).abs().max() < 1e-3


def test_recurrent_step_shapes():
    h, n, b = 2, 8, 3
    lam, p, b_vec, c = _stable_params(n)
    lam = lam[None].repeat(h, 1)
    p = p[None].repeat(h, 1)
    b_vec = b_vec[None].repeat(h, 1)
    c = c[None].repeat(h, 1)
    step = torch.full((h,), 0.05)
    a_bar, b_bar = discretize_dplr_dense(lam, p, p, b_vec, step)
    state = torch.zeros(b, h, n, dtype=torch.cfloat)
    new_state, y = s4_recurrent_step(state, torch.randn(b, h), a_bar, b_bar, c)
    assert new_state.shape == (b, h, n)
    assert y.shape == (b, h)
