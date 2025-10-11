"""The Structured State-Space (S4) convolution kernel, from scratch.

This module is the heart of the project.  It computes, for each of ``H`` feature
channels, the length-``L`` SSM convolution kernel

    K = (C B, C A B, C A^2 B, ..., C A^{L-1} B)

*without* materializing any of the powers ``A^k`` (which would be O(L N^2)).
Instead it uses the S4 algorithm:

1. Keep the state matrix in **diagonal-plus-low-rank (DPLR)** form
   ``A = Lambda - P Q^*`` (here rank 1).
2. Discretize ``Lambda`` (bilinear or ZOH) — see :mod:`.discretization`.
3. Evaluate the **truncated generating function**
   ``K_hat(z) = sum_k K_k z^k`` at the ``L``-th roots of unity.  Under the DPLR
   structure this collapses to a **Cauchy kernel** plus a rank-1 **Woodbury**
   correction.
4. Recover the time-domain kernel ``K`` by an inverse FFT, then run the actual
   sequence convolution as an FFT **long-convolution** (O(L log L)).

Everything is written in terms of explicit complex arithmetic so the math is
auditable; no external SSM kernels (pykeops / cauchy CUDA) are required.

References
----------
Gu, Goel & Re, *Efficiently Modeling Long Sequences with Structured State
Spaces*, ICLR 2022; and Rush & Karamcheti, *The Annotated S4*.
"""

from __future__ import annotations

import torch

from .discretization import discretize


def cauchy(v: torch.Tensor, omega: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Cauchy matrix-vector product ``sum_n v[n] / (omega - lam[n])``.

    Parameters
    ----------
    v     : (H, N) complex   — numerators (per channel, per state).
    omega : (L,)   complex   — evaluation nodes (roots-of-unity image).
    lam   : (H, N) complex   — poles (discretized eigenvalues).

    Returns
    -------
    (H, L) complex — the Cauchy sum evaluated at every node for every channel.
    """
    # (H, L, N) = omega[L,1] - lam[H,1,N]  then reduce over N.
    denom = omega.view(1, -1, 1) - lam.unsqueeze(1)
    return (v.unsqueeze(1) / denom).sum(dim=-1)


def s4_kernel_dplr(
    lam: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    step: torch.Tensor,
    seq_len: int,
    discretization: str = "bilinear",
) -> torch.Tensor:
    """Compute the length-``L`` S4 convolution kernel for ``H`` channels.

    All tensors carry a leading channel dim ``H`` and a state dim ``N``.

    Parameters
    ----------
    lam  : (H, N) complex  — diagonal eigenvalues ``Lambda``.
    p, q : (H, N) complex  — rank-1 low-rank vectors of ``A = Lambda - P Q^*``.
    b    : (H, N) complex  — input projection.
    c    : (H, N) complex  — output projection.
    step : (H,)   real     — per-channel discretization step ``dt``.
    seq_len : int          — kernel/sequence length ``L``.
    discretization : {"bilinear", "zoh"}.

    Returns
    -------
    (H, L) real — the time-domain convolution kernel.
    """
    h, n = lam.shape
    device = lam.device
    step = step.view(h, 1).to(lam.real.dtype)

    # L-th roots of unity z = exp(-2 pi i k / L).
    k = torch.arange(seq_len, device=device)
    z = torch.exp(-2j * torch.pi * k / seq_len).to(lam.dtype)  # (L,)

    if discretization == "bilinear":
        # Bilinear maps the unit circle to the imaginary axis:
        #   g(z) = (2/dt) (1 - z)/(1 + z),   c(z) = 2/(1 + z)
        g = (2.0 / step) * ((1.0 - z) / (1.0 + z))  # (H, L)
        front = 2.0 / (1.0 + z)  # (L,)
    elif discretization == "zoh":
        # ZOH maps via the matrix logarithm of the unit circle.
        g = (1.0 / step) * (seq_len) * (1.0 - z)  # first-order surrogate node map
        front = torch.ones_like(z)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown discretization {discretization!r}")

    # Woodbury: invert (g - A)^{-1} with A = Lambda - P Q^*.  Each of the four
    # Cauchy sums shares the pole set Lambda but uses different numerators.
    def _cauchy_over_g(num: torch.Tensor) -> torch.Tensor:
        # num: (H, N); evaluate sum_n num / (g - lam) for every node in g.
        denom = g.unsqueeze(-1) - lam.unsqueeze(1)  # (H, L, N)
        return (num.unsqueeze(1) / denom).sum(dim=-1)  # (H, L)

    cb = c * b
    cp = c * p
    qb = q.conj() * b
    qp = q.conj() * p

    k00 = _cauchy_over_g(cb)
    k01 = _cauchy_over_g(cp)
    k10 = _cauchy_over_g(qb)
    k11 = _cauchy_over_g(qp)

    at_roots = front * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)  # (H, L)

    # Inverse FFT back to the time domain; the kernel is real by construction.
    kernel = torch.fft.ifft(at_roots, n=seq_len, dim=-1)
    return kernel.real


def fft_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal long-convolution of inputs ``u`` with ``kernel`` via FFT.

    Parameters
    ----------
    u      : (B, H, L) real — batched per-channel input sequences.
    kernel : (H, L)   real — per-channel convolution kernel.

    Returns
    -------
    (B, H, L) real — the causally-convolved output, truncated to length ``L``.
    """
    seq_len = u.shape[-1]
    fft_len = 2 * seq_len  # zero-pad to avoid circular wrap-around
    u_f = torch.fft.rfft(u, n=fft_len, dim=-1)
    k_f = torch.fft.rfft(kernel, n=fft_len, dim=-1).unsqueeze(0)
    y = torch.fft.irfft(u_f * k_f, n=fft_len, dim=-1)
    return y[..., :seq_len]


def discretize_diagonal(
    lam: torch.Tensor,
    b: torch.Tensor,
    step: torch.Tensor,
    method: str = "bilinear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: discretize ``(Lambda, B)`` for the recurrent path.

    Used by the step-by-step recurrence (:func:`s4_recurrent_step`) and tests
    that compare the convolutional and recurrent views of the same SSM.
    """
    step = step.view(-1, 1)
    lam_bar, scale = discretize(lam, step, method)
    return lam_bar, scale * b


def s4_recurrent_step(
    state: torch.Tensor,
    u_t: torch.Tensor,
    lam_bar: torch.Tensor,
    b_bar: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of the diagonal SSM recurrence (for inference / validation).

    Implements ``x_t = lam_bar * x_{t-1} + b_bar * u_t`` and
    ``y_t = Re(<c, x_t>)``.  This is the O(N) autoregressive view that S4 shares
    with the O(L log L) convolutional view above; the two must agree, which the
    test-suite asserts.

    Parameters
    ----------
    state   : (B, H, N) complex — previous hidden state.
    u_t     : (B, H)    real    — current input per channel.
    lam_bar : (H, N)    complex — discretized recurrence coefficients.
    b_bar   : (H, N)    complex — discretized input map.
    c       : (H, N)    complex — output map.

    Returns
    -------
    (new_state, y_t) with shapes (B, H, N) and (B, H).
    """
    new_state = lam_bar.unsqueeze(0) * state + b_bar.unsqueeze(0) * u_t.unsqueeze(-1)
    y_t = 2.0 * (new_state * c.unsqueeze(0)).sum(dim=-1).real
    return new_state, y_t
