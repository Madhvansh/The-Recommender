"""Trainable S4 layer and residual S4 block.

:class:`S4Layer` wraps the from-scratch DPLR kernel (:mod:`.kernel`) in an
``nn.Module`` with learnable parameters:

- ``Lambda`` (complex diagonal) initialized from HiPPO-LegS,
- the rank-1 low-rank vector ``P`` (and ``Q = P`` for the symmetric LegS init),
- the input/output projections ``B``, ``C``,
- a per-channel feed-through ``D``,
- a log-parameterized step size ``log_dt`` sampled in a sensible band.

Complex parameters are stored as real ``(..., 2)`` tensors and viewed as complex
on the fly so that standard optimizers and ``state_dict`` serialization work
without custom hooks.

:class:`S4Block` is the full residual block used by the sequence models: a
pre-norm, the S4 mixing layer, a GLU-gated pointwise output projection, and
dropout — matching the block design from the S4 paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hippo import make_dplr_hippo
from .kernel import discretize_diagonal, fft_conv, s4_kernel_dplr, s4_recurrent_step


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """View a real ``(..., 2)`` parameter as a complex tensor."""
    return torch.view_as_complex(x)


class S4Layer(nn.Module):
    """A single-input-single-output (per-channel) S4 mixing layer.

    Models ``d_model`` independent SSMs (one per feature channel), each with a
    state of size ``state_size``.  The forward pass builds the convolution
    kernel from the DPLR parameters and applies it as an FFT long-convolution.

    Parameters
    ----------
    d_model : int    — number of feature channels ``H``.
    state_size : int — SSM state dimension ``N`` (default 64).
    dt_min, dt_max : float — band for the log-uniform step-size init.
    discretization : {"bilinear", "zoh"}.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        discretization: str = "bilinear",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.discretization = discretization

        lam, p, b, _ = make_dplr_hippo(state_size)  # (N,) each, complex
        # Broadcast the shared HiPPO init across all H channels.
        lam = lam.unsqueeze(0).repeat(d_model, 1)  # (H, N)
        p = p.unsqueeze(0).repeat(d_model, 1)
        b = b.unsqueeze(0).repeat(d_model, 1)
        c = torch.randn(d_model, state_size, dtype=torch.cfloat) * (state_size**-0.5)

        # Store complex params as real (..., 2) tensors so optimizers see floats.
        self.lam = nn.Parameter(torch.view_as_real(lam))
        self.p = nn.Parameter(torch.view_as_real(p))
        self.b = nn.Parameter(torch.view_as_real(b))
        self.c = nn.Parameter(torch.view_as_real(c))
        self.d = nn.Parameter(torch.ones(d_model))

        # Log-uniform step size in [dt_min, dt_max].
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
        log_dt = log_dt + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    # -- kernel construction -------------------------------------------------
    def _params(self):
        lam = _as_complex(self.lam)
        p = _as_complex(self.p)
        b = _as_complex(self.b)
        c = _as_complex(self.c)
        # Clamp the real part of Lambda to be strictly negative for stability.
        lam = torch.complex(-F.softplus(-lam.real), lam.imag)
        step = torch.exp(self.log_dt)
        return lam, p, b, c, step

    def kernel(self, seq_len: int) -> torch.Tensor:
        """Materialize the length-``seq_len`` convolution kernel (H, L)."""
        lam, p, b, c, step = self._params()
        return s4_kernel_dplr(
            lam, p, p, b, c, step, seq_len, discretization=self.discretization
        )

    # -- forward paths -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional forward pass.

        Parameters
        ----------
        x : (B, L, H) — batch, sequence length, channels.

        Returns
        -------
        (B, L, H) — mixed sequence (same shape).
        """
        b_sz, seq_len, h = x.shape
        u = x.transpose(1, 2)  # (B, H, L)
        k = self.kernel(seq_len)  # (H, L)
        y = fft_conv(u, k)  # (B, H, L)
        y = y + u * self.d.view(1, h, 1)  # skip / feed-through term
        return y.transpose(1, 2)  # (B, L, H)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: torch.Tensor | None = None):
        """Recurrent (autoregressive) single step, O(N) per token.

        Used for fast incremental inference and to validate that the recurrent
        and convolutional views coincide.

        Parameters
        ----------
        x_t   : (B, H) — current input per channel.
        state : (B, H, N) complex or None — previous state.

        Returns
        -------
        (y_t, new_state) with y_t shape (B, H).
        """
        lam, p, b, c, step = self._params()
        lam_bar, b_bar = discretize_diagonal(lam, b, step, self.discretization)
        if state is None:
            state = torch.zeros(
                x_t.shape[0], self.d_model, self.state_size,
                dtype=torch.cfloat, device=x_t.device,
            )
        new_state, y_t = s4_recurrent_step(state, x_t, lam_bar, b_bar, c)
        y_t = y_t + self.d.view(1, -1) * x_t
        return y_t, new_state


class S4Block(nn.Module):
    """Residual S4 block: pre-norm -> S4 -> GLU -> dropout, with a skip add."""

    def __init__(
        self,
        d_model: int,
        state_size: int = 64,
        dropout: float = 0.1,
        discretization: str = "bilinear",
        prenorm: bool = True,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model, state_size=state_size, discretization=discretization)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # GLU-style gated output projection.
        self.out_proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.prenorm:
            x = self.norm(x)
        x = self.s4(x)
        x = self.activation(x)
        x = self.out_proj(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        x = residual + x
        if not self.prenorm:
            x = self.norm(x)
        return x
