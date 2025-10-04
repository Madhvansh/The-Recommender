"""Discretization of the continuous-time state-space model.

A continuous linear SSM

    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

must be discretized with a step size ``dt`` (``Delta``) before it can be applied
to a discrete sequence.  S4 supports two discretization rules:

- **Bilinear (Tustin)**: a second-order accurate trapezoidal rule, the default
  used in the original S4 paper.
- **Zero-order hold (ZOH)**: exact for piecewise-constant inputs.

Both are implemented here for the *diagonal* state matrix used by the DPLR
kernel (``A = Lambda`` after the low-rank correction is folded into the
generating function), operating elementwise on the complex eigenvalues.
"""

from __future__ import annotations

import torch


def discretize_bilinear(
    lam: torch.Tensor,
    step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (trapezoidal) discretization of a diagonal SSM.

    For a scalar mode ``a`` with step ``dt`` the bilinear transform gives

        a_bar = (1 + dt*a/2) / (1 - dt*a/2)
        scale = dt / (1 - dt*a/2)        (the factor multiplying B)

    Parameters
    ----------
    lam  : (..., N) complex eigenvalues of the (diagonal) state matrix.
    step : (..., N) or (N,) real/complex step sizes ``dt``.

    Returns
    -------
    lam_bar : (..., N) discretized diagonal recurrence coefficients.
    scale   : (..., N) factor to apply to the (already discretized) input map.
    """
    half = step * lam / 2.0
    denom = 1.0 - half
    lam_bar = (1.0 + half) / denom
    scale = step / denom
    return lam_bar, scale


def discretize_zoh(
    lam: torch.Tensor,
    step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-order hold (exact) discretization of a diagonal SSM.

    For a scalar mode ``a``

        a_bar = exp(dt*a)
        scale = (exp(dt*a) - 1) / a       (the factor multiplying B)

    A stable Taylor fallback is used where ``|dt*a|`` is tiny to avoid 0/0.

    Returns
    -------
    lam_bar : (..., N) discretized diagonal recurrence coefficients.
    scale   : (..., N) factor to apply to the input map.
    """
    dt_a = step * lam
    lam_bar = torch.exp(dt_a)
    # (exp(dt*a) - 1) / a = dt * (exp(dt*a) - 1) / (dt*a); use the latter form so
    # the removable singularity at a -> 0 is handled by a series expansion.
    small = dt_a.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(dt_a), dt_a)
    ratio = torch.where(small, 1.0 + dt_a / 2.0, (lam_bar - 1.0) / safe)
    scale = step * ratio
    return lam_bar, scale


DISCRETIZERS = {
    "bilinear": discretize_bilinear,
    "zoh": discretize_zoh,
}


def discretize(
    lam: torch.Tensor,
    step: torch.Tensor,
    method: str = "bilinear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the requested discretization rule by name."""
    try:
        fn = DISCRETIZERS[method]
    except KeyError as exc:  # pragma: no cover - guarded by caller
        raise ValueError(
            f"unknown discretization {method!r}; choose from {sorted(DISCRETIZERS)}"
        ) from exc
    return fn(lam, step)
