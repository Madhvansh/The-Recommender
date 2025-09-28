"""HiPPO initialization for the S4 state matrix.

The S4 layer is parameterized by a state matrix ``A`` that, for good long-range
memory, is initialized to the **HiPPO-LegS** matrix.  S4 does not work with the
dense HiPPO matrix directly; instead it uses the fact that HiPPO-LegS is a
*normal-plus-low-rank* (NPLR) matrix:

    A = V (Lambda - p q^*) V^*          (with V unitary)

which we re-express in the **diagonal-plus-low-rank (DPLR)** form actually used
by the kernel:

    A = Lambda - p q^*

where ``Lambda`` is complex-diagonal and ``p, q`` are low-rank (rank-1 here)
column vectors, all in the eigenbasis ``V`` of the *normal* part.

This module derives ``Lambda``, ``p``, ``q`` and the input/output projections
``B``, ``C`` from the original HiPPO matrix following Gu et al., 2022
("Efficiently Modeling Long Sequences with Structured State Spaces", S4).

References
----------
- Gu, Goel, Re. *Efficiently Modeling Long Sequences with Structured State
  Spaces.* ICLR 2022.
- Gu et al. *On the Parameterization and Initialization of Diagonal State Space
  Models* (S4D), 2022 — for the diagonal initialization variants.
"""

from __future__ import annotations

import numpy as np
import torch


def make_hippo(state_size: int) -> np.ndarray:
    """Build the (dense, real) HiPPO-LegS state matrix ``A`` of shape (N, N).

    Uses the standard LegS construction

        A_{nk} = -sqrt((2n+1)(2k+1))   if n > k
                 -(n+1)                 if n == k
                  0                     if n < k

    Returned as the *negated* matrix used by S4 (so the continuous-time system
    ``x' = -A x + B u`` is stable, with eigenvalues in the left half-plane).
    """
    n = np.arange(state_size)
    row, col = np.meshgrid(n, n, indexing="ij")
    norm = np.sqrt((2 * row + 1) * (2 * col + 1))
    a = np.tril(norm, k=-1) + np.diag(n + 1)
    return -a  # (N, N)


def make_nplr_hippo(state_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the normal-plus-low-rank (NPLR) decomposition of HiPPO-LegS.

    Produces ``(A_normal, P, B)`` such that

        A = A_normal - P P^T

    where ``A_normal`` is a *normal* matrix (skew-symmetric plus a constant
    diagonal shift) and ``P`` is the rank-1 correction.  ``B`` is the HiPPO
    input projection.

    Returns
    -------
    A_normal : (N, N) real
    P        : (N,)   real, rank-1 correction column
    B        : (N,)   real, input vector
    """
    hippo = make_hippo(state_size)  # (N, N)

    n = np.arange(state_size)
    # Rank-1 term P such that hippo + P P^T is skew-symmetric (up to a 0.5*I shift).
    p = np.sqrt(n + 0.5)  # (N,)
    # Normal part = hippo + outer(p, p), shifted so it is exactly skew + 0.5 I.
    a_normal = hippo + p[:, None] * p[None, :]

    # HiPPO-LegS input projection.
    b = np.sqrt(2 * n + 1.0)  # (N,)
    return a_normal, p, b


def make_dplr_hippo(
    state_size: int,
    dtype: torch.dtype = torch.cfloat,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonalize HiPPO-LegS into the DPLR form used by the S4 kernel.

    We diagonalize the *normal* part ``A_normal = V Lambda V^*`` and rotate the
    low-rank vectors and input projection into the same eigenbasis ``V``:

        Lambda  = eig(A_normal)                 (complex diagonal)
        P       = V^* p                          (complex, rank-1)
        B       = V^* b                          (complex)

    so that the original system matrix is ``A = Lambda - P P^*`` in the V-basis.

    Returns
    -------
    Lambda : (N,) complex   — diagonal entries of the normal part
    P      : (N,) complex   — low-rank correction (rank 1)
    B      : (N,) complex   — input projection in the eigenbasis
    V      : (N, N) complex  — eigenvectors (for sanity checks / reconstruction)
    """
    a_normal, p, b = make_nplr_hippo(state_size)

    # A_normal is skew-symmetric + 0.5 I; split out the constant real shift so we
    # diagonalize a Hermitian (skew -> i*Hermitian) part with high numerical
    # accuracy via eigh.
    skew = a_normal - 0.5 * np.eye(state_size)  # skew-symmetric
    # eigh on the Hermitian matrix (i * skew) gives real eigenvalues w; the
    # eigenvalues of `skew` are then i*w, and of A_normal are 0.5 - ... wait:
    # eig(A_normal) = 0.5 + eig(skew) = 0.5 + i*w.
    w_imag, v = np.linalg.eigh(-1j * skew)  # -1j*skew is Hermitian
    lam = -0.5 + 1j * w_imag  # eigenvalues of A_normal (Re<0 for stability)

    vc = v.conj().T  # V^*
    p_c = vc @ p.astype(np.complex128)
    b_c = vc @ b.astype(np.complex128)

    lam_t = torch.as_tensor(lam, dtype=dtype)
    p_t = torch.as_tensor(p_c, dtype=dtype)
    b_t = torch.as_tensor(b_c, dtype=dtype)
    v_t = torch.as_tensor(v, dtype=dtype)
    return lam_t, p_t, b_t, v_t


def make_diagonal_hippo(
    state_size: int,
    variant: str = "legs",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """S4D-style diagonal-only initialization of ``Lambda``.

    A lightweight alternative to the full DPLR init for ablations.  ``variant``:

    - ``"legs"``: the diagonal part of the DPLR HiPPO-LegS eigenvalues.
    - ``"lin"``:  S4D-Lin, ``-1/2 + i*pi*n``.
    - ``"inv"``:  S4D-Inv, ``-1/2 + i*(N/pi)*(N/(2n+1) - 1)``.
    """
    if variant == "legs":
        lam, _, _, _ = make_dplr_hippo(state_size, dtype=dtype)
        return lam
    n = torch.arange(state_size)
    real = -0.5 * torch.ones(state_size)
    if variant == "lin":
        imag = torch.pi * n
    elif variant == "inv":
        imag = (state_size / torch.pi) * (state_size / (2 * n + 1) - 1)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown diagonal variant: {variant!r}")
    return torch.complex(real, imag).to(dtype)
