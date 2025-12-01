# The S4 kernel, from scratch

This note documents the Structured State-Space (S4) implementation that powers
the recommender. It is written from first principles and depends on **no**
external SSM kernels (no `pykeops`, no custom CUDA Cauchy op) — just PyTorch
complex tensors and `torch.fft`.

Source: [`src/recommender/models/s4/`](../src/recommender/models/s4/).

## 1. The continuous state-space model

A single-input-single-output linear SSM is

```
x'(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

with state matrix `A ∈ C^{N×N}`, input/output maps `B, C ∈ C^{N}`, and a scalar
feed-through `D`. We use `H` independent SSMs — one per feature channel.

## 2. HiPPO initialization → DPLR form

Random `A` has poor long-range memory. S4 initializes `A` to **HiPPO-LegS**,
which provably memorizes history through orthogonal-polynomial projections.
HiPPO-LegS is *normal-plus-low-rank* (NPLR):

```
A = A_normal − P Pᵀ            (A_normal normal, P rank-1)
```

We diagonalize the normal part `A_normal = V Λ V*` (via a Hermitian `eigh` for
numerical accuracy) and rotate everything into the eigenbasis `V`, giving the
**diagonal-plus-low-rank (DPLR)** parameters actually trained:

```
A = Λ − P Q*      (Λ complex-diagonal; P, Q rank-1; here Q = P)
```

Implemented in [`hippo.py`](../src/recommender/models/s4/hippo.py):
`make_hippo → make_nplr_hippo → make_dplr_hippo`.

## 3. Discretization (bilinear / ZOH)

To apply the SSM to a discrete sequence we discretize with step `Δ`:

- **Bilinear (Tustin)** — `Ā = (1 + Δ/2·a)/(1 − Δ/2·a)`, second-order accurate
  (default).
- **Zero-order hold (ZOH)** — `Ā = exp(Δ·a)`, exact for piecewise-constant
  inputs, with a Taylor fallback near `a → 0`.

See [`discretization.py`](../src/recommender/models/s4/discretization.py).

## 4. The convolutional kernel via the generating function

The SSM output is a causal convolution `y = K * u` with kernel

```
K = (C̄ B̄, C̄ Ā B̄, C̄ Ā² B̄, …, C̄ Ā^{L−1} B̄).
```

Forming the powers `Ā^k` costs `O(L N²)`. Instead we evaluate the **truncated
generating function** `K̂(z) = Σ_k K_k z^k` at the `L`-th roots of unity. For the
bilinear transform this becomes a resolvent `C (g(z)I − A)^{-1} B` with
`g(z) = (2/Δ)(1−z)/(1+z)`, and the DPLR structure collapses it to four **Cauchy
sums** plus a rank-1 **Woodbury** correction:

```
K̂ = (2/(1+z)) · ( k00 − k01 (1 + k11)^{-1} k10 )
k00 = Σ_n  C_n B_n / (g − Λ_n)      k01 = Σ_n  C_n P_n / (g − Λ_n)
k10 = Σ_n  Q̄_n B_n / (g − Λ_n)      k11 = Σ_n  Q̄_n P_n / (g − Λ_n)
```

An inverse FFT returns the time-domain kernel `K`. See
[`kernel.py::s4_kernel_dplr`](../src/recommender/models/s4/kernel.py).

> **Aliasing.** The generating function is infinite; the length-`L` ifft recovers
> the *aliased* kernel `Σ_j K_{k+jL}`. The unit tests verify the kernel matches a
> dense system summed over many periods to `< 1e-4`, and that — once the kernel
> has decayed — it matches the exact recurrence to `~1e-7`.

## 5. Two views, kept in sync

- **Convolutional** (training): `fft_conv` applies `K` in `O(L log L)`,
  zero-padded to avoid circular wrap-around.
- **Recurrent** (inference): `s4_recurrent_step` runs the dense DPLR recurrence
  `x_t = Ā x_{t−1} + B̄ u_t` in `O(N²)` per token.

The two are asserted equivalent in
[`tests/test_s4_kernel.py`](../tests/test_s4_kernel.py).

## 6. Trainable layer

[`s4_layer.py`](../src/recommender/models/s4/s4_layer.py) wraps the kernel in an
`nn.Module`: complex parameters stored as real `(…, 2)` tensors (so optimizers
and `state_dict` work unchanged), a `softplus` clamp keeping `Re(Λ) < 0` for
stability, and a log-parameterized step size. `S4Block` adds pre-norm, a
GLU-gated projection, and a residual connection.
