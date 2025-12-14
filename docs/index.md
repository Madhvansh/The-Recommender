---
title: The Recommender
description: State-space (S4) sequential recommender benchmarked against SASRec.
---

# The Recommender

A PyTorch sequential recommender built around a **from-scratch Structured
State-Space (S4)** kernel, framing recommendation as next-item prediction and
benchmarked head-to-head against a **SASRec** self-attention baseline.

[View the project on GitHub »](https://github.com/Madhvansh/The-Recommender)

## Results at a glance

**Amazon Reviews (Beauty)** — timestamp leave-one-out, 100 sampled negatives:

| Model | Hit@10 | NDCG@10 |
|---|---|---|
| SASRec (baseline) | 0.0802 | 0.0412 |
| **S4Rec (ours)** | **0.0890** | **0.0470** |
| Δ | **+11.0%** | **+14.1%** |

**S4 engine validation:** 86.8% on pixel-by-pixel sequential MNIST.

## Documentation

- [The S4 kernel, from scratch](s4_kernel.html) — HiPPO → DPLR → discretization
  → generating function (Cauchy + Woodbury) → FFT long-convolution.
- [Architecture](architecture.html) — end-to-end data → model → metrics flow and
  why the timestamp leave-one-out split.
- [Experiments & results](experiments.html) — full benchmark tables and
  reproduction commands.
- [Deployment](deployment.html) — train → serve (FastAPI) → Docker.

## Quickstart

```bash
pip install -e ".[dev]"
recommender compare --config configs/synthetic_quick.yaml   # no download needed
```

---

<small>MIT licensed · built by Madhvansh</small>
