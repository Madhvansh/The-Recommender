# The Recommender — State-Space Sequential Recommendation (S4)

A PyTorch sequential recommender that frames recommendation as **next-item
prediction** and uses a **from-scratch Structured State-Space (S4)** kernel as
its sequence backbone — benchmarked head-to-head against a **SASRec**
self-attention baseline on Amazon Reviews.

[![CI](https://github.com/Madhvansh/The-Recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Madhvansh/The-Recommender/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Highlights

- 🧩 **S4 kernel implemented from scratch** — DPLR state matrix from HiPPO-LegS,
  bilinear & ZOH discretization, and an FFT long-convolution via the
  Cauchy/Woodbury generating function. No `pykeops`, no custom CUDA — just
  PyTorch complex tensors. ([deep dive](docs/s4_kernel.md))
- 🔬 **Engine independently validated** at **86.8%** on pixel-by-pixel
  sequential MNIST, a standard long-range benchmark.
- 🏆 **Beats SASRec on Amazon Reviews**: **Hit@10 0.089 / NDCG@10 0.047**
  — **+11% / +14%** over the self-attention baseline.
- 🛡️ **Leakage-free evaluation** via a **timestamp leave-one-out** split (no
  random-split future-leakage), unit-tested.
- ⚡ **O(L log L)** sequence mixing vs SASRec's O(L²); intrinsically causal, no
  positional embeddings needed.
- 🚀 **End-to-end**: data download → training → evaluation → a FastAPI serving
  API + Docker image.

## Results

### Amazon Reviews (Beauty) — timestamp leave-one-out, 100 sampled negatives

| Model | Hit@10 | NDCG@10 | Hit@5 | NDCG@5 |
|---|---|---|---|---|
| SASRec (self-attention baseline) | 0.0802 | 0.0412 | 0.0547 | 0.0331 |
| **S4Rec (ours)** | **0.0890** | **0.0470** | **0.0612** | **0.0381** |
| **Δ vs SASRec** | **+11.0%** | **+14.1%** | +11.9% | +15.1% |

### S4 engine validation — sequence classification

| Model | Task | Test accuracy |
|---|---|---|
| `S4Classifier` | sequential MNIST (L=784) | **86.8%** |

Full details and reproduction commands: [docs/experiments.md](docs/experiments.md).

## How it works

```
history [i₁ … i_t] ─► item embedding ─► S4 backbone ─► tied scoring head ─► P(i_{t+1})
```

The S4 backbone keeps the state matrix in **diagonal-plus-low-rank** form,
evaluates the truncated generating function at roots of unity (collapsing to
Cauchy sums + a rank-1 Woodbury correction), and recovers the convolution kernel
by inverse FFT. The same SSM has an exact `O(N)` recurrent form for inference —
the two views are asserted equivalent in the test-suite.

See [docs/architecture.md](docs/architecture.md) and
[docs/s4_kernel.md](docs/s4_kernel.md).

## Install

```bash
git clone https://github.com/Madhvansh/The-Recommender.git
cd The-Recommender
pip install -e ".[dev]"     # installs torch, numpy, pandas, etc.
```

## Quickstart

No download required — run the synthetic smoke comparison on CPU:

```bash
recommender compare --config configs/synthetic_quick.yaml
```

Train on real data:

```bash
python scripts/download_data.py --category beauty       # -> data/raw/
recommender train   --config configs/s4rec_amazon_beauty.yaml
recommender compare --config configs/s4rec_amazon_beauty.yaml   # S4Rec vs SASRec
```

Validate the S4 engine on sequence classification:

```bash
python scripts/benchmark_s4_classification.py --epochs 30 --d-model 128
```

Serve recommendations (see [docs/deployment.md](docs/deployment.md)):

```bash
export RECO_NUM_ITEMS=12101
uvicorn recommender.serve:app --port 8000
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"history": [12, 45, 9], "k": 10}'
```

## Project layout

```
src/recommender/
├── models/
│   ├── s4/                 # the from-scratch S4 engine
│   │   ├── hippo.py        # HiPPO-LegS -> NPLR -> DPLR initialization
│   │   ├── discretization.py  # bilinear / ZOH
│   │   ├── kernel.py       # Cauchy+Woodbury generating function, FFT conv, recurrence
│   │   └── s4_layer.py     # trainable S4Layer / S4Block
│   ├── base.py             # shared embedding + tied scoring head
│   ├── s4rec.py            # S4 recommender
│   ├── sasrec.py           # SASRec self-attention baseline
│   └── s4_classifier.py    # S4 sequence-classification head (engine validation)
├── data/                   # Amazon loader, leave-one-out splits, torch datasets
├── training/               # losses, ranking metrics, Trainer
├── pipeline.py             # config -> loaders/model/trainer
├── cli.py                  # train / evaluate / compare
├── inference.py · serve.py # top-K recommender + FastAPI service
└── utils/                  # config, logging, seeding
scripts/    configs/    docs/    tests/    Dockerfile
```

## Features

**Core**
- From-scratch S4 (DPLR / HiPPO / bilinear+ZOH / FFT long-conv / recurrent step)
- S4Rec next-item recommender and a faithful SASRec baseline on a shared footing
- Timestamp leave-one-out split (leakage-free) with sampled-negative ranking
- Hit@K, NDCG@K, MRR with pessimistic tie-breaking
- Full-softmax and SASRec-style BCE objectives
- 30-test suite (kernel correctness, causality, leakage, metrics) + CI

**Extended**
- S4 engine validation on sequence classification (sMNIST + synthetic fallback)
- YAML-driven experiments and a `compare` command tabulating S4Rec-vs-SASRec deltas
- FastAPI serving + Docker image + deployment guide
- `bilinear`/`zoh` and `legs`/`lin`/`inv` S4(D) initialization variants for ablation
- Synthetic data generator so the whole pipeline runs offline

## Roadmap / future work

- **S5 / bidirectional & multi-head SSM** mixing and gated (Mamba-style) selective
  state spaces.
- **Real-data leaderboard** across more Amazon categories (Toys, Sports, Games)
  and MovieLens, with full-catalog (un-sampled) ranking metrics.
- **Hybrid S4 + attention** blocks and a learned fusion of the two backbones.
- **Side information**: item text/price/category features fused into the embedding.
- **Faster training**: associative-scan recurrence, mixed precision, and a fused
  Cauchy kernel.
- **Stateful serving**: per-user cached recurrent state for streaming updates.
- **Calibration & diversity** objectives beyond accuracy (coverage, novelty).

## References

- Gu, Goel & Ré. *Efficiently Modeling Long Sequences with Structured State
  Spaces (S4).* ICLR 2022.
- Gu et al. *On the Parameterization and Initialization of Diagonal State Space
  Models (S4D).* 2022.
- Rush & Karamcheti. *The Annotated S4.* 2022.
- Kang & McAuley. *Self-Attentive Sequential Recommendation (SASRec).* ICDM 2018.
- Hou et al. *Amazon Reviews 2023* (McAuley Lab).

## License

MIT — see [LICENSE](LICENSE).
