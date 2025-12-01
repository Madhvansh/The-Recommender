# Architecture

How the pieces fit together, from raw interactions to ranked recommendations.

```
 raw Amazon reviews
        │  scripts/download_data.py
        ▼
 data/raw/amazon-*.csv
        │  data/amazon.py  (k-core filter, id remap)
        ▼
 user → [(item, timestamp)]
        │  data/splits.py  (timestamp leave-one-out)
        ▼
 per-user (train_seq, valid_target, test_target)
        │  data/dataset.py  (left-pad, negative sampling)
        ▼
 ┌─────────────────────────────────────────────┐
 │  SequentialRecommender  (models/base.py)     │
 │   item embedding  ──tied──►  scoring head     │
 │            │                                  │
 │   ┌────────┴─────────┐                        │
 │   │   backbone        │                       │
 │   │  S4Rec  | SASRec  │                        │
 │   └──────────────────┘                        │
 └─────────────────────────────────────────────┘
        │  training/trainer.py  (AdamW, early stop)
        ▼
 Hit@K / NDCG@K / MRR   (training/metrics.py)
```

## Framing: recommendation as next-item prediction

A user's chronologically-ordered history `[i₁, …, i_t]` is fed to the model,
which predicts the next item `i_{t+1}`. Training uses the **shifted-sequence**
objective: at every position `k`, predict `i_{k+1}` from `[i₁, …, i_k]`.

## The two backbones (deliberately interchangeable)

Everything except the sequence mixer is shared between the two models — the same
embedding table, the same tied scoring head, the same objective, the same split
and metrics. Only `encode()` differs, so any performance gap is attributable to
the backbone, not the surrounding machinery.

| | **S4Rec** | **SASRec (baseline)** |
|---|---|---|
| Mixer | stacked S4 blocks | causal multi-head self-attention |
| Complexity | `O(L log L)` | `O(L²)` |
| Causality | intrinsic (one-sided conv) | enforced by a triangular mask |
| Positions | implicit (SSM is order-aware) | learned positional embeddings |
| Long-range memory | HiPPO state matrix | attention span |

- [`models/s4rec.py`](../src/recommender/models/s4rec.py)
- [`models/sasrec.py`](../src/recommender/models/sasrec.py) — faithful Kang &
  McAuley (2018) re-implementation.

## Why timestamp leave-one-out

A random train/test split can place a user's *future* interaction into the
history used to predict a *past* one — future leakage that inflates metrics. We
sort each user's events by time and hold out the **last** (test) and
**second-to-last** (validation) interactions, conditioning only on the strict
past. See [`data/splits.py`](../src/recommender/data/splits.py) and the
leakage tests in [`tests/test_data_splits.py`](../tests/test_data_splits.py).

## Evaluation protocol

For each user the held-out target is ranked against `N` sampled negatives the
user has not interacted with (default `N = 100`). We report Hit@K (recall of the
single relevant item), NDCG@K, and MRR, with **pessimistic** tie-breaking so
equal scores never inflate the metric.

## Training loop

[`training/trainer.py`](../src/recommender/training/trainer.py): AdamW
(`β = 0.9, 0.98`), optional linear warmup, gradient clipping, masked
cross-entropy (default) or SASRec-style BCE, early stopping on validation
NDCG@10, and best-checkpoint restore. The identical loop trains both models.
