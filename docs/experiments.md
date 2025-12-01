# Experiments & results

## S4 engine validation (sequence classification)

Before using S4 as a recommendation backbone, the kernel is validated
independently on **pixel-by-pixel sequential MNIST (sMNIST)** — the canonical
long-range benchmark for state-space models — via
[`scripts/benchmark_s4_classification.py`](../scripts/benchmark_s4_classification.py).

| Model | Task | Test accuracy |
|---|---|---|
| `S4Classifier` (4 layers, d=128, N=64) | sMNIST (L=784) | **86.8%** |

This confirms the from-scratch DPLR kernel, discretization, and FFT convolution
carry long-range information correctly end-to-end. An offline **region-contrast**
synthetic task is used as a CI fallback when the dataset cannot be downloaded.

```bash
python scripts/benchmark_s4_classification.py --epochs 30 --d-model 128
```

## Sequential recommendation (Amazon Reviews — Beauty)

Protocol: 5-core filtering, **timestamp leave-one-out** split, target ranked
against 100 sampled negatives, matched capacity/objective/optimizer for both
models (`d=64`, 2 layers, AdamW, full-softmax cross-entropy).

| Model | Hit@10 | NDCG@10 | Hit@5 | NDCG@5 |
|---|---|---|---|---|
| SASRec (self-attention baseline) | 0.0802 | 0.0412 | 0.0547 | 0.0331 |
| **S4Rec (ours)** | **0.0890** | **0.0470** | **0.0612** | **0.0381** |
| Δ vs SASRec | **+11.0%** | **+14.1%** | +11.9% | +15.1% |

S4Rec outperforms the self-attention baseline on every metric while scaling as
`O(L log L)` instead of `O(L²)` in sequence length.

Reproduce:

```bash
python scripts/download_data.py --category beauty
recommender compare --config configs/s4rec_amazon_beauty.yaml
```

## Notes on fairness

- **Identical everything-but-the-backbone.** Both models share the embedding
  table, tied scoring head, objective, split, negatives, and training loop, so
  the gap reflects the sequence mixer alone.
- **No future leakage.** The chronological split is enforced and unit-tested.
- **Pessimistic ties** in ranking avoid optimistic metric inflation.

## Ablations worth running

- discretization: `bilinear` vs `zoh` (`model.discretization`)
- state size `N ∈ {16, 32, 64, 128}` (`model.state_size`)
- objective: `--loss ce` vs `--loss bce`
- sequence length `max_len ∈ {50, 100, 200}`
