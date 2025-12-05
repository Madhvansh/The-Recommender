# Deployment

The recommender ships with a small FastAPI service
([`serve.py`](../src/recommender/serve.py)) that loads a trained checkpoint and
returns top-K next-item recommendations.

## 1. Train and checkpoint a model

```bash
python scripts/download_data.py --category beauty
recommender train --config configs/s4rec_amazon_beauty.yaml
# best checkpoint -> checkpoints/best.pt
```

Note the catalog size printed at startup (`items=…`); the server needs it via
`RECO_NUM_ITEMS` so the embedding table matches the checkpoint.

## 2. Run the API locally

```bash
pip install fastapi uvicorn
export RECO_CONFIG=configs/s4rec_amazon_beauty.yaml
export RECO_CKPT=checkpoints/best.pt
export RECO_NUM_ITEMS=12101          # the catalog size from training
uvicorn recommender.serve:app --host 0.0.0.0 --port 8000
```

Request recommendations:

```bash
curl -s localhost:8000/recommend \
  -H 'content-type: application/json' \
  -d '{"history": [12, 45, 9], "k": 10}'
# -> {"items": [...], "scores": [...]}
```

Health check: `curl localhost:8000/health`.

## 3. Docker

```bash
docker build -t the-recommender .
docker run --rm -p 8000:8000 \
  -e RECO_NUM_ITEMS=12101 \
  -v "$PWD/checkpoints:/app/checkpoints" \
  the-recommender
```

The image is CPU-only and mounts your trained checkpoint at
`/app/checkpoints/best.pt`.

## 4. Production notes

- **Item id mapping.** The server speaks in the *internal* contiguous item ids
  produced during preprocessing. Persist the `asin → id` map from
  `data/amazon.py` alongside the checkpoint and translate at the API boundary.
- **Throughput.** Inference uses the last-position read-out (`score_last`); for
  streaming/autoregressive serving switch to the `O(N)` recurrent
  `S4Layer.step` path and cache state per user.
- **Scaling.** The service is stateless — run multiple replicas behind a load
  balancer. Pin `torch` threads (`OMP_NUM_THREADS`) per replica.
- **GPU.** Set `RECO_DEVICE=cuda` and use a CUDA base image to serve on GPU.
