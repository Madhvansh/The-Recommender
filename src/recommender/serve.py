"""Minimal FastAPI service for serving recommendations.

Loads a trained checkpoint once at startup and exposes a ``/recommend`` endpoint
that returns the top-K next items for a user's interaction history.

Run
---
    export RECO_CONFIG=configs/s4rec_amazon_beauty.yaml
    export RECO_CKPT=checkpoints/best.pt
    export RECO_NUM_ITEMS=12101
    uvicorn recommender.serve:app --host 0.0.0.0 --port 8000

Then:
    curl -s localhost:8000/recommend -H 'content-type: application/json' \
        -d '{"history": [12, 45, 9], "k": 10}'

FastAPI / uvicorn are optional extras (``pip install fastapi uvicorn``); the rest
of the library does not depend on them.
"""

from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "serving requires fastapi + uvicorn: pip install fastapi uvicorn"
    ) from exc

from .inference import Recommender


class RecommendRequest(BaseModel):
    history: list[int]
    k: int = 10
    exclude_seen: bool = True


class RecommendResponse(BaseModel):
    items: list[int]
    scores: list[float]


def _load_recommender() -> Recommender:
    config = os.environ.get("RECO_CONFIG", "configs/s4rec_amazon_beauty.yaml")
    ckpt = os.environ.get("RECO_CKPT", "checkpoints/best.pt")
    num_items = int(os.environ["RECO_NUM_ITEMS"])  # required: catalog size
    device = os.environ.get("RECO_DEVICE", "cpu")
    return Recommender.from_checkpoint(config, ckpt, num_items, device=device)


app = FastAPI(title="The Recommender", version="0.4.0")
_recommender: Recommender | None = None


@app.on_event("startup")
def _startup() -> None:
    global _recommender
    _recommender = _load_recommender()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _recommender is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if _recommender is None:  # pragma: no cover - startup guard
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.history:
        raise HTTPException(status_code=400, detail="history must be non-empty")
    rec = _recommender.recommend(req.history, k=req.k, exclude_seen=req.exclude_seen)
    return RecommendResponse(items=rec.items, scores=rec.scores)
