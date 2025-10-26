"""Models: the S4 backbone, the S4 recommender, and the SASRec baseline."""

from .base import SequentialRecommender
from .s4rec import S4Rec
from .sasrec import SASRec

MODELS = {
    "s4rec": S4Rec,
    "sasrec": SASRec,
}


def build_model(name: str, **kwargs) -> SequentialRecommender:
    """Factory: instantiate a recommender by config name."""
    try:
        cls = MODELS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from exc
    # Drop kwargs a given backbone does not accept.
    import inspect

    sig = inspect.signature(cls.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**accepted)


__all__ = ["SequentialRecommender", "S4Rec", "SASRec", "MODELS", "build_model"]
