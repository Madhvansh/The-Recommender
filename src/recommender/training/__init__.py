"""Training: objectives, ranking metrics, and the train/eval loop."""

from .losses import bpr_bce_loss, masked_cross_entropy
from .metrics import compute_metrics, hit_at_k, ndcg_at_k, ranks_of_target
from .trainer import Trainer, evaluate, resolve_device

__all__ = [
    "masked_cross_entropy",
    "bpr_bce_loss",
    "compute_metrics",
    "hit_at_k",
    "ndcg_at_k",
    "ranks_of_target",
    "Trainer",
    "evaluate",
    "resolve_device",
]
