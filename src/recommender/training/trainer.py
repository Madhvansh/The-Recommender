"""Training / evaluation loop for sequential recommenders.

Handles device placement, the optimizer + (optional) warmup, the masked
next-item objective, periodic validation with ranked metrics, early stopping on
validation NDCG@10, and best-checkpoint restoration.  The same loop trains both
S4Rec and SASRec so the comparison is apples-to-apples.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..utils.config import Config
from ..utils.logging import get_logger
from .losses import bpr_bce_loss, masked_cross_entropy
from .metrics import compute_metrics, ranks_of_target

logger = get_logger(__name__)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@dataclass
class TrainState:
    best_metric: float = float("-inf")
    best_epoch: int = -1
    best_state: dict | None = None
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Score every eval example and aggregate ranked metrics.

    The eval candidate set has the true item at column 0; we gather the model's
    scores for exactly those candidates and rank within them.
    """
    model.eval()
    all_ranks = []
    for item_seq, candidates in loader:
        item_seq = item_seq.to(device)
        candidates = candidates.to(device)
        hidden = model.forward(item_seq)                  # (B, L, d)
        lengths = (item_seq != model.pad_idx).sum(dim=1)
        last = (lengths - 1).clamp(min=0)
        last_hidden = hidden[torch.arange(hidden.size(0)), last]  # (B, d)
        cand_emb = model.item_emb(candidates)             # (B, C, d)
        scores = torch.einsum("bd,bcd->bc", last_hidden, cand_emb)
        all_ranks.append(ranks_of_target(scores, target_col=0).cpu())
    ranks = torch.cat(all_ranks)
    return compute_metrics(ranks, ks)


class Trainer:
    """Drives optimization and early stopping for one model."""

    def __init__(
        self,
        model,
        cfg: Config,
        loss_type: str = "ce",
        ckpt_dir: str | Path = "checkpoints",
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.loss_type = loss_type
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.98),
        )
        self.state = TrainState()

    def _lr_scale(self, step: int) -> float:
        w = self.cfg.train.warmup_steps
        if w <= 0:
            return 1.0
        return min(1.0, step / max(1, w))

    def _train_step(self, batch) -> float:
        item_seq, pos, neg, mask = (t.to(self.device) for t in batch)
        if self.loss_type == "ce":
            logits = self.model.sequence_logits(item_seq)
            loss = masked_cross_entropy(logits, pos, mask)
        elif self.loss_type == "bce":
            loss = bpr_bce_loss(self.model, item_seq, pos, neg, mask)
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"unknown loss_type {self.loss_type!r}")
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()
        return loss.item()

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> TrainState:
        ks = self.cfg.train.ks
        global_step = 0
        base_lrs = [g["lr"] for g in self.opt.param_groups]
        for epoch in range(1, self.cfg.train.epochs + 1):
            self.model.train()
            t0 = time.time()
            running = 0.0
            for batch in train_loader:
                global_step += 1
                scale = self._lr_scale(global_step)
                for g, base in zip(self.opt.param_groups, base_lrs):
                    g["lr"] = base * scale
                running += self._train_step(batch)
            train_loss = running / max(1, len(train_loader))

            if epoch % self.cfg.train.eval_every == 0:
                metrics = evaluate(self.model, valid_loader, self.device, ks)
                metrics["epoch"] = epoch
                metrics["train_loss"] = train_loss
                self.state.history.append(metrics)
                monitor = metrics.get("ndcg@10", metrics.get(f"ndcg@{ks[-1]}"))
                logger.info(
                    "epoch %3d | loss %.4f | val NDCG@10 %.4f | Hit@10 %.4f | %.1fs",
                    epoch, train_loss, monitor, metrics.get("hit@10", float("nan")),
                    time.time() - t0,
                )
                if monitor > self.state.best_metric:
                    self.state.best_metric = monitor
                    self.state.best_epoch = epoch
                    self.state.best_state = copy.deepcopy(self.model.state_dict())
                    torch.save(self.state.best_state, self.ckpt_dir / "best.pt")
                elif epoch - self.state.best_epoch >= self.cfg.train.patience:
                    logger.info("early stopping at epoch %d (best %d)",
                                epoch, self.state.best_epoch)
                    break

        if self.state.best_state is not None:
            self.model.load_state_dict(self.state.best_state)
        return self.state

    def test(self, test_loader: DataLoader) -> dict:
        return evaluate(self.model, test_loader, self.device, self.cfg.train.ks)
