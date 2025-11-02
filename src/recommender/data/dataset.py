"""PyTorch datasets and collation for sequential recommendation.

Sequences are **left-padded** to a fixed ``max_len`` so the most recent
interaction always sits at the last position — this matters because the scoring
head reads off the final position, and S4's causal convolution treats position 0
as the oldest event.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import UserSplit, train_windows


def left_pad(ids: list[int], max_len: int, pad_idx: int = 0) -> list[int]:
    """Left-pad / left-truncate ``ids`` to exactly ``max_len``."""
    ids = ids[-max_len:]
    return [pad_idx] * (max_len - len(ids)) + ids


class TrainSequenceDataset(Dataset):
    """Next-item training pairs with negative sampling for the BCE objective.

    Each example yields a padded input sequence, the shifted positive targets,
    one sampled negative per position, and a mask marking real (non-pad)
    positions.  The negatives let us train with the SASRec-style binary objective;
    the full-softmax objective ignores them.
    """

    def __init__(
        self,
        splits: dict[int, UserSplit],
        num_items: int,
        max_len: int,
        pad_idx: int = 0,
        seed: int = 0,
    ) -> None:
        self.num_items = num_items
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.rng = np.random.default_rng(seed)

        self.examples: list[tuple[list[int], set[int]]] = []
        for split in splits.values():
            inputs, targets = train_windows(split.train_seq, max_len)
            if not inputs:
                continue
            self.examples.append((split.train_seq, set(split.train_seq)))

    def __len__(self) -> int:
        return len(self.examples)

    def _sample_negative(self, seen: set[int]) -> int:
        while True:
            neg = int(self.rng.integers(1, self.num_items + 1))
            if neg not in seen:
                return neg

    def __getitem__(self, idx: int):
        seq, seen = self.examples[idx]
        inputs, targets = train_windows(seq, self.max_len)

        inp = left_pad(inputs, self.max_len, self.pad_idx)
        pos = left_pad(targets, self.max_len, self.pad_idx)
        mask = [1 if t != self.pad_idx else 0 for t in pos]
        neg = [
            self._sample_negative(seen) if m else self.pad_idx for m in mask
        ]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )


class EvalSequenceDataset(Dataset):
    """Evaluation examples: a padded history, the held-out target, and a fixed
    set of sampled negatives for ranked metrics.

    The target is ranked against ``num_negatives`` items the user has not
    interacted with (the standard sampled-metric protocol).  Using a fixed RNG
    seed per user makes the candidate set deterministic across runs.
    """

    def __init__(
        self,
        splits: dict[int, UserSplit],
        num_items: int,
        max_len: int,
        which: str = "test",
        num_negatives: int = 100,
        pad_idx: int = 0,
        seed: int = 0,
    ) -> None:
        assert which in {"valid", "test"}
        self.num_items = num_items
        self.max_len = max_len
        self.num_negatives = num_negatives
        self.pad_idx = pad_idx

        self.items: list[tuple[list[int], int, set[int]]] = []
        for user in sorted(splits):
            split = splits[user]
            if which == "valid":
                seq, target = split.valid_seq, split.valid_target
            else:
                seq, target = split.test_seq, split.test_target
            if not seq:
                continue
            seen = set(split.test_seq) | {split.test_target, split.valid_target}
            self.items.append((seq, target, seen))
        self._seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        seq, target, seen = self.items[idx]
        rng = np.random.default_rng(self._seed + idx)
        negatives: list[int] = []
        while len(negatives) < self.num_negatives:
            cand = int(rng.integers(1, self.num_items + 1))
            if cand not in seen:
                negatives.append(cand)
        candidates = [target] + negatives  # index 0 is always the true target
        inp = left_pad(seq, self.max_len, self.pad_idx)
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(candidates, dtype=torch.long),
        )
