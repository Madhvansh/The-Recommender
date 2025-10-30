"""Train / validation / test splitting for sequential recommendation.

We use a **timestamp leave-one-out** split, which is the protocol that prevents
the *future-leakage* present in naive random splits:

For each user, interactions are sorted by timestamp.  The chronologically **last**
item becomes the test target, the **second-to-last** becomes the validation
target, and everything before is training history.  Crucially, the model only
ever conditions on items that occurred *before* the target, so no future
information leaks into a prediction — unlike a random split, where a future
purchase can end up in the training history used to predict a past one.

This module operates on a mapping ``user -> [(item, timestamp), ...]`` and emits
per-user history/target tuples for each split.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserSplit:
    """Per-user histories and targets for the three splits."""

    train_seq: list[int]   # history used for training (all but last two)
    valid_seq: list[int]   # history to condition on for the validation target
    valid_target: int      # second-to-last item
    test_seq: list[int]    # history to condition on for the test target
    test_target: int       # last item


def leave_one_out(
    interactions: dict[int, list[tuple[int, float]]],
    min_seq_len: int = 3,
) -> dict[int, UserSplit]:
    """Build timestamp leave-one-out splits.

    Parameters
    ----------
    interactions : mapping user_id -> list of (item_id, timestamp).
    min_seq_len  : minimum number of interactions required to keep a user
        (need >= 3 so that train/valid/test targets are all distinct).

    Returns
    -------
    mapping user_id -> :class:`UserSplit`.
    """
    splits: dict[int, UserSplit] = {}
    for user, events in interactions.items():
        if len(events) < min_seq_len:
            continue
        # Stable sort by timestamp; ties keep their original (insertion) order.
        ordered = [item for item, _ in sorted(events, key=lambda e: e[1])]
        test_target = ordered[-1]
        valid_target = ordered[-2]
        splits[user] = UserSplit(
            train_seq=ordered[:-2],
            valid_seq=ordered[:-2],
            valid_target=valid_target,
            test_seq=ordered[:-1],
            test_target=test_target,
        )
    return splits


def train_windows(
    train_seq: list[int],
    max_len: int,
) -> tuple[list[int], list[int]]:
    """Build the (input, shifted-target) pair for next-item training.

    Given a training history ``[i_1, ..., i_n]`` the model is trained to predict
    ``i_{k+1}`` from ``[i_1, ..., i_k]`` at every position, i.e. the target is the
    input shifted left by one.  The most recent ``max_len`` items are kept.

    Returns
    -------
    (inputs, targets) — equal-length id lists (un-padded).
    """
    seq = train_seq[-(max_len + 1):]
    if len(seq) < 2:
        return [], []
    inputs = seq[:-1]
    targets = seq[1:]
    return inputs, targets
