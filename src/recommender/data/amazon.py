"""Amazon Reviews data loading and preprocessing.

Loads the standard Amazon Reviews *k-core* interaction files (e.g. the Beauty,
Toys, or Sports categories from Julian McAuley's collection), filters to a
``k``-core, remaps user/item ids to contiguous integers, and produces the
``user -> [(item, timestamp)]`` mapping consumed by :mod:`.splits`.

The raw files are not committed; fetch them with ``scripts/download_data.py``.
For unit tests and CI a :func:`synthetic_interactions` generator produces data
with the same schema so the full pipeline can run without any download.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    """A fully-preprocessed sequential-recommendation dataset."""

    interactions: dict[int, list[tuple[int, float]]]  # user -> [(item, ts)]
    num_users: int
    num_items: int
    name: str = "amazon"

    @property
    def num_interactions(self) -> int:
        return sum(len(v) for v in self.interactions.values())


def _iter_raw_rows(path: Path):
    """Yield (user, item, timestamp) from a raw Amazon file.

    Supports both the legacy gzipped-JSON-lines format (``reviewerID`` /
    ``asin`` / ``unixReviewTime``) and the newer CSV ratings dumps
    (``item,user,rating,timestamp``).
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        first = fh.readline()
        fh.seek(0)
        is_json = first.strip().startswith("{")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if is_json:
                row = json.loads(line)
                user = row.get("reviewerID") or row.get("user_id")
                item = row.get("asin") or row.get("parent_asin") or row.get("item_id")
                ts = row.get("unixReviewTime") or row.get("timestamp")
                if user is None or item is None or ts is None:
                    continue
                yield user, item, float(ts)
            else:
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                item, user, _rating, ts = parts[:4]
                yield user, item, float(ts)


def _build_dataset(
    rows,
    k_core: int,
    name: str,
) -> Dataset:
    """Filter to a k-core and remap ids to contiguous integers (items 1-indexed)."""
    user_events: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for user, item, ts in rows:
        user_events[user].append((item, ts))

    # Iterative k-core filtering on the user-item bipartite graph.
    while True:
        item_counts: dict[str, int] = defaultdict(int)
        for events in user_events.values():
            for item, _ in events:
                item_counts[item] += 1
        changed = False
        for user in list(user_events):
            kept = [(i, t) for i, t in user_events[user] if item_counts[i] >= k_core]
            if len(kept) != len(user_events[user]):
                changed = True
            if len(kept) < k_core:
                del user_events[user]
                changed = True
            else:
                user_events[user] = kept
        if not changed:
            break

    # Remap to contiguous ids; items start at 1 (0 is the padding token).
    user_ids: dict[str, int] = {}
    item_ids: dict[str, int] = {}
    interactions: dict[int, list[tuple[int, float]]] = {}
    for user, events in user_events.items():
        uid = user_ids.setdefault(user, len(user_ids))
        mapped = []
        for item, ts in events:
            iid = item_ids.setdefault(item, len(item_ids) + 1)
            mapped.append((iid, ts))
        interactions[uid] = mapped

    return Dataset(
        interactions=interactions,
        num_users=len(user_ids),
        num_items=len(item_ids),
        name=name,
    )


def load_amazon(
    path: str | Path,
    k_core: int = 5,
    name: str | None = None,
) -> Dataset:
    """Load and preprocess a raw Amazon Reviews file into a :class:`Dataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with scripts/download_data.py first."
        )
    return _build_dataset(_iter_raw_rows(path), k_core, name or path.stem)


def synthetic_interactions(
    num_users: int = 300,
    num_items: int = 120,
    avg_len: int = 18,
    seed: int = 0,
    name: str = "synthetic",
) -> Dataset:
    """Generate synthetic sequential data with mild sequential structure.

    Each user walks a noisy "preference cluster" so that next-item prediction is
    learnable (better than random), which lets the end-to-end pipeline and tests
    run without downloading anything.
    """
    rng = np.random.default_rng(seed)
    n_clusters = max(2, num_items // 20)
    cluster_of = rng.integers(0, n_clusters, size=num_items + 1)
    by_cluster = {c: np.where(cluster_of == c)[0] for c in range(n_clusters)}
    by_cluster = {c: v[v >= 1] for c, v in by_cluster.items() if (v >= 1).any()}
    clusters = list(by_cluster)

    interactions: dict[int, list[tuple[int, float]]] = {}
    for u in range(num_users):
        length = max(5, int(rng.poisson(avg_len)))
        cluster = clusters[rng.integers(0, len(clusters))]
        events = []
        ts = float(rng.integers(1_000_000, 2_000_000))
        for _ in range(length):
            if rng.random() < 0.15:  # occasionally drift to another cluster
                cluster = clusters[rng.integers(0, len(clusters))]
            pool = by_cluster[cluster]
            item = int(pool[rng.integers(0, len(pool))])
            ts += float(rng.integers(1, 10_000))
            events.append((item, ts))
        interactions[u] = events

    return Dataset(
        interactions=interactions,
        num_users=num_users,
        num_items=num_items,
        name=name,
    )
