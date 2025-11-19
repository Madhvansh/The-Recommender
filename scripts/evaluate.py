#!/usr/bin/env python3
"""Thin wrapper to evaluate a checkpoint (delegates to the package CLI).

    python scripts/evaluate.py --config configs/s4rec_amazon_beauty.yaml \
        --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
