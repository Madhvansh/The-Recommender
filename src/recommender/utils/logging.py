"""Lightweight logging utilities.

Uses ``rich`` for pretty console output when available, falling back to the
standard library otherwise so the package has no hard dependency at import time.
"""

from __future__ import annotations

import logging

try:  # pragma: no cover - optional dependency
    from rich.logging import RichHandler

    _HANDLER: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
    _FMT = "%(message)s"
except Exception:  # pragma: no cover
    _HANDLER = logging.StreamHandler()
    _FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "recommender", level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _HANDLER
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
