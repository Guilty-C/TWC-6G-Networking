"""Utility helpers for deterministic randomness."""
from __future__ import annotations

import random


def set_all_seeds(seed: int = 2025) -> random.Random:
    """Seed Python's pseudo random generator and return a ``Random`` instance."""

    random.seed(seed)
    return random.Random(seed)


__all__ = ["set_all_seeds"]
