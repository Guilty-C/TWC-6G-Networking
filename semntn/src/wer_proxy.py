"""Packet and semantic error proxy utilities.

These helpers deliberately avoid external dependencies so they can run in
restricted environments where ``numpy`` is unavailable.  The functions mimic the
behaviour of their previous implementations using only the standard library.
"""
from __future__ import annotations

import math


def _clip(value: float, low: float, high: float) -> float:
    """Return *value* clipped to the ``[low, high]`` interval."""

    if value < low:
        return low
    if value > high:
        return high
    return value


def per_from_snr_db(snr_db: float, steep: float = 1.2, mid_db: float = 1.0) -> float:
    """Approximate the packet error rate from an effective SNR value.

    The mapping follows a logistic curve so that higher SNRs lead to lower PERs.
    It reproduces the shape used in the original project while keeping the
    result bounded inside ``[0, 1]``.  ``steep`` controls the slope around the
    transition point and ``mid_db`` shifts the midpoint of the curve.
    """

    x = float(snr_db) - float(mid_db)
    # Guard against overflow in ``exp`` for large negative or positive inputs.
    if x > 60:
        return 0.0
    if x < -60:
        return 1.0
    per = 1.0 / (1.0 + math.exp(steep * x))
    return _clip(per, 0.0, 1.0)


def wer_from_per(per: float, sem_weight: float, base_alpha: float = 1.2) -> float:
    """Map PER to semantic WER under a semantic weight estimate."""

    sem_weight = max(float(sem_weight), 1e-6)
    wer = base_alpha * float(per) / sem_weight
    return _clip(wer, 0.0, 1.0)


__all__ = ["per_from_snr_db", "wer_from_per"]
