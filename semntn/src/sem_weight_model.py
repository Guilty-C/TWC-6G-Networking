"""Light-weight semantic weight estimator.

The original project relied on ``numpy`` for a small logistic regression style
model.  To keep the repository runnable without external wheels we reimplement
that model using pure Python operations while maintaining the same interface.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


class SemWeightModel:
    """Estimate semantic importance weights from simple hand-crafted features."""

    def __init__(self, w_min: float = 1.0, w_max: float = 3.0) -> None:
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        # Coefficients roughly matching the original behaviour.
        self._weights = (0.8, 0.6, 0.4, 0.3, 0.2)
        self._bias = -0.5

    @staticmethod
    def _squash(z: float) -> float:
        if z < -60.0:
            return 0.0
        if z > 60.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def infer_w_sem(self, feat_vec: Iterable[float]) -> float:
        feats: Sequence[float] = tuple(float(x) for x in feat_vec)
        total = self._bias
        for w, f in zip(self._weights, feats):
            total += w * f
        score = self._squash(total)
        return self.w_min + (self.w_max - self.w_min) * score


__all__ = ["SemWeightModel"]
