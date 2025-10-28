"""Semantic weight utility with EMA smoothing.

get_sem_weight(state)->float in [0,1]

State may include optional instantaneous signals to guide weight:
- 'snr_db': higher SNR => lower semantic penalty (slightly)
- 'keyword_flag': 1 if important keyword; boosts weight
- 'prosody_energy': proxy for voice energy; increases weight

The raw score is squashed to [0,1], then smoothed via EMA with alpha.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class _EMASmoother:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.initialized = False
        self.y = 0.0

    def update(self, x: float) -> float:
        if not self.initialized:
            self.y = float(x)
            self.initialized = True
        else:
            self.y = (1.0 - self.alpha) * self.y + self.alpha * float(x)
        return float(self.y)


_SMOOTHER = _EMASmoother(alpha=0.08)


def _raw_sem_signal(state: Dict[str, float]) -> float:
    snr = float(state.get("snr_db", 0.0))
    kw = float(state.get("keyword_flag", 0.0))
    energy = float(state.get("prosody_energy", 0.0))
    # Normalize individual components
    snr_term = 1.0 / (1.0 + np.exp(0.6 * (snr - 4.0)))  # higher snr -> lower weight
    kw_term = np.clip(kw, 0.0, 1.0)
    energy_term = 1.0 / (1.0 + np.exp(-0.4 * (energy - 0.5)))
    # Combine conservatively
    s = 0.5 * kw_term + 0.35 * energy_term + 0.15 * snr_term
    return float(np.clip(s, 0.0, 1.0))


def get_sem_weight(state: Dict[str, float]) -> float:
    """Return semantic weight in [0,1] with EMA smoothing."""
    raw = _raw_sem_signal(state)
    return float(np.clip(_SMOOTHER.update(raw), 0.0, 1.0))