"""QoE utilities.

Provides:
- pesq_to_mos(pesq): clamp/scale PESQ to MOS range
- mos_surrogate(state, P, B): QoE proxy composing rate & PER

This module is lightweight and numpy-first. No heavy dependencies.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .link_models import rate, per


def pesq_to_mos(pesq: float) -> float:
    """Map PESQ (≈[1, 4.5]) to MOS [1, 4.5].

    For simplicity we clamp to [1, 4.5] and return as-is.
    """
    return float(np.clip(float(pesq), 1.0, 4.5))


def _normalize(x: float, lo: float, hi: float) -> float:
    hi = float(hi); lo = float(lo)
    if hi <= lo + 1e-12:
        return 0.0
    return float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))


def mos_surrogate(state: Dict[str, float], P: float, B: float) -> float:
    """QoE proxy combining bitrate and PER.

    The idea:
    - Convert effective bitrate to MOS-like utility via logistic
    - Apply loss via (1 - PER)^gamma
    - Add small semantic bonus if state provides 'sem_weight' in [0,1]
    Returns MOS in [1, 4.5].
    """
    q_bps = rate(P, B, state)
    p = per(P, B, state)
    # bitrate utility normalized by a reference scale
    q_ref = float(state.get("q_ref_bps", 2400.0))
    u_rate = 1.0 / (1.0 + math.exp(-1.2 * (q_bps / max(q_ref, 1e-3) - 0.5)))
    # reliability utility
    gamma = float(state.get("mos_gamma", 1.2))
    u_rel = (1.0 - p) ** gamma
    # semantic bonus
    sem_w = float(state.get("sem_weight", 0.0))
    bonus = 0.15 * sem_w  # small uplift
    u = float(np.clip(u_rate * u_rel + bonus, 0.0, 1.0))
    # map to MOS range [1, 4.5]
    mos_lo, mos_hi = 1.0, 4.5
    mos = mos_lo + u * (mos_hi - mos_lo)
    return float(np.clip(mos, mos_lo, mos_hi))