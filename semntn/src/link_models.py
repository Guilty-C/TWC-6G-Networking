"""Lightweight link models for rate and PER using effective SNR.

Pure-Python, numpy-first. Provides:
- rate(P, B, state): logistic rate approximation from effective SNR
- per(P, B, state): packet error rate via logistic in dB space

Effective SNR model:
  snr_eff_db = snr_db + 10*log10(max(P*B, eps)) - c_rate_db(q_bps)

If `state` includes `q_bps`, we apply a rate penalty on the effective SNR
to reflect higher codec rate requiring higher SNR.

Configuration (optional) via state keys:
- 'logistic_rate': {'r_max': float, 'mid_db': float, 'steep': float}
- 'per_model': {'mid_db': float, 'steep': float}
- 'slot_sec': float (used by helpers)

Fallbacks are sensible if not provided.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import numpy as np

# Optional PER abstraction curve cache
_PER_CURVE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _load_per_curve() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load PER vs SNR(dB) curve if available under data/link_abstraction.

    Expected CSV format with header: snr_db,per
    Returns (snr_db_array, per_array) or None if not found.
    """
    global _PER_CURVE
    if _PER_CURVE is not None:
        return _PER_CURVE
    try:
        import os
        from pathlib import Path
        base = Path("semntn/data/link_abstraction/per_curve.csv")
        if not base.exists():
            return None
        import pandas as pd
        df = pd.read_csv(base)
        x = df["snr_db"].to_numpy(dtype=float)
        y = df["per"].to_numpy(dtype=float)
        if x.size == 0 or y.size == 0:
            return None
        # Ensure sorted by snr_db
        idx = np.argsort(x)
        _PER_CURVE = (x[idx], np.clip(y[idx], 0.0, 1.0))
        return _PER_CURVE
    except Exception:
        return None


def _rate_penalty_db(q_bps: float, q_ref: float = 300.0, c_db: float = 3.0) -> float:
    """Higher codec rate => higher required SNR. Smooth penalty in dB.

    penalty ≈ c_db * log2(q/q_ref + 1)
    """
    q = max(float(q_bps), 1.0)
    return float(c_db * math.log2(q / q_ref + 1.0))


def _snr_eff_db(P_w: float, B_units: float, state: Dict[str, float]) -> float:
    snr_db = float(state.get("snr_db", 0.0))
    eps = 1e-9
    # Base effective SNR scaling with power and number of subbands
    gain_db = 10.0 * math.log10(max(P_w * B_units, eps))

    # Optional rate penalty if codec rate provided
    q_bps = float(state.get("q_bps", 0.0))
    penalty_db = _rate_penalty_db(q_bps) if q_bps > 0.0 else 0.0

    return float(snr_db + gain_db - penalty_db)


def per(P: float, B: float, state: Dict[str, float]) -> float:
    """Packet error rate via EESM/MIESM abstraction if curve provided,
    otherwise fall back to logistic in effective SNR (dB).

    Curve lookup: semntn/data/link_abstraction/per_curve.csv
    """
    snr_eff = _snr_eff_db(P, B, state)
    curve = _load_per_curve()
    if curve is not None:
        snr_arr, per_arr = curve
        # interpolate, extrapolate flat at ends
        p = float(np.interp(snr_eff, snr_arr, per_arr))
        return float(np.clip(p, 0.0, 1.0))
    # fallback logistic
    per_cfg = state.get("per_model", {})
    steep = float(per_cfg.get("steep", 1.2))
    mid_db = float(per_cfg.get("mid_db", 1.0))
    x = (snr_eff - mid_db)
    p = 1.0 / (1.0 + math.exp(steep * x))
    return float(np.clip(p, 0.0, 1.0))


def rate(P: float, B: float, state: Dict[str, float]) -> float:
    """Logistic rate approximation from effective SNR.

    r(P,B) ≈ r_max * logistic(snr_eff_db - mid_db), scaled by B.
    Returns bits-per-second.
    """
    snr_eff = _snr_eff_db(P, B, state)
    cfg = state.get("logistic_rate", {})
    r_max = float(cfg.get("r_max", 2400.0))  # codec-like scale
    steep = float(cfg.get("steep", 0.8))
    mid_db = float(cfg.get("mid_db", 3.0))
    # logistic in [0,1]
    z = 1.0 / (1.0 + math.exp(-steep * (snr_eff - mid_db)))
    # scale by B to reflect parallel subbands
    return float(r_max * z * max(B, 1.0))


def throughput_bits(P: float, B: float, state: Dict[str, float]) -> Tuple[float, float, float]:
    """Convenience: return (q_bps, per, S_bits_per_slot).

    Uses `rate()` as offered bitrate, PER() as loss, and slot_sec to
    produce serviced bits per slot.
    """
    q_bps = rate(P, B, state)
    p = per(P, B, state)
    slot_sec = float(state.get("slot_sec", 0.02))
    S_bits = float(q_bps * (1.0 - p) * slot_sec)
    return float(q_bps), float(p), float(S_bits)