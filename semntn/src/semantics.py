"""Semantic distortion (sWER) utilities.

Implements a sigmoid on linear SINR to approximate semantic WER.
Parameters `a` and `b` are optionally loaded from `configs/link_semantic.yaml`.
If the config does not exist or keys are missing, defaults are used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

_CACHE: Tuple[float, float] | None = None


def _load_params() -> Tuple[float, float]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    a_default, b_default = -0.35, 0.0
    cfg_path = Path("configs") / "link_semantic.yaml"
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            a = float(data.get("a", a_default))
            b = float(data.get("b", b_default))
            _CACHE = (a, b)
            return a, b
        except Exception:
            pass
    _CACHE = (a_default, b_default)
    return _CACHE


def get_sem_params() -> Tuple[float, float]:
    """Expose (a, b) parameters for external callers."""
    return _load_params()


def swer_from_sinr(sinr_db: float | np.ndarray, a: float | None = None, b: float | None = None) -> np.ndarray:
    """Compute semantic WER from SINR in dB.

    y = sigmoid(a * sinr + b), where sinr = 10**(sinr_db/10).
    Returns clipped values in [0,1].
    If `a` or `b` is None, they are loaded from configs/link_semantic.yaml.
    """
    if a is None or b is None:
        a_cfg, b_cfg = _load_params()
        a = a if a is not None else a_cfg
        b = b if b is not None else b_cfg
    sinr = 10.0 ** (np.asarray(sinr_db, dtype=float) / 10.0)
    z = float(a) * sinr + float(b)
    y = 1.0 / (1.0 + np.exp(-z))
    return np.clip(y, 0.0, 1.0)


# === Guard helpers on linear SINR ===
def swer_from_sinr_linear(sinr_lin, a, b):
    """Semantic WER from linear SINR using logistic.

    z = a * sinr_lin + b; y = sigmoid(z); clipped to [0,1].
    """
    z = float(a) * np.asarray(sinr_lin, dtype=float) + float(b)
    y = 1.0 / (1.0 + np.exp(-z))
    return np.clip(y, 0.0, 1.0)


def sinr_lin_for_swer_target(swer_target, a, b):
    """Invert logistic to get required linear SINR for target sWER.

    swer = 1/(1+exp(-(a*sinr_lin + b))) → sinr_lin = (logit(swer)-b)/a
    """
    swer = np.clip(float(swer_target), 1e-6, 1.0 - 1e-6)
    logit = np.log(swer / (1.0 - swer))
    return (logit - float(b)) / float(a)