from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def precompute_mu_true(env) -> Dict[int, float]:
    r"""Pre-compute :math:`\mu_{true}` for each action using the env's mapping."""

    mu = {}
    for action in env.enumerate_action_space():
        mu[action] = float(env.expected_reward_unit_interval(action))
    return mu


def compute_pseudo_regret(actions: Iterable[int], mu_true: Dict[int, float]) -> np.ndarray:
    r"""Return the cumulative pseudo-regret curve for ``actions``.

    Parameters
    ----------
    actions:
        Sequence of chosen actions ``a_t``.
    mu_true:
        Mapping from action id to :math:`\mu_{true}(a)` on the shared ``[0, 1]``
        scale.
    """

    actions_list: List[int] = list(actions)
    if not actions_list:
        return np.zeros(0, dtype=float)

    opt = float(max(mu_true.values()))
    regret = np.zeros(len(actions_list), dtype=float)
    cumulative = 0.0
    for idx, action in enumerate(actions_list):
        mu_a = mu_true[int(action)]
        cumulative += opt - mu_a
        regret[idx] = cumulative
    return regret


def correlation_vs_sqrt_t(regret: np.ndarray) -> float:
    r"""Pearson correlation between ``R(T)`` and ``sqrt(T)``."""

    if regret.size == 0:
        return float("nan")
    start = max(0, int(0.1 * regret.size))
    tail_regret = regret[start:]
    sqrt_t = np.sqrt(np.arange(start + 1, regret.size + 1, dtype=float))
    return float(np.corrcoef(tail_regret, sqrt_t)[0, 1])


def slope_vs_sqrt_t(regret: np.ndarray) -> float:
    r"""Slope of the least-squares fit ``R(T) ≈ slope * sqrt(T) + b``."""

    if regret.size == 0:
        return 0.0
    start = max(0, int(0.1 * regret.size))
    tail_regret = regret[start:]
    sqrt_t = np.sqrt(np.arange(start + 1, regret.size + 1, dtype=float))
    slope, _ = np.polyfit(sqrt_t, tail_regret, deg=1)
    return float(slope)
