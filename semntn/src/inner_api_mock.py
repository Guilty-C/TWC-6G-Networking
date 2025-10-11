"""Baseline inner loop oracle used for sanity checks."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

from wer_proxy import per_from_snr_db, wer_from_per


def _rate_penalty_db(q_bps: float, q_ref: float = 300.0, c_db: float = 3.0) -> float:
    """Heuristic rate penalty so that higher bit-rates require higher SNR."""

    q_bps = max(float(q_bps), 1e-9)
    return float(c_db * math.log2(q_bps / q_ref + 1.0))


def _enumerate_actions(action_space: Dict[str, Iterable[float]]) -> Tuple[Tuple[float, float, int], ...]:
    q_list = tuple(float(x) for x in action_space.get("q_bps", [300, 600, 1200, 2400]))
    p_list = tuple(float(x) for x in action_space.get("p_grid_w", [0.2, 0.4, 0.8, 1.2, 1.6]))
    b_list = tuple(int(x) for x in action_space.get("b_subbands", [1, 2, 3]))
    actions = []
    for q in q_list:
        for p in p_list:
            for b in b_list:
                actions.append((q, p, b))
    return tuple(actions)


def pick_action_and_estimate(
    ctx: Dict[str, float],
    action_space: Dict[str, Iterable[float]],
    Q_t: float,
    J_t: float,
    V: float,
    sem_weight: float,
    Q_scale: float = 20000.0,
    J_scale: float = 500.0,
) -> Dict[str, float]:
    """Return the best action according to a deterministic oracle."""

    actions = _enumerate_actions(action_space)
    if not actions:
        raise RuntimeError("Action space is empty")
    snr_db = float(ctx.get("snr_db", 0.0))
    E_bar = float(ctx.get("E_bar", 0.5))
    slot_sec = float(ctx.get("slot_sec", 0.02))
    A_bits = float(ctx.get("A_bits", 600.0 * slot_sec))

    best_cost = float("inf")
    best = None
    for q, p, b in actions:
        snr_eff_db = snr_db + 10.0 * math.log10(max(p * b, 1e-6)) - _rate_penalty_db(q)
        per = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
        swwer = wer_from_per(per, sem_weight)
        S_bits = float(q * (1.0 - per) * slot_sec)
        E_hat = float(p)

        drift_term = (Q_t / Q_scale) * (A_bits - S_bits) + (J_t / J_scale) * (E_hat - E_bar)
        cost = drift_term + V * swwer
        if cost < best_cost:
            best_cost = cost
            best = {
                "action": (q, p, b, "fixed"),
                "S_bits": S_bits,
                "E_hat": E_hat,
                "SWWER_hat": swwer,
                "snr_eff_db": float(snr_eff_db),
                "per": float(per),
                "A_bits": A_bits,
            }
    return best


__all__ = ["pick_action_and_estimate"]
