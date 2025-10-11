"""Outer Lyapunov loop with pluggable inner controller."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from sem_weight_model import SemWeightModel
from utils_seed import set_all_seeds


def _get_inner(inner_mode: str):
    mode = (inner_mode or "mock").strip().lower()
    if mode == "mock":
        import inner_api_mock as inner
    elif mode == "ucb":
        import inner_api_ucb as inner
    else:
        raise ValueError(f"Unknown inner_mode: {inner_mode}")
    return inner


def _mean(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    return sum(seq) / len(seq)


def _rolling_features(samples: List[float], win: int) -> Tuple[float, float, float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    head = samples[-win:]
    K = len(head)
    energy = [max(x, 0.0) for x in head]
    energy_mean = _mean(energy)
    diff = [head[i] - head[i - 1] for i in range(1, K)]
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diff]
    zcr = 0.0
    if signs:
        changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        zcr = changes / max(len(signs), 1)
    positive = [max(x, 0.0) for x in head]
    if positive:
        idx = list(range(1, len(positive) + 1))
        numerator = sum(p * i for p, i in zip(positive, idx))
        denominator = sum(positive) + 1e-6
        spec_centroid = numerator / denominator
    else:
        spec_centroid = 0.0
    snr_mean = _mean(head)
    keyword_flag = 1.0 if max(head) > 8.0 else 0.0
    return energy_mean, zcr, spec_centroid, snr_mean, keyword_flag


def run_episode(cfg: Dict, snr_db: List[float], V: int, inner_mode: str = "mock") -> Dict[str, float]:
    seed = int(cfg.get("seed", 2025))
    set_all_seeds(seed)
    T = int(min(cfg.get("T", len(snr_db)), len(snr_db)))
    K = int(cfg.get("K_head_pkts", 5))
    slot_sec = float(cfg.get("slot_sec", 0.02))

    arrivals = cfg.get("arrivals", {})
    A_bps = float(arrivals.get("A_bps", 600.0))
    A_bits = A_bps * slot_sec

    budgets = cfg.get("queues_budget", {})
    E_bar = float(budgets.get("E_per_slot", 0.5))

    lyap_cfg = cfg.get("lyap", {})
    Q_scale = float(lyap_cfg.get("Q_scale", 1e4))
    J_scale = float(lyap_cfg.get("J_scale", 1e2))

    action_space = cfg.get("actions", {})
    sem_cfg = cfg.get("sem_weight", {})
    model = SemWeightModel(w_min=sem_cfg.get("w_min", 1.0), w_max=sem_cfg.get("w_max", 3.0))

    inner = _get_inner(inner_mode)
    reset_fn = getattr(inner, "reset_state", None)
    if callable(reset_fn):
        reset_fn()

    Q = 0.0
    J = 0.0
    swwer_hist: List[float] = []
    Q_hist: List[float] = []
    J_hist: List[float] = []

    block_counts: Dict[Tuple[float, float, int], int] = {}
    block_per_sum = 0.0
    block_steps = 0
    report_interval = max(int(cfg.get("log_interval", 2000)), 1)

    for t in range(T):
        left = max(0, t - K + 1)
        feat = _rolling_features(snr_db[left : t + 1], K)
        w_sem = model.infer_w_sem(feat)

        ctx = {
            "snr_db": float(snr_db[t]),
            "E_bar": E_bar,
            "slot_sec": slot_sec,
            "A_bits": A_bits,
        }

        pick = inner.pick_action_and_estimate(ctx, action_space, Q, J, V, w_sem, Q_scale=Q_scale, J_scale=J_scale)

        S_bits = float(pick["S_bits"])
        E_hat = float(pick["E_hat"])
        swwer = float(pick["SWWER_hat"])

        ctx["S_bits_obs"] = S_bits
        ctx["E_obs"] = E_hat
        ctx["swwer_obs"] = swwer
        if "per" in pick:
            ctx["per_obs"] = float(pick["per"])

        inner.pick_action_and_estimate(ctx, action_space, Q, J, V, w_sem, Q_scale=Q_scale, J_scale=J_scale)

        Q = max(Q - S_bits, 0.0) + A_bits
        J = max(J + E_hat - E_bar, 0.0)

        swwer_hist.append(swwer)
        Q_hist.append(Q)
        J_hist.append(J)

        action_key = tuple(pick.get("action", (0.0, 0.0, 0)))[:3]
        block_counts[action_key] = block_counts.get(action_key, 0) + 1
        block_per_sum += float(ctx.get("per_obs", pick.get("per", 0.0)))
        block_steps += 1

        if (t + 1) % report_interval == 0 or t + 1 == T:
            total = max(sum(block_counts.values()), 1)
            freq_parts = [
                f"{int(q)}/{p:.1f}/{int(b)}:{count/total:.2f}"
                for (q, p, b), count in sorted(block_counts.items())
            ]
            per_mean = block_per_sum / max(block_steps, 1)
            print(
                f"[inner-log] t={t + 1} freq=" + ",".join(freq_parts) + f" mean_per={per_mean:.3f}"
            )
            block_counts.clear()
            block_per_sum = 0.0
            block_steps = 0

    return {
        "V": int(V),
        "SWWER_mean": _mean(swwer_hist),
        "Q_mean": _mean(Q_hist),
        "J_mean": _mean(J_hist),
    }


__all__ = ["run_episode"]
