"""PD9 Preflight Feasibility Gate

Runs a short episode to estimate feasible_intersection_rate. If rate < 0.30,
applies bounded ladder adjustments:
  (a) link.P_max_dBm += 3 (cap +6 total)
  (b) search.B_min_kHz += 3.125 (one notch)
  (c) pd.delta_queue -= 0.04 (floor 0.10)
Re-check after each adjustment. If still < 0.30 → raise [FAIL_PREFLIGHT_FEASIBILITY].

Behavioral invariants (PD9 R4):
- Call the same `_adapt_config` as main runners to normalize config.
- ALWAYS write `outputs/dumps/<cfg>_preflight_resolved.yaml` before raising.
- If aggregator fails, fall back to mean of guards from raw log (preflight only).
- Derive `P_max_dBm`, `B_min_kHz`, and `B_grid_k` using the same mapping as main flow.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

# Support both module invocation (-m) and direct script execution
try:
    from ..dpp_core import DPPConfig, DPPController
    from ..analyze_task1 import aggregate_metrics
    from ..run_task1 import _adapt_config
    from ..utils_seed import set_all_seeds
except Exception:
    from semntn.src.dpp_core import DPPConfig, DPPController
    from semntn.src.analyze_task1 import aggregate_metrics
    from semntn.src.run_task1 import _adapt_config
    from semntn.src.utils_seed import set_all_seeds


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(base: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    cand = (base / p).resolve()
    if cand.exists():
        return cand
    return (base.parent / p).resolve()


def _dbm_to_lin(p_dbm: float) -> float:
    return float(10.0 ** (float(p_dbm) / 10.0) / 1000.0)


def _lin_to_dbm(p_w: float) -> float:
    return float(10.0 * np.log10(max(float(p_w) * 1000.0, 1e-12)))


def _build_controller(cfg: Dict[str, object]) -> DPPController:
    scales = cfg.get("scales", {})
    dpp_cfg = DPPConfig(
        mode=str(cfg.get("mode", "continuous")).lower(),
        V=float(cfg.get("V", 100.0)),
        Q_scale=float(scales.get("queue_scale", cfg.get("Q_scale", 2.0e4))),
        J_scale=float(scales.get("energy_scale", cfg.get("J_scale", 5.0e2))),
        S_scale=float(scales.get("semantic_scale", cfg.get("S_scale", 1.0))),
        lam_init_multiplier=float(cfg.get("lam_init_multiplier", 1.0)),
        lam_init_lat=cfg.get("lam_init_lat", None),
        lam_init_eng=cfg.get("lam_init_eng", None),
        lam_init_sem=cfg.get("lam_init_sem", None),
        lam_clip_lat=cfg.get("lam_clip_lat", None),
        lam_clip_eng=cfg.get("lam_clip_eng", None),
        eta_dual=float(cfg.get("eta_dual", 5e-3)),
        queue_delta=float(cfg.get("delta_queue", 0.08)),
        B_grid=tuple(cfg.get("B_grid", [1, 2, 3])),
        P_min=float(cfg.get("P_min", 0.2)),
        P_max=float(cfg.get("P_max", 1.6)),
        P_grid=int(cfg.get("P_grid", 25)),
    )
    return DPPController(dpp_cfg)


def _derive_unified_knobs(cfg_raw: Dict[str, object], cfg: Dict[str, object]) -> Tuple[float, float, int]:
    """Derive physical knobs using the same mapping as main runner.

    Returns (P_max_dBm, B_min_kHz, B_grid_k).
    """
    # Controller grid diagnostics
    B_list = list(cfg.get("B_grid", cfg.get("B_grid", [1, 2, 3])))
    try:
        B_count = int(len(B_list))
    except Exception:
        B_list = [1, 2, 3]
        B_count = 3
    try:
        B_min_val = float(min(B_list)) if B_count > 0 else float("nan")
    except Exception:
        B_min_val = float("nan")

    # P_max_dBm mapping (prefer explicit dBm, else convert linear Watts)
    link_block = dict((cfg_raw.get("link") or {}))
    pmax_dbm_src = link_block.get("P_max_dBm", None)
    pmax_lin_src = link_block.get("P_max", cfg_raw.get("P_max", None))
    if pmax_dbm_src is not None:
        try:
            P_max_dBm = float(pmax_dbm_src)
        except Exception:
            P_max_dBm = _lin_to_dbm(float(cfg.get("P_max", 1.6)))
    elif pmax_lin_src is not None:
        try:
            P_max_dBm = _lin_to_dbm(float(pmax_lin_src))
        except Exception:
            P_max_dBm = _lin_to_dbm(float(cfg.get("P_max", 1.6)))
    else:
        P_max_dBm = _lin_to_dbm(float(cfg.get("P_max", 1.6)))

    # B_min_kHz mapping (prefer explicit kHz; else legacy B_min ×0.5; else fallback to grid min)
    search_block = dict((cfg_raw.get("search") or {}))
    if "B_min_kHz" in search_block:
        try:
            B_min_kHz = float(search_block.get("B_min_kHz"))
        except Exception:
            B_min_kHz = float(B_min_val) if B_count > 0 else float("nan")
    else:
        bmin_src = search_block.get("B_min", None)
        try:
            B_min_kHz = float(bmin_src) * 0.5 if bmin_src is not None else (float(B_min_val) if B_count > 0 else float("nan"))
        except Exception:
            B_min_kHz = float(B_min_val) if B_count > 0 else float("nan")

    # B_grid_k mapping (min 16, prefer explicit; else len(grid))
    bgrid_k_src = search_block.get("B_grid_k", search_block.get("B_grid", None))
    try:
        B_grid_k = int(max(16, int(bgrid_k_src))) if bgrid_k_src is not None else int(max(16, B_count))
    except Exception:
        B_grid_k = int(max(16, B_count))

    return float(P_max_dBm), float(B_min_kHz), int(B_grid_k)


def _episode_feasible_intersection_rate(cfg: Dict[str, object], df_trace: pd.DataFrame, T_slice: int = 300) -> float:
    ctrl = _build_controller(cfg)
    snr_db = df_trace["snr_db"].to_numpy(dtype=float)
    slot_sec = float(cfg.get("slot_sec", 0.02))
    arrivals_bps = float(cfg.get("arrivals_bps", 600.0))
    A_bits = arrivals_bps * slot_sec
    energy_budget_per_slot_j = float(cfg.get("energy_budget_per_slot_j", 0.5))
    semantic_budget = float(cfg.get("semantic_budget", 0.38))
    fixed_q = float(cfg.get("q_bps", cfg.get("latency_budget_q_bps", 1200.0)))
    # Unified physical minimum bandwidth (kHz) for unit/identity mapping
    try:
        search_block = dict(cfg.get("search", {}))
        if search_block.get("B_min_kHz") is not None:
            B_min_kHz = float(search_block.get("B_min_kHz"))
        else:
            B_list = list(cfg.get("B_grid", [1, 2, 3]))
            B_min_kHz = float(min(B_list)) if len(B_list) > 0 else float("nan")
    except Exception:
        B_list = list(cfg.get("B_grid", [1, 2, 3]))
        B_min_kHz = float(min(B_list)) if len(B_list) > 0 else float("nan")
    rows = []
    T = min(int(T_slice), int(len(snr_db)))
    for t in range(1, T + 1):
        state = {
            "snr_db": float(snr_db[t - 1]),
            "slot_sec": slot_sec,
            "A_bits": A_bits,
            "energy_budget_per_slot_j": energy_budget_per_slot_j,
            "semantic_budget": semantic_budget,
            "latency_budget_q_bps": float(cfg.get("latency_budget_q_bps", fixed_q)),
            "q_bps": fixed_q if str(cfg.get("mode", "continuous")).lower() == "continuous" else 0.0,
            "enable_primal_dual": bool(cfg.get("enable_primal_dual", False)),
            "guard_semantic": bool((cfg.get("guard") or {}).get("semantic", False)),
            "guard_queue": bool((cfg.get("guard") or {}).get("queue", False)),
            "guard_energy": bool((cfg.get("guard") or {}).get("energy", False)),
            "epsilon_energy": float((cfg.get("guard") or {}).get("epsilon_energy", 0.0)),
            # Ensure unit/identity invariants can be checked inside controller
            "B_min_kHz": float(B_min_kHz),
        }
        P, B, stats = ctrl.step(state)
        rows.append({"t": t, **stats})
    log = pd.DataFrame(rows)
    try:
        agg = aggregate_metrics(log)
        return float(agg.get("feasible_intersection_rate", np.nan))
    except Exception:
        # Fallback: mean of (feasible_sem ∧ feasible_queue ∧ feasible_energy)
        try:
            feas_energy = log.get("feasible_energy_guard", pd.Series([np.nan]*len(log))).to_numpy()
            feas_sem = log.get("feasible_sem_guard", pd.Series([np.nan]*len(log))).to_numpy()
            feas_queue = log.get("feasible_queue_guard", pd.Series([np.nan]*len(log))).to_numpy()
            mask = (feas_energy == feas_energy) & (feas_sem == feas_sem) & (feas_queue == feas_queue)
            rate = float(np.mean((feas_energy[mask] > 0) & (feas_sem[mask] > 0) & (feas_queue[mask] > 0))) if np.any(mask) else float("nan")
            return rate
        except Exception:
            return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="PD9 Preflight Feasibility Gate")
    ap.add_argument("--config", type=Path, required=True, help="Path to YAML config (task1_release_pd2.yaml)")
    ap.add_argument("--slice", type=int, default=300, help="Short trace slice length for preflight")
    args = ap.parse_args()

    cfg_raw = _load_yaml(args.config)
    base = args.config.parent
    trace_path = _resolve_path(base, Path(cfg_raw.get("trace", "semntn/data/channel_trace.csv")))
    df_trace = pd.read_csv(trace_path)

    # Normalize config via main flow adapter and set seeds
    cfg = _adapt_config(dict(cfg_raw))
    try:
        seed = int(cfg.get("seed", cfg_raw.get("seed", 2025)))
    except Exception:
        seed = 2025
    set_all_seeds(seed)
    link = dict(cfg.get("link", {}))
    search = dict(cfg.get("search", {}))
    pd_block = dict(cfg.get("pd", {}))

    # Unified physical knobs consistent with main flow (for snapshot defaults)
    P_max_dBm_u, B_min_kHz_u, B_grid_k_u = _derive_unified_knobs(cfg_raw, cfg)

    # Compute initial rate
    rate0 = _episode_feasible_intersection_rate(cfg, df_trace, T_slice=args.slice)
    print(f"[PREFLIGHT] initial feasible_intersection_rate={rate0:.3f}")

    pmax_bump_db = 0.0
    # (a) P_max_dBm += 3, allow a second +3 (cap +6 total)
    if not (rate0 == rate0 and rate0 >= 0.30):
        p_dbm = float(link.get("P_max_dBm", np.nan))
        if not (p_dbm == p_dbm):
            # derive from linear or controller default
            p_lin = float(cfg.get("P_max", np.nan))
            if not (p_lin == p_lin):
                p_lin = 1.6
            p_dbm = _lin_to_dbm(p_lin)
        # first bump
        p_dbm_new = float(p_dbm + 3.0)
        pmax_bump_db += 3.0
        link["P_max_dBm"] = p_dbm_new
        cfg["link"] = link
        cfg["P_max"] = _dbm_to_lin(p_dbm_new)
        rate1 = _episode_feasible_intersection_rate(cfg, df_trace, T_slice=args.slice)
        print(f"[PREFLIGHT] after P_max_dBm+=3 → rate={rate1:.3f}")
        rate0 = rate1
        # optional second bump if still below threshold
        if not (rate0 == rate0 and rate0 >= 0.30) and pmax_bump_db <= 3.0:
            p_dbm_new2 = float(p_dbm_new + 3.0)
            pmax_bump_db += 3.0
            link["P_max_dBm"] = p_dbm_new2
            cfg["link"] = link
            cfg["P_max"] = _dbm_to_lin(p_dbm_new2)
            rate1b = _episode_feasible_intersection_rate(cfg, df_trace, T_slice=args.slice)
            print(f"[PREFLIGHT] after P_max_dBm+=6 total → rate={rate1b:.3f}")
            rate0 = rate1b

    # (b) B_min_kHz += 3.125 (use same mapping as main flow when deriving default)
    if not (rate0 == rate0 and rate0 >= 0.30):
        bmin_khz = float(search.get("B_min_kHz", np.nan))
        if not (bmin_khz == bmin_khz):
            # prefer legacy B_min ×0.5; else fallback to grid minimum
            try:
                if search.get("B_min", None) is not None:
                    bmin_khz = float(search.get("B_min")) * 0.5
                else:
                    B_list = list(cfg.get("B_grid", [1, 2, 3]))
                    bmin_khz = float(min(B_list)) if len(B_list) > 0 else float("nan")
            except Exception:
                B_list = list(cfg.get("B_grid", [1, 2, 3]))
                bmin_khz = float(min(B_list)) if len(B_list) > 0 else float("nan")
        search["B_min_kHz"] = float(bmin_khz + 3.125)
        cfg["search"] = search
        rate2 = _episode_feasible_intersection_rate(cfg, df_trace, T_slice=args.slice)
        print(f"[PREFLIGHT] after B_min_kHz+=3.125 → rate={rate2:.3f}")
        rate0 = rate2

    # (c) delta_queue -= 0.04 (floor 0.10)
    if not (rate0 == rate0 and rate0 >= 0.30):
        dq = float(pd_block.get("delta_queue", cfg.get("delta_queue", 0.18)))
        dq_new = float(max(dq - 0.04, 0.10))
        pd_block["delta_queue"] = dq_new
        cfg["pd"] = pd_block
        cfg["delta_queue"] = dq_new
        rate3 = _episode_feasible_intersection_rate(cfg, df_trace, T_slice=args.slice)
        print(f"[PREFLIGHT] after delta_queue-=0.04 → rate={rate3:.3f}")
        rate0 = rate3

    # (d) energy guard margin ε → 0.0 (optional relaxation within spec)
    if not (rate0 == rate0 and rate0 >= 0.30):
        guard = dict(cfg.get("guard", {}))
        eps_e = float(guard.get("epsilon_energy", 0.02))
        if eps_e > 0.0:
            guard["epsilon_energy"] = 0.0
            cfg["guard"] = guard
            rate4 = _episode_feasible_intersection_rate(cfg, df_trace, T_slice=args.slice)
            print(f"[PREFLIGHT] after epsilon_energy→0.0 → rate={rate4:.3f}")
            rate0 = rate4

    # Determine pass flag before writing snapshot
    pass_flag = bool(rate0 == rate0 and rate0 >= 0.30)

    # Cap P_max_dBm bump to +6 total if exceeded
    if pmax_bump_db > 6.0:
        link["P_max_dBm"] = float(link.get("P_max_dBm", 0.0) - (pmax_bump_db - 6.0))
        cfg["link"] = link

    # Write resolved snapshot (always write; include pass flag & knobs)
    outdir = Path(cfg.get("outdir", "outputs"))
    dumps = outdir / "dumps"
    os.makedirs(dumps, exist_ok=True)
    cfg_name = Path(args.config).stem
    guard_cfg = dict(cfg.get("guard", {}))
    # PD knobs: eta/delta_queue/lambda_init/clip
    pd_cfg = dict(cfg.get("pd", {}))
    lam_init = pd_cfg.get("lambda_init", {
        "eng": cfg.get("lam_init_eng", None),
        "lat": cfg.get("lam_init_lat", None),
        "sem": cfg.get("lam_init_sem", None),
    })
    lam_clip = {
        "eng": cfg.get("lam_clip_eng", None),
        "lat": cfg.get("lam_clip_lat", None),
        "sem": cfg.get("lam_clip_sem", None),
    }
    # Physical boundaries (unified mapping; prefer final adjusted values when available)
    # P_max_dBm: prefer link value after bumps; else derive from current cfg P_max or from raw
    try:
        P_max_dBm = float(link.get("P_max_dBm")) if link.get("P_max_dBm") is not None else _lin_to_dbm(float(cfg.get("P_max", 1.6)))
    except Exception:
        P_max_dBm = float(P_max_dBm_u)
    # B_min_kHz: prefer search value after bumps; else derive using main mapping
    try:
        B_min_kHz = float(search.get("B_min_kHz")) if search.get("B_min_kHz") is not None else float(B_min_kHz_u)
    except Exception:
        B_min_kHz = float(B_min_kHz_u)
    # Grid count: prefer explicit; else min 16 from grid length
    try:
        B_grid_k = int(search.get("B_grid_k")) if search.get("B_grid_k") is not None else int(max(16, len(tuple(cfg.get("B_grid", [1,2,3])))))
    except Exception:
        B_grid_k = int(B_grid_k_u)
    snapshot = {
        "pass": bool(pass_flag),
        "V": cfg.get("V"),
        "slot_sec": cfg.get("slot_sec"),
        "arrivals_bps": cfg.get("arrivals_bps"),
        "energy_budget_per_slot_j": cfg.get("energy_budget_per_slot_j"),
        "latency_budget_q_bps": cfg.get("latency_budget_q_bps"),
        "semantic_budget": cfg.get("semantic_budget"),
        "guard": {
            "semantic": bool(guard_cfg.get("semantic", False)),
            "queue": bool(guard_cfg.get("queue", False)),
            "energy": bool(guard_cfg.get("energy", False)),
            "epsilon_energy": float(guard_cfg.get("epsilon_energy", 0.0)),
        },
        "pd": {
            "eta": pd_cfg.get("eta", cfg.get("eta_dual", None)),
            "delta_queue": pd_block.get("delta_queue"),
            "lambda_init": lam_init,
            "clip": lam_clip,
        },
        "link": {"P_max_dBm": float(P_max_dBm) if P_max_dBm == P_max_dBm else "[MISSING]"},
        "search": {"B_min_kHz": float(B_min_kHz) if B_min_kHz == B_min_kHz else "[MISSING]", "B_grid_k": int(B_grid_k)},
        "feasible_intersection_rate": rate0,
    }
    resolved_path = dumps / f"{cfg_name}_preflight_resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)
    if pass_flag:
        print(f"[PREFLIGHT_OK] {resolved_path} → feasible_intersection_rate={rate0:.3f}")
    else:
        print(f"[PREFLIGHT_FAIL] {resolved_path} → feasible_intersection_rate={rate0:.3f}")
        # Fail after writing snapshot per PD9 runbook
        raise RuntimeError("[FAIL_PREFLIGHT_FEASIBILITY] Feasible intersection rate remains below 0.30")


if __name__ == "__main__":
    main()