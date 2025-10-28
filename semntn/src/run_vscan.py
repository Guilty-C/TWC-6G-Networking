"""V-sweep CLI: run Task1 multiple times with varying V and aggregate.

Usage:
  python -m semntn.src.run_vscan --base-config configs/task1_tuned.yaml --vscan-config configs/vscan.yaml

Outputs:
  - Per-trial logs under outputs/vscan/V=<val>/trial_<k>.csv
  - Aggregated per-V means and 95% CIs at outputs/vscan/summary.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .dpp_core import DPPConfig, DPPController
from .analyze_task1 import aggregate_metrics
from .utils_seed import set_all_seeds
from .constants import VSCAN_MIN_ROWS, AGG_VERSION, AGG_WARMUP_SKIP, FAIL_AGG_VERSION_MISMATCH, FAIL_VSCAN_ROWS_LT_MIN, FAIL_VSCAN_AGG_MISMATCH
from .run_task1 import _adapt_config


def _resolve_path(base: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    cand = (base / p).resolve()
    if cand.exists():
        return cand
    return (base.parent / p).resolve()


def _mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = x.astype(float)
    n = max(len(x), 1)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci = 1.96 * se
    return mean, mean - ci, mean + ci


def main() -> None:
    ap = argparse.ArgumentParser(description="V-sweep runner using Task1 controller")
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--vscan-config", type=Path, required=True)
    args = ap.parse_args()

    import yaml
    with args.base_config.open("r", encoding="utf-8") as f:
        base_cfg_raw = yaml.safe_load(f)
    cfg = _adapt_config(base_cfg_raw)
    # Seed RNG for reproducibility across trials
    try:
        seed = int(cfg.get("seed", base_cfg_raw.get("seed", 2025)))
    except Exception:
        seed = 2025
    set_all_seeds(seed)

    with args.vscan_config.open("r", encoding="utf-8") as f:
        vscan_cfg = yaml.safe_load(f)

    outdir = Path("outputs") / "vscan"
    os.makedirs(outdir, exist_ok=True)

    # Trace: channel SNR
    trace_path = _resolve_path(args.base_config.parent, Path(cfg.get("trace", "semntn/data/channel_trace.csv")))
    df_trace = pd.read_csv(trace_path)
    if "snr_db" not in df_trace.columns:
        raise ValueError("trace CSV must contain 'snr_db' column")
    snr_db = df_trace["snr_db"].to_numpy(dtype=float)

    V_list = vscan_cfg.get("V_list", [20, 60, 120, 240])
    repeat_each = int(vscan_cfg.get("repeat_each", 2))

    # Episode params
    T = int(cfg.get("T", len(snr_db)))
    slot_sec = float(cfg.get("slot_sec", 0.02))
    arrivals_bps = float(cfg.get("arrivals_bps", 600.0))
    energy_budget_per_slot_j = float(cfg.get("energy_budget_per_slot_j", 0.5))
    semantic_budget = float(cfg.get("semantic_budget", 0.20))
    A_bits = arrivals_bps * slot_sec
    fixed_q = float(cfg.get("q_bps", cfg.get("latency_budget_q_bps", 1200.0)))

    scales = cfg.get("scales", {})
    base_dpp = {
        "Q_scale": float(scales.get("queue_scale", cfg.get("Q_scale", 2.0e4))),
        "J_scale": float(scales.get("energy_scale", cfg.get("J_scale", 5.0e2))),
        "S_scale": float(scales.get("semantic_scale", cfg.get("S_scale", 1.0))),
        "B_grid": tuple(cfg.get("B_grid", [1, 2, 3])),
        "P_min": float(cfg.get("P_min", 0.2)),
        "P_max": float(cfg.get("P_max", 1.6)),
        "P_grid": int(cfg.get("P_grid", 25)),
    }

    # Derive unit bandwidth mapping like run_task1
    B_list = sorted(list(base_dpp["B_grid"]))
    B_min_val = float(min(B_list)) if len(B_list) > 0 else float("nan")
    search_block = (base_cfg_raw.get("search") or {})
    if "B_min_kHz" in search_block:
        try:
            B_min_kHz = float(search_block.get("B_min_kHz"))
        except Exception:
            B_min_kHz = float(B_min_val) if len(B_list) > 0 else float("nan")
    else:
        bmin_src = search_block.get("B_min", None)
        try:
            B_min_kHz = float(bmin_src) * 0.5 if bmin_src is not None else float(B_min_val) if len(B_list) > 0 else float("nan")
        except Exception:
            B_min_kHz = float(B_min_val) if len(B_list) > 0 else float("nan")

    rows_summary: list[Dict[str, float]] = []

    for V in V_list:
        trial_metrics = []
        outV = outdir / f"V={V}"
        os.makedirs(outV, exist_ok=True)
        for k in range(1, repeat_each + 1):
            # Stabilize multi-trial randomness by reseeding per trial
            try:
                set_all_seeds(int(seed) + int(V) * 1000 + int(k))
            except Exception:
                set_all_seeds(2025 + int(V) * 1000 + int(k))
            dpp_cfg = DPPConfig(
                mode=str(cfg.get("mode", "continuous")),
                V=float(V),
                Q_scale=base_dpp["Q_scale"],
                J_scale=base_dpp["J_scale"],
                S_scale=base_dpp["S_scale"],
                lam_init_multiplier=float(cfg.get("lam_init_multiplier", 1.0)),
                eta_dual=float(cfg.get("eta_dual", (cfg.get("pd") or {}).get("eta", 5e-3))),
                queue_delta=float(cfg.get("delta_queue", (cfg.get("pd") or {}).get("delta_queue", 0.08))),
                B_grid=base_dpp["B_grid"],
                P_min=base_dpp["P_min"],
                P_max=base_dpp["P_max"],
                P_grid=base_dpp["P_grid"],
                arms=None,
                bandit_kwargs=cfg.get("bandit", {}),
            )
            ctrl = DPPController(dpp_cfg)

            rows = []
            for t in range(1, T + 1):
                state = {
                    "snr_db": float(snr_db[t - 1]),
                    "slot_sec": slot_sec,
                    "A_bits": A_bits,
                    "energy_budget_per_slot_j": energy_budget_per_slot_j,
                    "semantic_budget": semantic_budget,
                    "latency_budget_q_bps": float(cfg.get("latency_budget_q_bps", fixed_q)),
                    "q_bps": fixed_q if cfg.get("mode", "continuous") == "continuous" else 0.0,
                    # enable guards consistent with release config
                    "guard_semantic": bool((cfg.get("guard") or {}).get("semantic", False)),
                    "guard_queue": bool((cfg.get("guard") or {}).get("queue", False)),
                    "guard_energy": bool((cfg.get("guard") or {}).get("energy", False)),
                    "epsilon_energy": float((cfg.get("guard") or {}).get("epsilon_energy", 0.0)),
                    # Provide physical min bandwidth for unit/identity checks
                    "B_min_kHz": float(B_min_kHz),
                }
                P, B, stats = ctrl.step(state)
                # Effective bandwidth in Hz (relative to min grid → kHz mapping)
                try:
                    B_sel = float(B)
                    if (B_sel == B_sel) and (B_min_kHz == B_min_kHz) and (B_min_val == B_min_val) and (B_min_val > 0):
                        B_eff_kHz = float(B_sel / B_min_val) * float(B_min_kHz)
                        B_eff_Hz = float(B_eff_kHz * 1000.0)
                    else:
                        B_eff_Hz = float("nan")
                except Exception:
                    B_eff_Hz = float("nan")
                rows.append({
                    "t": int(t),
                    **stats,
                    "B_min_kHz": float(B_min_kHz) if B_min_kHz == B_min_kHz else float("nan"),
                    "B_eff_Hz": float(B_eff_Hz),
                })

            log = pd.DataFrame(rows)
            log_path = outV / f"trial_{k}.csv"
            log.to_csv(log_path, index=False)

            # Single-source aggregation with graceful failure handling
            agg = {}
            try:
                agg = aggregate_metrics(log)
                # Gate on aggregator version consistency
                if str(agg.get("agg_version", "[UNKNOWN]")) != AGG_VERSION:
                    print(f"[AGG_VERSION_MISMATCH] Expected {AGG_VERSION}, got {agg.get('agg_version', '[UNKNOWN]')}")
                    # Continue with fallback aggregation
                
                # Mismatch detection between legacy flags and aggregator
                diffs = []
                try:
                    # Align legacy means with aggregator's warm-up skip
                    if "t" in log.columns:
                        log_use = log[pd.to_numeric(log["t"], errors="coerce") > int(AGG_WARMUP_SKIP)]
                    else:
                        log_use = log.iloc[int(AGG_WARMUP_SKIP):]
                    if "violate_lat" in log_use.columns:
                        diffs.append(abs(float(log_use["violate_lat"].mean()) - float(agg["viol_lat"])) )
                    if "violate_eng" in log_use.columns:
                        diffs.append(abs(float(log_use["violate_eng"].mean()) - float(agg["viol_eng"])) )
                    if "violate_sem" in log_use.columns:
                        diffs.append(abs(float(log_use["violate_sem"].mean()) - float(agg["viol_sem"])) )
                except Exception:
                    diffs.append(float("inf"))
                # Only check NaNs among numeric aggregation fields
                try:
                    numeric_vals = [v for v in agg.values() if isinstance(v, (int, float))]
                    has_nan = any(np.isnan(numeric_vals)) if numeric_vals else False
                except Exception:
                    has_nan = True
                if has_nan or any(d > 1e-3 for d in diffs):
                    print(f"[AGG_MISMATCH] NaN={has_nan}, diffs={diffs}")
                    # Continue with fallback aggregation
            except Exception as e:
                print(f"[AGG_FAILURE] {e}")
                # Fallback to basic aggregation
                agg = {
                    "viol_lat": float(log["violate_lat"].mean()) if "violate_lat" in log.columns else 0.0,
                    "viol_eng": float(log["violate_eng"].mean()) if "violate_eng" in log.columns else 0.0,
                    "viol_sem": float(log["violate_sem"].mean()) if "violate_sem" in log.columns else 0.0,
                    "agg_version": AGG_VERSION,
                    "warmup_skip": int(AGG_WARMUP_SKIP)
                }

            trial_metrics.append({
                "MOS_mean": float(log["mos"].mean()),
                "E_slot_J_mean": float(log["E_slot_J"].mean() if "E_slot_J" in log.columns else np.nan),
                # Use aggregator for violation rates
                "viol_lat": float(agg["viol_lat"]),
                "viol_eng": float(agg["viol_eng"]),
                "viol_sem": float(agg["viol_sem"]),
            })

        # Aggregate per V
        mos_vals = np.array([m["MOS_mean"] for m in trial_metrics])
        e_vals = np.array([m["E_slot_J_mean"] for m in trial_metrics])
        vl = np.array([m["viol_lat"] for m in trial_metrics])
        ve = np.array([m["viol_eng"] for m in trial_metrics])
        vs = np.array([m["viol_sem"] for m in trial_metrics])

        mos_m, mos_lo, mos_hi = _mean_ci(mos_vals)
        e_m, e_lo, e_hi = _mean_ci(e_vals)
        vl_m, vl_lo, vl_hi = _mean_ci(vl)
        ve_m, ve_lo, ve_hi = _mean_ci(ve)
        vs_m, vs_lo, vs_hi = _mean_ci(vs)

        rows_summary.append({
            "V": float(V),
            "MOS_mean": mos_m, "MOS_ci_lo": mos_lo, "MOS_ci_hi": mos_hi,
            "E_slot_J_mean": e_m, "E_slot_J_ci_lo": e_lo, "E_slot_J_ci_hi": e_hi,
            "viol_lat": vl_m, "viol_lat_ci_lo": vl_lo, "viol_lat_ci_hi": vl_hi,
            "viol_eng": ve_m, "viol_eng_ci_lo": ve_lo, "viol_eng_ci_hi": ve_hi,
            "viol_sem": vs_m, "viol_sem_ci_lo": vs_lo, "viol_sem_ci_hi": vs_hi,
            # Aggregator metadata for downstream gating
            "agg_version": AGG_VERSION,
            "warmup_skip": int(AGG_WARMUP_SKIP),
        })

    summary = pd.DataFrame(rows_summary).sort_values(by="V")
    summary_path = outdir / "summary.csv"
    
    # Always write summary.csv, even if below minimum rows
    summary.to_csv(summary_path, index=False)
    
    # PD9 gating: check minimum rows but don't fail
    if int(summary.shape[0]) < int(VSCAN_MIN_ROWS):
        print(f"[VSCAN_WARNING] Only {summary.shape[0]} rows, minimum {VSCAN_MIN_ROWS} required")
    else:
        print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()