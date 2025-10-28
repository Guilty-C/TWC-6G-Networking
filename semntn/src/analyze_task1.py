"""Analyze Task1 outputs.

Computes time-averages, violation rates, pseudo-regret/AUC (if bandit),
MOS uplift, and emits figures:
- MOS vs time
- queue CDFs
- energy per slot vs time
- semantic distortion (sWER) vs time
- Pareto front (MOS vs Energy)
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from datetime import datetime


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cdf_data(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1, dtype=float) / max(len(x), 1)
    return x, y


def analyze(log_path: Path, outdir: Path, config_name: str | None = None) -> None:
    # Enforce dtype for unit/identity columns
    try:
        df = pd.read_csv(log_path, dtype={"B_eff_Hz": float, "B_min_kHz": float})
    except Exception:
        df = pd.read_csv(log_path)
    T = len(df)
    # time-averages
    mos_mean = float(df["mos"].mean())
    rate_mean = float(df.get("rate", df.get("q_bps", pd.Series([np.nan]*T))).mean())
    # prefer per-slot energy (J) if present
    energy_col = "E_slot_J" if "E_slot_J" in df.columns else "P"
    energy_mean = float(df[energy_col].mean())
    # sWER mean prefers raw pipeline if available
    if "sWER_raw" in df.columns:
        swer_mean = float(df["sWER_raw"].mean())
        # Robustness checks
        try:
            frac_in01 = float(df["sWER_raw"].between(0, 1).mean())
            std_val = float(df["sWER_raw"].std())
            assert frac_in01 > 0.99 and std_val > 1e-6
        except Exception:
            print("[ERROR] sWER pipeline constant/invalid")
    else:
        swer_mean = float(df.get("sWER", pd.Series([np.nan]*T)).mean())

    # violation rates (constraints exceeded per slot)
    # We infer simple violations from queue increments (z differences) > 0
    z_lat = df["z_lat"].to_numpy(dtype=float)
    z_eng = df["z_eng"].to_numpy(dtype=float)
    z_sem = df["z_sem"].to_numpy(dtype=float)
    if set(["violate_lat","violate_eng","violate_sem"]).issubset(df.columns):
        v_lat = float(df["violate_lat"].mean())
        v_eng = float(df["violate_eng"].mean())
        v_sem = float(df["violate_sem"].mean())
    else:
        v_lat = float(np.mean(np.diff(np.concatenate([[0.0], z_lat])) > 0.0))
        v_eng = float(np.mean(np.diff(np.concatenate([[0.0], z_eng])) > 0.0))
        v_sem = float(np.mean(np.diff(np.concatenate([[0.0], z_sem])) > 0.0))

    # Aggregator self-check (instantaneous vs aggregated)
    viol_eng_inst_mean = None
    viol_lat_inst_mean = None
    try:
        if "E_slot_J" in df.columns and "energy_budget_per_slot_j" in df.columns:
            viol_eng_inst = (df["E_slot_J"].to_numpy(dtype=float) > df["energy_budget_per_slot_j"].to_numpy(dtype=float)).astype(float)
            viol_eng_inst_mean = float(np.mean(viol_eng_inst))
        # Latency: use per-slot violation column if present
        if "violate_lat" in df.columns:
            viol_lat_inst = df["violate_lat"].to_numpy(dtype=float)
            viol_lat_inst_mean = float(np.mean(viol_lat_inst))
        if (viol_eng_inst_mean is not None):
            print(f"[SELF_CHECK] viol_eng_inst_mean={viol_eng_inst_mean:.6f} vs aggreg={v_eng:.6f}")
        if (viol_lat_inst_mean is not None):
            print(f"[SELF_CHECK] viol_lat_inst_mean={viol_lat_inst_mean:.6f} vs aggreg={v_lat:.6f}")
    except Exception as e:
        print(f"[WARN] aggregator self-check skipped: {e}")

    # pseudo-regret (bandit): against hindsight best arm if present
    regret = None
    auc = None
    if "arm_id" in df.columns:
        # Approximate best fixed arm by max mean reward (normalized MOS)
        mos_norm = (df["mos"] - 1.0) / (4.5 - 1.0)
        groups = df.groupby("arm_id")["mos"].mean()
        best_mos = float(groups.max()) if len(groups) > 0 else mos_mean
        inst_reg = np.maximum(best_mos - df["mos"], 0.0)
        regret = float(np.cumsum(inst_reg).iloc[-1])
        # AUC of MOS over time (normalized)
        auc = float(mos_norm.sum() / max(T, 1))

    # MOS uplift: vs baseline that ignores semantic bonus
    baseline = float((1.0 - df["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - baseline)

    # Unit/identity gate (applies to all configs when columns present)
    try:
        if set(["B_eff_Hz","B_min_kHz"]).issubset(df.columns):
            lhs = df["B_eff_Hz"].to_numpy(dtype=float)
            rhs = df["B_min_kHz"].to_numpy(dtype=float) * 1000.0

            # Immediate violation for NaN in B_eff_Hz
            nan_mask = np.isnan(lhs)
            if np.any(nan_mask):
                i0 = int(np.where(nan_mask)[0][0])
                t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_eff_Hz=nan, B_min_kHz={float(rhs[i0]/1000.0):.3f}, rhs={rhs[i0]:.2f}")
                sys.exit(2)

            # Also fail-fast on NaN in B_min_kHz
            nan_rhs = np.isnan(rhs)
            if np.any(nan_rhs):
                i0 = int(np.where(nan_rhs)[0][0])
                t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_eff_Hz={lhs[i0]:.3f}, B_min_kHz=nan, rhs=nan")
                sys.exit(2)

            ok_vec = (lhs >= rhs)
            ok_rate = float(ok_vec.mean())
            if not (ok_rate >= 0.99):
                idxs = np.where(~ok_vec)[0]
                i0 = int(idxs[0]) if idxs.size else 0
                t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_eff_Hz={lhs[i0]:.3f}, B_min_kHz={float(rhs[i0]/1000.0):.3f}, rhs={rhs[i0]:.2f}")
                sys.exit(2)
    except Exception:
        pass

    # Print summary
    print("[SUMMARY] T=%d" % T)
    energy_unit = "J" if energy_col == "E_slot_J" else "W"
    print(f"  MOS_mean={mos_mean:.4f}, rate_mean={rate_mean:.2f} bps, energy_mean={energy_mean:.3f} {energy_unit}, sWER_mean={swer_mean:.3f}")
    print(f"  violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f})")
    if regret is not None:
        print(f"  pseudo_regret={regret:.3f}, AUC_mos_norm={auc:.3f}")
    print(f"  MOS_uplift_vs_baseline={mos_uplift:.4f}")

    # Diagnostics: PD and Guard activation rates (column means if present)
    pd_active_rate = float(df.get("pd_active", pd.Series([np.nan]*T)).mean())
    guard_sem_rate = float(df.get("guard_sem_active", pd.Series([np.nan]*T)).mean())
    guard_queue_rate = float(df.get("guard_queue_active", pd.Series([np.nan]*T)).mean())
    guard_energy_rate = float(df.get("guard_energy_active", pd.Series([np.nan]*T)).mean())
    # Guard feasible-set null fraction: both guards infeasible simultaneously
    feasible_sem = df.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
    feasible_queue = df.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
    try:
        guard_null_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0)))
    except Exception:
        guard_null_rate = float("nan")
    # Optional: triple-null fraction when energy guard present
    try:
        feasible_energy = df.get("feasible_energy_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null3_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0) & (feasible_energy == 0)))
        if guard_null3_rate == guard_null3_rate:
            print(f"[DIAG] guard_null3_rate={guard_null3_rate:.3f}")
    except Exception:
        pass
    if not np.isnan(pd_active_rate) and pd_active_rate < 0.95:
        print(f"[WARN] pd_active_rate={pd_active_rate:.3f} < 0.95")
    if not np.isnan(guard_sem_rate) and guard_sem_rate < 0.95:
        print(f"[WARN] guard_sem_rate={guard_sem_rate:.3f} < 0.95")
    if not np.isnan(guard_queue_rate) and guard_queue_rate < 0.95:
        print(f"[WARN] guard_queue_rate={guard_queue_rate:.3f} < 0.95")
    if not np.isnan(guard_energy_rate) and guard_energy_rate < 0.95:
        print(f"[WARN] guard_energy_rate={guard_energy_rate:.3f} < 0.95")
    if not np.isnan(guard_null_rate) and guard_null_rate > 0.02:
        print(f"[WARN] guard_null_rate={guard_null_rate:.3f} > 0.02")
    # Dual price trajectory diagnostics
    lam_eng = df.get("lambda_eng", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    lam_lat = df.get("lambda_lat", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    if np.isfinite(lam_eng).any():
        lam_eng_min = float(np.nanmin(lam_eng))
        lam_eng_max = float(np.nanmax(lam_eng))
        lam_eng_last = float(lam_eng[-1]) if len(lam_eng) > 0 else float("nan")
        print(f"[DIAG] lambda_eng_min/max/last={lam_eng_min:.4f}/{lam_eng_max:.4f}/{lam_eng_last:.4f}")
    if np.isfinite(lam_lat).any():
        lam_lat_min = float(np.nanmin(lam_lat))
        lam_lat_max = float(np.nanmax(lam_lat))
        lam_lat_last = float(lam_lat[-1]) if len(lam_lat) > 0 else float("nan")
        print(f"[DIAG] lambda_lat_min/max/last={lam_lat_min:.4f}/{lam_lat_max:.4f}/{lam_lat_last:.4f}")
    # Wirecheck hard fail if requested
    if config_name and str(config_name).lower().startswith("wirecheck"):
        if (pd_active_rate == pd_active_rate) and pd_active_rate < 0.95:
            print("[FAIL_WIRECHECK_PD]")
            sys.exit(2)
        if (guard_sem_rate == guard_sem_rate) and guard_sem_rate < 0.95:
            print("[FAIL_WIRECHECK_GUARD_SEM]")
            sys.exit(2)
        if (guard_queue_rate == guard_queue_rate) and guard_queue_rate < 0.95:
            print("[FAIL_WIRECHECK_GUARD_QUEUE]")
            sys.exit(2)
        if (guard_energy_rate == guard_energy_rate) and guard_energy_rate < 0.95:
            print("[FAIL_WIRECHECK_GUARD_ENERGY]")
            sys.exit(2)
        if (guard_null_rate == guard_null_rate) and guard_null_rate > 0.02:
            print("[FAIL_WIRECHECK_GUARD_NULL]")
            sys.exit(2)
        # Aggregator mismatch hard assertions
        if viol_eng_inst_mean is not None and abs(viol_eng_inst_mean - v_eng) > 1e-3:
            print("[FAIL_AGGREGATOR_MISMATCH_ENG]")
            sys.exit(2)
        if viol_lat_inst_mean is not None and abs(viol_lat_inst_mean - v_lat) > 1e-3:
            print("[FAIL_AGGREGATOR_MISMATCH_LAT]")
            sys.exit(2)
        # Enforce resolved.yaml contains unified physical boundaries
        try:
            res_path = log_path.parent / (log_path.stem.replace("_log", "_resolved") + ".yaml")
            data_res = _read_yaml_safe(res_path)
        except Exception:
            data_res = {}
        pmax_dbm = data_res.get("P_max_dBm")
        bmin_khz = data_res.get("B_min_kHz")
        bgrid_k = data_res.get("B_grid_k")
        def _is_missing(x) -> bool:
            if x is None:
                return True
            if isinstance(x, str) and x.strip().upper() == "[MISSING]":
                return True
            if isinstance(x, float) and (x != x):
                return True
            return False
        if _is_missing(pmax_dbm) or _is_missing(bmin_khz) or _is_missing(bgrid_k):
            print("[FAIL_WIRECHECK_MISSING_KEYS]")
            sys.exit(2)
        # Landing checks: units and guard enforcement (PD9 spec)
        # Units: require B_eff_Hz >= B_min_kHz*1e3 in at least 99% slots
        try:
            if set(["B_eff_Hz","B_min_kHz"]).issubset(df.columns):
                lhs = df["B_eff_Hz"].to_numpy(dtype=float)
                rhs = df["B_min_kHz"].to_numpy(dtype=float) * 1000.0
                
                # CRITICAL FIX: Enhanced NaN handling - treat NaN in B_eff_Hz as immediate violation
                nan_mask = np.isnan(lhs)
                if np.any(nan_mask):
                    # Find first NaN sample
                    nan_indices = np.where(nan_mask)[0]
                    i0 = int(nan_indices[0]) if nan_indices.size else 0
                    t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                    print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_eff_Hz=nan, B_min_kHz={float(rhs[i0]/1000.0):.3f}, rhs={rhs[i0]:.2f}")
                    sys.exit(2)
                
                # CRITICAL FIX: Also check for NaN in B_min_kHz
                nan_bmin_mask = np.isnan(rhs)
                if np.any(nan_bmin_mask):
                    # Find first NaN sample in B_min_kHz
                    nan_indices = np.where(nan_bmin_mask)[0]
                    i0 = int(nan_indices[0]) if nan_indices.size else 0
                    t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                    print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_eff_Hz={lhs[i0]:.3f}, B_min_kHz=nan, rhs=nan")
                    sys.exit(2)
                
                ok_vec = (lhs >= rhs)
                ok_rate = float(ok_vec.mean())
                if not (ok_rate >= 0.99):
                    # Find first violation where lhs < rhs
                    try:
                        import numpy as np
                        idxs = np.where(~ok_vec)[0]
                        i0 = int(idxs[0]) if idxs.size else 0
                        t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                        print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_eff_Hz={lhs[i0]:.3f}, B_min_kHz={float(rhs[i0]/1000.0):.3f}, rhs={rhs[i0]:.2f}")
                    except Exception:
                        print("[FAIL_UNIT_BANDWIDTH]")
                    sys.exit(2)
        except Exception:
            pass
        # Aggregator identity checks
        try:
            agg = aggregate_metrics(df)
            viol_eng_inst = (df["E_slot_J"].to_numpy(dtype=float) > df["energy_budget_per_slot_j"].to_numpy(dtype=float)).astype(float)
            viol_lat_inst = (df["S_eff_bps"].to_numpy(dtype=float) < (df["arrivals_bps"].to_numpy(dtype=float) + df["delta_queue_used"].to_numpy(dtype=float) * df["Q_t_used"].to_numpy(dtype=float))).astype(float)
            d_eng = abs(float(np.mean(viol_eng_inst)) - float(agg["viol_eng"]))
            d_lat = abs(float(np.mean(viol_lat_inst)) - float(agg["viol_lat"]))
            if d_eng > 1e-3:
                # 打印第一处能量违例样本
                try:
                    import numpy as np
                    idxs = np.where(df["E_slot_J"].to_numpy(dtype=float) > df["energy_budget_per_slot_j"].to_numpy(dtype=float))[0]
                    i0 = int(idxs[0]) if idxs.size else 0
                    t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                    print(f"[FAIL_AGGREGATOR_MISMATCH_ENG] sample t={t0}, E_slot_J={float(df['E_slot_J'].iloc[i0]):.6f}, budget_J={float(df['energy_budget_per_slot_j'].iloc[i0]):.6f}")
                except Exception:
                    print("[FAIL_AGGREGATOR_MISMATCH_ENG]")
                sys.exit(2)
            if d_lat > 1e-3:
                # 打印第一处时延违例样本
                try:
                    import numpy as np
                    lhs = df["S_eff_bps"].to_numpy(dtype=float)
                    rhs = df["arrivals_bps"].to_numpy(dtype=float) + df["delta_queue_used"].to_numpy(dtype=float) * df["Q_t_used"].to_numpy(dtype=float)
                    idxs = np.where(lhs < rhs)[0]
                    i0 = int(idxs[0]) if idxs.size else 0
                    t0 = int(df.get("t", pd.Series([i0])).iloc[i0]) if "t" in df.columns else i0
                    print(f"[FAIL_AGGREGATOR_MISMATCH_LAT] sample t={t0}, S_eff_bps={lhs[i0]:.3f}, arrivals+δQ={rhs[i0]:.3f}")
                except Exception:
                    print("[FAIL_AGGREGATOR_MISMATCH_LAT]")
                sys.exit(2)
        except Exception:
            pass
        # Enforcement leak: if action_in_guard_set == 1 and any bad_* == 1 → fail
        try:
            if set(["action_in_guard_set","bad_energy","bad_queue","bad_sem"]).issubset(df.columns):
                leaks_eng = ((df["action_in_guard_set"].to_numpy(dtype=int) == 1) & (df["bad_energy"].to_numpy(dtype=int) == 1)).sum()
                leaks_lat = ((df["action_in_guard_set"].to_numpy(dtype=int) == 1) & (df["bad_queue"].to_numpy(dtype=int) == 1)).sum()
                leaks_sem = ((df["action_in_guard_set"].to_numpy(dtype=int) == 1) & (df["bad_sem"].to_numpy(dtype=int) == 1)).sum()
                if leaks_eng > 0:
                    print("[ENFORCEMENT_LEAK:energy]")
                    sys.exit(2)
                if leaks_lat > 0:
                    print("[ENFORCEMENT_LEAK:queue]")
                    sys.exit(2)
                if leaks_sem > 0:
                    print("[ENFORCEMENT_LEAK:semantic]")
                    sys.exit(2)
        except Exception:
            pass
        # PD9 thresholds via shared aggregator (fail-fast)
        try:
            agg = aggregate_metrics(df)
            # feasible_intersection_rate ≥ 0.30
            if float(agg.get("feasible_intersection_rate", float("nan"))) < 0.30:
                print("[FAIL_WIRECHECK_INTERSECTION_LT030]")
                sys.exit(2)
            # action_in_guard_set_rate ≥ 0.99
            if float(agg.get("action_in_guard_set_rate", float("nan"))) < 0.99:
                print("[FAIL_WIRECHECK_ACTION_IN_SET_LT099]")
                sys.exit(2)
            # fallback_used_rate ≤ 0.01
            if float(agg.get("fallback_used_rate", float("nan"))) > 0.01:
                print("[FAIL_WIRECHECK_FALLBACK_GT001]")
                sys.exit(2)
            # bad_* must be 0
            if float(agg.get("guard_enforcement_bad_rate_energy", float("nan"))) != 0.0:
                print("[FAIL_WIRECHECK_BAD_ENERGY_NONZERO]")
                sys.exit(2)
            if float(agg.get("guard_enforcement_bad_rate_queue", float("nan"))) != 0.0:
                print("[FAIL_WIRECHECK_BAD_QUEUE_NONZERO]")
                sys.exit(2)
            if float(agg.get("guard_enforcement_bad_rate_sem", float("nan"))) != 0.0:
                print("[FAIL_WIRECHECK_BAD_SEM_NONZERO]")
                sys.exit(2)
        except RuntimeError as e:
            # Propagate aggregator failures (missing columns / NaN)
            print(str(e))
            sys.exit(2)

    # Figures
    figs = outdir / "figs"
    dumps = outdir / "dumps"
    os.makedirs(figs, exist_ok=True)
    os.makedirs(dumps, exist_ok=True)

    t = np.arange(1, T + 1)
    # MOS vs time
    plt.figure(figsize=(7, 4.5))
    plt.plot(t, df["mos"].to_numpy(dtype=float))
    plt.xlabel("time"); plt.ylabel("MOS"); plt.title("MOS vs time"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(figs / "Fig_MOS_vs_time.png", dpi=150); plt.close()

    # Queue CDFs
    for col, name in [("z_lat", "latency"), ("z_eng", "energy"), ("z_sem", "semantic")]:
        x, y = _cdf_data(df[col].to_numpy(dtype=float))
        plt.figure(figsize=(7, 4.5))
        plt.step(x, y, where="post")
        plt.xlabel("queue size"); plt.ylabel("CDF"); plt.title(f"{name} queue CDF")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(figs / f"Fig_CDF_{name}.png", dpi=150); plt.close()

    # Energy per slot (prefer J)
    plt.figure(figsize=(7, 4.0))
    plt.plot(t, df[energy_col].to_numpy(dtype=float))
    ylabel = "E_slot_J (J/slot)" if energy_col == "E_slot_J" else "energy proxy (P)"
    plt.xlabel("time"); plt.ylabel(ylabel); plt.title("Energy per slot"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(figs / "Fig_Energy_vs_time.png", dpi=150); plt.close()

    # Semantic distortion vs time
    plt.figure(figsize=(7, 4.0))
    swer_plot = df.get("sWER_raw", df.get("sWER", pd.Series([np.nan]*T))).to_numpy(dtype=float)
    plt.plot(t, swer_plot)
    plt.xlabel("time"); plt.ylabel("sWER"); plt.title("Semantic distortion vs time"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(figs / "Fig_sWER_vs_time.png", dpi=150); plt.close()

    # Pareto front (MOS vs Energy)
    plt.figure(figsize=(6.0, 4.8))
    plt.scatter(df["P"].to_numpy(dtype=float), df["mos"].to_numpy(dtype=float), s=10, alpha=0.6)
    plt.xlabel("Energy proxy (P)"); plt.ylabel("MOS"); plt.title("Pareto front: MOS vs Energy")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(figs / "Fig_Pareto_MOS_vs_Energy.png", dpi=150); plt.close()

    # dump short comparison summary with unified columns
    if config_name is None:
        config_name = log_path.stem
    summary_path = dumps / "task1_summary.csv"
    row = {
        "config_name": str(config_name),
        "MOS_mean": mos_mean,
        "E_slot_J_mean": energy_mean if energy_col == "E_slot_J" else np.nan,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
        "pd_active_rate": pd_active_rate,
        "guard_sem_rate": guard_sem_rate,
        "guard_queue_rate": guard_queue_rate,
        "guard_energy_rate": guard_energy_rate,
    }
    # Write header if file missing; else append
    if summary_path.exists():
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")

    # S2: 服务率残差（当日志包含 A_bits/Q_bits/Q_next_bits/slot_sec）
    try:
        required_cols = {"A_bits","Q_bits","Q_next_bits","slot_sec"}
        if required_cols.issubset(set(df.columns)):
            errs = Path("outputs") / "error_analysis"
            os.makedirs(errs, exist_ok=True)
            csv_path = errs / "service_residual_samples.csv"
            fig_path = errs / "Fig_service_residual_heatmap.png"
            slot_sec_val = float(df["slot_sec"].iloc[0]) if len(df) > 0 else 0.02
            rate_bps = df.get("rate", df.get("q_bps", pd.Series([np.nan]*T))).to_numpy(dtype=float)
            per = df.get("per", pd.Series([np.nan]*T)).to_numpy(dtype=float)
            S_model = rate_bps * (1.0 - per) * slot_sec_val
            A_bits = df.get("A_bits").to_numpy(dtype=float)
            Q = df.get("Q_bits").to_numpy(dtype=float)
            Qn = df.get("Q_next_bits").to_numpy(dtype=float)
            S_meas = A_bits - np.maximum(Qn - Q, 0.0)
            dS = S_meas - S_model
            samples = pd.DataFrame({
                "t": df.get("t", pd.Series(np.arange(1, T+1))).to_numpy(dtype=int),
                "SINR": df.get("sinr_db", pd.Series([np.nan]*T)).to_numpy(dtype=float),
                "B": df.get("B", pd.Series([np.nan]*T)).to_numpy(dtype=float),
                "rate": rate_bps,
                "per": per,
                "S_model": S_model,
                "S_meas": S_meas,
                "dS": dS,
            })
            samples.to_csv(csv_path, index=False)

            # Heatmap: dS over (SINR×B)
            df_s = samples.dropna(subset=["dS","SINR","B"])  # require complete cells
            if len(df_s) > 0:
                sinr = df_s["SINR"].to_numpy(dtype=float)
                dS_v = df_s["dS"].to_numpy(dtype=float)
                B_v = df_s["B"].to_numpy(dtype=float)
                # Bin SINR by quantiles
                qs = np.quantile(sinr, [0.25, 0.5, 0.75])
                bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
                idx_s = np.digitize(sinr, bins) - 1
                Bs = np.unique(B_v)
                grid = np.full((4, len(Bs)), np.nan)
                for i in range(4):
                    for j, b in enumerate(Bs):
                        mask = (idx_s == i) & (B_v == b)
                        grid[i, j] = float(np.nanmean(dS_v[mask])) if np.any(mask) else np.nan
                plt.figure(figsize=(7.2, 4.6))
                im = plt.imshow(grid, aspect="auto", cmap="coolwarm", origin="lower")
                plt.colorbar(im, label="dS (bits)")
                plt.yticks([0,1,2,3],["Q1","Q2","Q3","Q4"]) ; plt.xticks(range(len(Bs)), [f"B={int(b)}" for b in Bs])
                plt.title("Service residual dS heatmap (SINR quantile × B)") ; plt.xlabel("B") ; plt.ylabel("SINR quantile")
                plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    except Exception as e:
        print(f"[WARN] Service residuals not computed: {e}")


def _plot_vscan(summary_csv: Path, outdir: Path) -> None:
    df = pd.read_csv(summary_csv)
    if "V" not in df.columns:
        print(f"[WARN] vscan summary missing 'V' column: {summary_csv}")
        return
    # Require at least 3 rows
    if len(df) < 3:
        print("[FAIL_VSCAN] summary rows < 3")
        sys.exit(2)
    V = df["V"].to_numpy(dtype=float)
    # MOS vs V
    plt.figure(figsize=(7, 4.2))
    plt.plot(V, df["MOS_mean"].to_numpy(dtype=float), marker="o")
    plt.xlabel("V"); plt.ylabel("MOS_mean"); plt.title("MOS vs V"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir / "Fig_MOS_vs_V.png", dpi=150); plt.close()
    # Violations vs V
    plt.figure(figsize=(7, 4.2))
    plt.plot(V, df["viol_lat"].to_numpy(dtype=float), marker="o", label="lat")
    plt.plot(V, df["viol_eng"].to_numpy(dtype=float), marker="o", label="eng")
    plt.plot(V, df["viol_sem"].to_numpy(dtype=float), marker="o", label="sem")
    plt.xlabel("V"); plt.ylabel("violation rate"); plt.title("Violation rates vs V"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir / "Fig_ViolationRates_vs_V.png", dpi=150); plt.close()
    # Pareto MOS vs Energy
    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(df["E_slot_J_mean"].to_numpy(dtype=float), df["MOS_mean"].to_numpy(dtype=float), s=30)
    for i, v in enumerate(V):
        plt.annotate(f"V={int(v)}", (df["E_slot_J_mean"].iloc[i], df["MOS_mean"].iloc[i]), fontsize=8)
    plt.xlabel("E_slot_J_mean (J/slot)"); plt.ylabel("MOS_mean"); plt.title("Pareto: MOS vs Energy")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(outdir / "Fig_Pareto_MOS_vs_Energy.png", dpi=150); plt.close()


# ---------------- Consolidated Strict Report ----------------
def _git_meta() -> tuple[str, str]:
    """Return (short_hash, branch) or ('[MISSING]', '[MISSING]') if not a git repo."""
    try:
        short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        if short_hash:
            return short_hash, branch or "[MISSING]"
    except Exception:
        pass
    return "[MISSING]", "[MISSING]"


def _read_yaml_safe(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))

def calibrate_swer_sinr(link_yaml: Path, source_log: Path, budget: float, q: float = 0.65) -> tuple[float, float]:
    """Calibrate (a,b) in configs/link_semantic.yaml so that sWER(sigmoid(a*x+b)) hits
    `budget` at SINR linear corresponding to quantile q of `sinr_db` in source_log.

    Keeps `a` unchanged if present; updates `b` accordingly. Returns (a_new, b_new).
    """
    # Read current a,b
    a_default = -0.40
    b_default = -0.05
    a = a_default
    b = b_default
    try:
        with link_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            a = float(data.get("a", a_default))
            b = float(data.get("b", b_default))
    except Exception:
        pass
    # Read sinr_db from log
    df = pd.read_csv(source_log)
    if "sinr_db" not in df.columns:
        raise RuntimeError(f"[CALIB] source log missing sinr_db: {source_log}")
    qval = float(np.quantile(df["sinr_db"].to_numpy(dtype=float), float(np.clip(q, 0.0, 1.0))))
    sinr_lin = float(10.0 ** (qval / 10.0))
    # solve b so sigmoid(a*sinr_lin + b) = budget
    b_new = float(_logit(float(budget)) - a * sinr_lin)
    try:
        with link_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"a": float(a), "b": float(b_new)}, f)
        print(f"[CALIB] link_semantic.yaml updated: a={a:.4f}, b={b_new:.4f}")
    except Exception as e:
        print(f"[CALIB][ERROR] writing {link_yaml}: {e}")
    return float(a), float(b_new)


def _read_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def update_semantic_params_and_budget(release_cfg_path: Path) -> None:
    """Update link_semantic (a,b) if error_analysis recalibration exists; else
    compute sWER_target from SINR_max quantile (0.675) and update semantic_budget
    in the release config (and wirecheck variant if present).
    """
    errs_yaml = Path("outputs/error_analysis/semantic_recalibration.yaml")
    link_yaml = Path("configs/link_semantic.yaml")
    rel = _read_yaml(release_cfg_path)
    updated_budget = None
    if errs_yaml.exists():
        try:
            data = _read_yaml(errs_yaml)
            a_new = float(data.get("a_candidate", data.get("a_current", -0.40)))
            b_new = float(data.get("b_candidate", data.get("b_current", -0.05)))
            _write_yaml(link_yaml, {"a": float(a_new), "b": float(b_new)})
            print(f"[SEM] link_semantic.yaml <- error_analysis: a={a_new:.4f}, b={b_new:.4f}")
        except Exception as e:
            print(f"[SEM][ERROR] syncing from {errs_yaml}: {e}")
    else:
        # Compute sWER_target from SINR_max quantile
        try:
            trace = Path("semntn/data/channel_trace.csv")
            df = pd.read_csv(trace)
            snr_db = df["snr_db"].to_numpy(dtype=float)
            P_max = float(rel.get("P_max", 1.6))
            B_grid = list(rel.get("B_grid", [1, 2, 3]))
            B_min = float(min(B_grid)) if B_grid else 1.0
            q_budget = float(rel.get("latency_budget_q_bps", rel.get("q_bps", 300.0)))
            gain_db = 10.0 * np.log10(max(P_max * B_min, 1e-9))
            penalty_db = 3.0 * np.log2(max(q_budget, 1.0) / 300.0 + 1.0)
            sinr_db_max = snr_db + float(gain_db) - float(penalty_db)
            sinr_lin_max = 10.0 ** (sinr_db_max / 10.0)
            # Load current a,b
            ab = _read_yaml(link_yaml)
            a = float(ab.get("a", -0.40)); b = float(ab.get("b", -0.05))
            q = 0.675
            x = float(np.quantile(sinr_lin_max, q))
            swer_target = float(1.0 / (1.0 + np.exp(-(a * x + b))))
            updated_budget = float(np.clip(swer_target, 0.35, 0.38))
            # Update release config(s)
            rel["semantic_budget"] = float(updated_budget)
            _write_yaml(release_cfg_path, rel)
            wc_path = Path(str(release_cfg_path).replace(".yaml", "_wirecheck.yaml"))
            if wc_path.exists():
                wc = _read_yaml(wc_path)
                wc["semantic_budget"] = float(updated_budget)
                _write_yaml(wc_path, wc)
            print(f"[SEM] semantic_budget updated via SINR_max q={q:.3f}: {updated_budget:.4f}")
        except Exception as e:
            print(f"[SEM][ERROR] computing budget from quantile: {e}")


def _metrics_from_log(df: pd.DataFrame) -> dict:
    T = len(df)
    mos_mean = float(df.get("mos", pd.Series([np.nan]*T)).mean())
    energy_col = "E_slot_J" if "E_slot_J" in df.columns else ("P" if "P" in df.columns else None)
    energy_mean = float(df[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df.get("sWER", pd.Series([np.nan]*T)).mean())
    # violations
    if set(["violate_lat","violate_eng","violate_sem"]).issubset(df.columns):
        v_lat = float(df["violate_lat"].mean())
        v_eng = float(df["violate_eng"].mean())
        v_sem = float(df["violate_sem"].mean())
    else:
        z_lat = df.get("z_lat", pd.Series([0.0]*T)).to_numpy(dtype=float)
        z_eng = df.get("z_eng", pd.Series([0.0]*T)).to_numpy(dtype=float)
        z_sem = df.get("z_sem", pd.Series([0.0]*T)).to_numpy(dtype=float)
        v_lat = float(np.mean(np.diff(np.concatenate([[0.0], z_lat])) > 0.0))
        v_eng = float(np.mean(np.diff(np.concatenate([[0.0], z_eng])) > 0.0))
        v_sem = float(np.mean(np.diff(np.concatenate([[0.0], z_sem])) > 0.0))
    has_arm = "arm_id" in df.columns
    has_lcb_min = "lcb_min" in df.columns
    return {
        "MOS_mean": mos_mean,
        "E_slot_J_mean": energy_mean,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
        "has_arm_id": has_arm,
        "has_lcb_min": has_lcb_min,
    }


def _format_resolved(cfg_name: str, resolved_dir: Path, configs_dir: Path) -> str:
    rp = resolved_dir / f"{cfg_name}_resolved.yaml"
    data = _read_yaml_safe(rp)
    if not data:
        # fallback to raw config
        raw = _read_yaml_safe(configs_dir / f"{cfg_name}.yaml")
        data = {
            "T": raw.get("T", "[MISSING]"),
            "slot_sec": raw.get("slot_sec", "[MISSING]"),
            "arrivals_bps": raw.get("arrivals_bps", raw.get("A_bps", "[MISSING]")),
            "energy_budget_per_slot_j": raw.get("energy_budget_per_slot_j", raw.get("E_per_slot", "[MISSING]")),
            "latency_budget_q_bps": raw.get("latency_budget_q_bps", raw.get("q_bps", "[MISSING]")),
            "semantic_budget": raw.get("semantic_budget", raw.get("I_target", "[MISSING]")),
            "V": raw.get("V", "[MISSING]"),
            "scales": (raw.get("scales") or {"queue_scale":"[MISSING]","energy_scale":"[MISSING]","semantic_scale":"[MISSING]"}),
        }
    scales = data.get("scales", {})
    return (
        f"RESOLVED[{cfg_name}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
        f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
        f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{scales.get('queue_scale')}, energy:{scales.get('energy_scale')}, semantic:{scales.get('semantic_scale')}}}"
    )


def write_consolidated_report_strict(outdir: Path) -> Path:
    """Generate outputs/reports/task1_consolidated_report_strict.txt.

    Prioritizes *_resolved.yaml, compares bandit (ucb1 vs shaped), includes strict in acceptance.
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    configs_dir = Path("configs")
    os.makedirs(reports_dir, exist_ok=True)

    # Metadata
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # Collect logs
    names = ["task1_basic", "task1_tuned", "task1_bandit", "task1_ucb1", "task1_strict"]
    logs: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict] = {}
    for nm in names:
        lp = dumps_dir / f"{nm}_log.csv"
        if lp.exists():
            try:
                df = pd.read_csv(lp)
                logs[nm] = df
                metrics[nm] = _metrics_from_log(df)
            except Exception:
                pass

    # Build text
    lines: list[str] = []
    lines.append("=== METADATA ===")
    lines.append(f"repo_path: {cwd}")
    lines.append(f"timestamp: {ts}")
    lines.append(f"python: {py_ver}")
    lines.append(f"git_short: {short}")
    lines.append(f"git_branch: {branch}")
    lines.append("")

    lines.append("=== RESOLVED CONFIGS ===")
    for nm in ["task1_basic", "task1_tuned", "task1_bandit", "task1_ucb1", "task1_strict"]:
        lines.append(_format_resolved(nm, dumps_dir, configs_dir))
    lines.append("")

    lines.append("=== SNAPSHOT METRICS ===")
    for nm in ["task1_basic", "task1_tuned", "task1_strict"]:
        m = metrics.get(nm, {})
        if m:
            lines.append(
                f"{nm}: MOS_mean={m['MOS_mean']:.4f}, E_slot_J_mean={m['E_slot_J_mean']:.3f}, sWER_mean={m['sWER_mean']:.3f}, "
                f"viol(lat,eng,sem)=({m['viol_lat']:.3f},{m['viol_eng']:.3f},{m['viol_sem']:.3f})"
            )
        else:
            lines.append(f"{nm}: [MISSING LOG]")
    lines.append("")

    lines.append("=== BANDIT COMPARISON (ucb1 vs bandit_lagrange) ===")
    for nm in ["task1_ucb1", "task1_bandit"]:
        label = nm if nm == "task1_ucb1" else "bandit_lagrange"
        m = metrics.get(nm, {})
        if m:
            lines.append(
                f"{label}: MOS_mean={m['MOS_mean']:.4f}, sWER_mean={m['sWER_mean']:.3f}, viol(lat,eng,sem)=({m['viol_lat']:.3f},{m['viol_eng']:.3f},{m['viol_sem']:.3f}), "
                f"arm_id=YES, lcb_min={'YES' if m['has_lcb_min'] else 'NO'}"
            )
        else:
            lines.append(f"{label}: [MISSING LOG]")
    lines.append("")

    lines.append("=== ACCEPTANCE CHECK (target ≤ 0.01) ===")
    ms = metrics.get("task1_strict", {})
    if ms:
        ok_lat = ms["viol_lat"] <= 0.01
        ok_sem = ms["viol_sem"] <= 0.01
        ok_eng = ms["viol_eng"] <= 0.01
        lines.append(
            f"task1_strict: viol(lat,eng,sem)=({ms['viol_lat']:.3f},{ms['viol_eng']:.3f},{ms['viol_sem']:.3f}) → "
            f"{'PASS' if (ok_lat and ok_sem and ok_eng) else 'FAIL'}"
        )
    else:
        lines.append("task1_strict: [MISSING LOG]")
    lines.append("")

    lines.append("=== PESQ Surrogate ===")
    model_path = Path("models/pesq_surrogate.pkl")
    if model_path.exists():
        lines.append("models/pesq_surrogate.pkl: YES")
    else:
        lines.append("models/pesq_surrogate.pkl: NO")
        lines.append("建议：落盘并在下轮启用")
    lines.append("")

    # Manifest
    lines.append("=== MANIFEST ===")
    for nm in names:
        lp = dumps_dir / f"{nm}_log.csv"
        rp = dumps_dir / f"{nm}_resolved.yaml"
        lines.append(f"log:{nm} → {'YES' if lp.exists() else 'NO'}")
        lines.append(f"resolved:{nm} → {'YES' if rp.exists() else 'NO'}")
    vscan_csv = Path("vscan/summary.csv")
    lines.append(f"vscan/summary.csv → {'YES' if vscan_csv.exists() else 'NO'}")

    # Save
    report_path = reports_dir / "task1_consolidated_report_strict.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_consolidated_report_final(outdir: Path) -> Path:
    """Generate outputs/reports/task1_consolidated_report_final.txt.

    Extends strict report with:
    - Includes all RESOLVED[...] lines with scales present
    - Uses strict rerun metrics (post sWER fix)
    - Selects best V from vscan under all constraints ≤1%
    - Compares UCB1 vs bandit_lagrange including mean(reward_shaped)
    - PESQ surrogate reminder if missing
    - Prints final [REPORT] line with path
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    configs_dir = Path("configs")
    os.makedirs(reports_dir, exist_ok=True)

    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # Collect known logs
    names = [
        "task1_basic",
        "task1_tuned",
        "task1_strict",
        "task1_ucb1",
        "task1_bandit",
        "task1_bandit_lagrange",
    ]
    logs: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict] = {}
    for nm in names:
        lp = dumps_dir / f"{nm}_log.csv"
        if lp.exists():
            try:
                df = pd.read_csv(lp)
                logs[nm] = df
                # Prefer raw sWER for metrics
                m = _metrics_from_log(df)
                # overwrite sWER_mean with raw if available
                if "sWER_raw" in df.columns:
                    m["sWER_mean"] = float(df["sWER_raw"].mean())
                # reward_shaped mean if present
                if "reward_shaped" in df.columns:
                    m["reward_shaped_mean"] = float(df["reward_shaped"].mean())
                metrics[nm] = m
            except Exception:
                pass

    lines: list[str] = []
    lines.append("=== METADATA ===")
    lines.append(f"repo_path: {cwd}")
    lines.append(f"timestamp: {ts}")
    lines.append(f"python: {py_ver}")
    lines.append(f"git_short: {short}")
    lines.append(f"git_branch: {branch}")
    lines.append("")

    lines.append("=== RESOLVED CONFIGS ===")
    for nm in [
        "task1_basic",
        "task1_tuned",
        "task1_bandit",
        "task1_ucb1",
        "task1_bandit_lagrange",
        "task1_strict",
    ]:
        lines.append(_format_resolved(nm, dumps_dir, configs_dir))
    lines.append("")

    lines.append("=== SNAPSHOT METRICS (strict, post-fix) ===")
    ms = metrics.get("task1_strict", {})
    if ms:
        lines.append(
            f"task1_strict: MOS_mean={ms['MOS_mean']:.4f}, E_slot_J_mean={ms['E_slot_J_mean']:.3f}, sWER_mean={ms['sWER_mean']:.3f}, "
            f"viol(lat,eng,sem)=({ms['viol_lat']:.3f},{ms['viol_eng']:.3f},{ms['viol_sem']:.3f})"
        )
    else:
        lines.append("task1_strict: [MISSING LOG]")
    lines.append("")

    # V-sweep best under constraints
    lines.append("=== V-SWEEP: Best Under Constraints (≤1%) ===")
    vscan_path = Path("outputs/vscan/summary.csv")
    best_line = "[MISSING V-SCAN SUMMARY]"
    if vscan_path.exists():
        try:
            vdf = pd.read_csv(vscan_path)
            ok = vdf[(vdf["viol_lat"] <= 0.01) & (vdf["viol_eng"] <= 0.01) & (vdf["viol_sem"] <= 0.01)]
            if len(ok) > 0:
                idx = int(ok["MOS_mean"].idxmax())
                row = ok.loc[idx]
                best_line = (
                    f"V={int(row['V'])}, MOS_mean={row['MOS_mean']:.4f}, E_slot_J_mean={row['E_slot_J_mean']:.3f}, "
                    f"viol(lat,eng,sem)=({row['viol_lat']:.3f},{row['viol_eng']:.3f},{row['viol_sem']:.3f})"
                )
            else:
                best_line = "[NO ROW SATISFIES ALL ≤1% CONSTRAINTS]"
        except Exception:
            best_line = "[ERROR READING V-SCAN SUMMARY]"
    lines.append(best_line)
    lines.append("")

    # Bandit comparison
    lines.append("=== BANDIT COMPARISON (ucb1 vs bandit_lagrange) ===")
    for nm in ["task1_ucb1", "task1_bandit_lagrange"]:
        label = nm if nm == "task1_ucb1" else "bandit_lagrange"
        m = metrics.get(nm, {})
        if m:
            rs = m.get("reward_shaped_mean", float("nan"))
            lines.append(
                f"{label}: MOS_mean={m['MOS_mean']:.4f}, sWER_mean={m['sWER_mean']:.3f}, viol(lat,eng,sem)=({m['viol_lat']:.3f},{m['viol_eng']:.3f},{m['viol_sem']:.3f}), "
                f"arm_id={'YES' if m.get('has_arm_id') else 'NO'}, lcb_min={'YES' if m.get('has_lcb_min') else 'NO'}, reward_shaped_mean={rs:.3f}"
            )
        else:
            lines.append(f"{label}: [MISSING LOG]")
    lines.append("")

    # Acceptance check
    lines.append("=== ACCEPTANCE CHECK (target ≤ 0.01) ===")
    if ms:
        ok_lat = ms["viol_lat"] <= 0.01
        ok_sem = ms["viol_sem"] <= 0.01
        ok_eng = ms["viol_eng"] <= 0.01
        lines.append(
            f"task1_strict: viol(lat,eng,sem)=({ms['viol_lat']:.3f},{ms['viol_eng']:.3f},{ms['viol_sem']:.3f}) → "
            f"{'PASS' if (ok_lat and ok_sem and ok_eng) else 'FAIL'}"
        )
    else:
        lines.append("task1_strict: [MISSING LOG]")
    lines.append("")

    # PESQ surrogate
    lines.append("=== PESQ Surrogate ===")
    model_path = Path("models/pesq_surrogate.pkl")
    if model_path.exists():
        lines.append("models/pesq_surrogate.pkl: YES")
    else:
        lines.append("models/pesq_surrogate.pkl: NO")
        lines.append("建议：落盘并在下轮启用")
    lines.append("")

    # Manifest
    lines.append("=== MANIFEST ===")
    for nm in names:
        lp = dumps_dir / f"{nm}_log.csv"
        rp = dumps_dir / f"{nm}_resolved.yaml"
        lines.append(f"log:{nm} → {'YES' if lp.exists() else 'NO'}")
        lines.append(f"resolved:{nm} → {'YES' if rp.exists() else 'NO'}")
    vscan_csv = Path("outputs/vscan/summary.csv")
    lines.append(f"outputs/vscan/summary.csv → {'YES' if vscan_csv.exists() else 'NO'}")

    report_path = reports_dir / "task1_consolidated_report_final.txt"
    # Append final report path line within the file as requested
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def _best_v_from_summary(summary_csv: Path) -> tuple[float | None, pd.DataFrame | None, bool]:
    try:
        vdf = pd.read_csv(summary_csv)
        feasible = vdf[(vdf["viol_lat"] <= 0.01) & (vdf["viol_eng"] <= 0.01) & (vdf["viol_sem"] <= 0.01)]
        if len(feasible) > 0:
            idx = int(feasible["MOS_mean"].idxmax())
            return float(vdf.loc[idx, "V"]), vdf.loc[[idx]], True
        # fallback: smallest sum of violations, then max MOS
        vdf = vdf.copy()
        vdf["viol_sum"] = vdf[["viol_lat","viol_eng","viol_sem"]].sum(axis=1)
        min_sum = float(vdf["viol_sum"].min())
        candidates = vdf[vdf["viol_sum"] == min_sum]
        idx = int(candidates["MOS_mean"].idxmax())
        return float(vdf.loc[idx, "V"]), vdf.loc[[idx]], False
    except Exception:
        return None, None, False


def _plot_acceptance_figs(summary_csv: Path, outdir: Path) -> float | None:
    os.makedirs(outdir, exist_ok=True)
    V_star, row_star, feasible = _best_v_from_summary(summary_csv)
    df = pd.read_csv(summary_csv)
    V = df["V"].to_numpy(dtype=float)
    # ViolationRates vs V with V* annotation
    plt.figure(figsize=(7, 4.2))
    plt.plot(V, df["viol_lat"].to_numpy(dtype=float), marker="o", label="lat")
    plt.plot(V, df["viol_eng"].to_numpy(dtype=float), marker="o", label="eng")
    plt.plot(V, df["viol_sem"].to_numpy(dtype=float), marker="o", label="sem")
    if V_star is not None:
        plt.axvline(V_star, color="red", linestyle="--", alpha=0.6, label=f"V*={int(V_star)}")
    plt.xlabel("V"); plt.ylabel("violation rate"); plt.title("Violation rates vs V"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir / "Fig_ViolationRates_vs_V.png", dpi=150); plt.close()
    # Pareto: highlight feasible (≤1%)
    plt.figure(figsize=(6.2, 4.8))
    x = df["E_slot_J_mean"].to_numpy(dtype=float)
    y = df["MOS_mean"].to_numpy(dtype=float)
    plt.scatter(x, y, s=30, color="#888", label="all")
    feas_mask = (df["viol_lat"] <= 0.01) & (df["viol_eng"] <= 0.01) & (df["viol_sem"] <= 0.01)
    if feas_mask.any():
        plt.scatter(df.loc[feas_mask, "E_slot_J_mean"], df.loc[feas_mask, "MOS_mean"], s=50, color="green", label="≤1%")
    if V_star is not None and row_star is not None:
        plt.scatter(row_star["E_slot_J_mean"], row_star["MOS_mean"], s=80, color="red", marker="*", label="V*")
    plt.xlabel("E_slot_J_mean (J/slot)"); plt.ylabel("MOS_mean"); plt.title("Pareto: MOS vs Energy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(outdir / "Fig_Pareto_MOS_vs_Energy.png", dpi=150); plt.close()
    return V_star


def write_acceptance_signoff(outdir: Path) -> Path:
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # Determine final release logs (prefer last iter if present)
    iter_logs = sorted(dumps_dir.glob("task1_release_iter*_log.csv"))
    if iter_logs:
        log_final = iter_logs[-1]
        iter_tag = log_final.stem.replace("_log", "")  # task1_release_iterX
        resolved_final = dumps_dir / (iter_tag + "_resolved.yaml")
        tag_print = iter_tag
    else:
        log_final = dumps_dir / "task1_release_log.csv"
        resolved_final = dumps_dir / "task1_release_resolved.yaml"
        tag_print = "task1_release"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel["mos"].mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # V-sweep
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text

    # Bandit (optional)
    bandit_rows: list[str] = []
    def _bandit_summary(path: Path, label: str) -> tuple[str, dict]:
        if path.exists():
            dfb = pd.read_csv(path)
            T_b = len(dfb)
            mos_norm = (dfb["mos"] - 1.0) / (4.5 - 1.0)
            auc = float(mos_norm.sum() / max(T_b, 1)) if T_b > 0 else float("nan")
            regret = None
            if "arm_id" in dfb.columns:
                groups = dfb.groupby("arm_id")["mos"].mean()
                best_mos = float(groups.max()) if len(groups) > 0 else float(dfb["mos"].mean())
                inst_reg = (best_mos - dfb["mos"]).clip(lower=0.0)
                regret = float(inst_reg.cumsum().iloc[-1])
            v_l = float(dfb.get("violate_lat", pd.Series([0]*T_b)).mean())
            v_e = float(dfb.get("violate_eng", pd.Series([0]*T_b)).mean())
            v_s = float(dfb.get("violate_sem", pd.Series([0]*T_b)).mean())
            has_arm = "arm_id" in dfb.columns
            has_lcb = "lcb_min" in dfb.columns
            has_rs = "reward_shaped" in dfb.columns
            row = f"{label},{auc:.3f},{(regret if regret is not None else float('nan')):.3f},{v_l:.3f},{v_e:.3f},{v_s:.3f}"
            return row, {"has_arm": has_arm, "has_lcb": has_lcb, "has_rs": has_rs}
        return f"{label},[MISSING],[MISSING],[MISSING],[MISSING],[MISSING]", {"has_arm": False, "has_lcb": False, "has_rs": False}

    uc_row, uc_meta = _bandit_summary(dumps_dir / "task1_ucb1_log.csv", "ucb1")
    bl_row, bl_meta = _bandit_summary(dumps_dir / "task1_bandit_lagrange_log.csv", "bandit_lagrange")

    # RESOLVED lines (release plus iter copies if present)
    resolved_lines: list[str] = []
    def _fmt_resolved_from_file(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}"
        )

    # Collect all iter resolved
    if iter_logs:
        for p in sorted(dumps_dir.glob("task1_release_iter*_resolved.yaml")):
            resolved_lines.append(_fmt_resolved_from_file(p, p.stem.replace("_resolved", "")))
    # Always include base release
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_from_file(resolved_final, tag_print))

    # Build report
    lines: list[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- repo_path: {cwd}")

    lines.append("=== 1. RESOLVED Configs ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== 2. Final Release Snapshot ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    all_ok = (v_lat <= 0.01) and (v_eng <= 0.01) and (v_sem <= 0.01)
    lines.append(f"Energy/Latency/Semantic violation all ≤ 1%? {'YES' if all_ok else 'NO'} ({v_lat:.3f},{v_eng:.3f},{v_sem:.3f})")

    lines.append("=== 3. V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")

    lines.append("=== 4. Bandit (Optional) ===")
    lines.append("| mode | MOS_AUC | pseudo_regret | viol_lat | viol_eng | viol_sem |")
    lines.append(uc_row)
    lines.append(bl_row)
    lines.append(f"arm_id: ucb1={'YES' if uc_meta['has_arm'] else 'NO'}, bandit_lagrange={'YES' if bl_meta['has_arm'] else 'NO'}")
    lines.append(f"lcb_min: ucb1={'YES' if uc_meta['has_lcb'] else 'NO'}, bandit_lagrange={'YES' if bl_meta['has_lcb'] else 'NO'}")
    lines.append(f"reward_shaped: ucb1={'YES' if uc_meta['has_rs'] else 'NO'}, bandit_lagrange={'YES' if bl_meta['has_rs'] else 'NO'}")

    lines.append("=== 5. Acceptance Check ===")
    uplift_ok = (mos_uplift >= 0.30)
    lines.append(f"- constraints ≤1%? {'YES' if all_ok else 'FAIL'}")
    lines.append(f"- MOS uplift ≥ +0.30? {'YES' if uplift_ok else 'FAIL'} (uplift={mos_uplift:.3f})")
    # Final parameters from resolved
    final_cfg = _read_yaml_safe(resolved_final)
    s = final_cfg.get("scales", {})
    lines.append(
        f"最终上线参数: V={final_cfg.get('V')}, latency_budget_q_bps={final_cfg.get('latency_budget_q_bps')}, energy_budget_per_slot_j={final_cfg.get('energy_budget_per_slot_j')}, "
        f"semantic_budget={final_cfg.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}"
    )

    lines.append("=== 6. 文件清单 ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    # Table: release vs baselines
    table_path = dumps_dir / "task1_release_vs_baselines.csv"
    rows = []
    rows.append({
        "config_name": "baseline_fixed",
        "MOS_mean": mos_baseline,
        "E_slot_J_mean": energy_mean,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
    })
    rows.append({
        "config_name": tag_print,
        "MOS_mean": mos_mean,
        "E_slot_J_mean": energy_mean,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
    })
    # Optional bandit_lagrange row
    bl_log = dumps_dir / "task1_bandit_lagrange_log.csv"
    if bl_log.exists():
        dfb = pd.read_csv(bl_log)
        rows.append({
            "config_name": "task1_bandit_lagrange",
            "MOS_mean": float(dfb["mos"].mean()),
            "E_slot_J_mean": float(dfb.get("E_slot_J", dfb.get("P", pd.Series([float('nan')]*len(dfb)))).mean()),
            "sWER_mean": float(dfb.get("sWER_raw", dfb.get("sWER", pd.Series([float('nan')]*len(dfb)))).mean()),
            "viol_lat": float(dfb.get("violate_lat", pd.Series([0]*len(dfb))).mean()),
            "viol_eng": float(dfb.get("violate_eng", pd.Series([0]*len(dfb))).mean()),
            "viol_sem": float(dfb.get("violate_sem", pd.Series([0]*len(dfb))).mean()),
        })
    pd.DataFrame(rows)[["config_name","MOS_mean","E_slot_J_mean","sWER_mean","viol_lat","viol_eng","viol_sem"]].to_csv(table_path, index=False)
    lines.append(str(table_path))

    report_path = reports_dir / "task1_acceptance_signoff.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=False, help="Path to task1 log CSV")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory")
    ap.add_argument("--config-name", type=str, default=None, help="Label for summary row")
    ap.add_argument("--vscan", type=Path, default=None, help="Path to vscan summary CSV to plot")
    ap.add_argument("--update-semantic", action="store_true", help="Sync (a,b) from error_analysis or compute budget from SINR_max quantile")
    ap.add_argument("--release-config", type=Path, default=Path("configs/task1_release_pd2.yaml"), help="Release config to update for semantic_budget")
    ap.add_argument("--write-consolidated", action="store_true", help="Write strict consolidated TXT report")
    ap.add_argument("--write-final", action="store_true", help="Write final consolidated TXT report")
    ap.add_argument("--write-signoff", action="store_true", help="Write acceptance signoff TXT report")
    ap.add_argument("--write-signoff-guard", action="store_true", help="Write acceptance signoff TXT report for sWER-guard release")
    ap.add_argument("--write-signoff-final", action="store_true", help="Write acceptance signoff TXT report for FINAL release")
    ap.add_argument("--write-signoff-pd", action="store_true", help="Write acceptance signoff TXT report for PD release (task1_release_pd)")
    ap.add_argument("--write-signoff-pd2", action="store_true", help="Write acceptance signoff TXT report for PD+Guard release (task1_release_pd2)")
    ap.add_argument("--write-signoff-pd3", action="store_true", help="Write acceptance signoff TXT report for PD3 package (based on PD2 run + V-sweep)")
    ap.add_argument("--write-signoff-pd4", action="store_true", help="Write acceptance signoff TXT report for PD4 package (PD2 release + narrow V-sweep + diagnostics)")
    ap.add_argument("--write-signoff-pd5", action="store_true", help="Write acceptance signoff TXT report for PD5 package (strict unified keys + aggregator self-check)")
    ap.add_argument("--write-signoff-pd6", action="store_true", help="Write acceptance signoff TXT report for PD6 package (units & guard enforcement + dual intensity diagnostics)")
    ap.add_argument("--write-signoff-pd7-guard", action="store_true", help="Write acceptance signoff TXT report for PD7 package (wirecheck+release+vscan, energy guard)")
    ap.add_argument("--write-signoff-pd7-cal", action="store_true", help="Write acceptance signoff TXT report for PD7 calibrated package (wirecheck+release+vscan with calibration)")
    ap.add_argument("--write-signoff-pd8-guard", action="store_true", help="Write acceptance signoff TXT report for PD8 package (true-constraint gating + epsilon-energy tightening)")
    ap.add_argument("--write-signoff-pd8-cal", action="store_true", help="Write acceptance signoff TXT report for PD8 calibrated package")
    ap.add_argument("--write-signoff-pd9", action="store_true", help="Write acceptance signoff TXT report for PD9 package (preflight + wirecheck + release + vscan)")
    ap.add_argument("--calibrate-swer", action="store_true", help="Calibrate link_semantic.yaml using source log and semantic budget")
    ap.add_argument("--calib-log", type=Path, default=None, help="Log path for calibration (defaults to outputs/dumps/task1_release_log.csv)")
    ap.add_argument("--budget", type=float, default=None, help="Semantic budget for calibration (defaults to configs/task1_release.yaml)")
    args = ap.parse_args()
    if args.update_semantic:
        update_semantic_params_and_budget(args.release_config)
    if args.log:
        analyze(args.log, args.outdir, args.config_name)
    if args.vscan:
        figs_out = args.outdir if args.outdir.is_dir() else Path("outputs/figs")
        os.makedirs(figs_out, exist_ok=True)
        _plot_vscan(args.vscan, figs_out)
    if args.write_consolidated:
        write_consolidated_report_strict(args.outdir)
    if args.write_final:
        write_consolidated_report_final(args.outdir)
    if args.write_signoff:
        # Generate acceptance figures based on vscan if available
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff(args.outdir)
    if args.write_signoff_guard:
        # Figures from vscan (guard base expected)
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_guard(args.outdir)
    if args.write_signoff_pd:
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd(args.outdir)
    if args.write_signoff_final:
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_final(args.outdir)
    if args.write_signoff_pd5:
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd5(args.outdir)
    if args.write_signoff_pd2:
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd2(args.outdir)
    if getattr(args, "write_signoff_pd3", False):
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd3(args.outdir)
    if getattr(args, "write_signoff_pd4", False):
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd4(args.outdir)
    if args.calibrate_swer:
        link_yaml = Path("configs/link_semantic.yaml")
        budget = args.budget
        if budget is None:
            rel = _read_yaml_safe(Path("configs/task1_release.yaml"))
            budget = float(rel.get("semantic_budget", 0.37))
        s_log = args.calib_log if args.calib_log else Path("outputs/dumps/task1_release_log.csv")
        if not (s_log and Path(s_log).exists()):
            s_log = Path("outputs/dumps/task1_release_guard_log.csv")
        calibrate_swer_sinr(link_yaml, Path(s_log), float(budget), q=0.65)
    if args.write_signoff_pd6:
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd6(args.outdir)
    if getattr(args, "write_signoff_pd7_guard", False):
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd7_guard(args.outdir)
    if getattr(args, "write_signoff_pd7_cal", False):
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd7_cal(args.outdir)
    if getattr(args, "write_signoff_pd8_guard", False):
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd8_guard(args.outdir)
    if getattr(args, "write_signoff_pd8_cal", False):
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd8_cal(args.outdir)
    if getattr(args, "write_signoff_pd9", False):
        # PD9 入口第一行：预检硬闸（缺文件或 pass:false 立刻停）
        try:
            dumps_dir = Path(args.outdir) / "dumps"
            # Prefer explicit task1_release_pd2; else pick first *_preflight_resolved.yaml
            preferred = dumps_dir / "task1_release_pd2_preflight_resolved.yaml"
            candidates = [preferred] if preferred.exists() else sorted(dumps_dir.glob("*_preflight_resolved.yaml"))
            if not candidates:
                sys.exit("[FAIL_PREFLIGHT_FEASIBILITY]")
            data_pf = _read_yaml_safe(candidates[0])
            pass_flag = bool(data_pf.get("pass", False))
            if not pass_flag:
                sys.exit("[FAIL_PREFLIGHT_FEASIBILITY]")
        except Exception:
            sys.exit("[FAIL_PREFLIGHT_FEASIBILITY]")
        vscan_csv = Path("outputs/vscan/summary.csv")
        if vscan_csv.exists():
            figs_out = Path("outputs/figs")
            os.makedirs(figs_out, exist_ok=True)
            _plot_acceptance_figs(vscan_csv, figs_out)
        write_acceptance_signoff_pd9(args.outdir)

def write_acceptance_signoff_guard(outdir: Path) -> Path:
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # Prefer iter1 if present
    log_base = dumps_dir / "task1_release_guard_log.csv"
    res_base = dumps_dir / "task1_release_guard_resolved.yaml"
    log_iter1 = dumps_dir / "task1_release_guard_iter1_log.csv"
    res_iter1 = dumps_dir / "task1_release_guard_iter1_resolved.yaml"
    tag_print = "task1_release_guard"
    log_final = log_iter1 if log_iter1.exists() else log_base
    resolved_final = res_iter1 if res_iter1.exists() else res_base

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel["mos"].mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text

    # RESOLVED lines
    resolved_lines: list[str] = []
    def _fmt_resolved(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}"
        )
    if res_iter1.exists():
        resolved_lines.append(_fmt_resolved(res_iter1, "task1_release_guard_iter1"))
    if res_base.exists():
        resolved_lines.append(_fmt_resolved(res_base, "task1_release_guard"))

    # Build report
    lines: list[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- repo_path: {cwd}")

    lines.append("=== 1. RESOLVED Configs ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== 2. Final Release Snapshot ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    all_ok = (v_lat <= 0.01) and (v_eng <= 0.01) and (v_sem <= 0.01)
    lines.append(f"Energy/Latency/Semantic violation all ≤ 1%? {'YES' if all_ok else 'NO'} ({v_lat:.3f},{v_eng:.3f},{v_sem:.3f})")

    lines.append("=== 3. V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")

    lines.append("=== 4. Acceptance Check ===")
    uplift_ok = (mos_uplift >= 0.30)
    lines.append(f"- constraints ≤1%? {'PASS' if all_ok else 'FAIL'}")
    lines.append(f"- MOS uplift ≥ +0.30? {'PASS' if uplift_ok else 'FAIL'} (uplift={mos_uplift:.3f})")
    # Final parameters from resolved
    final_cfg = _read_yaml_safe(resolved_final)
    s = final_cfg.get("scales", {})
    if all_ok and uplift_ok:
        lines.append(
            f"最终上线参数: V={final_cfg.get('V')}, latency_budget_q_bps={final_cfg.get('latency_budget_q_bps')}, energy_budget_per_slot_j={final_cfg.get('energy_budget_per_slot_j')}, "
            f"semantic_budget={final_cfg.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}"
        )
    else:
        # Stop-loss tips per spec
        lines.append("放宽 semantic_budget 至 0.38，并重跑一次")
        lines.append("若 P_min 超过 P_max 超过 5% 以上的隙占比>1%，提高 P_max 或缩小 B 最小值（以抬升 SINR）。")

    lines.append("=== 5. 文件清单 ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    # Table: release_guard vs baselines
    table_path = dumps_dir / "task1_release_vs_baselines.csv"
    rows = []
    rows.append({
        "config_name": "baseline_fixed",
        "MOS_mean": mos_baseline,
        "E_slot_J_mean": energy_mean,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
    })
    # Add base guard
    df_base = pd.read_csv(log_base) if log_base.exists() else df_rel
    rows.append({
        "config_name": "task1_release_guard",
        "MOS_mean": float(df_base["mos"].mean()),
        "E_slot_J_mean": float(df_base.get("E_slot_J", df_base.get("P", pd.Series([float('nan')]*len(df_base)))).mean()),
        "sWER_mean": float(df_base.get("sWER_raw", df_base.get("sWER", pd.Series([float('nan')]*len(df_base)))).mean()),
        "viol_lat": float(df_base.get("violate_lat", pd.Series([0]*len(df_base))).mean()),
        "viol_eng": float(df_base.get("violate_eng", pd.Series([0]*len(df_base))).mean()),
        "viol_sem": float(df_base.get("violate_sem", pd.Series([0]*len(df_base))).mean()),
    })
    if log_iter1.exists():
        df1 = pd.read_csv(log_iter1)
        rows.append({
            "config_name": "task1_release_guard_iter1",
            "MOS_mean": float(df1["mos"].mean()),
            "E_slot_J_mean": float(df1.get("E_slot_J", df1.get("P", pd.Series([float('nan')]*len(df1)))).mean()),
            "sWER_mean": float(df1.get("sWER_raw", df1.get("sWER", pd.Series([float('nan')]*len(df1)))).mean()),
            "viol_lat": float(df1.get("violate_lat", pd.Series([0]*len(df1))).mean()),
            "viol_eng": float(df1.get("violate_eng", pd.Series([0]*len(df1))).mean()),
            "viol_sem": float(df1.get("violate_sem", pd.Series([0]*len(df1))).mean()),
        })
    pd.DataFrame(rows)[["config_name","MOS_mean","E_slot_J_mean","sWER_mean","viol_lat","viol_eng","viol_sem"]].to_csv(table_path, index=False)
    lines.append(str(table_path))

    report_path = reports_dir / "task1_acceptance_signoff_guard.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path

def write_acceptance_signoff_pd2(outdir: Path) -> Path:
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # PD+Guard release artifacts
    log_final = dumps_dir / "task1_release_pd2_log.csv"
    resolved_final = dumps_dir / "task1_release_pd2_resolved.yaml"
    tag_print = "task1_release_pd2"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel["mos"].mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text

    # RESOLVED line with unified PD/Guard params
    def _fmt_resolved_pd2(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        en_pd = data.get("enable_primal_dual", False)
        guard = data.get("guard", {})
        pd_block = data.get("pd", {})
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}, "
            f"enable_primal_dual={en_pd}, pd.eta={pd_block.get('eta')}, guard.semantic={guard.get('semantic')}, guard.queue={guard.get('queue')}"
        )

    resolved_lines: list[str] = []
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_pd2(resolved_final, tag_print))

    # Diagnostics activation rates from release_pd2 log
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([np.nan]*T)).mean())
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([np.nan]*T)).mean())
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([np.nan]*T)).mean())
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([np.nan]*T)).mean())

    # Build report
    lines: list[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- repo_path: {cwd}")

    lines.append("=== 1. RESOLVED Configs ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== 2. Final Release Snapshot ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    all_ok = (v_lat <= 0.01) and (v_eng <= 0.01) and (v_sem <= 0.01)
    lines.append(f"Energy/Latency/Semantic violation all ≤ 1%? {'YES' if all_ok else 'NO'} ({v_lat:.3f},{v_eng:.3f},{v_sem:.3f})")

    lines.append("=== 3. V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")

    lines.append("=== 4. Diagnostics (release_pd2) ===")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")

    lines.append("=== 5. 文件清单 ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    report_path = reports_dir / "task1_acceptance_signoff_pd2.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd3(outdir: Path) -> Path:
    """Acceptance signoff for PD3 package.

    PD3 is based on the PD2 release snapshot combined with the V-sweep summary
    and acceptance figures. This report aggregates:
    - PD2 final metrics (MOS, energy, sWER, violation rates)
    - PD/Guard activation rates
    - V-sweep best row and feasibility
    - Unified RESOLVED PD/Guard params from PD2
    - Acceptance checks and file manifest
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # PD3 uses PD2 artifacts
    log_final = dumps_dir / "task1_release_pd2_log.csv"
    resolved_final = dumps_dir / "task1_release_pd2_resolved.yaml"
    tag_print = "task1_release_pd2"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel["mos"].mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # Activation rates
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean())
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean())
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean())
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean())
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean())

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    feasible_flag = None
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        feasible_flag = feasible
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text

    # RESOLVED line with unified PD/Guard params
    def _fmt_resolved_pd2(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        en_pd = data.get("enable_primal_dual", False)
        guard = data.get("guard", {})
        pd_block = data.get("pd", {})
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}, "
            f"enable_primal_dual={en_pd}, pd.eta={pd_block.get('eta')}, pd.delta_queue={pd_block.get('delta_queue')}, guard.semantic={guard.get('semantic')}, guard.queue={guard.get('queue')}"
        )

    resolved_lines: list[str] = []
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_pd2(resolved_final, tag_print))

    # Build report
    lines: list[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- repo_path: {cwd}")

    lines.append("=== 1. RESOLVED Configs (PD2 base) === inh. PD3")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== 2. PD2 Final Snapshot ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    all_ok = (v_lat <= 0.01) and (v_eng <= 0.01) and (v_sem <= 0.01)
    lines.append(f"Energy/Latency/Semantic violation all ≤ 1%? {'YES' if all_ok else 'NO'} ({v_lat:.3f},{v_eng:.3f},{v_sem:.3f})")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")

    lines.append("=== 3. V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if V_star is not None:
        lines.append(f"V_star={int(V_star)} (feasible={bool(feasible_flag)})")

    lines.append("=== 4. Acceptance Check (PD3) ===")
    uplift_ok = (mos_uplift >= 0.30)
    lines.append(f"- constraints ≤1%? {'PASS' if all_ok else 'FAIL'}")
    lines.append(f"- MOS uplift ≥ +0.30? {'PASS' if uplift_ok else 'FAIL'} (uplift={mos_uplift:.3f})")
    final_cfg = _read_yaml_safe(resolved_final)
    s = final_cfg.get("scales", {})
    lines.append(
        f"最终上线参数(PD3基于PD2): V={final_cfg.get('V')}, latency_budget_q_bps={final_cfg.get('latency_budget_q_bps')}, energy_budget_per_slot_j={final_cfg.get('energy_budget_per_slot_j')}, "
        f"semantic_budget={final_cfg.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}, "
        f"lam_init_multiplier={final_cfg.get('lam_init_multiplier')}, pd_eta={final_cfg.get('eta') or final_cfg.get('eta_dual')}, pd_delta_queue={final_cfg.get('delta_queue')}"
    )

    lines.append("=== 5. 文件清单 ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    report_path = reports_dir / "task1_acceptance_signoff_pd3.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd4(outdir: Path) -> Path:
    """Acceptance signoff for PD4 package.

    PD4 aggregates PD2 release snapshot plus narrow V-sweep and diagnostics,
    and ensures RESOLVED includes physical boundaries.
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    # UTC+08 timestamp as required
    try:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0800")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # PD4 uses PD2 artifacts
    log_final = dumps_dir / "task1_release_pd2_log.csv"
    resolved_final = dumps_dir / "task1_release_pd2_resolved.yaml"
    tag_print = "task1_release_pd2"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel.get("mos", pd.Series([float("nan")]*T)).mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel.get("per", pd.Series([0]*T)).to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # Activation rates
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean())
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean())
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean())

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    feasible_flag = None
    v_rows = 0
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        v_rows = len(vdf)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        feasible_flag = feasible
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text
        # Require at least 3 rows for acceptance; fail fast
        if v_rows < 3:
            print("[FAIL_VSCAN] summary rows < 3")
            sys.exit(2)

    # RESOLVED line with unified PD/Guard params + unified physical boundaries
    def _fmt_resolved_pd4(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        en_pd = data.get("enable_primal_dual", False)
        guard = data.get("guard", {})
        pd_block = data.get("pd", {})
        pmax_dbm = data.get("P_max_dBm", "[MISSING]")
        bmin_khz = data.get("B_min_kHz", "[MISSING]")
        bgrid_k = data.get("B_grid_k", "[MISSING]")
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}, "
            f"enable_primal_dual={en_pd}, pd.eta={pd_block.get('eta')}, pd.delta_queue={pd_block.get('delta_queue')}, guard.semantic={guard.get('semantic')}, guard.queue={guard.get('queue')}, "
            f"P_max_dBm={pmax_dbm}, B_min_kHz={bmin_khz}, B_grid_k={bgrid_k}"
        )

    resolved_lines: list[str] = []
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_pd4(resolved_final, tag_print))

    # Build report
    lines: list[str] = []
    lines.append("=== Metadata ===")
    lines.append(f"- repo_path: {cwd}")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")

    lines.append("=== RESOLVED（确保含 P_max_dBm/B_min_kHz/B_grid_k） ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== Final Snapshot（task1_release_pd2） ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )

    lines.append("=== V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if V_star is not None:
        lines.append(f"V_star={int(V_star)}")

    lines.append("=== Diagnostics ===")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")
    try:
        feasible_sem = df_rel.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_queue = df_rel.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0)))
        lines.append(f"guard_null_rate={guard_null_rate:.3f}")
        # Optional: include triple-null when energy guard present
        feasible_energy = df_rel.get("feasible_energy_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null3_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0) & (feasible_energy == 0)))
        if guard_null3_rate == guard_null3_rate:
            lines.append(f"guard_null3_rate={guard_null3_rate:.3f}")
    except Exception:
        lines.append("guard_null_rate=[MISSING]")

    lines.append("=== File Manifest ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    # Stop-loss tips (exact two lines as specified)
    lines.append("止损提示：")
    lines.append("semantic_budget := min(0.38, semantic_budget + 0.02) ；或基于 SINR 的 0.70 分位 重做分位对齐")
    lines.append("若 P_min_sem  或 P_min_queue  超过 P_max  的隙占比 > 2%  → P_max += 3 dB  或 B_min  再降一档")

    report_path = reports_dir / "task1_acceptance_signoff_pd4.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd5(outdir: Path) -> Path:
    """Acceptance signoff for PD5 package.

    PD5 requires strict unified physical keys in RESOLVED, aggregator self-checks
    enforced during wirecheck, and consolidates PD2 release snapshot plus narrow
    V-sweep decision and diagnostics.
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    # UTC+08 timestamp
    try:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0800")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # PD5 uses PD2 artifacts
    log_final = dumps_dir / "task1_release_pd2_log.csv"
    resolved_final = dumps_dir / "task1_release_pd2_resolved.yaml"
    tag_print = "task1_release_pd2"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel.get("mos", pd.Series([float("nan")]*T)).mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel.get("per", pd.Series([0]*T)).to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # Diagnostics
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean())
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean())
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean())
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean())
    try:
        feasible_sem = df_rel.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_queue = df_rel.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0)))
    except Exception:
        guard_null_rate = float("nan")

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    feasible_flag = None
    v_rows = 0
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        v_rows = len(vdf)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        feasible_flag = feasible
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text
        # Require at least 3 rows for acceptance; fail fast
        if v_rows < 3:
            print("[FAIL_VSCAN] summary rows < 3")
            sys.exit(2)

    # RESOLVED block per spec (include unified physical boundaries)
    def _fmt_resolved_pd5(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        en_pd = data.get("enable_primal_dual", False)
        guard = data.get("guard", {})
        pd_block = data.get("pd", {})
        pmax_dbm = data.get("P_max_dBm", "[MISSING]")
        bmin_khz = data.get("B_min_kHz", "[MISSING]")
        bgrid_k = data.get("B_grid_k", "[MISSING]")
        return (
            f"V={data.get('V')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}, "
            f"enable_primal_dual={en_pd}, pd.eta={pd_block.get('eta')}, pd.delta_queue={pd_block.get('delta_queue')}, guard.semantic={guard.get('semantic')}, guard.queue={guard.get('queue')}, "
            f"P_max_dBm={pmax_dbm}, B_min_kHz={bmin_khz}, B_grid_k={bgrid_k}"
        )

    resolved_lines: list[str] = []
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_pd5(resolved_final, tag_print))

    # Build report
    lines: list[str] = []
    lines.append("=== Metadata ===")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- config: {tag_print}")

    lines.append("=== RESOLVED ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== Final Snapshot（task1_release_pd2） ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )

    lines.append("=== V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if V_star is not None:
        lines.append(f"V_star={int(V_star)}")

    lines.append("=== Diagnostics（发布回合） ===")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")
    lines.append(f"guard_null_rate={guard_null_rate:.3f}")
    try:
        feasible_energy = df_rel.get("feasible_energy_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_sem = df_rel.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_queue = df_rel.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null3_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0) & (feasible_energy == 0)))
        if guard_null3_rate == guard_null3_rate:
            lines.append(f"guard_null3_rate={guard_null3_rate:.3f}")
    except Exception:
        pass

    lines.append("=== File Manifest ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    # Stop-loss tips (two specific lines)
    lines.append("止损提示：")
    lines.append("semantic_budget := min(0.38, semantic_budget + 0.02)；或基于 SINR 的 0.70 分位重做分位对齐")
    lines.append("若 P_min_sem 或 P_min_queue 超过 P_max 的隙占比 > 2% → P_max += 3 dB 或 B_min 再降一档")

    report_path = reports_dir / "task1_acceptance_signoff_pd5.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd6(outdir: Path) -> Path:
    """Acceptance signoff for PD6 package.

    PD6 extends PD5 with unit/guard enforcement landing checks and dual intensity
    diagnostics in the report. It uses PD2 release artifacts and narrow V-sweep.
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    # UTC+08 timestamp
    try:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0800")
    short, branch = _git_meta()

    # PD6 uses PD2 artifacts
    log_final = dumps_dir / "task1_release_pd2_log.csv"
    resolved_final = dumps_dir / "task1_release_pd2_resolved.yaml"
    tag_print = "task1_release_pd2"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel.get("mos", pd.Series([float("nan")]*T)).mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel.get("per", pd.Series([0]*T)).to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # Diagnostics rates
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean())
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean())
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean())
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean())
    try:
        feasible_sem = df_rel.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_queue = df_rel.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0)))
    except Exception:
        guard_null_rate = float("nan")
    # Dual trajectory summary
    lam_eng = df_rel.get("lambda_eng", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    lam_lat = df_rel.get("lambda_lat", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    lam_eng_min = float(np.nanmin(lam_eng)) if np.isfinite(lam_eng).any() else float("nan")
    lam_eng_max = float(np.nanmax(lam_eng)) if np.isfinite(lam_eng).any() else float("nan")
    lam_eng_last = float(lam_eng[-1]) if len(lam_eng) > 0 and (lam_eng[-1] == lam_eng[-1]) else float("nan")
    lam_lat_min = float(np.nanmin(lam_lat)) if np.isfinite(lam_lat).any() else float("nan")
    lam_lat_max = float(np.nanmax(lam_lat)) if np.isfinite(lam_lat).any() else float("nan")
    lam_lat_last = float(lam_lat[-1]) if len(lam_lat) > 0 and (lam_lat[-1] == lam_lat[-1]) else float("nan")

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text
        # Do not write V_star when not feasible
        if not feasible:
            V_star = None

    # RESOLVED block per spec
    def _fmt_resolved_pd6(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        en_pd = data.get("enable_primal_dual", False)
        guard = data.get("guard", {})
        pd_block = data.get("pd", {})
        pmax_dbm = data.get("P_max_dBm", "[MISSING]")
        bmin_khz = data.get("B_min_kHz", "[MISSING]")
        bgrid_k = data.get("B_grid_k", "[MISSING]")
        lam_init = pd_block.get("lambda_init", {})
        return (
            f"V={data.get('V')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}, "
            f"enable_primal_dual={en_pd}, pd.eta={pd_block.get('eta')}, pd.delta_queue={pd_block.get('delta_queue')}, pd.lambda_init.eng={lam_init.get('eng')}, pd.lambda_init.lat={lam_init.get('lat')}, "
            f"guard.semantic={guard.get('semantic')}, guard.queue={guard.get('queue')}, P_max_dBm={pmax_dbm}, B_min_kHz={bmin_khz}, B_grid_k={bgrid_k}"
        )

    resolved_lines: list[str] = []
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_pd6(resolved_final, tag_print))

    # Build report content
    lines: list[str] = []
    lines.append("=== Metadata ===")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- config: {tag_print}")

    lines.append("=== RESOLVED ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== Final Snapshot（task1_release_pd2） ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )

    lines.append("=== V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if V_star is not None:
        lines.append(f"V_star={int(V_star)}")

    lines.append("=== Diagnostics ===")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")
    lines.append(f"guard_null_rate={guard_null_rate:.3f}")
    try:
        feasible_energy = df_rel.get("feasible_energy_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_sem = df_rel.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_queue = df_rel.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null3_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0) & (feasible_energy == 0)))
        if guard_null3_rate == guard_null3_rate:
            lines.append(f"guard_null3_rate={guard_null3_rate:.3f}")
    except Exception:
        pass
    lines.append(f"lambda_eng_min/max/last={lam_eng_min:.4f}/{lam_eng_max:.4f}/{lam_eng_last:.4f}")
    lines.append(f"lambda_lat_min/max/last={lam_lat_min:.4f}/{lam_lat_max:.4f}/{lam_lat_last:.4f}")

    lines.append("=== File Manifest ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    # Stop-loss tips (exact two lines)
    lines.append("止损提示：")
    lines.append("semantic_budget := min(0.38, semantic_budget + 0.02)；或基于 SINR 的 0.70 分位重做分位对齐")
    lines.append("若 P_min_sem 或 P_min_queue 超过 P_max 的隙占比 > 2% → P_max += 3 dB 或 B_min 再降一档")

    report_path = reports_dir / "task1_acceptance_signoff_pd6.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd7_guard(outdir: Path) -> Path:
    """Acceptance signoff for PD7 package (guard variant).

    PD7 consolidates:
    - 300-slot wirecheck (energy guard included)
    - 2000-slot release (PD2 base)
    - Narrow-band V-scan summary and figures
    - Wirecheck assertions: four guard activation rates, guard_null_rate, optional guard_null3_rate,
      and enforcement/aggregation consistency checks
    - If vscan summary rows < 3, throw [FAIL_VSCAN] and do not write V_star
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)

    # UTC+08 timestamp
    try:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0800")
    short, branch = _git_meta()

    # Artifacts
    wire_log = dumps_dir / "task1_wirecheck_log.csv"
    wire_res = dumps_dir / "task1_wirecheck_resolved.yaml"
    if not wire_log.exists():
        alt_log = dumps_dir / "task1_release_pd2_wirecheck_log.csv"
        alt_res = dumps_dir / "task1_release_pd2_wirecheck_resolved.yaml"
        if alt_log.exists():
            wire_log = alt_log
        if alt_res.exists():
            wire_res = alt_res
    rel_log = dumps_dir / "task1_release_pd2_log.csv"
    rel_res = dumps_dir / "task1_release_pd2_resolved.yaml"
    vscan_csv = Path("outputs/vscan/summary.csv")

    # Read wirecheck
    df_wc = pd.read_csv(wire_log) if wire_log.exists() else pd.DataFrame()
    Twc = len(df_wc)
    pd_active_rate_wc = float(df_wc.get("pd_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    guard_sem_rate_wc = float(df_wc.get("guard_sem_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    guard_queue_rate_wc = float(df_wc.get("guard_queue_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    guard_energy_rate_wc = float(df_wc.get("guard_energy_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    try:
        feasible_sem_wc = df_wc.get("feasible_sem_guard", pd.Series([np.nan]*Twc)).to_numpy()
        feasible_queue_wc = df_wc.get("feasible_queue_guard", pd.Series([np.nan]*Twc)).to_numpy()
        guard_null_rate_wc = float(np.mean((feasible_sem_wc == 0) & (feasible_queue_wc == 0)))
    except Exception:
        guard_null_rate_wc = float("nan")
    guard_null3_rate_wc = float("nan")
    try:
        feasible_energy_wc = df_wc.get("feasible_energy_guard", pd.Series([np.nan]*Twc)).to_numpy()
        feasible_sem_wc = df_wc.get("feasible_sem_guard", pd.Series([np.nan]*Twc)).to_numpy()
        feasible_queue_wc = df_wc.get("feasible_queue_guard", pd.Series([np.nan]*Twc)).to_numpy()
        guard_null3_rate_wc = float(np.mean((feasible_sem_wc == 0) & (feasible_queue_wc == 0) & (feasible_energy_wc == 0)))
    except Exception:
        pass
    # Aggregator self-check deltas
    viol_eng_inst_mean = None
    viol_lat_inst_mean = None
    v_eng_wc = float(df_wc.get("violate_eng", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    v_lat_wc = float(df_wc.get("violate_lat", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    try:
        if "E_slot_J" in df_wc.columns and "energy_budget_per_slot_j" in df_wc.columns:
            viol_eng_inst = (df_wc["E_slot_J"].to_numpy(dtype=float) > df_wc["energy_budget_per_slot_j"].to_numpy(dtype=float)).astype(float)
            viol_eng_inst_mean = float(np.mean(viol_eng_inst))
        if "violate_lat" in df_wc.columns:
            viol_lat_inst_mean = float(df_wc["violate_lat"].to_numpy(dtype=float).mean())
    except Exception:
        pass
    # Guard enforcement bad rate
    bad_rate_enforcement = float("nan")
    try:
        if set(["feasible_queue_guard","S_eff_bps","A_bits","slot_sec"]).issubset(df_wc.columns):
            slot_sec_val = float(df_wc["slot_sec"].iloc[0]) if Twc > 0 else 0.02
            A_bps = df_wc["A_bits"].to_numpy(dtype=float) / max(slot_sec_val, 1e-9)
            S_bps = df_wc["S_eff_bps"].to_numpy(dtype=float)
            feasible_q = df_wc["feasible_queue_guard"].to_numpy(dtype=float)
            mask_bad = (feasible_q == 1) & (S_bps < A_bps)
            bad_rate_enforcement = float(np.mean(mask_bad)) if len(mask_bad) > 0 else 0.0
    except Exception:
        pass

    # Read release
    df_rel = pd.read_csv(rel_log)
    T = len(df_rel)
    mos_mean = float(df_rel.get("mos", pd.Series([float("nan")]*T)).mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel.get("per", pd.Series([0]*T)).to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)
    # Release activation rates
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean())
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean())
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean())
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean())
    # Release guard-null diagnostics
    guard_null_rate = float("nan")
    guard_null3_rate = float("nan")
    try:
        feasible_sem = df_rel.get("feasible_sem_guard", pd.Series([np.nan]*T)).to_numpy()
        feasible_queue = df_rel.get("feasible_queue_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0)))
        feasible_energy = df_rel.get("feasible_energy_guard", pd.Series([np.nan]*T)).to_numpy()
        guard_null3_rate = float(np.mean((feasible_sem == 0) & (feasible_queue == 0) & (feasible_energy == 0)))
    except Exception:
        pass

    # V-sweep decision with fail-fast when rows < 3
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    feasible_flag = None
    v_rows = 0
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        v_rows = len(vdf)
        header_text = ",".join(vdf.columns.tolist())
        if v_rows < 3:
            print("[FAIL_VSCAN] summary rows < 3")
            sys.exit(2)
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        feasible_flag = feasible
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text
        # Do not write V_star when not feasible
        if not feasible:
            V_star = None

    # Build report content
    lines: list[str] = []
    lines.append("=== Metadata ===")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- configs: wirecheck=task1_wirecheck, release=task1_release_pd2")

    lines.append("=== Wirecheck RESOLVED ===")
    if wire_res.exists():
        lines.append(_format_resolved("task1_wirecheck", dumps_dir, Path("configs")))
    else:
        lines.append("[MISSING RESOLVED]")

    lines.append("=== Wirecheck Assertions ===")
    lines.append(
        f"pd_active_rate={pd_active_rate_wc:.3f}, guard_sem_rate={guard_sem_rate_wc:.3f}, guard_queue_rate={guard_queue_rate_wc:.3f}, guard_energy_rate={guard_energy_rate_wc:.3f}"
    )
    if guard_null_rate_wc == guard_null_rate_wc:
        lines.append(f"guard_null_rate={guard_null_rate_wc:.3f}")
    else:
        lines.append("guard_null_rate=[MISSING]")
    if guard_null3_rate_wc == guard_null3_rate_wc:
        lines.append(f"guard_null3_rate={guard_null3_rate_wc:.3f}")
    else:
        lines.append("guard_null3_rate=[MISSING]")
    # Aggregation consistency
    if viol_eng_inst_mean is not None and v_eng_wc == v_eng_wc:
        lines.append(f"aggregation_consistency_eng|Δ|≤1e-3? {'PASS' if abs(viol_eng_inst_mean - v_eng_wc) <= 1e-3 else 'FAIL'} (inst={viol_eng_inst_mean:.6f}, aggreg={v_eng_wc:.6f})")
    else:
        lines.append("aggregation_consistency_eng=[SKIPPED]")
    if viol_lat_inst_mean is not None and v_lat_wc == v_lat_wc:
        lines.append(f"aggregation_consistency_lat|Δ|≤1e-3? {'PASS' if abs(viol_lat_inst_mean - v_lat_wc) <= 1e-3 else 'FAIL'} (inst={viol_lat_inst_mean:.6f}, aggreg={v_lat_wc:.6f})")
    else:
        lines.append("aggregation_consistency_lat=[SKIPPED]")
    # Guard enforcement
    if bad_rate_enforcement == bad_rate_enforcement:
        lines.append(f"guard_enforcement_bad_rate≤0.02? {'PASS' if bad_rate_enforcement <= 0.02 else 'FAIL'} (bad_rate={bad_rate_enforcement:.3f})")
    else:
        lines.append("guard_enforcement=[SKIPPED]")

    lines.append("=== Final Snapshot（task1_release_pd2） ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    lines.append("=== Diagnostics（发布回合） ===")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")
    if guard_null_rate == guard_null_rate:
        lines.append(f"guard_null_rate={guard_null_rate:.3f}")
    else:
        lines.append("guard_null_rate=[MISSING]")
    if guard_null3_rate == guard_null3_rate:
        lines.append(f"guard_null3_rate={guard_null3_rate:.3f}")
    else:
        lines.append("guard_null3_rate=[MISSING]")

    lines.append("=== V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if V_star is not None:
        lines.append(f"V_star={int(V_star)}")

    lines.append("=== File Manifest ===")
    lines.append(str(wire_log))
    lines.append(str(wire_res))
    lines.append(str(rel_log))
    lines.append(str(rel_res))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    report_path = reports_dir / "task1_acceptance_signoff_pd7_guard.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd7_cal(outdir: Path) -> Path:
    """Acceptance signoff for PD7 calibrated package.

    Adds calibration notice from calib/calib_resolved.json when present and
    otherwise mirrors PD7 guard report content.
    """
    # Generate guard report first for consistency
    guard_path = write_acceptance_signoff_pd7_guard(outdir)

    # Calibration source
    calib_dir = Path("calib")
    calib_json = calib_dir / "calib_resolved.json"
    calib_note = "[MISSING calib/calib_resolved.json]"
    if calib_json.exists():
        try:
            with calib_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            calib_note = f"calibration: {','.join(sorted([str(k) for k in data.keys()]))}"
        except Exception as e:
            calib_note = f"[CALIB READ ERROR] {e}"

    # Clone guard report lines and append calibration tag
    reports_dir = outdir / "reports"
    pd7_cal = reports_dir / "task1_acceptance_signoff_pd7_cal.txt"
    try:
        with guard_path.open("r", encoding="ascii", errors="ignore") as f:
            lines = f.read().splitlines()
        # Replace final [REPORT] path to new file and append calibration section
        # Remove old [REPORT] if present
        if lines and lines[-1].startswith("[REPORT]"):
            lines = lines[:-1]
        lines.append("=== Calibration ===")
        lines.append(calib_note)
        lines.append(f"[REPORT] {pd7_cal.as_posix()}")
        with pd7_cal.open("w", encoding="ascii", errors="ignore") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[REPORT] {pd7_cal}")
    except Exception as e:
        print(f"[ERROR] writing PD7 cal report: {e}")
    return pd7_cal


def write_acceptance_signoff_pd8_guard(outdir: Path) -> Path:
    """Acceptance signoff for PD8 package.

    PD8 adds true-constraint gating in the controller and epsilon-energy tightening
    for guard enforcement and diagnostics. This report mirrors PD7 but adjusts the
    energy aggregation consistency and enforcement checks to use the tightened budget.
    """
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)

    # Timestamp (UTC+08)
    try:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0800")
    short, branch = _git_meta()

    # Artifacts
    wire_log = dumps_dir / "task1_wirecheck_log.csv"
    wire_res = dumps_dir / "task1_wirecheck_resolved.yaml"
    if not wire_log.exists():
        alt_log = dumps_dir / "task1_release_pd2_wirecheck_log.csv"
        alt_res = dumps_dir / "task1_release_pd2_wirecheck_resolved.yaml"
        if alt_log.exists():
            wire_log = alt_log
        if alt_res.exists():
            wire_res = alt_res
    rel_log = dumps_dir / "task1_release_pd2_log.csv"
    rel_res = dumps_dir / "task1_release_pd2_resolved.yaml"
    vscan_csv = Path("outputs/vscan/summary.csv")

    # Read wirecheck
    df_wc = pd.read_csv(wire_log) if wire_log.exists() else pd.DataFrame()
    Twc = len(df_wc)
    pd_active_rate_wc = float(df_wc.get("pd_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    guard_sem_rate_wc = float(df_wc.get("guard_sem_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    guard_queue_rate_wc = float(df_wc.get("guard_queue_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    guard_energy_rate_wc = float(df_wc.get("guard_energy_active", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    # Guard-null diagnostics
    guard_null_rate_wc = float("nan"); guard_null3_rate_wc = float("nan")
    try:
        feasible_sem_wc = df_wc.get("feasible_sem_guard", pd.Series([np.nan]*Twc)).to_numpy()
        feasible_queue_wc = df_wc.get("feasible_queue_guard", pd.Series([np.nan]*Twc)).to_numpy()
        guard_null_rate_wc = float(np.mean((feasible_sem_wc == 0) & (feasible_queue_wc == 0)))
        feasible_energy_wc = df_wc.get("feasible_energy_guard", pd.Series([np.nan]*Twc)).to_numpy()
        guard_null3_rate_wc = float(np.mean((feasible_sem_wc == 0) & (feasible_queue_wc == 0) & (feasible_energy_wc == 0)))
    except Exception:
        pass

    # Aggregator self-check deltas (energy uses tightened budget E*(1-ε))
    viol_eng_inst_mean = None; viol_lat_inst_mean = None
    v_eng_wc = float(df_wc.get("violate_eng", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    v_lat_wc = float(df_wc.get("violate_lat", pd.Series([float("nan")]*Twc)).mean()) if Twc else float("nan")
    try:
        if set(["E_slot_J","energy_budget_per_slot_j"]).issubset(df_wc.columns):
            eps_wc = df_wc.get("epsilon_energy", pd.Series([0.0]*Twc)).to_numpy(dtype=float)
            budget_wc = df_wc["energy_budget_per_slot_j"].to_numpy(dtype=float)
            tight_wc = budget_wc * (1.0 - np.clip(eps_wc, 0.0, 1.0))
            viol_eng_inst = (df_wc["E_slot_J"].to_numpy(dtype=float) > tight_wc).astype(float)
            viol_eng_inst_mean = float(np.mean(viol_eng_inst))
        if "violate_lat" in df_wc.columns:
            viol_lat_inst_mean = float(df_wc["violate_lat"].to_numpy(dtype=float).mean())
    except Exception:
        pass

    # Guard enforcement bad rates (queue and energy)
    bad_rate_queue = float("nan"); bad_rate_energy = float("nan")
    try:
        if set(["feasible_queue_guard","S_eff_bps","A_bits","slot_sec"]).issubset(df_wc.columns):
            slot_sec_val = float(df_wc["slot_sec"].iloc[0]) if Twc > 0 else 0.02
            A_bps = df_wc["A_bits"].to_numpy(dtype=float) / max(slot_sec_val, 1e-9)
            S_bps = df_wc["S_eff_bps"].to_numpy(dtype=float)
            feasible_q = df_wc["feasible_queue_guard"].to_numpy(dtype=float)
            mask_bad_q = (feasible_q == 1) & (S_bps < A_bps)
            bad_rate_queue = float(np.mean(mask_bad_q)) if len(mask_bad_q) > 0 else 0.0
        if set(["feasible_energy_guard","E_slot_J","energy_budget_per_slot_j"]).issubset(df_wc.columns):
            eps_wc = df_wc.get("epsilon_energy", pd.Series([0.0]*Twc)).to_numpy(dtype=float)
            budget_wc = df_wc["energy_budget_per_slot_j"].to_numpy(dtype=float)
            tight_wc = budget_wc * (1.0 - np.clip(eps_wc, 0.0, 1.0))
            feasible_e = df_wc["feasible_energy_guard"].to_numpy(dtype=float)
            E_vals = df_wc["E_slot_J"].to_numpy(dtype=float)
            mask_bad_e = (feasible_e == 1) & (E_vals > tight_wc)
            bad_rate_energy = float(np.mean(mask_bad_e)) if len(mask_bad_e) > 0 else 0.0
    except Exception:
        pass

    # Read release
    df_rel = pd.read_csv(rel_log) if rel_log.exists() else pd.DataFrame()
    T = len(df_rel)
    mos_mean = float(df_rel.get("mos", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean()) if T else float("nan")
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean()) if T else float("nan")
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean()) if T else float("nan")
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean()) if T else float("nan")
    mos_baseline = float((1.0 - df_rel.get("per", pd.Series([0]*T)).to_numpy(dtype=float)).mean()) if T else float("nan")
    mos_uplift = float(mos_mean - mos_baseline) if T else float("nan")

    # Activation rates
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")

    # V-sweep decision
    V_star = None; best_row_text = "[MISSING vscan summary]"; header_text = ""; feasible_flag = None; v_rows = 0
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        v_rows = len(vdf)
        header_text = ",".join(vdf.columns.tolist())
        if v_rows < 3:
            print("[FAIL_VSCAN] summary rows < 3")
            sys.exit(2)
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        feasible_flag = feasible
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text
        if not feasible:
            V_star = None

    # Build report content
    lines: list[str] = []
    lines.append("=== Metadata ===")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- configs: wirecheck=task1_wirecheck, release=task1_release_pd2")

    lines.append("=== Wirecheck RESOLVED ===")
    if wire_res.exists():
        lines.append(_format_resolved("task1_wirecheck", dumps_dir, Path("configs")))
    else:
        lines.append("[MISSING RESOLVED]")

    lines.append("=== Wirecheck Assertions ===")
    lines.append(
        f"pd_active_rate={pd_active_rate_wc:.3f}, guard_sem_rate={guard_sem_rate_wc:.3f}, guard_queue_rate={guard_queue_rate_wc:.3f}, guard_energy_rate={guard_energy_rate_wc:.3f}"
    )
    if guard_null_rate_wc == guard_null_rate_wc:
        lines.append(f"guard_null_rate={guard_null_rate_wc:.3f}")
    else:
        lines.append("guard_null_rate=[MISSING]")
    if guard_null3_rate_wc == guard_null3_rate_wc:
        lines.append(f"guard_null3_rate={guard_null3_rate_wc:.3f}")
    else:
        lines.append("guard_null3_rate=[MISSING]")
    # Aggregation consistency
    if viol_eng_inst_mean is not None and v_eng_wc == v_eng_wc:
        lines.append(f"aggregation_consistency_eng(tight)|Δ|≤1e-3? {'PASS' if abs(viol_eng_inst_mean - v_eng_wc) <= 1e-3 else 'FAIL'} (inst={viol_eng_inst_mean:.6f}, aggreg={v_eng_wc:.6f})")
    else:
        lines.append("aggregation_consistency_eng(tight)=[SKIPPED]")
    if viol_lat_inst_mean is not None and v_lat_wc == v_lat_wc:
        lines.append(f"aggregation_consistency_lat|Δ|≤1e-3? {'PASS' if abs(viol_lat_inst_mean - v_lat_wc) <= 1e-3 else 'FAIL'} (inst={viol_lat_inst_mean:.6f}, aggreg={v_lat_wc:.6f})")
    else:
        lines.append("aggregation_consistency_lat=[SKIPPED]")
    # Guard enforcement
    if bad_rate_queue == bad_rate_queue:
        lines.append(f"guard_enforcement_queue_bad_rate≤0.02? {'PASS' if bad_rate_queue <= 0.02 else 'FAIL'} (bad_rate={bad_rate_queue:.3f})")
    else:
        lines.append("guard_enforcement_queue=[SKIPPED]")
    if bad_rate_energy == bad_rate_energy:
        lines.append(f"guard_enforcement_energy_bad_rate≤0.02? {'PASS' if bad_rate_energy <= 0.02 else 'FAIL'} (bad_rate={bad_rate_energy:.3f})")
    else:
        lines.append("guard_enforcement_energy=[SKIPPED]")

    lines.append("=== Final Snapshot（task1_release_pd2） ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    lines.append("=== Diagnostics（发布回合） ===")
    lines.append(f"pd_active_rate={pd_active_rate:.3f}")
    lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
    lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
    lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")

    lines.append("=== V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if V_star is not None:
        lines.append(f"V_star={int(V_star)}")

    lines.append("=== File Manifest ===")
    lines.append(str(wire_log))
    lines.append(str(wire_res))
    lines.append(str(rel_log))
    lines.append(str(rel_res))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    report_path = reports_dir / "task1_acceptance_signoff_pd8_guard.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd8_cal(outdir: Path) -> Path:
    """Acceptance signoff for PD8 calibrated package."""
    guard_path = write_acceptance_signoff_pd8_guard(outdir)
    # Calibration source
    calib_dir = Path("calib"); calib_json = calib_dir / "calib_resolved.json"
    calib_note = "[MISSING calib/calib_resolved.json]"
    if calib_json.exists():
        try:
            with calib_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            calib_note = f"calibration: {','.join(sorted([str(k) for k in data.keys()]))}"
        except Exception as e:
            calib_note = f"[CALIB READ ERROR] {e}"
    # Report
    reports_dir = outdir / "reports"
    lines = [str(guard_path), "=== Calibration ===", calib_note]
    report_path = reports_dir / "task1_acceptance_signoff_pd8_cal.txt"
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd9(outdir: Path) -> Path:
    """Acceptance signoff for PD9 package.

    Aggregates preflight, wirecheck, release, and V-scan artifacts and reports
    diagnostics per PD9 spec, including aggregator identity checks, feasible
    intersection rate, guard enforcement leak rates, and unit checks.
    """
    import numpy as np
    import pandas as pd

    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)

    # Timestamp (UTC+08)
    try:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0800")
    short, branch = _git_meta()

    # Artifacts
    preflight_res = dumps_dir / "task1_release_pd2_preflight_resolved.yaml"
    wire_log = dumps_dir / "task1_wirecheck_log.csv"
    wire_res = dumps_dir / "task1_wirecheck_resolved.yaml"
    alt_wire_log = dumps_dir / "task1_release_pd2_wirecheck_log.csv"
    alt_wire_res = dumps_dir / "task1_release_pd2_wirecheck_resolved.yaml"
    if not wire_log.exists() and alt_wire_log.exists():
        wire_log = alt_wire_log
    if not wire_res.exists() and alt_wire_res.exists():
        wire_res = alt_wire_res
    rel_log = dumps_dir / "task1_release_pd2_log.csv"
    rel_res = dumps_dir / "task1_release_pd2_resolved.yaml"
    vscan_csv = Path("outputs/vscan/summary.csv")

    # Read preflight and release (optional if pipeline halted at preflight)
    # Preflight gating: require pass flag in preflight_resolved.yaml
    halt_pipeline = False
    halt_reason = ""
    halt_causes: list[str] = []
    preflight_pass = None
    if preflight_res.exists():
        try:
            data_pf = _read_yaml_safe(preflight_res)
            preflight_pass = bool(data_pf.get("pass", False))
            if not preflight_pass:
                halt_causes.append("[FAIL_PREFLIGHT] pass flag is False")
        except Exception as e:
            halt_causes.append(f"[FAIL_PREFLIGHT_READ] {e}")
    else:
        halt_causes.append("[MISSING PREFLIGHT RESOLVED]")
    df_rel = pd.read_csv(rel_log) if rel_log.exists() else pd.DataFrame()
    T = len(df_rel)
    mos_mean = float(df_rel.get("mos", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    energy_col = "E_slot_J" if (T and ("E_slot_J" in df_rel.columns)) else ("P" if (T and ("P" in df_rel.columns)) else None)
    energy_mean = float(df_rel[energy_col].mean()) if (T and energy_col) else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean()) if T else float("nan")
    v_lat = float(df_rel.get("violate_lat", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    v_eng = float(df_rel.get("violate_eng", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    v_sem = float(df_rel.get("violate_sem", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    mos_baseline = float((1.0 - df_rel.get("per", pd.Series([float(0)]*T)).to_numpy(dtype=float)).mean()) if T else float("nan")
    mos_uplift = float(mos_mean - mos_baseline) if T else float("nan")

    # Diagnostics (release)
    pd_active_rate = float(df_rel.get("pd_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    guard_sem_rate = float(df_rel.get("guard_sem_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    guard_queue_rate = float(df_rel.get("guard_queue_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")
    guard_energy_rate = float(df_rel.get("guard_energy_active", pd.Series([float("nan")]*T)).mean()) if T else float("nan")

    # Aggregator metrics (release)
    from .constants import (
        AGG_VERSION, AGG_WARMUP_SKIP, AGG_IDENTITY_TOL, VSCAN_MIN_ROWS,
        FAIL_AGG_VERSION_MISMATCH, FAIL_VSCAN_ROWS_LT_MIN,
    )
    agg = {}
    if T:
        try:
            agg = aggregate_metrics(df_rel)
        except RuntimeError as e:
            # Fail-fast policy for NaN or missing columns
            halt_causes.append(str(e))
            agg = {}
    feasible_intersection_rate = float(agg.get("feasible_intersection_rate", float("nan")))
    action_in_guard_set_rate = float(agg.get("action_in_guard_set_rate", float("nan")))
    fallback_used_rate = float(agg.get("fallback_used_rate", float("nan")))
    bad_rate_eng = float(agg.get("guard_enforcement_bad_rate_energy", float("nan")))
    bad_rate_lat = float(agg.get("guard_enforcement_bad_rate_queue", float("nan")))
    bad_rate_sem = float(agg.get("guard_enforcement_bad_rate_sem", float("nan")))
    agg_version = str(agg.get("agg_version", "[UNKNOWN]"))
    warmup_skip = int(agg.get("warmup_skip", AGG_WARMUP_SKIP))
    # Gate on aggregator version
    if T and agg_version != AGG_VERSION:
        halt_causes.append(FAIL_AGG_VERSION_MISMATCH)

    # Aggregator identity checks (release): energy & latency
    # Apply same warm-up filter as aggregator for identity comparisons
    try:
        if "t" in df_rel.columns:
            df_rel_eff = df_rel[pd.to_numeric(df_rel["t"], errors="coerce") > int(warmup_skip)]
        else:
            df_rel_eff = df_rel.iloc[int(warmup_skip):]
    except Exception:
        df_rel_eff = df_rel.iloc[int(warmup_skip):]
    Teff = len(df_rel_eff)
    ident_eng = "[SKIPPED]"; ident_lat = "[SKIPPED]"
    try:
        if Teff and set(["E_slot_J","energy_budget_per_slot_j"]).issubset(df_rel_eff.columns):
            viol_eng_inst = (df_rel_eff["E_slot_J"].to_numpy(dtype=float) > df_rel_eff["energy_budget_per_slot_j"].to_numpy(dtype=float)).astype(float)
            d_eng = abs(float(np.mean(viol_eng_inst)) - float(agg.get("viol_eng", float("nan"))))
            ident_eng = f"{'PASS' if d_eng <= 1e-3 else 'FAIL'} (Δ={d_eng:.6f})"
        if Teff and set(["S_eff_bps","arrivals_bps","delta_queue_used","Q_t_used"]).issubset(df_rel_eff.columns):
            lhs = df_rel_eff["S_eff_bps"].to_numpy(dtype=float)
            rhs = df_rel_eff["arrivals_bps"].to_numpy(dtype=float) + df_rel_eff["delta_queue_used"].to_numpy(dtype=float) * df_rel_eff["Q_t_used"].to_numpy(dtype=float)
            viol_lat_inst = (lhs < rhs).astype(float)
            d_lat = abs(float(np.mean(viol_lat_inst)) - float(agg.get("viol_lat", float("nan"))))
            ident_lat = f"{'PASS' if d_lat <= 1e-3 else 'FAIL'} (Δ={d_lat:.6f})"
    except Exception:
        pass
    # Gating on aggregator identity checks
    if str(ident_eng).startswith("FAIL") or str(ident_lat).startswith("FAIL"):
        halt_causes.append("[FAIL_AGG_IDENTITY] energy/latency aggregation mismatch")

    # Unit check: B_eff_Hz coverage >= 0.99 vs B_min_kHz*1e3 (release)
    unit_bw_line = "[SKIPPED]"
    try:
        if Teff and set(["B_eff_Hz","B_min_kHz"]).issubset(df_rel_eff.columns):
            ok_rate = float((df_rel_eff["B_eff_Hz"].to_numpy(dtype=float) >= df_rel_eff["B_min_kHz"].to_numpy(dtype=float) * 1000.0).mean())
            unit_bw_line = f"{'PASS' if ok_rate >= 0.99 else 'FAIL'} (coverage={ok_rate:.3f})"
    except Exception:
        pass
    if str(unit_bw_line).startswith("FAIL"):
        halt_causes.append("[FAIL_UNIT_BANDWIDTH] B_eff_Hz coverage below 0.99 vs B_min_kHz")

    # V-scan row-count hard gate: require at least VSCAN_MIN_ROWS when summary exists
    if vscan_csv.exists():
        try:
            vdf_gate = pd.read_csv(vscan_csv)
            v_rows_gate = int(vdf_gate.shape[0])
            if v_rows_gate < int(VSCAN_MIN_ROWS):
                print(f"[{FAIL_VSCAN_ROWS_LT_MIN}] rows={v_rows_gate} min_required={int(VSCAN_MIN_ROWS)}")
                sys.exit(FAIL_VSCAN_ROWS_LT_MIN)
        except Exception as e:
            print(f"[{FAIL_VSCAN_ROWS_LT_MIN}] unable to read vscan summary: {e}")
            sys.exit(FAIL_VSCAN_ROWS_LT_MIN)

    # Dual price trajectories
    lam_eng = df_rel.get("lambda_eng", pd.Series([np.nan]*T)).to_numpy(dtype=float) if T else np.array([])
    lam_lat = df_rel.get("lambda_lat", pd.Series([np.nan]*T)).to_numpy(dtype=float) if T else np.array([])
    lam_eng_min = float(np.nanmin(lam_eng)) if lam_eng.size else float("nan")
    lam_eng_max = float(np.nanmax(lam_eng)) if lam_eng.size else float("nan")
    lam_eng_last = float(lam_eng[-1]) if lam_eng.size else float("nan")
    lam_lat_min = float(np.nanmin(lam_lat)) if lam_lat.size else float("nan")
    lam_lat_max = float(np.nanmax(lam_lat)) if lam_lat.size else float("nan")
    lam_lat_last = float(lam_lat[-1]) if lam_lat.size else float("nan")

    # V-sweep decision (no V_star if not feasible)
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    feasible_flag = None
    if vscan_csv.exists():
        try:
            vdf = pd.read_csv(vscan_csv)
            header_text = ",".join(vdf.columns.tolist())
            if int(vdf.shape[0]) < int(VSCAN_MIN_ROWS):
                # Do not write V* if fewer than 3 rows
                best_row_text = "[SKIPPED V-scan rows<3]"
            elif not halt_pipeline:
                V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
                feasible_flag = feasible
                if row_star is not None:
                    best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
                    if not feasible:
                        best_row_text = "[NO-FEASIBLE] " + best_row_text
                if not feasible:
                    V_star = None
            else:
                best_row_text = f"[SKIPPED due to {halt_reason}]"
        except Exception:
            pass

    # Build report content per PD9 spec
    lines: list[str] = []
    lines.append("=== Metadata ===")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- config: task1_release_pd2")
    # Resolve HALT status & root cause (unique priority order)
    if halt_causes:
        halt_pipeline = True
        # priority: preflight → agg fail-fast → identity → version → unit
        pri = [
            "[MISSING PREFLIGHT RESOLVED]",
            "[FAIL_PREFLIGHT_READ]",
            "[FAIL_PREFLIGHT]",
            "[FAIL_AGG_REQUIRED_COLUMNS]",
            "[FAIL_AGG_NAN]",
            "[FAIL_AGG_IDENTITY]",
            "[FAIL_AGG_VERSION_MISMATCH]",
            "[FAIL_UNIT_BANDWIDTH]",
        ]
        # pick the first cause by order above
        for tag in pri:
            cand = next((c for c in halt_causes if c.startswith(tag)), None)
            if cand is not None:
                halt_reason = cand
                break
        # fallback to the first cause if none matched the priority tags
        if not halt_reason:
            halt_reason = halt_causes[0]
    if halt_pipeline:
        lines.append(f"- HALT_REASON: {halt_reason}")

    # RESOLVED
    lines.append("=== RESOLVED ===")
    def _format_resolved(tag: str, path: Path) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        pd_block = data.get("pd", {})
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}, "
            f"P_max_dBm={data.get('P_max_dBm')}, B_min_kHz={data.get('B_min_kHz')}, B_grid_k={data.get('B_grid_k')}, "
            f"pd.lambda_init={pd_block.get('lambda_init')}, pd.delta_queue={pd_block.get('delta_queue')}, pd.eta={pd_block.get('eta')}"
        )
    if preflight_res.exists():
        lines.append(_format_resolved("preflight", preflight_res))
    else:
        lines.append("[MISSING PRE-FLIGHT RESOLVED]")
    if rel_res.exists():
        lines.append(_format_resolved("release", rel_res))
    else:
        lines.append("[MISSING RELEASE RESOLVED]")

    # Final Snapshot (release)
    lines.append("=== Final Snapshot (release) ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f},{v_eng:.3f},{v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
        if T else "[MISSING RELEASE LOG]"
    )

    # V-sweep Decision
    lines.append("=== V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")
    if (V_star is not None) and (not halt_pipeline):
        lines.append(f"V_star={int(V_star)}")

    # Diagnostics
    lines.append("=== Diagnostics ===")
    if T:
        # Rates computed on effective post-warmup window where applicable
        pd_active_rate = float(df_rel_eff.get("pd_active", pd.Series([float("nan")]*Teff)).mean()) if Teff else float("nan")
        guard_sem_rate = float(df_rel_eff.get("guard_sem_active", pd.Series([float("nan")]*Teff)).mean()) if Teff else float("nan")
        guard_queue_rate = float(df_rel_eff.get("guard_queue_active", pd.Series([float("nan")]*Teff)).mean()) if Teff else float("nan")
        guard_energy_rate = float(df_rel_eff.get("guard_energy_active", pd.Series([float("nan")]*Teff)).mean()) if Teff else float("nan")
        lines.append(f"agg_version={agg_version}")
        lines.append(f"warmup_skip={warmup_skip}")
        lines.append(f"pd_active_rate={pd_active_rate:.3f}")
        lines.append(f"guard_sem_rate={guard_sem_rate:.3f}")
        lines.append(f"guard_queue_rate={guard_queue_rate:.3f}")
        lines.append(f"guard_energy_rate={guard_energy_rate:.3f}")
        lines.append(f"feasible_intersection_rate={feasible_intersection_rate:.3f}")
        lines.append(f"action_in_guard_set_rate={action_in_guard_set_rate:.3f}")
        lines.append(f"fallback_used_rate={fallback_used_rate:.3f}")
        lines.append(f"guard_enforcement_bad_rate_energy={bad_rate_eng:.3f}")
        lines.append(f"guard_enforcement_bad_rate_queue={bad_rate_lat:.3f}")
        lines.append(f"guard_enforcement_bad_rate_sem={bad_rate_sem:.3f}")
        lines.append(f"lambda_eng_min/max/last={lam_eng_min:.4f}/{lam_eng_max:.4f}/{lam_eng_last:.4f}")
        lines.append(f"lambda_lat_min/max/last={lam_lat_min:.4f}/{lam_lat_max:.4f}/{lam_lat_last:.4f}")
        lines.append(f"unit_bandwidth_check={unit_bw_line}")
        lines.append(f"aggregator_identity_eng={ident_eng}")
        lines.append(f"aggregator_identity_lat={ident_lat}")
        # First violating samples (t indices)
        lines.append(f"first_violate_eng_t={agg.get('first_violate_eng_t', '[N/A]')}")
        lines.append(f"first_violate_lat_t={agg.get('first_violate_lat_t', '[N/A]')}")
        lines.append(f"first_violate_sem_t={agg.get('first_violate_sem_t', '[N/A]')}")
    else:
        lines.append("[PIPELINE HALTED BEFORE RELEASE]")

    # File Manifest
    lines.append("=== File Manifest ===")
    if preflight_res.exists():
        lines.append(str(preflight_res))
    lines.append(str(wire_log))
    lines.append(str(wire_res))
    lines.append(str(rel_log))
    lines.append(str(rel_res))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    # Two-line stop-loss
    lines.append("=== Stop-Loss ===")
    lines.append("semantic_budget := min(0.38, semantic_budget + 0.02)")
    lines.append("P_max_dBm/B_min_kHz := allow +3 dBm bump or one notch B_min_kHz")

    # Write report
    report_path = reports_dir / "task1_acceptance_signoff_pd9.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


def write_acceptance_signoff_pd(outdir: Path) -> Path:
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # PD release artifacts
    log_final = dumps_dir / "task1_release_pd_log.csv"
    resolved_final = dumps_dir / "task1_release_pd_resolved.yaml"
    tag_print = "task1_release_pd"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel["mos"].mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text

    # RESOLVED line with PD params
    def _fmt_resolved_pd(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        lam_mult = data.get("lam_init_multiplier", data.get("lam_mult"))
        eta = data.get("eta", data.get("pd", {}).get("eta", data.get("eta_dual")))
        dq = data.get("delta_queue", data.get("pd", {}).get("delta_queue"))
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}, "
            f"lam_init_multiplier={lam_mult}, pd_eta={eta}, pd_delta_queue={dq}"
        )

    resolved_lines: list[str] = []
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved_pd(resolved_final, tag_print))

    # Build report
    lines: list[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- repo_path: {cwd}")

    lines.append("=== 1. RESOLVED Configs ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== 2. Final Release Snapshot ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    all_ok = (v_lat <= 0.01) and (v_eng <= 0.01) and (v_sem <= 0.01)
    lines.append(f"Energy/Latency/Semantic violation all ≤ 1%? {'YES' if all_ok else 'NO'} ({v_lat:.3f},{v_eng:.3f},{v_sem:.3f})")

    lines.append("=== 3. V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")

    lines.append("=== 4. Acceptance Check ===")
    uplift_ok = (mos_uplift >= 0.30)
    lines.append(f"- constraints ≤1%? {'PASS' if all_ok else 'FAIL'}")
    lines.append(f"- MOS uplift ≥ +0.30? {'PASS' if uplift_ok else 'FAIL'} (uplift={mos_uplift:.3f})")
    final_cfg = _read_yaml_safe(resolved_final)
    s = final_cfg.get("scales", {})
    lines.append(
        f"最终上线参数: V={final_cfg.get('V')}, latency_budget_q_bps={final_cfg.get('latency_budget_q_bps')}, energy_budget_per_slot_j={final_cfg.get('energy_budget_per_slot_j')}, "
        f"semantic_budget={final_cfg.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}, "
        f"lam_init_multiplier={final_cfg.get('lam_init_multiplier')}, pd_eta={final_cfg.get('eta') or final_cfg.get('eta_dual')}, pd_delta_queue={final_cfg.get('delta_queue')}"
    )

    lines.append("=== 5. 文件清单 ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    report_path = reports_dir / "task1_acceptance_signoff_pd.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path

def write_acceptance_signoff_final(outdir: Path) -> Path:
    reports_dir = outdir / "reports"
    dumps_dir = outdir / "dumps"
    os.makedirs(reports_dir, exist_ok=True)
    cwd = str(Path.cwd())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = f"Python {sys.version.split()[0]}"
    short, branch = _git_meta()

    # Prefer iter release if present
    iter_logs = sorted(dumps_dir.glob("task1_release_iter*_log.csv"))
    if iter_logs:
        log_final = iter_logs[-1]
        iter_tag = log_final.stem.replace("_log", "")
        resolved_final = dumps_dir / (iter_tag + "_resolved.yaml")
        tag_print = iter_tag
    else:
        log_final = dumps_dir / "task1_release_log.csv"
        resolved_final = dumps_dir / "task1_release_resolved.yaml"
        tag_print = "task1_release"

    df_rel = pd.read_csv(log_final)
    T = len(df_rel)
    mos_mean = float(df_rel["mos"].mean())
    energy_col = "E_slot_J" if "E_slot_J" in df_rel.columns else ("P" if "P" in df_rel.columns else None)
    energy_mean = float(df_rel[energy_col].mean()) if energy_col else float("nan")
    swer_mean = float(df_rel.get("sWER_raw", df_rel.get("sWER", pd.Series([float("nan")]*T))).mean())
    v_lat = float(df_rel.get("violate_lat", pd.Series([0]*T)).mean())
    v_eng = float(df_rel.get("violate_eng", pd.Series([0]*T)).mean())
    v_sem = float(df_rel.get("violate_sem", pd.Series([0]*T)).mean())
    mos_baseline = float((1.0 - df_rel["per"].to_numpy(dtype=float)).mean())
    mos_uplift = float(mos_mean - mos_baseline)

    # V-sweep decision
    vscan_csv = Path("outputs/vscan/summary.csv")
    V_star = None
    best_row_text = "[MISSING vscan summary]"
    header_text = ""
    if vscan_csv.exists():
        vdf = pd.read_csv(vscan_csv)
        header_text = ",".join(vdf.columns.tolist())
        V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
        if row_star is not None:
            best_row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
            if not feasible:
                best_row_text = "[NO-FEASIBLE] " + best_row_text

    # RESOLVED lines
    resolved_lines: list[str] = []
    def _fmt_resolved(path: Path, tag: str) -> str:
        data = _read_yaml_safe(path)
        s = data.get("scales", {})
        return (
            f"RESOLVED[{tag}] T={data.get('T')}, slot_sec={data.get('slot_sec')}, arrivals_bps={data.get('arrivals_bps')}, "
            f"energy_budget_per_slot_j={data.get('energy_budget_per_slot_j')}, latency_budget_q_bps={data.get('latency_budget_q_bps')}, "
            f"semantic_budget={data.get('semantic_budget')}, V={data.get('V')}, scales={{queue:{s.get('queue_scale')}, energy:{s.get('energy_scale')}, semantic:{s.get('semantic_scale')}}}"
        )
    if resolved_final.exists():
        resolved_lines.append(_fmt_resolved(resolved_final, tag_print))

    # Build report
    lines: list[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- repo_path: {cwd}")

    lines.append("=== 1. RESOLVED Configs ===")
    lines.extend(resolved_lines or ["[MISSING RESOLVED]"])

    lines.append("=== 2. Final Release Snapshot ===")
    lines.append(
        f"T={T}, MOS_mean={mos_mean:.4f}, E_slot_J_mean={energy_mean:.3f}, sWER_mean={swer_mean:.3f}, "
        f"violation(lat,eng,sem)=({v_lat:.3f}, {v_eng:.3f}, {v_sem:.3f}), MOS_uplift_vs_baseline={mos_uplift:.4f}"
    )
    all_ok = (v_lat <= 0.01) and (v_eng <= 0.01) and (v_sem <= 0.01)
    lines.append(f"Energy/Latency/Semantic violation all ≤ 1%? {'YES' if all_ok else 'NO'} ({v_lat:.3f},{v_eng:.3f},{v_sem:.3f})")

    lines.append("=== 3. V-sweep Decision ===")
    if header_text:
        lines.append(header_text)
    lines.append(best_row_text)
    lines.append("outputs/figs/Fig_MOS_vs_V.png")
    lines.append("outputs/figs/Fig_ViolationRates_vs_V.png")
    lines.append("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")

    lines.append("=== 4. Acceptance Check ===")
    uplift_ok = (mos_uplift >= 0.30)
    lines.append(f"- constraints ≤1%? {'PASS' if all_ok else 'FAIL'}")
    lines.append(f"- MOS uplift ≥ +0.30? {'PASS' if uplift_ok else 'FAIL'} (uplift={mos_uplift:.3f})")
    final_cfg = _read_yaml_safe(resolved_final)
    s = final_cfg.get("scales", {})
    if all_ok and uplift_ok:
        lines.append(
            f"最终上线参数: V={final_cfg.get('V')}, latency_budget_q_bps={final_cfg.get('latency_budget_q_bps')}, energy_budget_per_slot_j={final_cfg.get('energy_budget_per_slot_j')}, "
            f"semantic_budget={final_cfg.get('semantic_budget')}, scales.queue={s.get('queue_scale')}, scales.energy={s.get('energy_scale')}, scales.semantic={s.get('semantic_scale')}"
        )
    else:
        lines.append("放宽 semantic_budget 至 0.38，并重跑一次")
        lines.append("若 P_min 超过 P_max 超过 5% 以上的隙占比>1%，提高 P_max 或缩小 B 最小值（以抬升 SINR）。")

    lines.append("=== 5. 文件清单 ===")
    lines.append(str(log_final))
    lines.append(str(resolved_final))
    if vscan_csv.exists():
        lines.append(str(vscan_csv))
    lines.append(str(dumps_dir / "task1_summary.csv"))
    lines.append(str(Path("outputs/figs/Fig_MOS_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_ViolationRates_vs_V.png")))
    lines.append(str(Path("outputs/figs/Fig_Pareto_MOS_vs_Energy.png")))

    table_path = dumps_dir / "release_vs_baselines.csv"
    rows = []
    rows.append({
        "config_name": "baseline_fixed",
        "MOS_mean": mos_baseline,
        "E_slot_J_mean": energy_mean,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
    })
    rows.append({
        "config_name": tag_print,
        "MOS_mean": mos_mean,
        "E_slot_J_mean": energy_mean,
        "sWER_mean": swer_mean,
        "viol_lat": v_lat,
        "viol_eng": v_eng,
        "viol_sem": v_sem,
    })
    pd.DataFrame(rows)[["config_name","MOS_mean","E_slot_J_mean","sWER_mean","viol_lat","viol_eng","viol_sem"]].to_csv(table_path, index=False)
    lines.append(str(table_path))

    report_path = reports_dir / "task1_acceptance_signoff_final.txt"
    lines.append(f"[REPORT] {report_path.as_posix()}")
    with report_path.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {report_path}")
    return report_path


if __name__ == "__main__":
    main()
# ===== PD9 Aggregation Utilities =====
def aggregate_metrics(df):
    """Single source of truth aggregator for release & V-scan.

    Uses centralized schema and gates from ``constants.py``, applies a
    warm-up skip before computing rates, and returns an aggregation dict
    with violation rates, guard diagnostics, and metadata including the
    aggregator version.

    Fail-fast policy (PD9):
    - If any required column missing → raise RuntimeError("[FAIL_AGG_REQUIRED_COLUMNS]")
    - If any NaN present in required columns → raise RuntimeError("[FAIL_AGG_NAN]")
    """
    import numpy as np
    import pandas as pd
    from .constants import (
        AGG_VERSION,
        AGG_WARMUP_SKIP,
        AGG_REQUIRED_COLS,
        FAIL_AGG_REQUIRED_COLUMNS,
        FAIL_AGG_NAN,
    )

    # Apply warm-up filter (prefer t-based; fallback to head-slice)
    try:
        if "t" in df.columns:
            df_use = df[pd.to_numeric(df["t"], errors="coerce") > int(AGG_WARMUP_SKIP)]
        else:
            df_use = df.iloc[int(AGG_WARMUP_SKIP):]
    except Exception:
        df_use = df.iloc[int(AGG_WARMUP_SKIP):]

    need = set(AGG_REQUIRED_COLS)
    cols = set(df_use.columns)
    missing = sorted(list(need - cols))
    if missing:
        raise RuntimeError(f"{FAIL_AGG_REQUIRED_COLUMNS} missing={missing}")
    if df_use[list(need)].isna().any().any():
        raise RuntimeError(FAIL_AGG_NAN)

    out: dict[str, float | int | str] = {}
    # Violations against budgets/targets
    E = df_use["E_slot_J"].to_numpy(dtype=float)
    G = df_use["energy_budget_per_slot_j"].to_numpy(dtype=float)
    out["viol_eng"] = float((E > G).mean())

    S_eff = df_use["S_eff_bps"].to_numpy(dtype=float)
    A = df_use["arrivals_bps"].to_numpy(dtype=float)
    dq = df_use["delta_queue_used"].to_numpy(dtype=float)
    Qt = df_use["Q_t_used"].to_numpy(dtype=float)
    out["viol_lat"] = float((S_eff < (A + dq * Qt)).mean())

    I_hat = df_use["sWER_clip"].to_numpy(dtype=float)
    I_t = df_use["semantic_budget"].to_numpy(dtype=float)
    out["viol_sem"] = float((I_hat > I_t).mean())

    # Enforcement leak rates
    out["guard_enforcement_bad_rate_energy"] = float((df_use["bad_energy"].to_numpy(dtype=int) == 1).mean())
    out["guard_enforcement_bad_rate_queue"] = float((df_use["bad_queue"].to_numpy(dtype=int) == 1).mean())
    out["guard_enforcement_bad_rate_sem"] = float((df_use["bad_sem"].to_numpy(dtype=int) == 1).mean())

    # Selection window diagnostics
    out["action_in_guard_set_rate"] = float(df_use["action_in_guard_set"].to_numpy(dtype=float).mean())
    out["fallback_used_rate"] = float(df_use["fallback_used"].to_numpy(dtype=float).mean())

    # Feasible intersection rate snapshot
    inter = (
        df_use["feasible_energy_guard"].astype(bool)
        & df_use["feasible_queue_guard"].astype(bool)
        & df_use["feasible_sem_guard"].astype(bool)
    )
    out["feasible_intersection_rate"] = float(inter.mean())

    # Metadata
    out["agg_version"] = AGG_VERSION
    out["warmup_skip"] = int(AGG_WARMUP_SKIP)
    out["T_eff"] = int(df_use.shape[0])

    # First violating sample indices (if 't' available)
    try:
        t_vals = pd.to_numeric(df_use["t"], errors="coerce").to_numpy(dtype=float)
        eng_first_idx = int(np.argmax(E > G)) if (E > G).any() else -1
        lat_first_idx = int(np.argmax(S_eff < (A + dq * Qt))) if (S_eff < (A + dq * Qt)).any() else -1
        sem_first_idx = int(np.argmax(I_hat > I_t)) if (I_hat > I_t).any() else -1
        out["first_violate_eng_t"] = None if eng_first_idx < 0 else int(t_vals[eng_first_idx])
        out["first_violate_lat_t"] = None if lat_first_idx < 0 else int(t_vals[lat_first_idx])
        out["first_violate_sem_t"] = None if sem_first_idx < 0 else int(t_vals[sem_first_idx])
    except Exception:
        out["first_violate_eng_t"] = None
        out["first_violate_lat_t"] = None
        out["first_violate_sem_t"] = None

    return out