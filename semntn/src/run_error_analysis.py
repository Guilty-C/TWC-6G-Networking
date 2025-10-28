"""Task1 error analysis: 误差—成因—对策 (S1–S6)

Reads existing outputs and configs, performs light-fit diagnostics, and emits:
- CSVs and figures under `outputs/error_analysis/`
- Structured TXT report at `outputs/reports/task1_error_analysis.txt`

Constraints:
- Do NOT rerun experiments > 2000 slots.
- Allow at most two micro experiments, each T<=200, as A/B.
- If required files are missing, mark as [MISSING] and continue.
"""
from __future__ import annotations

import os
import sys
import math
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from .semantics import get_sem_params, swer_from_sinr, sinr_lin_for_swer_target


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_yaml_safe(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _git_meta() -> Tuple[str, str]:
    try:
        short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        if short_hash:
            return short_hash, branch or "[MISSING]"
    except Exception:
        pass
    return "[MISSING]", "[MISSING]"


def _cdf_data(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1, dtype=float) / max(len(x), 1)
    return x, y


def _choose_log() -> Tuple[Path | None, str]:
    """Prefer bandit log; else release log; else any task1_*_log.csv under dumps.
    Returns (path, tag)."""
    dumps = Path("outputs") / "dumps"
    cand_order = [
        dumps / "task1_bandit_log.csv",
        dumps / "task1_release_log.csv",
    ]
    for p in cand_order:
        if p.exists():
            return p, p.stem.replace("_log", "")
    # fallback: any task1_*_log.csv
    if dumps.exists():
        for p in sorted(dumps.glob("task1_*_log.csv")):
            return p, p.stem.replace("_log", "")
    return None, "[MISSING]"


def _load_trace() -> Tuple[np.ndarray | None, str]:
    """Load trace sinr_db if available; else [MISSING]."""
    base = Path("semntn") / "data" / "channel_trace.csv"
    if base.exists():
        try:
            df = pd.read_csv(base)
            if "snr_db" in df.columns:
                return df["snr_db"].to_numpy(dtype=float), base.as_posix()
        except Exception:
            pass
    return None, "[MISSING]"


def _release_cfg() -> dict:
    return _read_yaml_safe(Path("configs") / "task1_release.yaml")


def _sinr_eff_db_max(snr_db: np.ndarray, P_max: float, B_min: float, q_budget: float) -> np.ndarray:
    gain_db = 10.0 * np.log10(max(P_max * B_min, 1e-9))
    penalty_db = 3.0 * np.log2(max(q_budget, 1.0) / 300.0 + 1.0)
    return np.asarray(snr_db, dtype=float) + float(gain_db) - float(penalty_db)


def s1_semantic_feasibility(outdir: Path) -> dict:
    errs = outdir / "error_analysis"
    _ensure_dir(errs)
    # params
    a, b = get_sem_params()
    rel = _release_cfg()
    semantic_budget = float(rel.get("semantic_budget", 0.35))
    # log or trace
    log_path, log_tag = _choose_log()
    df_log = None
    sinr_db_log = None
    if log_path and log_path.exists():
        try:
            df_log = pd.read_csv(log_path)
            if "sinr_db" in df_log.columns:
                sinr_db_log = df_log["sinr_db"].to_numpy(dtype=float)
        except Exception:
            pass
    trace, trace_path = _load_trace()
    # base config values
    P_max = float(rel.get("P_max", 1.6))
    B_grid = list(rel.get("B_grid", [1, 2, 3]))
    B_min = float(min(B_grid)) if B_grid else 1.0
    q_budget = float(rel.get("latency_budget_q_bps", rel.get("q_bps", 300.0)))
    slot_sec = float(rel.get("slot_sec", 0.02))

    # Observed or estimated SINR_max
    if sinr_db_log is not None:
        base = np.asarray(sinr_db_log, dtype=float)
    elif trace is not None:
        base = np.asarray(trace, dtype=float)
    else:
        base = np.array([], dtype=float)

    if base.size > 0:
        sinr_db_max = _sinr_eff_db_max(base, P_max, B_min, q_budget)
        sinr_lin_max = 10.0 ** (sinr_db_max / 10.0)
    else:
        sinr_db_max = np.array([], dtype=float)
        sinr_lin_max = np.array([], dtype=float)

    sinr_req_lin = float(sinr_lin_for_swer_target(semantic_budget, a, b))
    gamma_sem = sinr_lin_max - sinr_req_lin
    infeasible = (gamma_sem < 0).astype(int) if gamma_sem.size > 0 else np.array([], dtype=int)

    # CSV
    rows = []
    for i in range(len(sinr_lin_max)):
        rows.append({
            "t": i + 1,
            "SINR_max": float(sinr_lin_max[i]),
            "SINR_req": float(sinr_req_lin),
            "gamma_sem": float(gamma_sem[i]),
            "infeasible": int(infeasible[i]),
        })
    csv_path = errs / "semantic_margin.csv"
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    else:
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("t,SINR_max,SINR_req,gamma_sem,infeasible\n")
    # Figures
    # CDF of gamma
    fig1 = errs / "Fig_semantic_feasibility_CDF.png"
    if gamma_sem.size > 0:
        x, y = _cdf_data(gamma_sem)
        plt.figure(figsize=(7, 4.5)); plt.step(x, y, where="post")
        plt.xlabel("gamma_sem (linear)"); plt.ylabel("CDF"); plt.title("Semantic feasibility margin CDF")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(fig1, dpi=150); plt.close()
    else:
        plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] gamma_sem", ha="center")
        plt.axis("off"); plt.tight_layout(); plt.savefig(fig1, dpi=150); plt.close()
    # Infeasible rate overall and by SINR quantile bins
    fig2 = errs / "Fig_semantic_infeasible_rate.png"
    if base.size > 0 and infeasible.size > 0:
        overall = float(np.mean(infeasible))
        # quantile bins (quartiles)
        qs = np.quantile(base, [0.25, 0.5, 0.75])
        bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        labels = ["Q1", "Q2", "Q3", "Q4"]
        idx = np.digitize(base, bins) - 1
        rates = [float(np.mean(infeasible[idx == k])) if np.any(idx == k) else 0.0 for k in range(4)]
        plt.figure(figsize=(7, 4.5))
        plt.bar(["overall"] + labels, [overall] + rates)
        plt.ylabel("infeasible fraction"); plt.title("Semantic infeasible rate")
        plt.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(fig2, dpi=150); plt.close()
    else:
        plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] infeasible stats", ha="center")
        plt.axis("off"); plt.tight_layout(); plt.savefig(fig2, dpi=150); plt.close()

    return {
        "csv": csv_path.as_posix(),
        "fig_cdf": fig1.as_posix(),
        "fig_infeasible": fig2.as_posix(),
        "infeasible_overall": float(np.mean(infeasible)) if infeasible.size > 0 else float("nan"),
        "gamma_p10": float(np.quantile(gamma_sem, 0.10)) if gamma_sem.size > 0 else float("nan"),
        "gamma_median": float(np.quantile(gamma_sem, 0.50)) if gamma_sem.size > 0 else float("nan"),
        "gamma_p90": float(np.quantile(gamma_sem, 0.90)) if gamma_sem.size > 0 else float("nan"),
        "sources": {
            "log": log_path.as_posix() if log_path and log_path.exists() else "[MISSING]",
            "trace": trace_path,
            "release_cfg": Path("configs/task1_release.yaml").as_posix() if Path("configs/task1_release.yaml").exists() else "[MISSING]",
            "link_semantic": Path("configs/link_semantic.yaml").as_posix() if Path("configs/link_semantic.yaml").exists() else "[MISSING]",
        }
    }


def s2_service_residuals(outdir: Path) -> dict:
    errs = outdir / "error_analysis"
    _ensure_dir(errs)
    log_path, _ = _choose_log()
    csv_path = errs / "service_residual_samples.csv"
    fig_path = errs / "Fig_service_residual_heatmap.png"
    if not (log_path and log_path.exists()):
        # write empty CSV and placeholder figure
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("t,SINR,B,rate,per,S_model,S_meas,dS\n")
        plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] log", ha="center"); plt.axis("off")
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
        return {"csv": csv_path.as_posix(), "fig": fig_path.as_posix(), "bias_mean": float("nan"), "bias_var": float("nan")}
    df = pd.read_csv(log_path)
    slot_sec = float(_release_cfg().get("slot_sec", 0.02))
    # Build samples
    T = len(df)
    samples = []
    for t in range(T):
        sinr = float(df.get("sinr_db", pd.Series([float("nan")]*T)).iloc[t])
        B = float(df.get("B", pd.Series([float("nan")]*T)).iloc[t])
        rate = float(df.get("rate", df.get("q_bps", pd.Series([float("nan")]*T))).iloc[t])
        per = float(df.get("per", pd.Series([float("nan")]*T)).iloc[t])
        S_model = float(rate * (1.0 - per) * slot_sec) if not (math.isnan(rate) or math.isnan(per)) else float("nan")
        # S_meas via queue equation requires backlog; not available → [MISSING]
        S_meas = float("nan")
        dS = float(S_meas - S_model) if not (math.isnan(S_meas) or math.isnan(S_model)) else float("nan")
        samples.append({"t": t + 1, "SINR": sinr, "B": B, "rate": rate, "per": per, "S_model": S_model, "S_meas": S_meas, "dS": dS})
    pd.DataFrame(samples).to_csv(csv_path, index=False)

    # Heatmap of dS over (SINR,B) bins
    df_s = pd.DataFrame(samples)
    if df_s["dS"].notna().sum() > 0:
        # bin sinr into quantiles, B as categories
        sinr = df_s["SINR"].to_numpy(dtype=float)
        valid = np.isfinite(sinr) & df_s["dS"].notna().to_numpy()
        sinr_valid = sinr[valid]
        dS_valid = df_s["dS"].to_numpy(dtype=float)[valid]
        B_valid = df_s["B"].to_numpy(dtype=float)[valid]
        qs = np.quantile(sinr_valid, [0.25, 0.5, 0.75])
        bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        idx_s = np.digitize(sinr_valid, bins) - 1
        Bs = np.unique(B_valid)
        grid = np.full((4, len(Bs)), np.nan)
        for i in range(4):
            for j, b in enumerate(Bs):
                mask = (idx_s == i) & (B_valid == b)
                if np.any(mask):
                    grid[i, j] = float(np.mean(dS_valid[mask]))
        plt.figure(figsize=(7, 4.8))
        plt.imshow(grid, aspect="auto", cmap="coolwarm", interpolation="nearest")
        plt.colorbar(label="mean dS (bits)")
        plt.xticks(range(len(Bs)), [f"B={int(b)}" for b in Bs])
        plt.yticks(range(4), ["Q1", "Q2", "Q3", "Q4"])  # SINR quartiles
        plt.title("Service rate residual dS heatmap")
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
        bias_mean = float(np.nanmean(grid))
        bias_var = float(np.nanvar(grid))
    else:
        plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] dS", ha="center"); plt.axis("off")
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
        bias_mean = float("nan")
        bias_var = float("nan")
    # Bias direction: negative means model pessimistic
    bias_dir = "[MISSING]" if math.isnan(bias_mean) else ("negative" if bias_mean < 0 else ("positive" if bias_mean > 0 else "neutral"))
    return {"csv": csv_path.as_posix(), "fig": fig_path.as_posix(), "bias_mean": bias_mean, "bias_var": bias_var, "bias_dir": bias_dir}


def _micro_cfg_paths() -> Tuple[Path, Path]:
    return Path("configs") / "micro_dpp_fixed.yaml", Path("configs") / "micro_dpp_dual.yaml"


def _write_micro_cfgs() -> Tuple[Path, Path]:
    rel = _release_cfg()
    fixed, dual = _micro_cfg_paths()
    # Fixed DPP: eta_dual=0.0
    cfg_fixed = {**rel}
    cfg_fixed["T"] = 200
    cfg_fixed["eta_dual"] = 0.0
    # Dual: eta_dual in [1e-3,1e-2]
    cfg_dual = {**rel}
    cfg_dual["T"] = 200
    cfg_dual["eta_dual"] = float(cfg_dual.get("eta_dual", 5e-3))
    # Persist
    with fixed.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_fixed, f, sort_keys=False, allow_unicode=True)
    with dual.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dual, f, sort_keys=False, allow_unicode=True)
    return fixed, dual


def _run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def s3_micro_ab(outdir: Path) -> dict:
    errs = outdir / "error_analysis"
    _ensure_dir(errs)
    fixed, dual = _write_micro_cfgs()
    # Run two micro experiments (T<=200)
    _run_cmd([sys.executable, "-m", "semntn.src.run_task1", "--config", str(fixed)])
    _run_cmd([sys.executable, "-m", "semntn.src.run_task1", "--config", str(dual)])
    # Collect logs
    dumps = outdir / "dumps"
    log_fixed = dumps / f"{fixed.stem}_log.csv"
    log_dual = dumps / f"{dual.stem}_log.csv"
    rows = []
    def _summ(path: Path, mode: str) -> dict:
        if not path.exists():
            return {"mode": mode, "MOS_mean": float("nan"), "viol_lat": float("nan"), "viol_eng": float("nan"), "viol_sem": float("nan"),
                    "var_lat": float("nan"), "var_eng": float("nan"), "var_sem": float("nan")}
        df = pd.read_csv(path)
        mos_mean = float(df.get("mos", pd.Series([float("nan")]*len(df))).mean())
        v_lat = df.get("violate_lat", pd.Series([float("nan")]*len(df))).to_numpy(dtype=float)
        v_eng = df.get("violate_eng", pd.Series([float("nan")]*len(df))).to_numpy(dtype=float)
        v_sem = df.get("violate_sem", pd.Series([float("nan")]*len(df))).to_numpy(dtype=float)
        return {
            "mode": mode,
            "MOS_mean": mos_mean,
            "viol_lat": float(np.nanmean(v_lat)),
            "viol_eng": float(np.nanmean(v_eng)),
            "viol_sem": float(np.nanmean(v_sem)),
            "var_lat": float(np.nanvar(v_lat)),
            "var_eng": float(np.nanvar(v_eng)),
            "var_sem": float(np.nanvar(v_sem)),
        }
    rows.append(_summ(log_fixed, "fixed_dpp"))
    rows.append(_summ(log_dual, "primal_dual"))
    csv_path = errs / "micro_ab_summary.csv"
    pd.DataFrame(rows)[["mode","MOS_mean","viol_lat","viol_eng","viol_sem","var_lat","var_sem","var_eng"]].to_csv(csv_path, index=False)
    # Boxplot of violations
    fig_path = errs / "Fig_micro_violation_boxplot.png"
    if log_fixed.exists() and log_dual.exists():
        df_f = pd.read_csv(log_fixed)
        df_d = pd.read_csv(log_dual)
        data = [
            df_f.get("violate_lat", pd.Series([np.nan]*len(df_f))).to_numpy(dtype=float),
            df_f.get("violate_sem", pd.Series([np.nan]*len(df_f))).to_numpy(dtype=float),
            df_d.get("violate_lat", pd.Series([np.nan]*len(df_d))).to_numpy(dtype=float),
            df_d.get("violate_sem", pd.Series([np.nan]*len(df_d))).to_numpy(dtype=float),
        ]
        labels = ["fixed_lat", "fixed_sem", "dual_lat", "dual_sem"]
        plt.figure(figsize=(7, 4.5)); plt.boxplot(data, labels=labels)
        plt.title("Micro A/B violations boxplot")
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    else:
        plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] micro logs", ha="center"); plt.axis("off")
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    return {"csv": csv_path.as_posix(), "fig": fig_path.as_posix()}


def s4_nonstationarity(outdir: Path) -> dict:
    errs = outdir / "error_analysis"
    _ensure_dir(errs)
    log_path, tag = _choose_log()
    csv_path = errs / "nonstationary_regimes.csv"
    fig_v = errs / "Fig_regime_violations.png"
    fig_r = errs / "Fig_regime_regret.png"
    if not (log_path and log_path.exists()):
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("regime_id,t_start,t_end,sinr_mean,viol_lat,viol_eng,viol_sem,pseudo_regret\n")
        for ph in [fig_v, fig_r]:
            plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] log", ha="center"); plt.axis("off")
            plt.tight_layout(); plt.savefig(ph, dpi=150); plt.close()
        return {"csv": csv_path.as_posix(), "fig_viol": fig_v.as_posix(), "fig_reg": fig_r.as_posix()}
    df = pd.read_csv(log_path)
    T = len(df)
    sinr = df.get("sinr_db", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    # window size
    W = 50 if T >= 100 else max(10, T//4)
    means = pd.Series(sinr).rolling(W, min_periods=max(5, W//2)).mean().to_numpy()
    # detect regime breaks when diff > 2 dB
    breaks = [0]
    for i in range(W, T):
        if np.isfinite(means[i]) and np.isfinite(means[i-1]):
            if abs(means[i] - means[i-1]) > 2.0:
                if i - breaks[-1] >= W:
                    breaks.append(i)
    if breaks[-1] != T:
        breaks.append(T)
    # summarize regimes
    rows = []
    reg_ids = list(range(1, len(breaks)))
    v_lat = df.get("violate_lat", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    v_eng = df.get("violate_eng", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    v_sem = df.get("violate_sem", pd.Series([np.nan]*T)).to_numpy(dtype=float)
    pseudo_reg = None
    if "arm_id" in df.columns:
        # compute approximate pseudo-regret per regime against best mean MOS arm in regime
        mos = df.get("mos", pd.Series([np.nan]*T)).to_numpy(dtype=float)
        arm = df["arm_id"].to_numpy(dtype=int)
        pseudo_reg = np.zeros(T, dtype=float)
        for ridx in range(len(breaks)-1):
            s, e = breaks[ridx], breaks[ridx+1]
            seg = slice(s, e)
            # best arm by mean mos in segment
            seg_df = df.iloc[s:e]
            g = seg_df.groupby("arm_id")["mos"].mean()
            best_mos = float(g.max()) if len(g) > 0 else float(np.nan)
            pseudo_reg[seg] = np.maximum(best_mos - mos[seg], 0.0)
    for ridx in range(len(breaks)-1):
        s, e = breaks[ridx], breaks[ridx+1]
        seg = slice(s, e)
        rows.append({
            "regime_id": ridx + 1,
            "t_start": s + 1,
            "t_end": e,
            "sinr_mean": float(np.nanmean(sinr[seg])),
            "viol_lat": float(np.nanmean(v_lat[seg])),
            "viol_eng": float(np.nanmean(v_eng[seg])),
            "viol_sem": float(np.nanmean(v_sem[seg])),
            "pseudo_regret": float(np.nansum(pseudo_reg[seg])) if pseudo_reg is not None else float("nan"),
        })
    pd.DataFrame(rows)[["regime_id","t_start","t_end","sinr_mean","viol_lat","viol_eng","viol_sem","pseudo_regret"]].to_csv(csv_path, index=False)
    # Figures
    # violations per regime
    plt.figure(figsize=(7, 4.5))
    plt.bar([f"R{i}" for i in reg_ids], [r["viol_lat"] for r in rows], label="lat", alpha=0.7)
    plt.bar([f"R{i}" for i in reg_ids], [r["viol_sem"] for r in rows], label="sem", alpha=0.7)
    plt.bar([f"R{i}" for i in reg_ids], [r["viol_eng"] for r in rows], label="eng", alpha=0.7)
    plt.legend(); plt.title("Regime violations")
    plt.tight_layout(); plt.savefig(fig_v, dpi=150); plt.close()
    # pseudo regret
    plt.figure(figsize=(7, 4.5))
    plt.bar([f"R{i}" for i in reg_ids], [r["pseudo_regret"] for r in rows])
    plt.title("Regime pseudo regret")
    plt.tight_layout(); plt.savefig(fig_r, dpi=150); plt.close()
    return {"csv": csv_path.as_posix(), "fig_viol": fig_v.as_posix(), "fig_reg": fig_r.as_posix()}


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def s6_semantic_quantile_fit(outdir: Path) -> dict:
    errs = outdir / "error_analysis"
    _ensure_dir(errs)
    trace, trace_path = _load_trace()
    rel = _release_cfg()
    budget = float(rel.get("semantic_budget", 0.35))
    a, b = get_sem_params()
    yaml_out = errs / "semantic_recalibration.yaml"
    fig = errs / "Fig_swer_sigmoid_before_after.png"
    if trace is None:
        with yaml_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"a_current": float(a), "b_current": float(b), "a_candidate": float(a), "b_candidate": float(b), "source": trace_path}, f, sort_keys=False)
        plt.figure(figsize=(7, 4.5)); plt.text(0.5, 0.5, "[MISSING] trace", ha="center"); plt.axis("off")
        plt.tight_layout(); plt.savefig(fig, dpi=150); plt.close()
        return {"yaml": yaml_out.as_posix(), "fig": fig.as_posix(), "q": float("nan")}
    q = float(np.quantile(trace, 0.65))
    sinr_lin = float(10.0 ** (q / 10.0))
    # Fit b to align sWER(sigmoid(a*sinr_lin + b)) = budget
    b_new = float(_logit(budget) - a * sinr_lin)
    with yaml_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump({
            "a_current": float(a), "b_current": float(b), "a_candidate": float(a), "b_candidate": float(b_new),
            "budget": float(budget), "quantile": 0.65, "sinr_db_at_q": float(q), "sinr_lin_at_q": float(sinr_lin),
            "source": trace_path
        }, f, sort_keys=False)
    # Plot before/after curves over sinr range
    x_db = np.linspace(float(np.nanmin(trace)), float(np.nanmax(trace)), 200) if np.isfinite(trace).all() else np.linspace(-10, 30, 200)
    y_before = swer_from_sinr(x_db, a, b)
    y_after = swer_from_sinr(x_db, a, b_new)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_db, y_before, label=f"before (a={a:.3f}, b={b:.3f})")
    plt.plot(x_db, y_after, label=f"after (a={a:.3f}, b={b_new:.3f})")
    plt.axhline(budget, color="gray", linestyle="--", label="semantic_budget")
    plt.axvline(q, color="gray", linestyle=":", label="sinr q=0.65")
    plt.xlabel("SINR (dB)"); plt.ylabel("sWER"); plt.title("sWER sigmoid before/after (quantile alignment)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(fig, dpi=150); plt.close()
    return {"yaml": yaml_out.as_posix(), "fig": fig.as_posix(), "q": float(q)}


def _file_manifest(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        out.append(p if p and isinstance(p, str) else "[MISSING]")
    return out


def main() -> None:
    outdir = Path("outputs")
    _ensure_dir(outdir / "reports")
    # S1
    s1 = s1_semantic_feasibility(outdir)
    # S2
    s2 = s2_service_residuals(outdir)
    # S3 (micro A/B)
    s3 = s3_micro_ab(outdir)
    # S4
    s4 = s4_nonstationarity(outdir)
    # S6 (optional)
    s6 = s6_semantic_quantile_fit(outdir)

    # S5: structured TXT report
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    short, branch = _git_meta()
    py_ver = f"Python {sys.version.split()[0]}"
    cwd = str(Path.cwd())

    lines: List[str] = []
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- cwd: {cwd}")

    lines.append("\n=== 1. Semantic Feasibility ===")
    lines.append(f"- infeasible_overall: {s1.get('infeasible_overall', float('nan'))}")
    lines.append(f"- gamma_sem P10/Median/P90: {s1.get('gamma_p10', float('nan'))}, {s1.get('gamma_median', float('nan'))}, {s1.get('gamma_p90', float('nan'))}")
    lines.append(f"- figs: {s1['fig_cdf']}, {s1['fig_infeasible']}")

    lines.append("\n=== 2. Service Residuals ===")
    lines.append(f"- dS mean±var (mid-high SINR): {s2.get('bias_mean', float('nan'))} ± {s2.get('bias_var', float('nan'))}")
    lines.append(f"- bias_direction: {s2.get('bias_dir', '[MISSING]')}")
    lines.append(f"- fig: {s2['fig']}")

    lines.append("\n=== 3. DPP vs Primal–Dual (Micro A/B) ===")
    lines.append(f"- summary_csv: {s3['csv']}")
    lines.append(f"- fig: {s3['fig']}")

    lines.append("\n=== 4. Non-stationarity ===")
    lines.append(f"- regimes_csv: {s4['csv']}")
    lines.append(f"- figs: {s4['fig_viol']}, {s4['fig_reg']}")

    lines.append("\n=== 5. Actionable Fixes ===")
    # semantic infeasible > 2%
    inf_rate = s1.get('infeasible_overall', float('nan'))
    if isinstance(inf_rate, float) and inf_rate == inf_rate and inf_rate > 0.02:
        lines.append("* 建议：以当前 trace 的 SINR 分布，采用 60–70% 分位对齐重标定 (a,b)，或将 semantic_budget 放宽至可达边界。")
    else:
        lines.append("* 语义约束：总体不可达率低于阈值或数据缺失，无需立即更改；可保留量化拟合供参考。")
    # service residual bias
    dir_hint = s2.get('bias_dir', '[MISSING]')
    if dir_hint == 'negative':
        lines.append("* 抽象偏置：ΔS 在中高 SINR 区显著为负 → 抽象偏悲观；建议引入/校正 EESM/MIESM 的 PER–SINR 抽象。")
    elif dir_hint == 'positive':
        lines.append("* 抽象偏置：ΔS 为正 → 抽象偏乐观；建议重新校准 PER–SINR 对齐实测。")
    else:
        lines.append("* 抽象偏置：数据不足或偏置不显著；建议保留现有抽象并持续监测。")
    # micro A/B (primal–dual)
    lines.append("* 在线 λ：如 A/B 中 primal–dual 违约更低，建议上线在线 λ（primal–dual）。")
    # regime
    lines.append("* 非平稳：如存在 regime 变化，建议采用 SW-UCB/折扣 UCB 作为 bandit 默认策略以适应非平稳环境。")

    lines.append("\n=== 6. File Manifest ===")
    manifest = [
        s1.get("csv"), s1.get("fig_cdf"), s1.get("fig_infeasible"),
        s2.get("csv"), s2.get("fig"),
        s3.get("csv"), s3.get("fig"),
        s4.get("csv"), s4.get("fig_viol"), s4.get("fig_reg"),
        s6.get("yaml"), s6.get("fig"),
    ]
    lines.extend([f"- {p}" for p in _file_manifest(manifest)])
    rep = outdir / "reports" / "task1_error_analysis.txt"
    with rep.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] {rep}")


if __name__ == "__main__":
    main()