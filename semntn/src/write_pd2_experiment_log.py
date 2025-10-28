from __future__ import annotations

import os
import sys
import math
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import yaml


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _git_meta() -> Tuple[str, str]:
    try:
        short_hash = subprocess.check_output([
            "git", "rev-parse", "--short", "HEAD"
        ], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output([
            "git", "rev-parse", "--abbrev-ref", "HEAD"
        ], stderr=subprocess.DEVNULL).decode().strip()
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


def _fmt_float(x: Optional[float]) -> str:
    try:
        if x is None:
            return "[MISSING]"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "[MISSING]"
        return f"{float(x):.4f}"
    except Exception:
        return "[MISSING]"


def _fmt_bool(x: Optional[bool]) -> str:
    try:
        return "YES" if bool(x) else "NO"
    except Exception:
        return "[MISSING]"


def _inventory_line(rel: str) -> tuple[str, bool]:
    p = Path(rel)
    if p.exists():
        try:
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            return (f"EXIST {rel} size={st.st_size} bytes, mtime={mtime}", True)
        except Exception:
            return (f"EXIST {rel}", True)
    else:
        return (f"MISSING {rel}", False)


def _find_wirecheck_csv() -> Path | None:
    primary = Path("outputs/dumps/task1_wirecheck_log.csv")
    if primary.exists():
        return primary
    dumps = Path("outputs/dumps")
    if dumps.exists():
        # Prefer names containing 'wirecheck' and 'pd2'
        candidates = sorted(list(dumps.glob("*wirecheck*.csv")))
        if candidates:
            for c in candidates:
                if "pd2" in c.name.lower():
                    return c
            return candidates[0]
    return None


def _wirecheck_rates(csv_path: Path) -> tuple[float | None, float | None, float | None, float | None]:
    try:
        df = pd.read_csv(csv_path)
        T = len(df)
        pd_rate = float(df.get("pd_active", pd.Series([np.nan] * T)).mean())
        sem_rate = float(df.get("guard_sem_active", pd.Series([np.nan] * T)).mean())
        q_rate = float(df.get("guard_queue_active", pd.Series([np.nan] * T)).mean())
        eng_rate = float(df.get("guard_energy_active", pd.Series([np.nan] * T)).mean())
        return pd_rate, sem_rate, q_rate, eng_rate
    except Exception:
        return None, None, None, None


def _release_pd2_snapshot(csv_path: Path) -> dict:
    out = {
        "T": None,
        "MOS_mean": None,
        "E_slot_J_mean": None,
        "sWER_mean": None,
        "v_lat": None,
        "v_eng": None,
        "v_sem": None,
        "MOS_uplift": None,
    }
    try:
        df = pd.read_csv(csv_path)
        T = len(df)
        out["T"] = float(T)
        out["MOS_mean"] = float(df["mos"].mean())
        # Energy per slot
        energy_col = "E_slot_J" if "E_slot_J" in df.columns else ("P" if "P" in df.columns else None)
        out["E_slot_J_mean"] = float(df[energy_col].mean()) if energy_col else None
        # sWER
        if "sWER_raw" in df.columns:
            out["sWER_mean"] = float(df["sWER_raw"].mean())
        elif "sWER" in df.columns:
            out["sWER_mean"] = float(df["sWER"].mean())
        else:
            out["sWER_mean"] = None
        # Violations
        if set(["violate_lat", "violate_eng", "violate_sem"]).issubset(df.columns):
            out["v_lat"] = float(df["violate_lat"].mean())
            out["v_eng"] = float(df["violate_eng"].mean())
            out["v_sem"] = float(df["violate_sem"].mean())
        else:
            # Attempt fallback via z queue increments
            try:
                z_lat = df["z_lat"].to_numpy(dtype=float)
                z_eng = df["z_eng"].to_numpy(dtype=float)
                z_sem = df["z_sem"].to_numpy(dtype=float)
                out["v_lat"] = float(np.mean(np.diff(np.concatenate([[0.0], z_lat])) > 0.0))
                out["v_eng"] = float(np.mean(np.diff(np.concatenate([[0.0], z_eng])) > 0.0))
                out["v_sem"] = float(np.mean(np.diff(np.concatenate([[0.0], z_sem])) > 0.0))
            except Exception:
                out["v_lat"] = None
                out["v_eng"] = None
                out["v_sem"] = None
        # MOS uplift vs baseline
        try:
            baseline = float((1.0 - df["per"].to_numpy(dtype=float)).mean())
            out["MOS_uplift"] = float(out["MOS_mean"] - baseline)
        except Exception:
            out["MOS_uplift"] = None
    except Exception:
        pass
    return out


def _best_v_from_summary(summary_csv: Path) -> tuple[float | None, pd.DataFrame | None, bool]:
    try:
        vdf = pd.read_csv(summary_csv)
        feasible = vdf[(vdf["viol_lat"] <= 0.01) & (vdf["viol_eng"] <= 0.01) & (vdf["viol_sem"] <= 0.01)]
        if len(feasible) > 0:
            idx = int(feasible["MOS_mean"].idxmax())
            return float(vdf.loc[idx, "V"]), vdf.loc[[idx]], True
        # fallback: smallest sum of violations, then max MOS
        vdf = vdf.copy()
        vdf["viol_sum"] = vdf[["viol_lat", "viol_eng", "viol_sem"]].sum(axis=1)
        min_sum = float(vdf["viol_sum"].min())
        candidates = vdf[vdf["viol_sum"] == min_sum]
        idx = int(candidates["MOS_mean"].idxmax())
        return float(vdf.loc[idx, "V"]), vdf.loc[[idx]], False
    except Exception:
        return None, None, False


def _error_analysis_text_summary(path: Path) -> dict:
    res = {
        "semantic_infeas_rate": None,
        "gamma_sem_median": None,
        "service_residual_has_nan": None,
    }
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        # naive parse by token search
        for token in ["semantic_infeas_rate", "gamma_sem_median", "service_residual_has_nan"]:
            i = txt.find(token)
            if i >= 0:
                tail = txt[i:].splitlines()[0]
                # extract after '=' if present
                if "=" in tail:
                    val = tail.split("=", 1)[1].strip()
                    if token == "service_residual_has_nan":
                        res[token] = (val.upper().startswith("Y") or val.upper() == "TRUE")
                    else:
                        try:
                            res[token] = float(val)
                        except Exception:
                            res[token] = None
        return res
    except Exception:
        return res


def _semantic_margin_summary(path: Path) -> tuple[float | None, float | None, float | None]:
    try:
        df = pd.read_csv(path)
        col = None
        for c in df.columns:
            if c.lower().startswith("gamma_sem"):
                col = c
                break
        if col is None:
            return None, None, None
        x = df[col].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return None, None, None
        infeas_rate = float(np.mean(x < 0.0))
        p50 = float(np.quantile(x, 0.5))
        p90 = float(np.quantile(x, 0.9))
        return infeas_rate, p50, p90
    except Exception:
        return None, None, None


def _service_residual_summary(path: Path) -> tuple[float | None, float | None, int | None]:
    try:
        df = pd.read_csv(path)
        col = None
        for c in df.columns:
            if c.lower() == "ds" or c.lower().startswith("d_s"):
                col = c
                break
        if col is None:
            # try first numeric column
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    col = c
                    break
        if col is None:
            return None, None, None
        x = df[col].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return None, None, 0
        return float(np.mean(x)), float(np.std(x)), int(len(x))
    except Exception:
        return None, None, None


def main() -> None:
    # Targets and inputs
    out_log = Path("outputs/reports/task1_experiment_log.txt")
    _ensure_dir(out_log.parent)

    # Metadata
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    short, branch = _git_meta()
    py_ver = f"Python {sys.version.split()[0]}"
    cwd = str(Path.cwd())

    lines: List[str] = []
    # === 0. Metadata ===
    lines.append("=== 0. Metadata ===")
    lines.append(f"- time: {ts}")
    lines.append(f"- git_short: {short}")
    lines.append(f"- git_branch: {branch}")
    lines.append(f"- python: {py_ver}")
    lines.append(f"- cwd: {cwd}")

    # Inventory
    lines.append("=== 1. Run Inventory ===")
    inventory_paths = [
        "outputs/reports/task1_acceptance_signoff_pd2.txt",
        "outputs/dumps/task1_release_pd2_log.csv",
        "outputs/dumps/task1_release_pd2_resolved.yaml",
        "outputs/dumps/task1_wirecheck_log.csv",
        "outputs/vscan/summary.csv",
        "outputs/figs/Fig_MOS_vs_V.png",
        "outputs/figs/Fig_ViolationRates_vs_V.png",
        "outputs/figs/Fig_Pareto_MOS_vs_Energy.png",
        "outputs/reports/task1_error_analysis.txt",
        "outputs/error_analysis/semantic_margin.csv",
        "outputs/error_analysis/service_residual_samples.csv",
        "outputs/error_analysis/Fig_semantic_feasibility_CDF.png",
        "outputs/error_analysis/Fig_service_residual_heatmap.png",
    ]
    missing_files: List[str] = []

    # Special handling for wirecheck
    wirecheck_found = _find_wirecheck_csv()
    if wirecheck_found and wirecheck_found.as_posix() != "outputs/dumps/task1_wirecheck_log.csv":
        inventory_paths[3] = wirecheck_found.as_posix()

    for rel_path in inventory_paths:
        msg, ok = _inventory_line(rel_path)
        lines.append(msg)
        if not ok:
            missing_files.append(rel_path)

    # Config Snapshot
    lines.append("=== 2. Config Snapshot ===")
    resolved_path = Path("outputs/dumps/task1_release_pd2_resolved.yaml")
    cfg_pd2_path = Path("configs/task1_release_pd2.yaml")
    resolved = _read_yaml_safe(resolved_path)
    # fallback: attempt parse from acceptance signoff or base cfg
    acc_path = Path("outputs/reports/task1_acceptance_signoff_pd2.txt")
    base_cfg = _read_yaml_safe(cfg_pd2_path) if cfg_pd2_path.exists() else {}
    try:
        V = resolved.get("V", base_cfg.get("V"))
        slot_sec = resolved.get("slot_sec", base_cfg.get("slot_sec"))
        arrivals_bps = resolved.get("arrivals_bps", base_cfg.get("arrivals_bps"))
        energy_budget = resolved.get("energy_budget_per_slot_j", base_cfg.get("energy_budget_per_slot_j"))
        latency_budget = resolved.get("latency_budget_q_bps", base_cfg.get("latency_budget_q_bps"))
        semantic_budget = resolved.get("semantic_budget", base_cfg.get("semantic_budget"))
        scales = resolved.get("scales", base_cfg.get("scales", {}))
        enable_pd = resolved.get("enable_primal_dual", base_cfg.get("enable_primal_dual", False))
        pd_eta = (resolved.get("pd", {}) or {}).get("eta", (base_cfg.get("pd", {}) or {}).get("eta", base_cfg.get("eta_dual")))
        pd_delta = (resolved.get("pd", {}) or {}).get("delta_queue", (base_cfg.get("pd", {}) or {}).get("delta_queue", base_cfg.get("delta_queue")))
        guard_sem = (resolved.get("guard", {}) or {}).get("semantic", (base_cfg.get("guard", {}) or {}).get("semantic"))
        guard_queue = (resolved.get("guard", {}) or {}).get("queue", (base_cfg.get("guard", {}) or {}).get("queue"))
        P_max = base_cfg.get("P_max")
        B_grid = base_cfg.get("B_grid")
        B_min = (min(B_grid) if isinstance(B_grid, (list, tuple)) and len(B_grid) > 0 else None)
        B_len = (len(B_grid) if isinstance(B_grid, (list, tuple)) else None)
        lines.append(
            "RESOLVED: "
            f"V={_fmt_float(V)}, slot_sec={_fmt_float(slot_sec)} s, arrivals_bps={_fmt_float(arrivals_bps)} bps, "
            f"energy_budget_per_slot_j={_fmt_float(energy_budget)} J/slot, latency_budget_q_bps={_fmt_float(latency_budget)}, "
            f"semantic_budget={_fmt_float(semantic_budget)}, scales={{queue:{_fmt_float((scales or {}).get('queue_scale'))}, energy:{_fmt_float((scales or {}).get('energy_scale'))}, semantic:{_fmt_float((scales or {}).get('semantic_scale'))}}}, "
            f"enable_primal_dual={_fmt_bool(enable_pd)}, pd.eta={_fmt_float(pd_eta)}, pd.delta_queue={_fmt_float(pd_delta)}, "
            f"guard.semantic={_fmt_bool(guard_sem)}, guard.queue={_fmt_bool(guard_queue)}, P_max={_fmt_float(P_max)}, "
            f"B_min={_fmt_float(B_min)}, |B_grid|={_fmt_float(B_len)}"
        )
    except Exception:
        lines.append("RESOLVED: [MISSING]")

    # Wirecheck section
    lines.append("=== 3. Wirecheck ===")
    wc_path = wirecheck_found or Path("outputs/dumps/task1_wirecheck_log.csv")
    pd_rate, sem_rate, q_rate, eng_rate = (None, None, None, None)
    if wc_path and wc_path.exists():
        pd_rate, sem_rate, q_rate, eng_rate = _wirecheck_rates(wc_path)
    lines.append(f"pd_active_rate={_fmt_float(pd_rate)}, guard_sem_rate={_fmt_float(sem_rate)}, guard_queue_rate={_fmt_float(q_rate)}, guard_energy_rate={_fmt_float(eng_rate)}")
    def _ge95(x: Optional[float]) -> str:
        try:
            return "YES" if (x is not None and not math.isnan(x) and float(x) >= 0.95) else "NO"
        except Exception:
            return "NO"
    lines.append(f"ge_0.95? pd_active={_ge95(pd_rate)}, guard_sem={_ge95(sem_rate)}, guard_queue={_ge95(q_rate)}, guard_energy={_ge95(eng_rate)}")

    # Release PD2 snapshot
    lines.append("=== 4. Release PD2 Snapshot ===")
    rel_csv = Path("outputs/dumps/task1_release_pd2_log.csv")
    rel_snap = _release_pd2_snapshot(rel_csv)
    lines.append(
        f"T={_fmt_float(rel_snap['T'])}, MOS_mean={_fmt_float(rel_snap['MOS_mean'])}, E_slot_J_mean={_fmt_float(rel_snap['E_slot_J_mean'])}, "
        f"sWER_mean={_fmt_float(rel_snap['sWER_mean'])}, violation(lat,eng,sem)=({_fmt_float(rel_snap['v_lat'])}, {_fmt_float(rel_snap['v_eng'])}, {_fmt_float(rel_snap['v_sem'])}), "
        f"MOS_uplift_vs_baseline={_fmt_float(rel_snap['MOS_uplift'])}"
    )
    def _le01(x: Optional[float]) -> str:
        try:
            return "YES" if (x is not None and not math.isnan(x) and float(x) <= 0.01) else "NO"
        except Exception:
            return "NO"
    lines.append(
        f"violations_le_1pct? lat={_le01(rel_snap['v_lat'])}({_fmt_float(rel_snap['v_lat'])}), eng={_le01(rel_snap['v_eng'])}({_fmt_float(rel_snap['v_eng'])}), sem={_le01(rel_snap['v_sem'])}({_fmt_float(rel_snap['v_sem'])})"
    )

    # V-sweep coverage and decision
    lines.append("=== 5. V-sweep ===")
    vscan_csv = Path("outputs/vscan/summary.csv")
    header_text = "[MISSING]"
    row_text = "[MISSING]"
    K = None
    v_feasible = False
    try:
        if vscan_csv.exists():
            vdf = pd.read_csv(vscan_csv)
            header_text = ",".join(vdf.columns.tolist())
            K = int(len(vdf))
            V_star, row_star, feasible = _best_v_from_summary(vscan_csv)
            v_feasible = bool(feasible)
            if row_star is not None:
                row_text = ",".join(str(row_star.iloc[0][c]) for c in vdf.columns)
                if not feasible:
                    row_text = "[NO-FEASIBLE] " + row_text
    except Exception:
        pass
    lines.append(header_text)
    lines.append(f"rows={K if K is not None else '[MISSING]'}")
    if K is not None and K < 3:
        lines.append("[FAIL_VSCAN: rows<3]")
    lines.append(row_text)
    # Figures
    fig_paths = [
        "outputs/figs/Fig_MOS_vs_V.png",
        "outputs/figs/Fig_ViolationRates_vs_V.png",
        "outputs/figs/Fig_Pareto_MOS_vs_Energy.png",
    ]
    for fp in fig_paths:
        lines.append(fp if Path(fp).exists() else "[MISSING]")

    # Error Analysis summary
    lines.append("=== 6. Error Analysis ===")
    ea_txt = Path("outputs/reports/task1_error_analysis.txt")
    ea = _error_analysis_text_summary(ea_txt) if ea_txt.exists() else {
        "semantic_infeas_rate": None,
        "gamma_sem_median": None,
        "service_residual_has_nan": None,
    }
    lines.append(
        f"semantic_infeas_rate={_fmt_float(ea['semantic_infeas_rate'])}, "
        f"gamma_sem_median={_fmt_float(ea['gamma_sem_median'])}, "
        f"service_residual_has_nan={_fmt_bool(ea['service_residual_has_nan'])}"
    )
    # semantic_margin.csv
    sm_csv = Path("outputs/error_analysis/semantic_margin.csv")
    infeas_rate, p50, p90 = _semantic_margin_summary(sm_csv) if sm_csv.exists() else (None, None, None)
    if sm_csv.exists():
        lines.append(
            f"semantic_margin: overall_infeas_rate={_fmt_float(infeas_rate)}, gamma_sem_P50={_fmt_float(p50)}, gamma_sem_P90={_fmt_float(p90)}"
        )
    else:
        lines.append("semantic_margin: [MISSING]")
    # service_residual_samples.csv
    sr_csv = Path("outputs/error_analysis/service_residual_samples.csv")
    mu, sd, n = _service_residual_summary(sr_csv) if sr_csv.exists() else (None, None, None)
    if sr_csv.exists():
        lines.append(
            f"service_residual: dS_mean={_fmt_float(mu)}, dS_std={_fmt_float(sd)}, count={n if n is not None else '[MISSING]'}"
        )
    else:
        lines.append("service_residual: [MISSING]")

    # Exceptions and warnings (AUTO)
    lines.append("=== 7. Exceptions & Warnings ===")
    for rel_path in inventory_paths:
        if not Path(rel_path).exists():
            lines.append(f"[MISSING] {rel_path}")
    # wirecheck inactive
    if pd_rate is not None and not math.isnan(pd_rate) and pd_rate < 0.95:
        lines.append("[WARN_PD_OR_GUARD_INACTIVE] pd_active_rate<0.95")
    if sem_rate is not None and not math.isnan(sem_rate) and sem_rate < 0.95:
        lines.append("[WARN_PD_OR_GUARD_INACTIVE] guard_sem_rate<0.95")
    if q_rate is not None and not math.isnan(q_rate) and q_rate < 0.95:
        lines.append("[WARN_PD_OR_GUARD_INACTIVE] guard_queue_rate<0.95")
    if eng_rate is not None and not math.isnan(eng_rate) and eng_rate < 0.95:
        lines.append("[WARN_PD_OR_GUARD_INACTIVE] guard_energy_rate<0.95")
    # release violation exceeds
    for name in ["v_lat", "v_eng", "v_sem"]:
        val = rel_snap.get(name)
        if val is not None and not math.isnan(val) and val > 0.01:
            lines.append(f"[WARN_VIOLATION_EXCEEDS_TARGET] {name}={_fmt_float(val)}")
    # vscan insufficient
    if K is not None and K < 3:
        lines.append("[WARN_INSUFFICIENT_VSCAN_COVERAGE]")

    # Reproduction commands (record only)
    lines.append("=== 8. Reproduction Commands ===")
    lines.append("python -m semntn.src.run_task1 --config configs/task1_release_pd2.yaml")
    lines.append("python semntn/src/analyze_task1.py --log outputs/dumps/task1_release_pd2_log.csv --outdir outputs --config-name task1_release_pd2")
    lines.append("python -m semntn.src.run_vscan --base-config configs/task1_release_pd2.yaml --vscan-config configs/vscan.yaml")
    lines.append("python semntn/src/analyze_task1.py --vscan outputs/vscan/summary.csv --outdir outputs/figs")

    # Manifest
    lines.append("=== 9. File Manifest ===")
    manifest = set(inventory_paths)
    # include resolved and wirecheck found
    manifest.add("outputs/dumps/task1_release_pd2_log.csv")
    manifest.add("outputs/dumps/task1_release_pd2_resolved.yaml")
    if wirecheck_found:
        manifest.add(wirecheck_found.as_posix())
    for fp in fig_paths:
        manifest.add(fp)
    manifest.add("outputs/vscan/summary.csv")
    manifest.add("outputs/reports/task1_error_analysis.txt")
    manifest.add("outputs/error_analysis/semantic_margin.csv")
    manifest.add("outputs/error_analysis/service_residual_samples.csv")
    manifest.add("outputs/error_analysis/Fig_semantic_feasibility_CDF.png")
    manifest.add("outputs/error_analysis/Fig_service_residual_heatmap.png")
    for rel in sorted(manifest):
        lines.append(rel)

    # Write
    with out_log.open("w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
        f.write(f"[LOG] {out_log.as_posix()}\n")
    print(f"[LOG] {out_log.as_posix()}")


if __name__ == "__main__":
    main()