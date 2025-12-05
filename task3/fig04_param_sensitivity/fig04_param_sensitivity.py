"""
Path Usage Note:
All paths have been adapted to relative paths for independent execution.
Ensure this script is run from its own directory.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig_param_sensitivity(summary_csv: str, df_all: pd.DataFrame | None = None) -> None:
    ensure_outdir()
    df_source = df_all if df_all is not None and not df_all.empty else None
    if df_source is None and summary_csv and os.path.exists(summary_csv):
        try:
            df_source = pd.read_csv(summary_csv)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] fig04: cannot read summary {summary_csv}: {exc}")
            return
    if df_source is None or df_source.empty:
        print("[WARN] fig04: no data available for param sensitivity")
        return
    df_work = df_source.copy()
    if "file" not in df_work.columns and "file" in df_source.columns:
        df_work["file"] = df_source["file"]

    def _maybe_fill(col: str, pattern: str, cast) -> None:
        if col in df_work.columns and df_work[col].notna().any():
            return
        if "file" not in df_work.columns:
            return
        vals = []
        for f in df_work["file"]:
            m = re.search(pattern, os.path.basename(str(f)), re.IGNORECASE)
            vals.append(cast(m.group(1)) if m else np.nan)
        df_work[col] = vals

    _maybe_fill("window", r"w(\d+)", int)
    _maybe_fill("grid_density", r"g(\d+)", int)
    _maybe_fill("epsilon_t", r"eps(?:ilon)?(\d+\.?\d*)", float)
    _maybe_fill("L_P", r"lp(\d+\.?\d*)", float)
    _maybe_fill("L_B", r"lb(\d+\.?\d*)", float)

    if "qoe" not in df_work.columns and "mos_mean" in df_work.columns:
        df_work["qoe"] = df_work["mos_mean"]
    if "qoe" not in df_work.columns and "mos" in df_work.columns:
        df_work["qoe"] = df_work["mos"]

    params = ["window", "grid_density", "epsilon_t", "L_P", "L_B"]
    deltas = {}
    for p in params:
        if p in df_work.columns and df_work[p].notna().sum() > 1:
            grouped = df_work.dropna(subset=[p, "qoe"])
            if grouped.empty:
                continue
            means = grouped.groupby(p)["qoe"].mean()
            if len(means) > 1:
                deltas[p] = means.max() - means.min()
    if not deltas:
        print("[WARN] fig04: no parameter with multiple values to analyze")
        return

    plt.figure(figsize=(6, 4))
    names = list(deltas.keys())
    vals = [deltas[k] for k in names]
    plt.bar(names, vals, color="#4C72B0")
    plt.ylabel("Delta Ubar")
    plt.xlabel("parameter")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig04_param_sensitivity.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    summary_csv = "task3_summary.csv"
    fig_param_sensitivity(summary_csv, df_all)