"""
Path Usage Note:
All paths have been adapted to relative paths for independent execution.
Ensure this script is run from its own directory.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig_violation_over_time(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "t" not in df.columns:
        print("[WARN] fig11: missing data/time column")
        return
    violate_cols = []
    for cand in ["violate_lat", "violate_eng", "violate_sem", "q_violate_flag", "d_violate_flag", "energy_violate_flag", "violation_flag"]:
        if cand in df.columns:
            violate_cols.append(cand)
    if not violate_cols:
        print("[WARN] fig11: no violation columns found")
        return
    plt.figure(figsize=(8, 4))
    plotted = False
    t_int = pd.to_numeric(df["t"], errors="coerce").astype(int)
    for col in violate_cols:
        sub = df.copy()
        sub["t_int"] = t_int
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        grouped = sub.groupby("t_int")[col].mean().dropna()
        if grouped.empty:
            continue
        window = 200 if len(grouped) > 500 else 100
        smoothed = grouped.rolling(window=window, min_periods=max(10, window // 5)).mean()
        plt.plot(smoothed.index, smoothed.values, linewidth=2, label=col)
        plotted = True
    if not plotted:
        print("[WARN] fig11: nothing plotted")
        return
    plt.xlabel("t")
    plt.ylabel("violation prob")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig11_violation_over_time.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_violation_over_time(df_all)
