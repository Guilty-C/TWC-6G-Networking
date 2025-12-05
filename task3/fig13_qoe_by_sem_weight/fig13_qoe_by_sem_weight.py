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

def fig_qoe_by_sem_weight(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig13: empty dataframe")
        return
    work = df.copy()
    bins = None
    label = None
    if "sem_weight" in work.columns and work["sem_weight"].notna().sum() > 0:
        bins = pd.qcut(work["sem_weight"], q=5, duplicates="drop")
        label = "sem_weight"
    elif "sWER" in work.columns and work["sWER"].notna().sum() > 0:
        bins = pd.qcut(work["sWER"], q=5, duplicates="drop")
        label = "sWER proxy"
    if bins is None:
        print("[WARN] fig13: no sem_weight or sWER to bin")
        return
    work["bin"] = bins
    grouped = work.dropna(subset=["bin", "mos"]).groupby("bin")["mos"].mean()
    if grouped.empty:
        print("[WARN] fig13: empty grouped data")
        return
    plt.figure(figsize=(7, 4))
    plt.plot([str(b) for b in grouped.index], grouped.values, marker="o")
    plt.xlabel(label)
    plt.ylabel("QoE (mos)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig13_qoe_by_sem_weight.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_qoe_by_sem_weight(df_all)
