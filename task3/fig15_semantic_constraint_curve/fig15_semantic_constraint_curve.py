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

def fig_semantic_constraint_curve(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "sWER" not in df.columns or "mos" not in df.columns:
        print("[WARN] fig15: missing sWER/mos data")
        return
    thresholds = np.linspace(0.15, 0.25, 21)
    xs, ys = [], []
    for thr in thresholds:
        sub = df[pd.to_numeric(df["sWER"], errors="coerce") <= thr]
        if sub.empty:
            continue
        xs.append(thr)
        ys.append(pd.to_numeric(sub["mos"], errors="coerce").mean())
    if not xs:
        print("[WARN] fig15: no data after applying thresholds")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Dbar")
    plt.ylabel("Ubar")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig15_semantic_constraint_curve.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_semantic_constraint_curve(df_all)
