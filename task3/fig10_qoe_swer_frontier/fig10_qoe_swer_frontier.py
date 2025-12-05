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

def fig_qoe_swer_frontier(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "qoe" not in df.columns or "sWER" not in df.columns:
        print("[WARN] fig10: missing qoe/sWER data")
        return
    stats = df.dropna(subset=["qoe", "sWER"])
    if stats.empty:
        print("[WARN] fig10: no data after dropna")
        return
    agg = stats.groupby("algorithm").agg(Ubar=("qoe", "mean"), Dbar=("sWER", "mean")).reset_index()
    agg = agg.sort_values("Dbar")
    agg["frontier"] = np.maximum.accumulate(agg["Ubar"])
    plt.figure(figsize=(6, 4))
    plt.scatter(agg["Dbar"], agg["Ubar"], c="#4C72B0", label="algorithms")
    plt.plot(agg["Dbar"], agg["frontier"], color="#DD8452", linewidth=2, label="Pareto frontier")
    plt.xlabel("Dbar (sWER)")
    plt.ylabel("Ubar (QoE)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig10_qoe_swer_frontier.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_qoe_swer_frontier(df_all)
