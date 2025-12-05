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

def fig_swer_vs_snr(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "sWER" not in df.columns or "snr_db" not in df.columns:
        print("[WARN] fig09: missing data for sWER vs SNR")
        return
    plt.figure(figsize=(7, 4))
    plotted = False
    for alg, sub in df.groupby("algorithm"):
        vals = sub.dropna(subset=["sWER", "snr_db"])
        if vals.empty:
            continue
        snr_vals = pd.to_numeric(vals["snr_db"], errors="coerce")
        bins = np.linspace(snr_vals.min(), snr_vals.max(), 41)
        vals["snr_bin"] = pd.cut(snr_vals, bins, include_lowest=True)
        grouped = vals.groupby("snr_bin")["sWER"].mean()
        x = [b.mid for b in grouped.index]
        y = grouped.values
        plt.plot(x, y, linewidth=2, marker="o", label=alg)
        plotted = True
    if not plotted:
        print("[WARN] fig09: nothing plotted")
        return
    plt.xlabel("SNR (dB)")
    plt.ylabel("sWER")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig09_swer_vs_snr.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_swer_vs_snr(df_all)
