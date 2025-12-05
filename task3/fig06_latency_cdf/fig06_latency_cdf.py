"""
Path Usage Note:
All paths have been adapted to relative paths for independent execution.
Ensure this script is run from its own directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI, MIN_LATENCY_SAMPLES

def fig_latency_cdf(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig06: empty dataframe")
        return
    latency_cols = [c for c in df.columns if ("latency" in c.lower() or "decision" in c.lower()) and "slot" not in c.lower()]
    if not latency_cols:
        print("[WARN] fig06: no latency columns found")
        return
    col = latency_cols[0]
    plt.figure(figsize=(7, 4))
    plotted = False
    for alg, sub in df.groupby("algorithm"):
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(vals) < MIN_LATENCY_SAMPLES:
            print(f"[WARN] fig06: skip alg={alg} due to few latency samples")
            continue
        vals = np.sort(vals)
        cdf = np.linspace(0, 1, len(vals), endpoint=False)
        plt.plot(vals, cdf, linewidth=2, label=alg)
        plotted = True
    if not plotted:
        print("[WARN] fig06: nothing to plot")
        return
    plt.xlabel("latency (ms)")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig06_latency_cdf.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_latency_cdf(df_all)