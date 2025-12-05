import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig14_ridgeline(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "mos" not in df.columns:
        print("[WARN] fig14: missing mos data")
        return
    work = df.copy()
    if "sem_weight" in work.columns and work["sem_weight"].notna().sum() > 0:
        bins = pd.qcut(work["sem_weight"], q=5, duplicates="drop")
    elif "sWER" in work.columns and work["sWER"].notna().sum() > 0:
        bins = pd.qcut(work["sWER"], q=5, duplicates="drop")
    else:
        print("[WARN] fig14: no sem_weight/sWER to group")
        return
    work["bin"] = bins
    groups = [(str(b), sub["mos"].dropna().values) for b, sub in work.groupby("bin")]
    groups = [(name, vals) for name, vals in groups if len(vals) >= 10]
    if not groups:
        print("[WARN] fig14: insufficient samples per bin")
        return
    all_vals = np.concatenate([g[1] for g in groups])
    grid = np.linspace(all_vals.min(), all_vals.max(), 200)

    def kde(x: np.ndarray, grid_vals: np.ndarray) -> np.ndarray:
        if len(x) < 2:
            return np.zeros_like(grid_vals)
        bw = 0.3 * np.std(x) if np.std(x) > 1e-6 else 0.1
        diffs = (grid_vals[:, None] - x[None, :]) / bw
        kern = np.exp(-0.5 * diffs**2) / (np.sqrt(2 * np.pi) * bw)
        return kern.mean(axis=1)

    plt.figure(figsize=(8, 6))
    y_offset = 0
    for name, vals in groups:
        density = kde(np.array(vals), grid)
        plt.fill_between(grid, y_offset, y_offset + density, alpha=0.6, label=name)
        plt.plot(grid, y_offset + density, color="k", linewidth=0.8)
        y_offset += density.max() * 1.2
    plt.yticks([])
    plt.xlabel("QoE (mos)")
    plt.legend(title="group")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig14_ridgeline.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig14_ridgeline(df_all)
