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

def fig_grid_density_vs_regret(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig18: empty dataframe")
        return
    # Note: grid_density filling is handled by normalize_columns in load_all_dumps
    if "grid_density" not in df.columns:
        print("[WARN] fig18: grid_density not found")
        return
    if "regret_t" not in df.columns and "instant_regret" not in df.columns and "qoe" not in df.columns:
        print("[WARN] fig18: no regret/qoe data")
        return
    records = []
    for file, sub in df.groupby("file"):
        regret = None
        if "regret_t" in sub.columns:
            val = pd.to_numeric(sub["regret_t"], errors="coerce").dropna()
            regret = val.iloc[-1] if not val.empty else None
        if regret is None and "instant_regret" in sub.columns:
            regret = pd.to_numeric(sub["instant_regret"], errors="coerce").dropna().sum()
        if regret is None and "qoe" in sub.columns:
            q = pd.to_numeric(sub["qoe"], errors="coerce").dropna()
            if not q.empty:
                regret = (q.max() - q).clip(lower=0).sum()
        if regret is None:
            continue
        g_val = sub["grid_density"].dropna().iloc[0] if sub["grid_density"].notna().any() else np.nan
        records.append({"grid_density": g_val, "regret": regret})
    if not records:
        print("[WARN] fig18: no regret records")
        return
    rec_df = pd.DataFrame(records).dropna(subset=["grid_density"])
    if rec_df["grid_density"].nunique() <= 1:
        print("[WARN] fig18: only one grid_density value, skip plotting")
        return
    agg = rec_df.groupby("grid_density")["regret"].mean().reset_index().sort_values("grid_density")
    plt.figure(figsize=(7, 4))
    plt.plot(agg["grid_density"], agg["regret"], marker="o", linewidth=2)
    plt.xlabel("grid_density")
    plt.ylabel("R_T")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig18_grid_density_vs_regret.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_grid_density_vs_regret(df_all)
