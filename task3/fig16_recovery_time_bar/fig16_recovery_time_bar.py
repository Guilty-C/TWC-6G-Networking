"""
Path Usage Note:
All paths have been adapted to relative paths for independent execution.
Ensure this script is run from its own directory.
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig_recovery_time_bar(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "mos" not in df.columns or "t" not in df.columns:
        print("[WARN] fig16: missing mos/t data")
        return
    records = []
    seed_series = df["seed"] if "seed" in df.columns else pd.Series([0] * len(df))
    for (alg, file, seed), sub in df.groupby(["algorithm", "file", seed_series]):
        mos = pd.to_numeric(sub["mos"], errors="coerce").dropna()
        t_vals = pd.to_numeric(sub["t"], errors="coerce").reindex(mos.index)
        if len(mos) < 20:
            continue
        window = max(20, int(0.05 * len(mos)))
        roll = mos.rolling(window=window, min_periods=5).mean().dropna()
        if roll.empty:
            continue
        steady = roll.iloc[-min(len(roll), 200):].mean()
        thresh = 0.9 * steady
        mask = roll >= thresh
        if not mask.any():
            continue
        first_idx = mask[mask].index[0]
        t_rec = t_vals.loc[first_idx] if first_idx in t_vals.index else None
        if t_rec is None or not math.isfinite(t_rec):
            continue
        records.append({"algorithm": alg, "t_rec": t_rec})
    if not records:
        print("[WARN] fig16: no recovery records")
        return
    rec_df = pd.DataFrame(records)
    agg = rec_df.groupby("algorithm")["t_rec"].mean().reset_index()
    plt.figure(figsize=(7, 4))
    plt.bar(agg["algorithm"], agg["t_rec"], color="#C44E52")
    plt.ylabel("recovery time (slots)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig16_recovery_time_bar.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_recovery_time_bar(df_all)
