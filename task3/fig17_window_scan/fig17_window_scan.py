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

def fig_window_scan(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig17: empty dataframe")
        return
    # Note: window filling is handled by normalize_columns in load_all_dumps
    if "window" not in df.columns:
        print("[WARN] fig17: window not found")
        return
    if "regret_t" not in df.columns and "instant_regret" not in df.columns and "qoe" not in df.columns:
        print("[WARN] fig17: no regret/qoe data")
        return
    prefer_alg = "sw_ucb"
    algs = [prefer_alg] if prefer_alg in df["algorithm"].unique() else list(df["algorithm"].unique())
    records = []
    for alg in algs:
        sub_alg = df[df["algorithm"] == alg]
        if sub_alg.empty:
            continue
        for file, sub_file in sub_alg.groupby("file"):
            regret = None
            if "regret_t" in sub_file.columns:
                val = pd.to_numeric(sub_file["regret_t"], errors="coerce").dropna()
                regret = val.iloc[-1] if not val.empty else None
            if regret is None and "instant_regret" in sub_file.columns:
                regret = pd.to_numeric(sub_file["instant_regret"], errors="coerce").dropna().sum()
            if regret is None and "qoe" in sub_file.columns:
                q = pd.to_numeric(sub_file["qoe"], errors="coerce").dropna()
                if not q.empty:
                    regret = (q.max() - q).clip(lower=0).sum()
            if regret is None:
                continue
            w_val = sub_file["window"].dropna().iloc[0] if sub_file["window"].notna().any() else np.nan
            records.append({"algorithm": alg, "window": w_val, "regret": regret})
        if records:
            break
    if not records:
        print("[WARN] fig17: no regret records")
        return
    rec_df = pd.DataFrame(records).dropna(subset=["window"])
    agg = rec_df.groupby("window")["regret"].mean().reset_index().sort_values("window")
    plt.figure(figsize=(7, 4))
    plt.plot(agg["window"], agg["regret"], marker="o", linewidth=2)
    plt.xlabel("window size w")
    plt.ylabel("R_T")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig17_window_scan.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_window_scan(df_all)
