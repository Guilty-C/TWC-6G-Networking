"""
Path Usage Note:
All paths have been adapted to relative paths for independent execution.
Ensure this script is run from its own directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig_nonstationary_robustness(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig05: empty dataframe")
        return
    if "algorithm" not in df.columns:
        print("[WARN] fig05: missing algorithm column")
        return

    records = []
    seed_series = df["seed"] if "seed" in df.columns else pd.Series([0] * len(df))
    for (alg, file, seed), sub in df.groupby(["algorithm", "file", seed_series]):
        proxy = None
        if "snr_db" in sub.columns:
            snr = pd.to_numeric(sub["snr_db"], errors="coerce").dropna()
            if len(snr) > 1:
                proxy = np.mean(np.abs(np.diff(snr)))
        if proxy is None and "drift_level" in sub.columns:
            drift = pd.to_numeric(sub["drift_level"], errors="coerce").dropna()
            if not drift.empty:
                proxy = drift.max() - drift.min()
        if proxy is None and "V_T" in sub.columns:
            proxy = pd.to_numeric(sub["V_T"], errors="coerce").dropna().mean()
        if proxy is None and "S" in sub.columns:
            proxy = pd.to_numeric(sub["S"], errors="coerce").dropna().mean()
        if proxy is None:
            continue

        regret = None
        if "regret_t" in sub.columns:
            regret = pd.to_numeric(sub["regret_t"], errors="coerce").dropna()
            if not regret.empty:
                regret = regret.iloc[-1]
        if regret is None and "instant_regret" in sub.columns:
            regret = pd.to_numeric(sub["instant_regret"], errors="coerce").dropna().sum()
        if regret is None and "qoe" in sub.columns:
            qoe_vals = pd.to_numeric(sub["qoe"], errors="coerce").dropna()
            if not qoe_vals.empty:
                best = qoe_vals.max()
                regret = (best - qoe_vals).clip(lower=0).sum()
        if regret is None:
            continue
        records.append({"algorithm": alg, "proxy": proxy, "regret": regret})

    if not records:
        print("[WARN] fig05: no records for nonstationary robustness")
        return
    df_rec = pd.DataFrame(records)
    plt.figure(figsize=(7, 4))
    for alg, sub in df_rec.groupby("algorithm"):
        sub = sub.sort_values("proxy")
        bins = np.linspace(sub["proxy"].min(), sub["proxy"].max(), 8) if len(sub) > 6 else None
        if bins is not None:
            try:
                sub["bin"] = pd.cut(sub["proxy"], bins, include_lowest=True, duplicates='drop')
            except TypeError:
                # Fallback for older pandas versions that might not support duplicates arg in cut (though it should)
                # or just in case
                sub["bin"] = pd.cut(sub["proxy"], bins, include_lowest=True)
            
            if "bin" in sub.columns:
                sub_b = sub.groupby("bin")["regret"].mean()
                x = [b.mid for b in sub_b.index]
                y = sub_b.values
            else:
                 x = sub["proxy"].values
                 y = sub["regret"].values
        else:
            x = sub["proxy"].values
            y = sub["regret"].values
        plt.plot(x, y, marker="o", linewidth=2, label=alg)
    plt.xlabel("nonstationarity proxy")
    plt.ylabel("R_T")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig05_nonstationary_robustness.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_nonstationary_robustness(df_all)