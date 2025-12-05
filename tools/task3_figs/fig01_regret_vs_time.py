import os
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig_regret_vs_time(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "qoe" not in df.columns or "t" not in df.columns or "algorithm" not in df.columns:
        print("[WARN] fig01: insufficient data for regret_vs_time")
        return
    work = df.copy()
    work = work.dropna(subset=["qoe", "t", "algorithm"])
    if work.empty:
        print("[WARN] fig01: no valid rows after dropna")
        return
    work["t_int"] = pd.to_numeric(work["t"], errors="coerce").astype(int)
    pivot = (
        work.groupby(["t_int", "algorithm"])["qoe"]
        .mean()
        .unstack("algorithm")
        .dropna(how="all")
        .sort_index()
    )
    if pivot.empty or pivot.shape[1] < 1:
        print("[WARN] fig01: pivot empty")
        return
    valid_cols = [c for c in pivot.columns if pivot[c].count() > 10]
    pivot = pivot[valid_cols]
    common_t = pivot.dropna().index
    if len(common_t) == 0:
        print("[WARN] fig01: no common time points to align")
        return
    pivot = pivot.loc[common_t]
    best = pivot.max(axis=1)
    plt.figure(figsize=(8, 4))
    for alg in pivot.columns:
        regret = (best - pivot[alg]).clip(lower=0).cumsum()
        plt.plot(pivot.index, regret, label=alg, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig01_regret_vs_time.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_regret_vs_time(df_all)
