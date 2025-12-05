import os
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, select_best_V_scenario, _steady_state_stats, OUT_DIR, FIG_DPI

def fig_qoe_vs_V(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig02: empty dataframe")
        return
    if "qoe" not in df.columns or "V" not in df.columns:
        print("[WARN] fig02: required columns missing (qoe/V)")
        return
    df = df.dropna(subset=["qoe", "V"])
    df = df[df["algorithm"].notna()]
    df = df[df["algorithm"].str.lower() != "unknown"]
    if df.empty:
        print("[WARN] fig02: no data after filtering algorithms/qoe/V")
        return

    plt.figure(figsize=(7, 5))
    debug_rows = []
    plotted = False
    for alg, sub_alg in df.groupby("algorithm"):
        scene_df = select_best_V_scenario(sub_alg)
        stats = _steady_state_stats(scene_df, "qoe", extra_fields=["window", "grid_density"])
        if stats.empty:
            print(f"[WARN] fig02: skip alg={alg} due to empty stats")
            continue
        stats = stats.sort_values("V")
        plt.plot(stats["V"], stats["mean"], marker="o", linewidth=2, label=alg)
        plt.fill_between(
            stats["V"],
            stats["mean"] - stats["ci95"],
            stats["mean"] + stats["ci95"],
            alpha=0.2,
        )
        stats["metric"] = "qoe"
        stats["algorithm"] = alg
        debug_rows.append(stats)
        plotted = True

    if not plotted:
        print("[WARN] fig02: nothing to plot")
        return
    plt.xlabel("V")
    plt.ylabel("QoE (Ubar)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig02_qoe_vs_V.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    if debug_rows:
        debug_df = pd.concat(debug_rows, ignore_index=True)
        debug_path = os.path.join(OUT_DIR, "debug_fig02_qoe_vs_V.csv")
        debug_df.to_csv(debug_path, index=False)
        print(f"[INFO] Saved {fig_path} and debug CSV {debug_path}")
    else:
        print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_qoe_vs_V(df_all)
