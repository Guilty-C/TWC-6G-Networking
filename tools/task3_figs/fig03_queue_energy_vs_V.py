import os
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, select_best_V_scenario, _steady_state_stats, _extract_queue_energy, OUT_DIR, FIG_DPI

def fig_queue_energy_vs_V(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig03: empty dataframe")
        return
    if "V" not in df.columns or "algorithm" not in df.columns:
        print("[WARN] fig03: required columns missing (V/algorithm)")
        return

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    colors = plt.cm.tab10.colors
    debug_rows = []
    plotted = False
    for idx, (alg, sub_alg) in enumerate(df.groupby("algorithm")):
        if str(alg).lower() == "unknown":
            continue
        queue_series, energy_series = _extract_queue_energy(sub_alg)
        if queue_series is not None:
            sub_alg = sub_alg.assign(queue_metric=queue_series)
        if energy_series is not None:
            sub_alg = sub_alg.assign(energy_metric=energy_series)
        if "queue_metric" not in sub_alg.columns or "energy_metric" not in sub_alg.columns:
            print(f"[WARN] fig03: skip alg={alg} due to missing queue/energy columns")
            continue
        scene_df = select_best_V_scenario(sub_alg)
        stats_q = _steady_state_stats(scene_df, "queue_metric", extra_fields=["window", "grid_density"])
        stats_e = _steady_state_stats(scene_df, "energy_metric", extra_fields=["window", "grid_density"])
        if stats_q.empty and stats_e.empty:
            print(f"[WARN] fig03: skip alg={alg} due to empty stats")
            continue
        color = colors[idx % len(colors)]
        if not stats_q.empty:
            stats_q = stats_q.sort_values("V")
            ax1.plot(stats_q["V"], stats_q["mean"], marker="o", linewidth=2, color=color, label=f"{alg} Qbar")
            ax1.fill_between(
                stats_q["V"],
                stats_q["mean"] - stats_q["ci95"],
                stats_q["mean"] + stats_q["ci95"],
                color=color,
                alpha=0.15,
            )
        if not stats_e.empty:
            stats_e = stats_e.sort_values("V")
            ax2.plot(
                stats_e["V"],
                stats_e["mean"],
                marker="s",
                linewidth=2,
                color=color,
                linestyle="--",
                label=f"{alg} Ebar",
            )
            ax2.fill_between(
                stats_e["V"],
                stats_e["mean"] - stats_e["ci95"],
                stats_e["mean"] + stats_e["ci95"],
                color=color,
                alpha=0.1,
            )
        if not stats_q.empty or not stats_e.empty:
            merged = pd.merge(stats_q, stats_e, on=["algorithm", "V"], how="outer", suffixes=("_queue", "_energy"))
            debug_rows.append(merged)
            plotted = True

    if not plotted:
        print("[WARN] fig03: nothing plotted")
        return
    ax1.set_xlabel("V")
    ax1.set_ylabel("Qbar")
    ax2.set_ylabel("Ebar")
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="best")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig03_queue_energy_vs_V.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    if debug_rows:
        debug_df = pd.concat(debug_rows, ignore_index=True)
        debug_path = os.path.join(OUT_DIR, "debug_fig03_queue_energy_vs_V.csv")
        debug_df.to_csv(debug_path, index=False)
        print(f"[INFO] Saved {fig_path} and debug CSV {debug_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_queue_energy_vs_V(df_all)