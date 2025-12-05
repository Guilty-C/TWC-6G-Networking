import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI, PROJECT_ROOT, _extract_queue_energy

def fig_energy_efficiency_bar(summary_csv: str, df_all: pd.DataFrame) -> None:
    ensure_outdir()
    if (df_all is None or df_all.empty) and (not summary_csv or not os.path.exists(summary_csv)):
        print("[WARN] fig12: no data for energy efficiency")
        return
    alg_records = []
    if df_all is not None and not df_all.empty and "qoe" in df_all.columns:
        for alg, sub in df_all.groupby("algorithm"):
            q_series = pd.to_numeric(sub["qoe"], errors="coerce").dropna()
            queue_series, energy_series = _extract_queue_energy(sub)
            if energy_series is None:
                continue
            energy_series = energy_series.dropna()
            if len(q_series) == 0 or len(energy_series) == 0:
                continue
            t_max = pd.to_numeric(sub["t"], errors="coerce").max() if "t" in sub.columns else None
            burn_start = 0.1 * t_max if t_max and math.isfinite(t_max) else 0
            if "t" in sub.columns:
                mask = pd.to_numeric(sub["t"], errors="coerce") >= burn_start
                q_series = q_series[mask]
                energy_series = energy_series[mask]
            q_mean = q_series.mean()
            e_mean = energy_series.mean()
            if e_mean and e_mean > 0:
                alg_records.append({"algorithm": alg, "eff": q_mean / e_mean})
    if not alg_records and summary_csv and os.path.exists(summary_csv):
        try:
            summary = pd.read_csv(summary_csv)
            if {"qoe_mean", "energy_mean", "algorithm"}.issubset(set(summary.columns)):
                summary = summary.dropna(subset=["qoe_mean", "energy_mean"])
                summary["eff"] = summary["qoe_mean"] / summary["energy_mean"]
                alg_records = summary.groupby("algorithm")["eff"].mean().reset_index().to_dict("records")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] fig12: failed to read summary csv: {exc}")
            return
    if not alg_records:
        print("[WARN] fig12: no valid efficiency records")
        return
    eff_df = pd.DataFrame(alg_records)
    eff_df = eff_df.dropna(subset=["eff"])
    if eff_df.empty:
        print("[WARN] fig12: empty efficiency dataframe")
        return
    plt.figure(figsize=(7, 4))
    plt.bar(eff_df["algorithm"], eff_df["eff"], color="#55A868")
    plt.ylabel("Ubar / Ebar")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig12_energy_efficiency_bar.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    summary_csv = os.path.join(PROJECT_ROOT, "outputs", "dumps", "task3_summary.csv")
    fig_energy_efficiency_bar(summary_csv, df_all)
