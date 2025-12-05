import os
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, OUT_DIR, FIG_DPI, PROJECT_ROOT

def fig_snr_per_heat() -> None:
    ensure_outdir()
    path = os.path.join(PROJECT_ROOT, "outputs", "dumps", "task3_phys_scan.csv")
    if not os.path.exists(path):
        print(f"[WARN] fig07: missing {path}")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] fig07: failed to read phys scan: {exc}")
        return
    required = {"B", "snr_db", "per"}
    if not required.issubset(set(df.columns)):
        print("[WARN] fig07: required columns missing in phys scan")
        return
    grouped = df.groupby(["B", "snr_db"])["per"].mean().reset_index()
    pivot = grouped.pivot(index="B", columns="snr_db", values="per")
    if pivot.empty:
        print("[WARN] fig07: no data after grouping")
        return
    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", extent=[
        pivot.columns.min(),
        pivot.columns.max(),
        pivot.index.min(),
        pivot.index.max(),
    ])
    plt.colorbar(im, label="PER")
    plt.xlabel("snr_db")
    plt.ylabel("B (kHz)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig07_snr_per_heat.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    fig_snr_per_heat()