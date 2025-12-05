import os
import pandas as pd
import matplotlib.pyplot as plt
from common import ensure_outdir, load_all_dumps, OUT_DIR, FIG_DPI

def fig_pb_selection_heat(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig08: empty dataframe")
        return
    P_col = None
    if "P" in df.columns:
        P_col = "P"
    elif "power" in df.columns:
        P_col = "power"
    B_col = None
    if "B" in df.columns:
        B_col = "B"
    elif "bandwidth" in df.columns:
        B_col = "bandwidth"
    if P_col is None or B_col is None:
        print("[WARN] fig08: missing P/B columns")
        return
    work = df[[P_col, B_col]].dropna()
    if work.empty:
        print("[WARN] fig08: no P/B data")
        return
    work["P_round"] = pd.to_numeric(work[P_col], errors="coerce").round(1)
    work["B_round"] = pd.to_numeric(work[B_col], errors="coerce").round(1)
    grouped = work.groupby(["P_round", "B_round"]).size().reset_index(name="count")
    pivot = grouped.pivot(index="B_round", columns="P_round", values="count").fillna(0)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    plt.colorbar(im, label="count")
    plt.xlabel("P (dBm)")
    plt.ylabel("B (kHz)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig08_pb_selection_heat.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

if __name__ == "__main__":
    df_all = load_all_dumps()
    fig_pb_selection_heat(df_all)