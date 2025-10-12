from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_regret_curves(
    results: Dict[str, Dict],
    out_png: str,
    *,
    logx: bool = False,
    show_ref_sqrt: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))

    for policy_name, data in results.items():
        label = data.get("label", policy_name)
        T = data["T"]
        mean = data["mean"]
        lo = data["lo"]
        hi = data["hi"]
        plt.plot(T, mean, label=label)
        plt.fill_between(T, lo, hi, alpha=0.2)

    if show_ref_sqrt and "ucb" in results:
        data = results["ucb"]
        T = data["T"].astype(float)
        mean = data["mean"].astype(float)
        c = float(mean[-1] / np.sqrt(T[-1]))
        plt.plot(T, c * np.sqrt(T), "--", color="black", linewidth=1.5, label="C√T (ref)")

    plt.xlabel("T")
    plt.ylabel("Cumulative Regret R(T)")
    if logx:
        plt.xscale("log")

    all_hi = [np.max(data["hi"]) for data in results.values()]
    ymax = max(all_hi)
    plt.ylim(bottom=0.0, top=ymax * 1.05)

    seeds = None
    regret_note = "Pseudo-regret R(T) = Σ_t [max_a μ_true(a) - μ_true(a_t)]"
    norm_note = "Rewards mapped via map_to_unit_interval"
    for data in results.values():
        regs = data.get("regrets")
        if regs is not None:
            seeds = regs.shape[0]
            break
    caption = f"{regret_note}; {norm_note}; seeds={seeds if seeds is not None else '?'}"

    plt.title("Cumulative Pseudo-Regret")
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.5, -0.02, caption, ha="center", fontsize=8)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
