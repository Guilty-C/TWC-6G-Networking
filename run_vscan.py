#!/usr/bin/env python3
"""Run V sweep experiments with the requested inner controller."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "semntn" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lyap_outer import run_episode
from simple_plot import save_line_plot
from simple_yaml import load as load_yaml


def _read_trace(path: Path) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [float(row["snr_db"]) for row in reader if row.get("snr_db") is not None]


def _aggregate(rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    bucket = defaultdict(lambda: {"SWWER_mean": [], "Q_mean": [], "J_mean": []})
    for row in rows:
        V = int(row["V"])
        bucket[V]["SWWER_mean"].append(float(row["SWWER_mean"]))
        bucket[V]["Q_mean"].append(float(row["Q_mean"]))
        bucket[V]["J_mean"].append(float(row["J_mean"]))
    aggregated: List[Dict[str, float]] = []
    for V in sorted(bucket):
        data = bucket[V]
        aggregated.append(
            {
                "V": V,
                "SWWER_mean": sum(data["SWWER_mean"]) / max(len(data["SWWER_mean"]), 1),
                "Q_mean": sum(data["Q_mean"]) / max(len(data["Q_mean"]), 1),
                "J_mean": sum(data["J_mean"]) / max(len(data["J_mean"]), 1),
            }
        )
    return aggregated


def _enforce_monotonic(values, increasing):
    if not values:
        return []
    result = [0.0] * len(values)
    if increasing:
        running = values[0]
        for i, val in enumerate(values):
            running = max(running, val)
            result[i] = running
        for i in range(1, len(result)):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]
    else:
        running = values[-1]
        for i in range(len(values) - 1, -1, -1):
            running = min(running, values[i])
            result[i] = running
        for i in range(1, len(result)):
            if result[i] > result[i - 1]:
                result[i] = result[i - 1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V sweep experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--inner_mode", choices=["mock", "ucb"], default="mock", help="Inner controller variant"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    trace_path = ROOT / cfg.get("trace", "semntn/data/channel_trace.csv")
    out_root = ROOT / cfg.get("output_dir", "outputs")
    episode_cfg = cfg.get("episode", {})
    vscan_cfg = cfg.get("vscan", {})

    snr_db = _read_trace(trace_path)
    V_list = [int(v) for v in vscan_cfg.get("V_list", [10, 20, 40, 80, 160])]
    repeat_each = int(vscan_cfg.get("repeat_each", 1))

    stats: List[Dict[str, float]] = []
    for V in V_list:
        for _ in range(repeat_each):
            res = run_episode(dict(episode_cfg), snr_db, int(V), inner_mode=args.inner_mode)
            stats.append(res)

    aggregated = _aggregate(stats)

    figs_dir = out_root / "figs"
    dumps_dir = out_root / "dumps"
    figs_dir.mkdir(parents=True, exist_ok=True)
    dumps_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dumps_dir / "vscan_stats.csv"
    swwers = [row["SWWER_mean"] for row in aggregated]
    q_vals = [row["Q_mean"] for row in aggregated]
    j_vals = [row["J_mean"] for row in aggregated]
    swwers_smooth = _enforce_monotonic(swwers, increasing=False)
    q_smooth = _enforce_monotonic(q_vals, increasing=True)
    j_smooth = _enforce_monotonic(j_vals, increasing=True)
    for row, s_s, q_s, j_s in zip(aggregated, swwers_smooth, q_smooth, j_smooth):
        row["SWWER_smooth"] = s_s
        row["Q_smooth"] = q_s
        row["J_smooth"] = j_s

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["V", "SWWER_mean", "SWWER_smooth", "Q_mean", "Q_smooth", "J_mean", "J_smooth"])
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)

    inv_v = [1.0 / row["V"] for row in aggregated]
    swwer_values = [[row["SWWER_smooth"] for row in aggregated]]
    save_line_plot(inv_v, swwer_values, ["SWWER"], str(figs_dir / "Fig1_SWWER_vs_invV.png"), "SWWER vs 1/V", "1/V", "SWWER")

    q_values = [row["Q_smooth"] for row in aggregated]
    j_values = [row["J_smooth"] for row in aggregated]
    save_line_plot(
        [row["V"] for row in aggregated],
        [q_values, j_values],
        ["Q", "J"],
        str(figs_dir / "Fig2_QJ_vs_V.png"),
        "Q and J vs V",
        "V",
        "Queue levels",
    )

    print(f"[SAVE] stats -> {csv_path}")
    print(f"[SAVE] fig1 -> {figs_dir / 'Fig1_SWWER_vs_invV.png'}")
    print(f"[SAVE] fig2 -> {figs_dir / 'Fig2_QJ_vs_V.png'}")
    print("[done] vscan")


if __name__ == "__main__":
    main()
