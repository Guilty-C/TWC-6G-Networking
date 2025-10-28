"""Automatic micro-tuning loop for Task1 release config.

Rules (up to 3 rounds):
- Target violation tau = 0.01
- semantic_scale *= max(v_sem/tau, 1.0)**0.5
- queue_scale    *= max(v_lat/tau, 1.0)**0.5
- V := max(6, round(V * (tau / max((v_lat+v_sem)/2,1e-3))**0.5))
- After round 2: if v_sem > 0.05 then semantic_budget := min(0.38, semantic_budget+0.02)

Per round:
- Rerun run_task1 with updated configs/task1_release.yaml
- Save iter-specific copies: outputs/dumps/task1_release_iterX_log.csv, outputs/dumps/task1_release_iterX_resolved.yaml
- Append a summary row with config_name=task1_release@iterX
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import subprocess
import sys
import yaml
import pandas as pd


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _analyze(log_path: Path, outdir: Path, tag: str) -> dict:
    from .analyze_task1 import analyze  # local import to avoid heavy globals at import time
    analyze(log_path, outdir, config_name=tag)
    df = pd.read_csv(log_path)
    v_lat = float(df.get("violate_lat", pd.Series([0]*len(df))).mean())
    v_eng = float(df.get("violate_eng", pd.Series([0]*len(df))).mean())
    v_sem = float(df.get("violate_sem", pd.Series([0]*len(df))).mean())
    return {"v_lat": v_lat, "v_eng": v_eng, "v_sem": v_sem}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/task1_release.yaml"))
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()

    cfg_path = args.config
    dumps = Path("outputs") / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    tau = float(args.tau)

    # Iterate up to max-iters
    for it in range(1, args.max_iters + 1):
        print(f"[AUTO] Iteration {it}")
        # Run task1 with current config
        _run_cmd([sys.executable, "-m", "semntn.src.run_task1", "--config", str(cfg_path)])

        # Move/copy logs/resolved with iter suffix
        log_src = dumps / "task1_release_log.csv"
        res_src = dumps / "task1_release_resolved.yaml"
        if log_src.exists():
            log_dst = dumps / f"task1_release_iter{it}_log.csv"
            shutil.copyfile(log_src, log_dst)
        if res_src.exists():
            res_dst = dumps / f"task1_release_iter{it}_resolved.yaml"
            shutil.copyfile(res_src, res_dst)

        # Analyze and append summary row with tag
        tag = f"task1_release@iter{it}"
        metrics = _analyze(log_dst if log_src.exists() else log_src, Path("outputs"), tag)
        print(f"[AUTO] v_lat={metrics['v_lat']:.3f}, v_eng={metrics['v_eng']:.3f}, v_sem={metrics['v_sem']:.3f}")

        # Termination: all ≤ tau
        if metrics["v_lat"] <= tau and metrics["v_eng"] <= tau and metrics["v_sem"] <= tau:
            print("[AUTO] All violations ≤ tau; stop tuning.")
            break

        # Update config per rules
        cfg = _read_yaml(cfg_path)
        scales = cfg.get("scales", {})
        V = float(cfg.get("V", 8.0))
        # Gentle power updates
        q_mul = max(metrics["v_lat"] / tau, 1.0) ** 0.5
        s_mul = max(metrics["v_sem"] / tau, 1.0) ** 0.5
        scales["queue_scale"] = float(scales.get("queue_scale", 15000.0)) * q_mul
        scales["energy_scale"] = float(scales.get("energy_scale", 600.0))  # unchanged
        scales["semantic_scale"] = float(scales.get("semantic_scale", 6.0)) * s_mul
        # V update using average of (v_lat, v_sem)
        avg_vs = max((metrics["v_lat"] + metrics["v_sem"]) / 2.0, 1e-3)
        V_new = max(6, round(V * (tau / avg_vs) ** 0.5))
        cfg["V"] = float(V_new)
        cfg["scales"] = scales
        # After iter2 rule
        if it >= 2 and metrics["v_sem"] > 0.05:
            sb = float(cfg.get("semantic_budget", 0.35))
            cfg["semantic_budget"] = float(min(0.38, sb + 0.02))
        _write_yaml(cfg_path, cfg)
        print(f"[AUTO] Updated V={cfg['V']}, scales={{queue:{scales['queue_scale']:.1f}, energy:{scales['energy_scale']:.1f}, semantic:{scales['semantic_scale']:.1f}}}, semantic_budget={cfg['semantic_budget']}")

    print("[AUTO] Done.")


if __name__ == "__main__":
    main()