"""Parameter sweep runner for UCB 功能极限实验.

This script wraps :func:`lyap_outer.run_episode` and iterates over
configurable grids of (V, alpha, sigma_snr, K_factor, T).  Each
combination can be repeated multiple times to estimate the mean
statistics.  Results are incrementally appended to
``outputs/limit_tests/ucb_limit_stats.csv`` together with a JSON dump of
the per-run configuration for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import sys
import time
from typing import List, Sequence

import numpy as np
import pandas as pd
import yaml

from lyap_outer import run_episode


def _parse_scan(arg: str, dtype, name: str) -> List:
    if arg is None:
        return []
    if isinstance(arg, (list, tuple)):
        return [dtype(x) for x in arg]
    items = [x.strip() for x in str(arg).split(',') if x.strip()]
    if not items:
        return []
    try:
        return [dtype(x) for x in items]
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse --scan.{name}: {arg!r}") from exc


def _ensure_non_empty(values: Sequence, name: str) -> None:
    if not values:
        raise ValueError(f"scan list for {name} is empty")


def _tile_trace(base: np.ndarray, target_len: int) -> np.ndarray:
    if len(base) == 0:
        raise ValueError("channel trace is empty")
    if target_len <= len(base):
        return np.asarray(base[:target_len], dtype=float)
    reps = int(math.ceil(target_len / len(base)))
    tiled = np.tile(np.asarray(base, dtype=float), reps)
    return tiled[:target_len]


def _apply_channel_profile(base_snr_db: np.ndarray, sigma_snr: float,
                           k_factor: float, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(base_snr_db, dtype=float)
    if sigma_snr <= 0:
        noise = np.zeros_like(base)
    else:
        noise = rng.normal(loc=0.0, scale=float(sigma_snr), size=base.shape)

    if k_factor > 0:
        win = int(min(len(base), max(1, round(k_factor))))
        if win > 1:
            kernel = np.ones(win, dtype=float) / float(win)
            noise = np.convolve(noise, kernel, mode='same')
        bias_db = 10.0 * math.log10(1.0 + float(k_factor))
    else:
        bias_db = 0.0

    return base + noise + bias_db


def _derive_seed(base_seed: int, params: Sequence) -> int:
    key = "::".join(str(x) for x in params)
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    delta = int(h[:8], 16)
    return int((int(base_seed) + delta) % (2 ** 32))


def _format_float_for_path(x: float) -> str:
    return (f"{x:.6g}".replace('-', 'm').replace('.', 'p'))


def _load_existing_records(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    records = set()
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    int(float(row.get('V', 0))),
                    float(row.get('alpha', np.nan)),
                    float(row.get('sigma_snr', np.nan)),
                    float(row.get('K_factor', np.nan)),
                    int(float(row.get('T', 0))),
                    int(float(row.get('rep', 0))),
                )
            except (TypeError, ValueError):
                continue
            records.add(key)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="UCB limit experiment runner")
    parser.add_argument('--cfg', type=str, default='configs/sim_basic.yaml')
    parser.add_argument('--trace', type=str, default='data/channel_trace.csv')
    parser.add_argument('--use-mock-inner', type=int, choices=[0, 1], default=0)
    parser.add_argument('--outdir', type=str, default='outputs/limit_tests')
    parser.add_argument('--scan.V', dest='scan_V', type=str, required=True)
    parser.add_argument('--scan.alpha', dest='scan_alpha', type=str, required=True)
    parser.add_argument('--scan.sigma_snr', dest='scan_sigma', type=str, required=True)
    parser.add_argument('--scan.K_factor', dest='scan_k', type=str, required=True)
    parser.add_argument('--scan.T', dest='scan_T', type=str, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args(argv)

    scan_V = _parse_scan(args.scan_V, int, 'V')
    scan_alpha = _parse_scan(args.scan_alpha, float, 'alpha')
    scan_sigma = _parse_scan(args.scan_sigma, float, 'sigma_snr')
    scan_k = _parse_scan(args.scan_k, float, 'K_factor')
    scan_T = _parse_scan(args.scan_T, int, 'T')

    _ensure_non_empty(scan_V, 'V')
    _ensure_non_empty(scan_alpha, 'alpha')
    _ensure_non_empty(scan_sigma, 'sigma_snr')
    _ensure_non_empty(scan_k, 'K_factor')
    _ensure_non_empty(scan_T, 'T')
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f"config file not found: {args.cfg}")
    if not os.path.exists(args.trace):
        raise FileNotFoundError(f"channel trace not found: {args.trace}")

    with open(args.cfg, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    trace_df = pd.read_csv(args.trace)
    if 'snr_db' not in trace_df.columns:
        raise KeyError("trace file must contain 'snr_db' column")
    base_trace = trace_df['snr_db'].to_numpy(dtype=float)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    dump_dir = os.path.join(outdir, 'dumps')
    os.makedirs(dump_dir, exist_ok=True)

    csv_path = os.path.join(outdir, 'ucb_limit_stats.csv')
    header = ['V', 'alpha', 'sigma_snr', 'K_factor', 'T', 'rep',
              'final_cum_regret', 'SWWER_mean', 'Q_mean', 'J_mean',
              'run_dump_path', 'seed', 'timestamp']

    existing_records = _load_existing_records(csv_path)
    if existing_records:
        print(f"[info] resume mode: {len(existing_records)} existing records detected")

    use_mock_inner = bool(args.use_mock_inner)
    if not use_mock_inner:
        try:
            import inner_api_ucb as inner_module
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("failed to import inner_api_ucb") from exc
    else:
        inner_module = None

    combinations = list(itertools.product(scan_V, scan_alpha, scan_sigma, scan_k, scan_T))
    total_runs = len(combinations) * int(args.repeat)
    print(f"[plan] total combinations: {len(combinations)}, total runs (with repeat) = {total_runs}")

    # Prepare CSV header if file absent.
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    completed = 0
    for combo in combinations:
        V, alpha, sigma_snr, k_factor, T = combo
        for rep in range(1, args.repeat + 1):
            key = (int(V), float(alpha), float(sigma_snr), float(k_factor), int(T), int(rep))
            if key in existing_records:
                print(f"[skip] existing record for V={V}, alpha={alpha}, sigma={sigma_snr}, K={k_factor}, T={T}, rep={rep}")
                completed += 1
                continue

            derived_seed = _derive_seed(args.seed, key)
            cfg = copy.deepcopy(base_cfg)
            cfg['seed'] = int(derived_seed)
            cfg['T'] = int(T)

            trace_len = int(T)
            rng = np.random.default_rng(derived_seed)
            tiled_trace = _tile_trace(base_trace, trace_len)
            snr_profile = _apply_channel_profile(tiled_trace, sigma_snr, k_factor, rng)

            if not use_mock_inner and inner_module is not None:
                inner_module.set_ucb_config(alpha=alpha)

            dump_name = (
                f"run_V{V}_a{_format_float_for_path(alpha)}_"
                f"sig{_format_float_for_path(sigma_snr)}_"
                f"K{_format_float_for_path(k_factor)}_T{T}_rep{rep}_seed{derived_seed}.json"
            )
            dump_path = os.path.join(dump_dir, dump_name)

            start_ts = time.time()
            print(f"[run] V={V} alpha={alpha} sigma={sigma_snr} K={k_factor} T={T} rep={rep} seed={derived_seed}")
            try:
                result = run_episode(cfg, snr_profile, int(V), use_mock_inner=use_mock_inner)
            except Exception as exc:
                print(f"[error] run failed for combo {combo} rep={rep}: {exc}", file=sys.stderr)
                raise

            elapsed = time.time() - start_ts
            print(f"[done] elapsed={elapsed:.2f}s -> dump={dump_name}")

            payload = {
                'params': {
                    'V': int(V),
                    'alpha': float(alpha),
                    'sigma_snr': float(sigma_snr),
                    'K_factor': float(k_factor),
                    'T': int(T),
                    'rep': int(rep),
                    'seed': int(derived_seed),
                },
                'metrics': {k: float(v) for k, v in result.items() if isinstance(v, (int, float))},
                'raw_result': result,
                'cfg_overrides': {
                    'seed': int(derived_seed),
                    'T': int(T),
                },
                'notes': {
                    'sigma_snr': float(sigma_snr),
                    'K_factor': float(k_factor),
                    'elapsed_sec': float(elapsed),
                }
            }
            with open(dump_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)

            final_cum_regret = result.get('final_cum_regret', 'NA')
            if isinstance(final_cum_regret, (float, int)):
                final_cum_regret = float(final_cum_regret)
            else:
                final_cum_regret = 'NA'

            row = {
                'V': int(V),
                'alpha': float(alpha),
                'sigma_snr': float(sigma_snr),
                'K_factor': float(k_factor),
                'T': int(T),
                'rep': int(rep),
                'final_cum_regret': final_cum_regret,
                'SWWER_mean': float(result.get('SWWER_mean', float('nan'))),
                'Q_mean': float(result.get('Q_mean', float('nan'))),
                'J_mean': float(result.get('J_mean', float('nan'))),
                'run_dump_path': os.path.relpath(dump_path, start=outdir),
                'seed': int(derived_seed),
                'timestamp': dt.datetime.utcnow().isoformat(timespec='seconds'),
            }

            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(row)

            existing_records.add(key)
            completed += 1
            print(f"[progress] {completed}/{total_runs} rows written")

    print(f"[summary] completed {completed} runs; output -> {csv_path}")


if __name__ == '__main__':
    main()
