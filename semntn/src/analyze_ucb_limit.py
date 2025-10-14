"""Analysis and visualization tool for the UCB 功能极限 sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("input CSV has no data")
    return df


def _to_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _collect_regret_series(df: pd.DataFrame, outdir: str, fixed_sigma: float) -> Dict[str, Dict[str, np.ndarray]]:
    """Return averaged regret series grouped by alpha for the chosen sigma.

    The function attempts to read JSON dumps referenced in the CSV.  When a
    dump contains a ``regret_series`` entry, the values are averaged across
    repetitions.  If no series are available, an empty dict is returned.
    """

    if 'run_dump_path' not in df.columns:
        return {}

    mask = np.isclose(df['sigma_snr'].astype(float), float(fixed_sigma), atol=1e-6)
    df_sel = df.loc[mask]
    if df_sel.empty:
        return {}

    grouped: Dict[str, Dict[str, list]] = {}
    for _, row in df_sel.iterrows():
        dump_rel = row.get('run_dump_path', '')
        if not dump_rel or not isinstance(dump_rel, str):
            continue
        dump_path = os.path.join(outdir, dump_rel)
        if not os.path.exists(dump_path):
            continue
        try:
            with open(dump_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception:
            continue
        series = payload.get('raw_result', {}).get('regret_series')
        if series is None:
            series = payload.get('metrics', {}).get('regret_series')
        if series is None:
            continue
        try:
            arr = np.asarray(series, dtype=float)
        except Exception:
            continue
        alpha = row.get('alpha', np.nan)
        T = int(row.get('T', 0))
        key = f"alpha={alpha:.3g},T={T}"
        grouped.setdefault(key, {}).setdefault('series', []).append(arr)

    averaged: Dict[str, Dict[str, np.ndarray]] = {}
    for key, data in grouped.items():
        series_list = data.get('series', [])
        if not series_list:
            continue
        min_len = min(len(s) for s in series_list)
        stack = np.vstack([s[:min_len] for s in series_list])
        averaged[key] = {'mean': stack.mean(axis=0), 'std': stack.std(axis=0)}
    return averaged


def _plot_regret_vs_time(averaged, outdir: str, dpi: int, fixed_sigma: float) -> str:
    fig_path = os.path.join(outdir, f"regret_vs_time_sigma{fixed_sigma:.3g}.png")
    plt.figure(figsize=(7, 4))
    if not averaged:
        plt.axis('off')
        plt.text(0.5, 0.5, '[TBD] missing regret series', ha='center', va='center', fontsize=12)
    else:
        for label, stats in sorted(averaged.items()):
            mean = stats['mean']
            std = stats['std']
            t = np.arange(1, len(mean) + 1)
            plt.plot(t, mean, label=label)
            plt.fill_between(t, mean - std, mean + std, alpha=0.2)
        plt.xlabel('Time step')
        plt.ylabel('Cumulative regret')
        plt.title(f'Regret vs Time (sigma={fixed_sigma})')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    return fig_path


def _plot_regret_vs_logT(agg: pd.DataFrame, dpi: int, outdir: str) -> str | None:
    df = agg.dropna(subset=['final_cum_regret_mean'])
    if df.empty:
        fig_path = os.path.join(outdir, 'regret_vs_logT_TBD.png')
        plt.figure(figsize=(6.5, 4))
        plt.axis('off')
        plt.text(0.5, 0.5, '[TBD] missing regret series', ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi)
        plt.close()
        return None

    fig_path = os.path.join(outdir, 'regret_vs_logT.png')
    plt.figure(figsize=(7, 4))
    grouped = df.groupby(['alpha', 'sigma_snr'])
    for (alpha, sigma), data in grouped:
        T_vals = data['T'].to_numpy(dtype=float)
        regret = data['final_cum_regret_mean'].to_numpy(dtype=float)
        if len(T_vals) < 1:
            continue
        plt.plot(np.log10(T_vals), regret, marker='o', label=f"alpha={alpha:.3g}, sigma={sigma:.3g}")
    plt.xlabel('log10(T)')
    plt.ylabel('Final cumulative regret')
    plt.title('Final regret vs log(T)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    return fig_path


def _plot_swwer_vs_invV(agg: pd.DataFrame, dpi: int, outdir: str) -> str:
    df = agg.groupby('V', as_index=False)['SWWER_mean_mean'].mean()
    df['invV'] = 1.0 / np.maximum(df['V'].astype(float), 1e-9)
    fig_path = os.path.join(outdir, 'Fig1_SWWER_vs_invV.png')
    plt.figure(figsize=(6.5, 4))
    plt.plot(df['invV'], df['SWWER_mean_mean'], marker='o')
    plt.xlabel('1 / V')
    plt.ylabel('SWWER_mean')
    plt.title('SWWER vs 1/V')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    return fig_path


def _plot_qj_vs_V(agg: pd.DataFrame, dpi: int, outdir: str) -> str:
    df = agg.groupby('V', as_index=False)[['Q_mean_mean', 'J_mean_mean']].mean()
    fig_path = os.path.join(outdir, 'Fig2_QJ_vs_V.png')
    plt.figure(figsize=(6.5, 4))
    plt.plot(df['V'], df['Q_mean_mean'], marker='o', label='Q_mean')
    plt.plot(df['V'], df['J_mean_mean'], marker='s', label='J_mean')
    plt.xlabel('V')
    plt.ylabel('Queue metrics')
    plt.title('Q_mean and J_mean vs V')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    return fig_path


def _compute_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _analyze_regret_scaling(agg: pd.DataFrame) -> float | None:
    df = agg.dropna(subset=['final_cum_regret_mean'])
    if df.empty or len(df) < 2:
        return None
    if np.any(df['final_cum_regret_mean'] <= 0):
        return None
    logT = np.log(df['T'].astype(float))
    logR = np.log(df['final_cum_regret_mean'].astype(float))
    slope, _ = np.polyfit(logT, logR, 1)
    return float(slope)


def _write_summary(outdir: str, corr_stats: Dict[str, float], regret_slope: float | None,
                   regret_available: bool, notes: Sequence[str]) -> str:
    lines = ["# UCB Limit Summary", ""]
    lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat(timespec='seconds')} UTC")
    lines.append("")
    lines.append("## Correlation Checks")
    for name, val in corr_stats.items():
        if math.isnan(val):
            status = "N/A"
            lines.append(f"- {name}: N/A (insufficient data)")
        else:
            threshold = 0.7 if 'corr' in name else None
            if threshold is not None:
                passed = val > threshold
                status = "PASS" if passed else "FAIL"
                lines.append(f"- {name}: {val:.3f} ({status}, threshold={threshold})")
            else:
                lines.append(f"- {name}: {val:.3f}")

    lines.append("")
    lines.append("## Regret Scaling")
    if regret_available and regret_slope is not None:
        verdict = "PASS" if regret_slope < 1.0 else "WARN"
        lines.append(f"- log-log slope = {regret_slope:.3f} ({verdict}, target < 1)")
    else:
        lines.append("- [TBD: regret series missing]")

    if notes:
        lines.append("")
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")

    summary_path = os.path.join(outdir, 'limit_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Analyze UCB limit statistics')
    parser.add_argument('--in', dest='input_csv', type=str,
                        default='outputs/limit_tests/ucb_limit_stats.csv')
    parser.add_argument('--outdir', type=str, default='outputs/limit_tests')
    parser.add_argument('--fixed-sigma', type=float, default=1.0)
    parser.add_argument('--fig-dpi', type=int, default=140)
    args = parser.parse_args(argv)

    outdir = _ensure_outdir(args.outdir)
    csv_path = args.input_csv

    df = _load_csv(csv_path)
    df = _to_numeric(df, ['V', 'alpha', 'sigma_snr', 'K_factor', 'T', 'rep',
                         'final_cum_regret', 'SWWER_mean', 'Q_mean', 'J_mean'])

    group_cols = ['V', 'alpha', 'sigma_snr', 'K_factor', 'T']
    agg = df.groupby(group_cols).agg({
        'final_cum_regret': ['mean', 'std'],
        'SWWER_mean': ['mean', 'std'],
        'Q_mean': ['mean', 'std'],
        'J_mean': ['mean', 'std'],
    })
    agg.columns = ['_'.join(filter(None, col)).strip('_') for col in agg.columns.values]
    agg = agg.reset_index()

    averaged_regret = _collect_regret_series(df, outdir, args.fixed_sigma)
    regret_fig = _plot_regret_vs_time(averaged_regret, outdir, args.fig_dpi, args.fixed_sigma)
    regret_log_fig = _plot_regret_vs_logT(agg, args.fig_dpi, outdir)
    swwer_fig = _plot_swwer_vs_invV(agg, args.fig_dpi, outdir)
    qj_fig = _plot_qj_vs_V(agg, args.fig_dpi, outdir)

    df_by_V = agg.groupby('V', as_index=False).mean(numeric_only=True)
    corr_swwer_invV = _compute_corr(1.0 / np.maximum(df_by_V['V'].to_numpy(dtype=float), 1e-9),
                                    df_by_V['SWWER_mean_mean'].to_numpy(dtype=float))
    corr_Q_V = _compute_corr(df_by_V['V'].to_numpy(dtype=float),
                             df_by_V['Q_mean_mean'].to_numpy(dtype=float))
    corr_J_V = _compute_corr(df_by_V['V'].to_numpy(dtype=float),
                             df_by_V['J_mean_mean'].to_numpy(dtype=float))

    regret_available = bool(averaged_regret) or (regret_log_fig is not None)
    regret_slope = _analyze_regret_scaling(agg)

    notes = []
    if not averaged_regret:
        notes.append('Regret series not present in dumps; update inner/outer APIs to expose per-slot regret.')

    summary_path = _write_summary(
        outdir,
        {
            'corr(SWWER_mean, 1/V)': corr_swwer_invV,
            'corr(Q_mean, V)': corr_Q_V,
            'corr(J_mean, V)': corr_J_V,
        },
        regret_slope,
        regret_available,
        notes,
    )

    report_path = os.path.join('semntn', 'docs', 'ucb_limit_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(summary_path, 'r', encoding='utf-8') as src, open(report_path, 'w', encoding='utf-8') as dst:
        dst.write(src.read())

    print("[done] figures saved:")
    for path in [regret_fig, regret_log_fig, swwer_fig, qj_fig, summary_path, report_path]:
        if path:
            print(f" - {path}")


if __name__ == '__main__':
    main()
