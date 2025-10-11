import argparse, yaml, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from lyap_outer import run_episode

def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12: return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def _fit_line(x, y):
    """最小二乘拟合直线和 R²"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return a, b, r2, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--vscan', required=True)
    ap.add_argument('--trace', required=True)
    ap.add_argument('--use-mock-inner', type=int, default=1)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    with open(args.vscan, 'r', encoding='utf-8') as f:
        vscan = yaml.safe_load(f)
        print("vscan内容:", vscan)
    trace = pd.read_csv(args.trace)
    snr_db = trace['snr_db'].to_numpy().astype(float)

    V_list = vscan.get('V_list', [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,
                                  320,340,360,380,400,420,440,460,480,500,520,540,560,580,
                                  600,620,640,660,680,700,720,740,760,780,800,820,840,860,
                                  880,900,920,940,960,980,1000])
    repeat_each = int(vscan.get('repeat_each', 1))

    stats = []
    for V in V_list:
        for _ in range(repeat_each):
            res = run_episode(cfg, snr_db, int(V), use_mock_inner=bool(args.use_mock_inner))
            stats.append(res)

    df = pd.DataFrame(stats)
    dfg = df.groupby('V', sort=True).agg({'SWWER_mean':'mean','Q_mean':'mean','J_mean':'mean'}).reset_index()

    os.makedirs(os.path.join(args.outdir,'figs'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir,'dumps'), exist_ok=True)
    dump = os.path.join(args.outdir,'dumps','vscan_stats.csv')
    dfg.to_csv(dump, index=False)

    # Fig1: SWWER vs 1/V
    invV = 1.0/dfg['V'].astype(float).to_numpy()
    y = dfg['SWWER_mean'].to_numpy()
    plt.figure(figsize=(6,4))
    plt.plot(invV, y, marker='o')
    plt.xlabel('1/V'); plt.ylabel('mean SWWER'); plt.title('SWWER vs 1/V')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    fig1 = os.path.join(args.outdir,'figs','Fig1_SWWER_vs_invV.png')
    plt.savefig(fig1, dpi=150); plt.close()

    # Fig2: Q, J vs V （含线性拟合 + R²）
    Vv = dfg['V'].astype(float).to_numpy()
    Qn = _minmax(dfg['Q_mean'].to_numpy())
    Jn = _minmax(dfg['J_mean'].to_numpy())

    # 拟合线和 R²
    aQ, bQ, r2_Q, yQ = _fit_line(Vv, Qn)
    aJ, bJ, r2_J, yJ = _fit_line(Vv, Jn)

    plt.figure(figsize=(7,4.2))
    plt.scatter(Vv, Qn, label=f'Q (norm), R²={r2_Q:.3f}', color='C0', s=20)
    plt.plot(Vv, yQ, color='C0', linestyle='--', alpha=0.6)
    plt.scatter(Vv, Jn, label=f'J (norm), R²={r2_J:.3f}', color='C1', s=20)
    plt.plot(Vv, yJ, color='C1', linestyle='--', alpha=0.6)
    plt.xlabel('V'); plt.ylabel('Normalized Value [0–1]')
    plt.title('Q ↑ (blue) and J ↓ (orange) vs V with linear fit')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    fig2 = os.path.join(args.outdir,'figs','Fig2_QJ_vs_V_linfit.png')
    plt.savefig(fig2, dpi=150); plt.close()

    # Fig3: SWWER vs 1/V （线性拟合）
    y_swwer = dfg['SWWER_mean'].to_numpy()
    x_invV = 1.0 / np.maximum(Vv, 1e-9)
    aS, bS, r2_S, yS = _fit_line(x_invV, y_swwer)

    plt.figure(figsize=(7,4.2))
    plt.scatter(x_invV, y_swwer, label=f'SWWER vs 1/V, R²={r2_S:.3f}', color='C2', s=25)
    plt.plot(x_invV, yS, color='C2', linestyle='--', alpha=0.7)
    plt.xlabel('1 / V')
    plt.ylabel('mean SWWER')
    plt.title('SWWER vs 1/V with linear fit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig3 = os.path.join(args.outdir, 'figs', 'Fig3_SWWER_vs_invV_linfit.png')
    plt.savefig(fig3, dpi=150)
    plt.close()
    print(f"[SAVE] fig3 -> {fig3}")



    print(f"[SAVE] stats -> {dump}")
    print(f"[SAVE] fig1 -> {fig1}")
    print(f"[SAVE] fig2 -> {fig2}")
    print("[done] vscan")

if __name__ == '__main__':
    main()
