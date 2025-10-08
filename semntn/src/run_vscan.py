
import argparse, yaml, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from lyap_outer import run_episode

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
    trace = pd.read_csv(args.trace)
    snr_db = trace['snr_db'].to_numpy().astype(float)

    V_list = vscan.get('V_list', [10,20,40,80,160])
    repeat_each = int(vscan.get('repeat_each', 1))

    stats = []
    for V in V_list:
        for r in range(repeat_each):
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

    # Fig2: Q,J vs V
    plt.figure(figsize=(6,4))
    plt.plot(dfg['V'], dfg['Q_mean'], marker='o', label='mean Q')
    plt.plot(dfg['V'], dfg['J_mean'], marker='s', label='mean J')
    plt.xlabel('V'); plt.ylabel('Queue level'); plt.title('Q and J vs V')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    fig2 = os.path.join(args.outdir,'figs','Fig2_QJ_vs_V.png')
    plt.savefig(fig2, dpi=150); plt.close()

    print(f"[SAVE] stats -> {dump}")
    print(f"[SAVE] fig1 -> {fig1}")
    print(f"[SAVE] fig2 -> {fig2}")
    print("[done] vscan")

if __name__ == '__main__':
    main()
