
import argparse, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from utils_seed import set_all_seeds
from bandit_env import StationaryBernoulliBandit
from ucb import UCB1

ap=argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=20000)
ap.add_argument("--means", type=str, default="0.60,0.62,0.65,0.68,0.70")
ap.add_argument("--outdir", default="outputs")
args=ap.parse_args()

rng=set_all_seeds(2025)
means=[float(x) for x in args.means.split(",")]
env=StationaryBernoulliBandit(means, rng)
algo=UCB1(len(means))
best=env.best_mean

reg=[]; rew=[]; arm_hist=[]
cum=0.0
for t in range(1, args.T+1):
    a=algo.select()
    r=env.pull(a)
    algo.update(a,r)
    cum += (best - means[a])    # pseudo-regret to best fixed arm
    reg.append(cum); rew.append(r); arm_hist.append(a)

os.makedirs(f"{args.outdir}/figs", exist_ok=True)
os.makedirs(f"{args.outdir}/dumps", exist_ok=True)

# 保存日志
log=pd.DataFrame({"t":range(1,args.T+1),"arm":arm_hist,"reward":rew,"cum_regret":reg})
log.to_csv(f"{args.outdir}/dumps/ucb_log.csv", index=False)

# 画累计遗憾 & sqrt(t)参考线（归一到同终点，便于观感）
t=np.arange(1,args.T+1)
ref=np.sqrt(t); ref *= (reg[-1]/ref[-1])
plt.figure(figsize=(7,4.5))
plt.plot(t, reg, label="UCB1 cumulative regret")
plt.plot(t, ref, linestyle="--", label="c·√t (reference)")
plt.xlabel("time"); plt.ylabel("cumulative pseudo-regret")
plt.title("UCB1 on K-armed Bernoulli bandit")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{args.outdir}/figs/Fig4_UCB_regret.png", dpi=150); plt.close()

print(f"[SAVE] log  -> {args.outdir}/dumps/ucb_log.csv")
print(f"[SAVE] fig  -> {args.outdir}/figs/Fig4_UCB_regret.png")
print("[done] regret")
