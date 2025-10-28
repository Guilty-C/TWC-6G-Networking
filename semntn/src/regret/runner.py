import os, csv, json
import numpy as np

from .envs import BernoulliBandit, GaussianBandit, gaps_from_means
from .algorithms import UCB1, EpsilonGreedy, SemUCB
from .metrics import RegretMeter, order_check
from .plots import plot_regret_T, plot_regret_logT, plot_regret_over_logT


def _mk_outdir(d):
    os.makedirs(d, exist_ok=True)
    return d


def _build_env(env_cfg, seed):
    typ = str(env_cfg.get("type", "bernoulli")).lower()
    if typ == "bernoulli":
        return BernoulliBandit(env_cfg["means"], seed=seed)
    elif typ == "gaussian":
        return GaussianBandit(env_cfg["mus"], sigma=env_cfg.get("sigma", 0.1), seed=seed)
    else:
        raise ValueError(f"Unsupported env type: {typ}")


def _build_algo(algo_name, K, cfg):
    name = str(algo_name).lower()
    if name == "ucb1":
        return UCB1(K, **cfg.get("ucb1", {}))
    if name in ("eg", "epsilon_greedy", "epsgreedy"):
        return EpsilonGreedy(K, **cfg.get("eg", {}))
    if name in ("sem_ucb", "semucb"):
        return SemUCB(K, **cfg.get("sem_ucb", {}))
    raise ValueError(f"Unsupported algo: {algo_name}")


def run(cfg: dict):
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    env = _build_env(cfg["env"], seed=rng.integers(1 << 31))
    K = getattr(env, "K")
    T = int(cfg["env"]["T"])

    algo = _build_algo(cfg["algo"], K, cfg)
    # Regret computed w.r.t. expected means
    means = getattr(env, "means", getattr(env, "mus", None))
    meter = RegretMeter(opt_mean=float(np.max(means)))

    regrets = []
    # warm start: pull each arm once (for UCB counters stability)
    for a in range(K):
        r = env.pull(a)
        algo.update(a, r)
        regrets.append(meter.update(mu_a=means[a]))

    for _ in range(max(0, T - K)):
        a = algo.select()
        r = env.pull(a)
        algo.update(a, r)
        regrets.append(meter.update(mu_a=means[a]))

    out_dir = _mk_outdir(cfg["out_dir"])
    csv_path = os.path.join(out_dir, "regret_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "R_t"])
        for t, R in enumerate(regrets, 1):
            w.writerow([t, float(R)])

    plot_regret_T(regrets, os.path.join(out_dir, "Regret_vs_T.png"), dpi=cfg["plot"]["dpi"])
    plot_regret_logT(regrets, os.path.join(out_dir, "Regret_vs_logT.png"), dpi=cfg["plot"]["dpi"])
    plot_regret_over_logT(regrets, os.path.join(out_dir, "Regret_over_logT.png"), dpi=cfg["plot"]["dpi"])

    gaps = gaps_from_means(means)
    report = order_check(regrets, gaps, cfg["accept"])

    # 便于 CLI 打印/管线使用
    return report
