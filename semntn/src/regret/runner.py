from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np

from .algorithms import EpsilonGreedyPolicy, OraclePolicy, RandomPolicy, UCB1Policy
from .envs import BernoulliBandit, GaussianBandit
from .metrics import (
    compute_pseudo_regret,
    correlation_vs_sqrt_t,
    precompute_mu_true,
    slope_vs_sqrt_t,
)
from .plots import plot_regret_curves


POLICY_LABELS = {
    "ucb": "UCB1",
    "eps": "ε-Greedy",
    "rand": "Random",
    "oracle": "Oracle",
}


def _timestamp_dir(root: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out


def _build_env(env_cfg: Dict, seed: int):
    typ = str(env_cfg.get("type", "bernoulli")).lower()
    if typ == "bernoulli":
        return BernoulliBandit(env_cfg["means"], seed=seed)
    if typ == "gaussian":
        return GaussianBandit(
            env_cfg["mus"],
            sigma=env_cfg.get("sigma", 0.1),
            seed=seed,
        )
    raise ValueError(f"Unsupported env type: {typ}")


def _build_policy(name: str, num_actions: int, mu_true: Dict[int, float], seed: int, cfg: Dict):
    lname = name.lower()
    if lname in ("ucb", "ucb1"):
        return UCB1Policy(num_actions=num_actions, alpha=float(cfg.get("alpha", 2.0)))
    if lname in ("eps", "epsilon_greedy", "eg"):
        epsilon = float(cfg.get("epsilon", 0.05))
        return EpsilonGreedyPolicy(num_actions=num_actions, epsilon=epsilon, seed=seed)
    if lname in ("rand", "random"):
        return RandomPolicy(num_actions=num_actions, seed=seed)
    if lname == "oracle":
        return OraclePolicy(mu_true=mu_true)
    raise ValueError(f"Unsupported policy: {name}")


def _simulate_policy(env_cfg: Dict, policy_name: str, seeds: Iterable[int], policy_cfg: Dict) -> np.ndarray:
    regrets = []
    mu_true = None
    T = int(env_cfg["T"])

    for seed in seeds:
        env = _build_env(env_cfg, seed=seed)
        if mu_true is None:
            mu_true = precompute_mu_true(env)
        policy = _build_policy(policy_name, env.K, mu_true, seed, policy_cfg)

        actions: List[int] = []
        for t in range(1, T + 1):
            a = policy.select_action(t)
            reward_raw = env.pull(a)
            reward01 = env.map_reward_to_unit_interval(reward_raw)
            policy.update(a, reward01)
            actions.append(a)

        regret = compute_pseudo_regret(actions, mu_true)
        regrets.append(regret)

    return np.vstack(regrets)


def _aggregate_regrets(regrets: np.ndarray, confidence: float = 0.95) -> Dict[str, np.ndarray]:
    mean = regrets.mean(axis=0)
    n = regrets.shape[0]
    if n > 1:
        std = regrets.std(axis=0, ddof=1)
        z = 1.96 if confidence == 0.95 else 1.96
        half_width = z * std / np.sqrt(n)
    else:
        half_width = np.zeros_like(mean)
    return {
        "mean": mean,
        "lo": mean - half_width,
        "hi": mean + half_width,
    }


def run(
    cfg: Dict,
    outdir: str,
    make_plots: bool = True,
    save_csv: bool = True,
) -> Dict:
    env_cfg = cfg["env"]
    baselines = [str(x).lower() for x in cfg.get("baselines", ["ucb"])]
    seeds = cfg.get("seeds", [0])
    if not isinstance(seeds, (list, tuple)):
        raise ValueError("'seeds' must be a list of integers")
    seeds = [int(s) for s in seeds]

    for baseline in baselines:
        if baseline not in POLICY_LABELS:
            raise ValueError(f"Unknown baseline '{baseline}'. Supported: {sorted(POLICY_LABELS)}")

    policy_cfgs = {
        "ucb": cfg.get("ucb", cfg.get("ucb1", {})),
        "eps": {"epsilon": cfg.get("epsilon", cfg.get("eg", {}).get("epsilon", 0.05))},
        "rand": {},
        "oracle": {},
    }

    T = int(env_cfg["T"])
    timeline = np.arange(1, T + 1, dtype=int)
    results: Dict[str, Dict] = {}
    for policy_name in baselines:
        regrets = _simulate_policy(env_cfg, policy_name, seeds, policy_cfgs[policy_name])
        agg = _aggregate_regrets(regrets)
        results[policy_name] = {
            "label": POLICY_LABELS.get(policy_name, policy_name),
            "T": timeline,
            "mean": agg["mean"],
            "lo": agg["lo"],
            "hi": agg["hi"],
            "regrets": regrets,
        }

    if make_plots:
        plot_cfg = cfg.get("plot", {})
        plot_regret_curves(
            results,
            os.path.join(outdir, "regret.png"),
            logx=bool(plot_cfg.get("logx", False)),
            show_ref_sqrt=bool(plot_cfg.get("show_ref_sqrt", True)),
        )

    if save_csv:
        csv_path = os.path.join(outdir, "regret.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["t"]
            for policy_name in baselines:
                header.extend([
                    f"{policy_name}_mean",
                    f"{policy_name}_lo",
                    f"{policy_name}_hi",
                ])
            writer.writerow(header)
            for idx, t in enumerate(timeline):
                row = [t]
                for policy_name in baselines:
                    row.extend([
                        float(results[policy_name]["mean"][idx]),
                        float(results[policy_name]["lo"][idx]),
                        float(results[policy_name]["hi"][idx]),
                    ])
                writer.writerow(row)

    ucb_key = "ucb"
    oracle_key = "oracle"
    eps_key = "eps"
    rand_key = "rand"
    ucb_mean = results[ucb_key]["mean"]
    rand_mean = results[rand_key]["mean"]
    eps_mean = results[eps_key]["mean"]
    oracle_mean = results[oracle_key]["mean"]

    corr = correlation_vs_sqrt_t(ucb_mean)
    slope = slope_vs_sqrt_t(ucb_mean)
    c_star = float(ucb_mean[-1] / np.sqrt(T))

    pass_checks = (
        corr >= 0.98
        and ucb_mean[-1] < rand_mean[-1]
        and ucb_mean[-1] < eps_mean[-1]
        and ucb_mean[-1] > oracle_mean[-1]
    )

    summary = {"slope": slope, "C*": c_star, "pass": bool(pass_checks)}

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "correlation": corr}, f, indent=2)

    return summary


def run_with_timestamp(cfg: Dict, base_outdir: str, make_plots: bool = True, save_csv: bool = True) -> Dict:
    outdir = _timestamp_dir(base_outdir)
    return run(cfg, outdir=outdir, make_plots=make_plots, save_csv=save_csv)
