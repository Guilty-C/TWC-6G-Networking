"""Evaluation harness comparing vanilla UCB and LRCCUCB."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

import inner_api_ucb
from sem_weight_model import SemWeightModel
from utils_seed import set_all_seeds
from wer_proxy import per_from_snr_db, wer_from_per

logger = logging.getLogger(__name__)


def _rolling_features(snr_db: np.ndarray, win: int) -> np.ndarray:
    s = np.asarray(snr_db, dtype=float)
    K = min(win, len(s))
    head = s[:K]
    energy = np.maximum(head, 0.0)
    energy_mean = float(energy.mean())
    diff = np.diff(head)
    sign = np.sign(diff)
    zcr = float((np.abs(np.diff(sign)) > 0).sum()) / max(len(sign), 1)
    pos = np.maximum(head, 0.0)
    idx = np.arange(1, len(pos) + 1, dtype=float)
    spec_centroid = float((pos * idx).sum() / (pos.sum() + 1e-6))
    snr_mean = float(head.mean())
    keyword_flag = 1.0 if np.max(head) > 8.0 else 0.0
    return np.array([energy_mean, zcr, spec_centroid, snr_mean, keyword_flag], dtype=float)


class EvalConfig(BaseModel):
    seed: int = Field(2025)
    trace_csv: str = Field("data/channel_trace.csv")
    slot_sec: float = Field(0.02)
    K_head_pkts: int = Field(5)
    T: int = Field(600)
    E_per_slot: float = Field(0.5)
    A_bps: float = Field(600.0)
    Q_scale: float = Field(1e4)
    J_scale: float = Field(1e2)
    V: int = Field(50)
    change_slot: int = Field(300)
    recovery_window: int = Field(10)
    recovery_tol: float = Field(0.05)
    swwer_threshold: float = Field(0.4)
    bucket_count: int = Field(5)
    output_dir: str = Field("outputs/ucb_eval")
    episodes: int = Field(1)
    algorithms: List[str] = Field(default_factory=lambda: ["ucb", "lrc_cucb"])
    lrc_cucb: Dict[str, Any] = Field(default_factory=dict)
    V_scan: List[int] = Field(default_factory=lambda: [10, 30, 60, 120, 240])
    regret_T: int = Field(400)


@dataclass
class SlotLog:
    t: int
    algo: str
    action: Tuple[float, float, int, str]
    S_bits: float
    E_hat: float
    swwer: float
    per: float
    reward: float
    best_reward: float
    regret: float
    violation: int
    runtime_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["action"] = list(self.action)
        return d


def _action_space(cfg: EvalConfig) -> Dict[str, Iterable[float]]:
    return {
        "q_bps": [300, 600, 1200, 2400],
        "p_grid_w": [0.2, 0.4, 0.8, 1.2, 1.6],
        "b_subbands": [1, 2, 3],
    }


def _estimate_metrics(action: Tuple[float, float, int], ctx: Dict[str, float], sem_weight: float) -> Dict[str, float]:
    q, p, b = action
    snr_db = float(ctx.get("snr_db", 0.0))
    snr_eff_db = snr_db + 10.0 * math.log10(max(p * b, 1e-6)) - 3.5 * math.log2(q / 300.0 + 1.0)
    per = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
    slot_sec = float(ctx.get("slot_sec", 0.02))
    S_bits = float(q * (1.0 - per) * slot_sec)
    E_hat = float(p)
    swwer = float(wer_from_per(per, sem_weight))
    return {
        "S_bits": S_bits,
        "E_hat": E_hat,
        "SWWER_hat": swwer,
        "per": float(per),
        "snr_eff_db": float(snr_eff_db),
    }


def _reward_from_metrics(metrics: Dict[str, float], ctx: Dict[str, float], Q: float, J: float, V: float, cfg: EvalConfig) -> float:
    slot_sec = float(ctx.get("slot_sec", cfg.slot_sec))
    q_max = max(_action_space(cfg)["q_bps"])
    p_list = _action_space(cfg)["p_grid_w"]
    p_min, p_max = min(p_list), max(p_list)
    a = Q / cfg.Q_scale
    b = J / cfg.J_scale
    S_bits = metrics["S_bits"]
    E_hat = metrics["E_hat"]
    swwer = metrics["SWWER_hat"]
    r_raw = a * S_bits - b * E_hat - V * swwer
    r_min = -b * p_max - V * 1.0
    r_max = a * (q_max * slot_sec) - b * p_min
    if r_max - r_min < 1e-9:
        return 0.0
    return float((r_raw - r_min) / (r_max - r_min))


def simulate_episode(algo: str, cfg: EvalConfig, snr_trace: np.ndarray, lrc_cfg: Dict[str, Any]) -> Tuple[List[SlotLog], Dict[str, Any]]:
    inner_api_ucb.configure(algo=algo, config=lrc_cfg if algo == "lrc_cucb" else None)
    set_all_seeds(cfg.seed)
    action_space = _action_space(cfg)
    T = min(cfg.T, len(snr_trace))
    slot_sec = cfg.slot_sec
    A_bits = cfg.A_bps * slot_sec
    E_bar = cfg.E_per_slot
    model = SemWeightModel(w_min=1.0, w_max=3.0)
    Q = 0.0
    J = 0.0
    logs: List[SlotLog] = []
    best_rewards: List[float] = []
    snr_db = snr_trace[:T]
    for t in range(T):
        left = max(0, t - cfg.K_head_pkts + 1)
        feat = _rolling_features(snr_db[left : t + 1], cfg.K_head_pkts)
        w_sem = model.infer_w_sem(feat)
        ctx = {
            "snr_db": float(snr_db[t]),
            "E_bar": E_bar,
            "slot_sec": slot_sec,
            "A_bits": A_bits,
        }
        start = time.perf_counter()
        pick = inner_api_ucb.pick_action_and_estimate(ctx, action_space, Q, J, cfg.V, w_sem, Q_scale=cfg.Q_scale, J_scale=cfg.J_scale)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        metrics = {
            "S_bits": float(pick["S_bits"]),
            "E_hat": float(pick["E_hat"]),
            "SWWER_hat": float(pick["SWWER_hat"]),
            "per": float(pick.get("per", 0.0)),
        }
        reward = _reward_from_metrics(metrics, ctx, Q, J, cfg.V, cfg)
        best_reward = max(
            _reward_from_metrics(_estimate_metrics((float(q), float(p), int(b)), ctx, w_sem), ctx, Q, J, cfg.V, cfg)
            for q in action_space["q_bps"]
            for p in action_space["p_grid_w"]
            for b in action_space["b_subbands"]
        )
        regret = max(best_reward - reward, 0.0)
        best_rewards.append(best_reward)
        ctx_obs = dict(ctx)
        ctx_obs.update(
            {
                "S_bits_obs": metrics["S_bits"],
                "E_obs": metrics["E_hat"],
                "swwer_obs": metrics["SWWER_hat"],
            }
        )
        inner_api_ucb.pick_action_and_estimate(ctx_obs, action_space, Q, J, cfg.V, w_sem, Q_scale=cfg.Q_scale, J_scale=cfg.J_scale)
        Q = max(Q - metrics["S_bits"], 0.0) + A_bits
        J = max(J + metrics["E_hat"] - E_bar, 0.0)
        violation = int(metrics["E_hat"] > E_bar or metrics["SWWER_hat"] > cfg.swwer_threshold)
        logs.append(
            SlotLog(
                t=t,
                algo=algo,
                action=pick["action"],
                S_bits=metrics["S_bits"],
                E_hat=metrics["E_hat"],
                swwer=metrics["SWWER_hat"],
                per=metrics["per"],
                reward=reward,
                best_reward=best_reward,
                regret=regret,
                violation=violation,
                runtime_ms=runtime_ms,
            )
        )
    return logs, {"mean_best_reward": float(np.mean(best_rewards))}


def _compute_recovery_time(logs: List[SlotLog], cfg: EvalConfig) -> float:
    if not logs:
        return float("nan")
    change = min(cfg.change_slot, len(logs) - 1)
    pre = [log.swwer for log in logs[max(0, change - cfg.recovery_window) : change]]
    if not pre:
        return float("nan")
    baseline = float(np.mean(pre))
    for idx in range(change, len(logs)):
        window = [log.swwer for log in logs[idx : min(len(logs), idx + cfg.recovery_window)]]
        if not window:
            break
        if abs(np.mean(window) - baseline) <= cfg.recovery_tol * max(baseline, 1e-6):
            return float(idx - change)
    return float(len(logs) - change)


def _summarise(logs: List[SlotLog]) -> Dict[str, float]:
    swwer = np.array([log.swwer for log in logs])
    reward = np.array([log.reward for log in logs])
    regret = np.array([log.regret for log in logs])
    runtime = np.array([log.runtime_ms for log in logs])
    violations = np.array([log.violation for log in logs])
    return {
        "mean_swwer": float(swwer.mean()),
        "sample_efficiency": float(np.trapz(swwer) / max(len(swwer), 1)),
        "violation_rate": float(violations.mean()),
        "avg_reward": float(reward.mean()),
        "cum_regret": float(regret.cumsum()[-1]) if len(regret) else 0.0,
        "runtime_ms": float(runtime.mean()),
    }


def _v_scan(cfg: EvalConfig, snr: np.ndarray, lrc_cfg: Dict[str, Any], algo: str) -> pd.DataFrame:
    stats = []
    for V in cfg.V_scan:
        logs, _ = simulate_episode(algo, cfg.copy(update={"V": V}), snr, lrc_cfg)
        summary = _summarise(logs)
        summary.update({"V": V, "algo": algo})
        stats.append(summary)
    return pd.DataFrame(stats)


def _plot_vscan(df: pd.DataFrame, fig_dir: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for algo, sub in df.groupby("algo"):
        invV = 1.0 / sub["V"].astype(float).to_numpy()
        plt.plot(invV, sub["mean_swwer"], marker="o", label=f"{algo} (SWWER)")
    plt.xlabel("1/V")
    plt.ylabel("Mean SWWER")
    plt.title("V-scan for algorithms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(fig_dir, "ucb_vscan.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved V-scan figure to %s", path)


def _plot_regret(logs_by_algo: Dict[str, List[SlotLog]], cfg: EvalConfig, fig_dir: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for algo, logs in logs_by_algo.items():
        regret = np.array([log.regret for log in logs])
        cum = regret.cumsum()
        plt.plot(np.arange(1, len(cum) + 1), cum, label=f"{algo} regret")
    t = np.arange(1, cfg.regret_T + 1)
    ref = np.sqrt(t)
    if len(t) and any(len(v) for v in logs_by_algo.values()):
        scale = 1.0
        total = [np.array([log.regret for log in logs]).cumsum() for logs in logs_by_algo.values() if logs]
        if total:
            last = max(arr[-1] for arr in total if len(arr))
            scale = last / ref[min(len(ref), len(total[0])) - 1]
    plt.plot(t, ref * scale, linestyle="--", label="c·√T")
    plt.xlabel("slot")
    plt.ylabel("cumulative regret")
    plt.title("Regret comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(fig_dir, "ucb_regret.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved regret figure to %s", path)


def _plot_metrics(summary: pd.DataFrame, fig_dir: str) -> None:
    metrics = ["sample_efficiency", "violation_rate", "runtime_ms"]
    plt.figure(figsize=(8, 4))
    for idx, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), idx + 1)
        for algo, sub in summary.groupby("algo"):
            plt.bar(algo, sub[metric].mean())
        plt.title(metric)
        plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(fig_dir, "ucb_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved metrics bar chart to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UCB vs LRCCUCB evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = EvalConfig(**yaml.safe_load(handle))

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s")

    snr_trace = pd.read_csv(cfg.trace_csv)["snr_db"].to_numpy(dtype=float)
    dump_dir = os.path.join(cfg.output_dir, "dumps")
    fig_dir = os.path.join(cfg.output_dir, "figs")
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    all_logs: Dict[str, List[SlotLog]] = {}
    summaries: List[Dict[str, Any]] = []
    lrc_cfg = cfg.lrc_cucb or {}
    for algo in cfg.algorithms:
        logs, extra = simulate_episode(algo, cfg, snr_trace, lrc_cfg)
        all_logs[algo] = logs
        recovery = _compute_recovery_time(logs, cfg)
        summary = _summarise(logs)
        summary.update({"algo": algo, "recovery_time": recovery})
        summaries.append(summary)
        df = pd.DataFrame([log.to_dict() for log in logs])
        csv_path = os.path.join(dump_dir, f"{algo}_logs.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved per-slot log for %s -> %s", algo, csv_path)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(dump_dir, "ucb_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary table to %s", summary_path)

    _plot_metrics(summary_df, fig_dir)
    regret_path_data = {
        algo: [log.regret for log in logs] for algo, logs in all_logs.items()
    }
    with open(os.path.join(dump_dir, "ucb_regret_raw.json"), "w", encoding="utf-8") as handle:
        json.dump(regret_path_data, handle, indent=2)
    _plot_regret(all_logs, cfg, fig_dir)

    vscan_frames = []
    for algo in cfg.algorithms:
        v_df = _v_scan(cfg, snr_trace, lrc_cfg, algo)
        vscan_frames.append(v_df)
    vscan_df = pd.concat(vscan_frames, ignore_index=True)
    vscan_path = os.path.join(dump_dir, "ucb_vscan.csv")
    vscan_df.to_csv(vscan_path, index=False)
    logger.info("Saved V-scan stats to %s", vscan_path)
    _plot_vscan(vscan_df, fig_dir)

    summary_json = {
        "summaries": summary_df.to_dict(orient="records"),
        "vscan": vscan_df.to_dict(orient="records"),
    }
    with open(os.path.join(dump_dir, "ucb_eval_summary_raw.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
