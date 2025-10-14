from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wer_proxy import per_from_snr_db, wer_from_per


# ---- 全局：单 episode 内有效 ----
_AGENT: Optional["_UCBManager"] = None


@dataclass
class _UCBConfig:
    """统一管理内层 UCB 配置。"""

    mode: str = "ucb1"
    alpha: float = 1.0
    window: int = 0
    beta_discount: float = 0.0
    constraint_sensitive: bool = False  # [TBD]：约束感知触发逻辑
    delayed_feedback: bool = False
    eps_rand: float = 0.10

    def clone(self) -> "_UCBConfig":
        return replace(self)

    def update_from_dict(self, updates: Dict[str, object]) -> None:
        for key, value in updates.items():
            if key == "mode" and value is not None:
                mode = str(value).lower()
                if mode not in {"ucb1", "ucbv", "linucb", "sw_ucb"}:
                    raise ValueError(f"unsupported UCB mode: {value!r}")
                self.mode = mode
            elif key == "alpha" and value is not None:
                alpha = float(value)
                if alpha <= 0:
                    raise ValueError("alpha must be positive")
                self.alpha = alpha
            elif key == "window" and value is not None:
                win = int(value)
                if win < 0:
                    raise ValueError("window must be non-negative")
                self.window = win
            elif key == "beta_discount" and value is not None:
                beta = float(value)
                if beta < 0.0:
                    raise ValueError("beta_discount must be >= 0")
                self.beta_discount = beta
            elif key == "constraint_sensitive" and value is not None:
                self.constraint_sensitive = bool(value)
            elif key == "delayed_feedback" and value is not None:
                self.delayed_feedback = bool(value)
            elif key == "eps_rand" and value is not None:
                eps = float(value)
                if not (0.0 <= eps <= 1.0):
                    raise ValueError("eps_rand must lie in [0, 1]")
                self.eps_rand = eps
            else:
                # 未使用字段统一标为 [TBD]
                if value is not None:
                    raise KeyError(f"[TBD] unsupported config key: {key}")


_CFG = _UCBConfig()


class _RewardNormalizer:
    """使用 EWMA 对奖励做在线标准化。"""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.mu = 0.0
        self.sigma = 1.0

    def preview(self, r_raw: float) -> float:
        mu = self.mu if self.initialized else float(r_raw)
        sigma = max(self.sigma if self.initialized else 1.0, 1e-6)
        z = (float(r_raw) - mu) / sigma
        return float(np.clip(1.0 / (1.0 + np.exp(-0.75 * z)), 0.0, 1.0))

    def update(self, r_raw: float) -> float:
        x = float(r_raw)
        if not self.initialized:
            self.initialized = True
            self.mu = x
            self.sigma = 1.0
        else:
            self.mu = (1.0 - self.alpha) * self.mu + self.alpha * x
            abs_dev = abs(x - self.mu)
            self.sigma = (1.0 - self.alpha) * self.sigma + self.alpha * max(abs_dev, 1e-6)
        sigma = max(self.sigma, 1e-6)
        z = (x - self.mu) / sigma
        return float(np.clip(1.0 / (1.0 + np.exp(-0.75 * z)), 0.0, 1.0))


_NORMALIZER = _RewardNormalizer()


def _rate_penalty_db(q_bps: float, q_ref: float = 300.0, c_db: float = 3.5) -> float:
    """速率越高门限越高"""

    return float(c_db * np.log2(max(q_bps, 1.0) / q_ref + 1.0))


def _d_quant(q_bps: float) -> float:
    """[TBD] 量化失真占位实现。"""

    return float(1.0 / max(q_bps, 1.0))


def _compute_context_features(ctx: Dict[str, float], Q_t: float, J_t: float, V: float) -> np.ndarray:
    snr_db = float(ctx.get("snr_db", 0.0))
    snr_linear = 10.0 ** (snr_db / 10.0)
    logS = float(np.log(max(snr_linear, 1e-8)))
    Q_val = float(Q_t)
    J_val = float(J_t)
    return np.array([
        1.0,
        logS,
        logS ** 2,
        Q_val,
        J_val,
        Q_val * logS,
        J_val * logS,
    ], dtype=float)


def _compute_arm_features(action: Tuple[float, float, int]) -> np.ndarray:
    q, p, b = action
    _ = p  # [TBD]：如有需要可将功率纳入特征
    phi = np.array([
        1.0,
        float(q),
        _d_quant(float(q)),
    ], dtype=float)
    return phi


def _estimate_action_metrics(action: Tuple[float, float, int], ctx: Dict[str, float],
                             sem_weight: float) -> Dict[str, float]:
    q, p, b = action
    snr_db = float(ctx.get("snr_db", 0.0))
    slot_sec = float(ctx.get("slot_sec", 0.02))

    snr_eff_db = snr_db + 10.0 * np.log10(max(p * b, 1e-6)) - _rate_penalty_db(q)
    per = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
    swwer = wer_from_per(per, sem_weight)
    S_bits = float(q * (1.0 - per) * slot_sec)
    E_hat = float(p)

    return {
        "q": float(q),
        "S_bits": S_bits,
        "E_hat": E_hat,
        "swwer": float(swwer),
        "per": float(per),
        "snr_eff_db": float(snr_eff_db),
    }


def _compute_reward_raw(q_val: float, metrics: Dict[str, float], Q_t: float, J_t: float,
                        V: float, Q_scale: float, J_scale: float) -> Tuple[float, float, float, float]:
    a = float(Q_t) / float(Q_scale)
    b_w = float(J_t) / float(J_scale)
    V0 = 80.0
    V_norm = float(V) / (float(V) + V0)
    kE = 1.0
    kW = 4.0
    kQ = 0.0004

    r_raw = (
        a * metrics["S_bits"]
        - (b_w + kE * V_norm) * metrics["E_hat"]
        - (kW * V_norm) * metrics["swwer"]
        - kQ * float(V) * float(q_val)
    )
    return float(r_raw), float(V_norm), float(kE), float(kW)


class _UCBManager:
    """统一封装多种 UCB 策略。"""

    def __init__(self, actions: Sequence[Tuple[float, float, int]], cfg: _UCBConfig) -> None:
        self.actions: List[Tuple[float, float, int]] = list(actions)
        self.cfg = cfg.clone()
        self.rng = np.random.default_rng()
        self.K = len(self.actions)

        self.counts = np.zeros(self.K, dtype=float)
        self.total_reward = np.zeros(self.K, dtype=float)
        self.sq_reward = np.zeros(self.K, dtype=float)
        self.play_counts = np.zeros(self.K, dtype=int)

        self.time_step = 0
        self.beta = float(self.cfg.beta_discount)

        self.reward_history: Optional[List[deque]] = None
        self._hist_sums: Optional[np.ndarray] = None
        self._hist_sq_sums: Optional[np.ndarray] = None
        if self.cfg.window > 0:
            self.reward_history = [deque(maxlen=self.cfg.window) for _ in range(self.K)]
            self._hist_sums = np.zeros(self.K, dtype=float)
            self._hist_sq_sums = np.zeros(self.K, dtype=float)

        self.mode = self.cfg.mode
        self.eps_rand = float(np.clip(self.cfg.eps_rand, 0.0, 1.0))

        # LinUCB 结构
        self._context_dim = 7
        self._arm_dim = 3
        self._lin_dim = self._context_dim * self._arm_dim
        self._A = np.eye(self._lin_dim, dtype=float)
        self._b = np.zeros(self._lin_dim, dtype=float)
        self._A_inv: Optional[np.ndarray] = np.eye(self._lin_dim, dtype=float)
        self._theta: Optional[np.ndarray] = np.zeros(self._lin_dim, dtype=float)
        self._theta_stale = True

        self.pending_selection: Optional[Dict[str, object]] = None
        self.last_output: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------
    def plan_action(self, ctx: Dict[str, float], Q_t: float, J_t: float, V: float,
                    sem_weight: float, Q_scale: float, J_scale: float) -> Dict[str, object]:
        self.time_step += 1

        context_features = None
        if self.mode == "linucb":
            context_features = _compute_context_features(ctx, Q_t, J_t, V)

        idx, per_action = self._select_action(context_features)
        action = self.actions[idx]
        self.play_counts[idx] += 1

        metrics = _estimate_action_metrics(action, ctx, sem_weight)
        r_raw, V_norm, kE, kW = _compute_reward_raw(action[0], metrics, Q_t, J_t, V, Q_scale, J_scale)

        if self.cfg.delayed_feedback:
            r01 = _NORMALIZER.preview(r_raw)
            self.pending_selection = {
                "index": idx,
                "action": action,
                "context_features": context_features,
                "feature_vector": self._extract_feature_vector(action, context_features),
                "metrics_est": metrics,
                "reward_raw": r_raw,
            }
        else:
            feature_vector = self._extract_feature_vector(action, context_features)
            r01 = _NORMALIZER.update(r_raw)
            self._apply_update(idx, r01, feature_vector)

        estimated_rewards = {
            str(act): float(info.get("score", 0.0))
            for act, info in per_action.items()
        }

        debug_ucb = {}
        for act, info in per_action.items():
            payload = {k: float(v) for k, v in info.items() if k != "feature_vector"}
            if "feature_vector" in info:
                payload["feature_dim"] = len(info["feature_vector"])
            debug_ucb[str(act)] = payload

        result = {
            "action": (float(action[0]), float(action[1]), int(action[2]), "ucb"),
            "S_bits": metrics["S_bits"],
            "E_hat": metrics["E_hat"],
            "SWWER_hat": metrics["swwer"],
            "per": metrics["per"],
            "snr_eff_db": metrics["snr_eff_db"],
            "estimated_rewards": estimated_rewards,
            "debug_info": {
                "selected_index": idx,
                "reward01": float(r01),
                "r_raw": float(r_raw),
                "V_parameter": float(V),
                "V_norm": float(V_norm),
                "kE": kE,
                "kW": kW,
                "ucb_snapshot": debug_ucb,
                "mode": self.mode,
            },
        }

        self.last_output = result
        return result

    def ingest_feedback(self, ctx: Dict[str, float], Q_t: float, J_t: float, V: float,
                         sem_weight: float, Q_scale: float, J_scale: float) -> None:
        if not self.cfg.delayed_feedback or self.pending_selection is None:
            return

        pending = self.pending_selection
        idx = int(pending["index"])
        action = pending["action"]

        S_bits = float(ctx.get("S_bits_obs", pending["metrics_est"]["S_bits"]))
        E_obs = float(ctx.get("E_obs", pending["metrics_est"]["E_hat"]))
        swwer_obs = float(ctx.get("swwer_obs", pending["metrics_est"]["swwer"]))
        per_obs = float(ctx.get("per_obs", pending["metrics_est"]["per"]))

        metrics = {
            "S_bits": S_bits,
            "E_hat": E_obs,
            "swwer": swwer_obs,
            "per": per_obs,
            "snr_eff_db": pending["metrics_est"]["snr_eff_db"],
        }

        r_raw, _, _, _ = _compute_reward_raw(action[0], metrics, Q_t, J_t, V, Q_scale, J_scale)
        r01 = _NORMALIZER.update(r_raw)
        feature_vector = pending.get("feature_vector")
        self._apply_update(idx, r01, feature_vector)
        self.pending_selection = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _extract_feature_vector(self, action: Tuple[float, float, int],
                                context_features: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if self.mode != "linucb" or context_features is None:
            return None
        phi = _compute_arm_features(action)
        return np.kron(phi, context_features)

    def _apply_update(self, idx: int, reward: float, feature_vector: Optional[np.ndarray]) -> None:
        if self.cfg.window > 0 and self.reward_history is not None and self._hist_sums is not None:
            hist = self.reward_history[idx]
            hist.append(float(reward))
            self._hist_sums[idx] += reward
            self._hist_sq_sums[idx] += reward * reward
            if len(hist) > self.cfg.window:
                old = hist.popleft()
                self._hist_sums[idx] -= old
                self._hist_sq_sums[idx] -= old * old
            self.total_reward[idx] = self._hist_sums[idx]
            self.sq_reward[idx] = self._hist_sq_sums[idx]
            self.counts[idx] = float(len(hist))
        elif self.beta > 0.0:
            beta = self.beta
            self.total_reward[idx] = beta * self.total_reward[idx] + reward
            self.sq_reward[idx] = beta * self.sq_reward[idx] + reward * reward
            self.counts[idx] = beta * self.counts[idx] + 1.0
        else:
            self.total_reward[idx] += reward
            self.sq_reward[idx] += reward * reward
            self.counts[idx] += 1.0

        if self.mode == "linucb" and feature_vector is not None:
            self._update_linucb(feature_vector, reward)

    def _update_linucb(self, feature_vector: np.ndarray, reward: float) -> None:
        z = feature_vector.reshape(self._lin_dim)
        self._A += np.outer(z, z)
        self._b += z * reward
        self._A_inv = None
        self._theta_stale = True

    def _get_A_inv(self) -> np.ndarray:
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self._A)
        return self._A_inv

    def _get_theta(self) -> np.ndarray:
        if self._theta_stale or self._theta is None:
            self._theta = self._get_A_inv() @ self._b
            self._theta_stale = False
        return self._theta

    def _select_action(self, context_features: Optional[np.ndarray]) -> Tuple[int, Dict[str, Dict[str, float]]]:
        untried = np.where(self.play_counts == 0)[0]
        per_action: Dict[str, Dict[str, float]] = {}

        if len(untried) > 0:
            idx = int(self.rng.choice(untried))
            info = {"score": 0.0, "mean": 0.0, "bonus": 0.0}
            per_action[str(self.actions[idx])] = info
            return idx, per_action

        if self.mode == "linucb":
            return self._select_action_linucb(context_features)
        if self.mode == "ucbv":
            return self._select_action_ucbv()
        if self.mode == "sw_ucb":
            return self._select_action_sw_ucb()
        return self._select_action_ucb1()

    def _select_action_ucb1(self) -> Tuple[int, Dict[str, Dict[str, float]]]:
        counts = np.maximum(self.counts, 1e-9)
        means = self.total_reward / counts
        log_term = np.log(max(self.time_step, 2))
        bonus = np.sqrt(np.maximum(0.0, self.cfg.alpha * log_term / counts))
        scores = means + bonus

        info = {}
        for idx, action in enumerate(self.actions):
            info[str(action)] = {
                "mean": float(means[idx]),
                "bonus": float(bonus[idx]),
                "score": float(scores[idx]),
            }

        chosen = int(np.argmax(scores))
        if self.eps_rand > 0.0 and self.rng.random() < self.eps_rand:
            chosen = int(self.rng.integers(self.K))
        return chosen, info

    def _select_action_ucbv(self) -> Tuple[int, Dict[str, Dict[str, float]]]:
        counts = np.maximum(self.counts, 1e-9)
        means = self.total_reward / counts
        variances = np.clip(self.sq_reward / counts - means ** 2, 0.0, None)
        log_term = np.log(max(self.time_step, 2))
        bonus = (
            np.sqrt(np.maximum(0.0, 2.0 * variances * log_term / counts))
            + (3.0 * log_term / counts)
        ) * float(self.cfg.alpha)
        scores = means + bonus

        info = {}
        for idx, action in enumerate(self.actions):
            info[str(action)] = {
                "mean": float(means[idx]),
                "variance": float(variances[idx]),
                "bonus": float(bonus[idx]),
                "score": float(scores[idx]),
            }

        chosen = int(np.argmax(scores))
        if self.eps_rand > 0.0 and self.rng.random() < self.eps_rand:
            chosen = int(self.rng.integers(self.K))
        return chosen, info

    def _select_action_sw_ucb(self) -> Tuple[int, Dict[str, Dict[str, float]]]:
        # 滑动窗口/折扣 UCB：沿用 UCB1 形式，依赖更新阶段维护的有效样本数。
        return self._select_action_ucb1()

    def _select_action_linucb(self, context_features: Optional[np.ndarray]) -> Tuple[int, Dict[str, Dict[str, float]]]:
        if context_features is None:
            raise ValueError("context features required for LinUCB")

        theta = self._get_theta()
        A_inv = self._get_A_inv()

        best_idx = 0
        best_score = -np.inf
        info: Dict[str, Dict[str, float]] = {}

        for idx, action in enumerate(self.actions):
            phi = _compute_arm_features(action)
            feature_vector = np.kron(phi, context_features)
            mean = float(feature_vector @ theta)
            var = float(feature_vector @ (A_inv @ feature_vector))
            bonus = float(self.cfg.alpha * np.sqrt(max(var, 1e-9)))
            score = mean + bonus

            key = str(action)
            info[key] = {
                "mean": mean,
                "bonus": bonus,
                "variance": var,
                "score": score,
                "feature_vector": feature_vector,
            }

            if score > best_score:
                best_score = score
                best_idx = idx

        if self.eps_rand > 0.0 and self.rng.random() < self.eps_rand:
            best_idx = int(self.rng.integers(self.K))

        return best_idx, info


# ================================
# 工具
# ================================
def _init_agent(action_space: Dict[str, Sequence[float]]) -> None:
    global _AGENT
    q_list = action_space.get("q_bps", [300, 600, 1200, 2400])
    p_list = action_space.get("p_grid_w", [0.2, 0.4, 0.8, 1.2, 1.6])
    b_list = action_space.get("b_subbands", [1, 2, 3])

    actions: List[Tuple[float, float, int]] = []
    for q in q_list:
        for p in p_list:
            for b in b_list:
                actions.append((float(q), float(p), int(b)))
    _AGENT = _UCBManager(actions, _CFG)


def set_ucb_config(alpha=None, eps_rand=None, config: Optional[Dict[str, object]] = None, **extra) -> None:
    """配置 UCB 内层参数。"""

    updates: Dict[str, object] = {}
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dict when provided")
        updates.update(config)

    updates.update(extra)

    if alpha is not None:
        updates["alpha"] = alpha
    if eps_rand is not None:
        updates["eps_rand"] = eps_rand

    if updates:
        _CFG.update_from_dict(updates)


# ================================
# 核心 API
# ================================
def pick_action_and_estimate(ctx, action_space, Q_t, J_t, V, sem_weight,
                             Q_scale=1e4, J_scale=1e2):
    global _AGENT

    if _AGENT is None:
        _init_agent(action_space)

    if isinstance(ctx, dict) and "S_bits_obs" in ctx:
        _AGENT.ingest_feedback(ctx, Q_t, J_t, V, sem_weight, Q_scale, J_scale)
        return getattr(_AGENT, "last_output", {})

    return _AGENT.plan_action(ctx, Q_t, J_t, V, sem_weight, Q_scale, J_scale)


def reset_agent():
    global _AGENT
    _AGENT = None
    _NORMALIZER.reset()
