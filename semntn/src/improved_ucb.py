"""Lyapunov-aligned contextual UCB with constrained updates (LRCCUCB)."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from wer_proxy import per_from_snr_db, wer_from_per

logger = logging.getLogger(__name__)


class LRCCUCBConfig(BaseModel):
    """Hyper-parameters for LRCCUCB."""

    exploration_coef: float = Field(1.2, description="Exploration strength")
    safety_beta: float = Field(2.0, description="Safety inflation for LCB")
    safe_reward_floor: float = Field(-0.6, description="Minimum acceptable reward")
    forget_factor: float = Field(0.98, description="Forgetting factor for updates")
    delay_max: int = Field(2, description="Maximum supported feedback delay")
    reset_window: int = Field(40, description="Window size for soft reset detection")
    reset_threshold: float = Field(0.25, description="Drop threshold triggering reset")
    variance_clip: float = Field(5.0, description="Cap for variance proxy")
    context_scale_q: float = Field(1e4, description="Scale for Q drift term")
    context_scale_j: float = Field(1e2, description="Scale for J drift term")
    snr_ref_db: float = Field(5.0, description="Reference SNR for normalisation")


@dataclass
class DelayedFeedback:
    delay: int
    feature: NDArray[np.float64]
    reward: float


class DelayedBuffer:
    """Simple FIFO buffer to hold delayed rewards."""

    def __init__(self, max_delay: int):
        self.max_delay = max_delay
        self._queue: Deque[DelayedFeedback] = deque()

    def push(self, delay: int, feature: NDArray[np.float64], reward: float) -> None:
        delay = int(min(max(delay, 0), self.max_delay))
        self._queue.append(DelayedFeedback(delay, feature, reward))

    def advance(self) -> List[DelayedFeedback]:
        ready: List[DelayedFeedback] = []
        for _ in range(len(self._queue)):
            item = self._queue.popleft()
            if item.delay <= 0:
                ready.append(item)
            else:
                item.delay -= 1
                self._queue.append(item)
        return ready


class LRCCUCB:
    """Linear contextual UCB with Lyapunov-aligned reward shaping."""

    def __init__(self, action_space: Dict[str, Iterable[float]], config: Optional[LRCCUCBConfig] = None):
        self.config = config or LRCCUCBConfig()
        self.actions = self._enumerate_actions(action_space)
        self.buffer = DelayedBuffer(self.config.delay_max)
        self.context_dim = 6
        self.action_dim = 6
        self.theta = np.zeros(self.context_dim * self.action_dim, dtype=np.float64)
        self.A_inv = np.eye(self.context_dim * self.action_dim, dtype=np.float64)
        self._reward_window: Deque[float] = deque(maxlen=self.config.reset_window)
        self._baseline_reward: Optional[float] = None
        self._last = {}

    @staticmethod
    def _enumerate_actions(action_space: Dict[str, Iterable[float]]) -> List[Tuple[float, float, int]]:
        q_list = action_space.get("q_bps", [300, 600, 1200, 2400])
        p_list = action_space.get("p_grid_w", [0.2, 0.4, 0.8, 1.2, 1.6])
        b_list = action_space.get("b_subbands", [1, 2, 3])
        actions: List[Tuple[float, float, int]] = []
        for q in q_list:
            for p in p_list:
                for b in b_list:
                    actions.append((float(q), float(p), int(b)))
        return actions

    def _action_feature(self, action: Tuple[float, float, int]) -> NDArray[np.float64]:
        q, p, b = action
        return np.array(
            [
                1.0,
                q / 2400.0,
                p / 1.6,
                b / 3.0,
                np.log1p(q / 300.0) / np.log1p(8.0),
                np.log1p(p) / np.log1p(2.0),
            ],
            dtype=np.float64,
        )

    def _context_feature(
        self,
        ctx: Dict[str, float],
        sem_weight: float,
        Q_t: float,
        J_t: float,
        V: float,
    ) -> NDArray[np.float64]:
        snr = float(ctx.get("snr_db", 0.0))
        return np.array(
            [
                1.0,
                sem_weight,
                Q_t / self.config.context_scale_q,
                J_t / self.config.context_scale_j,
                V / 100.0,
                snr / max(self.config.snr_ref_db, 1e-3),
            ],
            dtype=np.float64,
        )

    def _feature_vector(self, context: NDArray[np.float64], action: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.kron(context, action)

    def _predict_reward(self, z: NDArray[np.float64]) -> Tuple[float, float]:
        mu = float(np.dot(self.theta, z))
        var = float(np.dot(z, self.A_inv @ z))
        var = min(max(var, 1e-9), self.config.variance_clip)
        sigma = np.sqrt(var)
        return mu, sigma

    def _physical_estimate(
        self,
        action: Tuple[float, float, int],
        ctx: Dict[str, float],
        sem_weight: float,
    ) -> Dict[str, float]:
        q, p, b = action
        snr_db = float(ctx.get("snr_db", 0.0))
        snr_eff_db = snr_db + 10.0 * np.log10(max(p * b, 1e-6)) - 3.5 * np.log2(q / 300.0 + 1.0)
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

    def _select_action(
        self,
        ctx: Dict[str, float],
        Q_t: float,
        J_t: float,
        V: float,
        sem_weight: float,
    ) -> Dict[str, float]:
        context = self._context_feature(ctx, sem_weight, Q_t, J_t, V)
        best_idx = 0
        best_ucb = -np.inf
        best_safe_idx = None
        best_safe_ucb = -np.inf
        preds: List[Dict[str, float]] = []
        features: List[NDArray[np.float64]] = []
        for idx, action in enumerate(self.actions):
            phi = self._action_feature(action)
            z = self._feature_vector(context, phi)
            mu, sigma = self._predict_reward(z)
            preds.append({"mu": mu, "sigma": sigma, "z": z, "action": action})
            ucb = mu + self.config.exploration_coef * sigma
            lcb = mu - self.config.safety_beta * sigma
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = idx
            if lcb >= self.config.safe_reward_floor and ucb > best_safe_ucb:
                best_safe_ucb = ucb
                best_safe_idx = idx
            features.append(z)
        chosen_idx = best_safe_idx if best_safe_idx is not None else best_idx
        info = preds[chosen_idx]
        est = self._physical_estimate(info["action"], ctx, sem_weight)
        self._last = {
            "feature": info["z"],
            "reward_mean": info["mu"],
            "ctx": dict(ctx),
            "sem_weight": float(sem_weight),
            "Q": float(Q_t),
            "J": float(J_t),
            "V": float(V),
            "action_idx": int(chosen_idx),
            "a": float(Q_t / max(ctx.get("Q_scale", 1.0), 1e-6)),
            "b": float(J_t / max(ctx.get("J_scale", 1.0), 1e-6)),
            "est": est,
        }
        return {
            "action": (float(info["action"][0]), float(info["action"][1]), int(info["action"][2]), "lrc_cucb"),
            **est,
        }

    def _lyapunov_reward(
        self,
        S_bits: float,
        E_hat: float,
        swwer: float,
        ctx: Dict[str, float],
        Q_t: float,
        J_t: float,
        V: float,
    ) -> float:
        slot_sec = float(ctx.get("slot_sec", 0.02))
        A_bits = float(ctx.get("A_bits", 600.0 * slot_sec))
        q_list = [a[0] for a in self.actions]
        p_list = [a[1] for a in self.actions]
        q_max = max(q_list)
        p_min, p_max = min(p_list), max(p_list)
        a = Q_t / self.config.context_scale_q
        b = J_t / self.config.context_scale_j
        r_raw = a * S_bits - b * E_hat - V * swwer
        r_min = -b * p_max - V * 1.0
        r_max = a * (q_max * slot_sec) - b * p_min
        if r_max - r_min < 1e-9:
            return 0.0
        return float(np.clip((r_raw - r_min) / (r_max - r_min), -1.0, 1.0))

    def _update(self, feature: NDArray[np.float64], reward: float) -> None:
        gamma = self.config.forget_factor
        self.theta *= gamma
        self.A_inv /= gamma
        gain = self.A_inv @ feature
        denom = 1.0 + float(feature.T @ gain)
        if denom <= 1e-9:
            return
        self.A_inv -= np.outer(gain, gain) / denom
        error = reward - float(feature @ self.theta)
        self.theta += gain * error
        self._reward_window.append(reward)
        if self._baseline_reward is None:
            self._baseline_reward = reward
        elif len(self._reward_window) == self._reward_window.maxlen:
            window_mean = float(np.mean(self._reward_window))
            if window_mean < self._baseline_reward - self.config.reset_threshold:
                logger.info("Soft reset triggered: reward mean %.3f -> %.3f", self._baseline_reward, window_mean)
                self.theta[:] = 0.0
                self.A_inv = np.eye(len(self.theta), dtype=np.float64)
                self._baseline_reward = window_mean
            else:
                self._baseline_reward = 0.8 * self._baseline_reward + 0.2 * window_mean

    def _process_ready(self) -> None:
        for item in self.buffer.advance():
            self._update(item.feature, item.reward)

    def observe(self, ctx: Dict[str, float]) -> None:
        if not self._last:
            return
        reward = self._lyapunov_reward(
            float(ctx.get("S_bits_obs", 0.0)),
            float(ctx.get("E_obs", 0.0)),
            float(ctx.get("swwer_obs", 0.0)),
            ctx,
            self._last.get("Q", 0.0),
            self._last.get("J", 0.0),
            self._last.get("V", 0.0),
        )
        delay = int(ctx.get("feedback_delay", 0))
        self.buffer.push(delay, self._last["feature"], reward)

    def step(
        self,
        ctx: Dict[str, float],
        action_space: Dict[str, Iterable[float]],
        Q_t: float,
        J_t: float,
        V: float,
        sem_weight: float,
    ) -> Dict[str, float]:
        self._process_ready()
        if "S_bits_obs" in ctx:
            self.observe(ctx)
            return {
                "action": (
                    float(self.actions[self._last.get("action_idx", 0)][0]),
                    float(self.actions[self._last.get("action_idx", 0)][1]),
                    int(self.actions[self._last.get("action_idx", 0)][2]),
                    "lrc_cucb",
                ),
                **self._last.get("est", {}),
            }
        return self._select_action(ctx, Q_t, J_t, V, sem_weight)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = LRCCUCBConfig()
    action_space = {"q_bps": [300, 600, 1200], "p_grid_w": [0.2, 0.5], "b_subbands": [1, 2]}
    algo = LRCCUCB(action_space, cfg)
    ctx = {"snr_db": 5.0, "slot_sec": 0.02, "A_bits": 600 * 0.02}
    res = algo.step(ctx, action_space, Q_t=500.0, J_t=20.0, V=50.0, sem_weight=0.8)
    logger.info("Selected action: %s", res["action"])
    ctx.update({"S_bits_obs": 20.0, "E_obs": 0.4, "swwer_obs": 0.1})
    algo.step(ctx, action_space, Q_t=500.0, J_t=20.0, V=50.0, sem_weight=0.8)
    logger.info("Update processed without errors")
