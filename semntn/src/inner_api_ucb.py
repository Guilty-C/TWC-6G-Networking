"""UCB-based inner controller with sliding-window and discounted rewards."""
from __future__ import annotations

import math
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from wer_proxy import per_from_snr_db, wer_from_per


@dataclass
class _Config:
    window: Optional[int] = 64
    gamma: float = 0.98
    cache_size: int = 512
    snr_quant: float = 0.25
    weight_quant: float = 0.05


@dataclass
class _Arm:
    action: Tuple[float, float, int]
    pulls: int = 0
    reward_sum: float = 0.0
    window: Deque[float] = field(default_factory=deque)
    discounted_sum: float = 0.0
    discounted_weight: float = 0.0

    def register(self, reward: float, cfg: _Config) -> None:
        reward = max(0.0, min(1.0, reward))
        self.pulls += 1
        self.reward_sum += reward
        if cfg.window:
            self.window.append(reward)
            while len(self.window) > cfg.window:
                self.window.popleft()
        if 0.0 < cfg.gamma < 1.0:
            self.discounted_sum = self.discounted_sum * cfg.gamma + reward
            self.discounted_weight = self.discounted_weight * cfg.gamma + 1.0
        else:
            self.discounted_sum += reward
            self.discounted_weight += 1.0

    def effective_mean(self, cfg: _Config) -> float:
        if cfg.window and self.window:
            return sum(self.window) / len(self.window)
        if 0.0 < cfg.gamma < 1.0 and self.discounted_weight > 0.0:
            return self.discounted_sum / self.discounted_weight
        if self.pulls:
            return self.reward_sum / self.pulls
        return 0.0

    def effective_count(self, cfg: _Config) -> float:
        if cfg.window and self.window:
            return float(len(self.window))
        if 0.0 < cfg.gamma < 1.0 and self.discounted_weight > 0.0:
            return max(self.discounted_weight, 1e-6)
        return float(max(self.pulls, 1))


class _UCBManager:
    def __init__(self, actions: Iterable[Tuple[float, float, int]], cfg: _Config):
        self.cfg = cfg
        self.arms: List[_Arm] = []
        for action in actions:
            q, p, b = action
            self.arms.append(_Arm((float(q), float(p), int(b))))
        self.t = 0

    def select(self) -> Tuple[int, Tuple[float, float, int]]:
        if not self.arms:
            raise RuntimeError("No actions available")
        self.t += 1
        for idx, arm in enumerate(self.arms):
            if arm.pulls == 0:
                return idx, arm.action
        best_idx = 0
        best_value = -1.0
        log_term = max(math.log(max(self.t, 2)), 1e-9)
        for idx, arm in enumerate(self.arms):
            mean = arm.effective_mean(self.cfg)
            count = arm.effective_count(self.cfg)
            bonus = math.sqrt(2.0 * log_term / max(count, 1e-6))
            value = mean + bonus
            if value > best_value:
                best_value = value
                best_idx = idx
        return best_idx, self.arms[best_idx].action

    def update(self, idx: int, reward: float) -> None:
        self.arms[idx].register(reward, self.cfg)


class _ActionCache:
    def __init__(self, capacity: int):
        self.capacity = max(int(capacity), 8)
        self._store: "OrderedDict[Tuple[int, float, float], Dict[str, float]]" = OrderedDict()

    def get(self, key: Tuple[int, float, float]) -> Optional[Dict[str, float]]:
        res = self._store.get(key)
        if res is not None:
            self._store.move_to_end(key)
        return res

    def put(self, key: Tuple[int, float, float], value: Dict[str, float]) -> None:
        if key in self._store:
            self._store[key] = value
            self._store.move_to_end(key)
            return
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


def _quantize(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return round(float(value) / step) * step


def _rate_penalty_db(q_bps: float, q_ref: float = 300.0, c_db: float = 3.5) -> float:
    q_bps = max(float(q_bps), 1e-9)
    return c_db * math.log2(q_bps / q_ref + 1.0)


def _enumerate_actions(action_space: Dict[str, Iterable[float]]) -> List[Tuple[float, float, int]]:
    tuples = action_space.get("tuples")
    actions: List[Tuple[float, float, int]]
    if tuples:
        actions = [ (float(q), float(p), int(b)) for q, p, b in tuples ]
    else:
        q_list = [float(x) for x in action_space.get("q_bps", [300, 600, 1200, 2400])]
        p_list = [float(x) for x in action_space.get("p_grid_w", [0.2, 0.4, 0.8, 1.2, 1.6])]
        b_list = [int(x) for x in action_space.get("b_subbands", [1, 2, 3])]
        actions = []
        for q in q_list:
            for p in p_list:
                for b in b_list:
                    actions.append((q, p, b))
    return _filter_dominated(actions)


def _filter_dominated(actions: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    filtered: List[Tuple[float, float, int]] = []
    for idx, candidate in enumerate(actions):
        dominated = False
        q_c, p_c, b_c = candidate
        for jdx, other in enumerate(actions):
            if idx == jdx:
                continue
            q_o, p_o, b_o = other
            if q_o >= q_c and p_o <= p_c and b_o >= b_c:
                if (q_o > q_c) or (p_o < p_c) or (b_o > b_c):
                    dominated = True
                    break
        if not dominated:
            filtered.append(candidate)
    return filtered


def _estimate_action(
    action: Tuple[float, float, int],
    ctx: Dict[str, float],
    sem_weight: float,
) -> Dict[str, float]:
    q, p, b = action
    snr_db = float(ctx.get("snr_db", 0.0))
    slot_sec = float(ctx.get("slot_sec", 0.02))
    snr_eff_db = snr_db + 10.0 * math.log10(max(p * b, 1e-6)) - _rate_penalty_db(q)
    per = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
    swwer = wer_from_per(per, sem_weight)
    S_bits = q * (1.0 - per) * slot_sec
    E_hat = p
    return {
        "action": (float(q), float(p), int(b), "ucb"),
        "S_bits": float(S_bits),
        "E_hat": float(E_hat),
        "SWWER_hat": float(swwer),
        "per": float(per),
        "snr_eff_db": float(snr_eff_db),
    }


_AGENT: Optional[_UCBManager] = None
_CACHE: Optional[_ActionCache] = None
_CONFIG = _Config()
_LAST_PICK: Optional[Dict[str, object]] = None


def _load_config(ctx: Dict[str, float]) -> _Config:
    user_cfg = ctx.get("ucb", {})
    cfg = _Config()
    if isinstance(user_cfg, dict):
        if "window" in user_cfg:
            window = int(user_cfg["window"])
            cfg.window = window if window > 0 else None
        if "gamma" in user_cfg:
            cfg.gamma = float(user_cfg["gamma"])
        if "cache_size" in user_cfg:
            cfg.cache_size = max(int(user_cfg["cache_size"]), 8)
        if "snr_quant" in user_cfg:
            cfg.snr_quant = max(float(user_cfg["snr_quant"]), 0.0)
        if "weight_quant" in user_cfg:
            cfg.weight_quant = max(float(user_cfg["weight_quant"]), 0.0)
    env_window = os.getenv("INNER_UCB_WINDOW")
    if env_window:
        try:
            val = int(env_window)
            cfg.window = val if val > 0 else None
        except ValueError:
            pass
    env_gamma = os.getenv("INNER_UCB_GAMMA")
    if env_gamma:
        try:
            cfg.gamma = float(env_gamma)
        except ValueError:
            pass
    env_cache = os.getenv("INNER_UCB_CACHE")
    if env_cache:
        try:
            cfg.cache_size = max(int(env_cache), 8)
        except ValueError:
            pass
    return cfg


def _ensure_agent(action_space: Dict[str, Iterable[float]], ctx: Dict[str, float]) -> None:
    global _AGENT, _CACHE, _CONFIG
    if _AGENT is None:
        _CONFIG = _load_config(ctx)
        actions = _enumerate_actions(action_space)
        _AGENT = _UCBManager(actions, _CONFIG)
        _CACHE = _ActionCache(_CONFIG.cache_size)


def _reward_from_obs(ctx: Dict[str, float]) -> float:
    swwer = float(ctx.get("swwer_obs", ctx.get("SWWER_hat", 1.0)))
    base = 1.0 - max(min(swwer, 1.0), 0.0)
    meta = _LAST_PICK.get("meta", {}) if _LAST_PICK else {}
    q_press = 0.0
    if meta:
        Q_scale = max(float(meta.get("Q_scale", 1.0)), 1e-6)
        J_scale = max(float(meta.get("J_scale", 1.0)), 1e-6)
        q_press = min(float(meta.get("Q", 0.0)) / Q_scale, 1.0)
        j_press = min(float(meta.get("J", 0.0)) / J_scale, 1.0)
        E_bar = float(meta.get("E_bar", 0.0))
    else:
        j_press = 0.0
        E_bar = float(ctx.get("E_bar", 0.0))
    service = float(ctx.get("S_bits_obs", 0.0))
    A_bits = float(meta.get("A_bits", service + 1e-6)) if meta else service + 1e-6
    service_ratio = min(service / max(A_bits, 1e-6), 1.0)
    energy_excess = max(float(ctx.get("E_obs", E_bar)) - E_bar, 0.0)
    reward = base - 0.3 * q_press - 0.2 * j_press - 0.1 * min(energy_excess, 1.0) + 0.2 * service_ratio
    v_level = float(meta.get("V", 0.0)) if meta else 0.0
    v_weight = min(v_level / (v_level + 50.0), 1.0)
    reward -= 0.4 * v_weight * (1.0 - base)
    return max(0.0, min(1.0, reward))


def pick_action_and_estimate(
    ctx: Dict[str, float],
    action_space: Dict[str, Iterable[float]],
    Q_t: float,
    J_t: float,
    V: float,
    sem_weight: float,
    Q_scale: float = 1e4,
    J_scale: float = 1e2,
) -> Dict[str, float]:
    """Select an action and provide approximate metrics."""

    global _LAST_PICK
    ctx = dict(ctx)
    ctx.setdefault("Q_scale", Q_scale)
    ctx.setdefault("J_scale", J_scale)
    _ensure_agent(action_space, ctx)

    if _LAST_PICK and "S_bits_obs" in ctx:
        reward = _reward_from_obs(ctx)
        idx = int(_LAST_PICK["idx"])
        _AGENT.update(idx, reward)
        cached_return = dict(_LAST_PICK["return"])
        if "per_obs" in ctx:
            cached_return["per"] = float(ctx["per_obs"])
        return cached_return

    idx, action = _AGENT.select()
    cache_key = None
    if _CACHE is not None:
        snr_q = _quantize(float(ctx.get("snr_db", 0.0)), _CONFIG.snr_quant)
        weight_q = _quantize(float(sem_weight), _CONFIG.weight_quant)
        cache_key = (idx, snr_q, weight_q)
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _LAST_PICK = {"idx": idx, "return": cached}
            return cached

    estimate = _estimate_action(action, ctx, sem_weight)
    if cache_key is not None and _CACHE is not None:
        _CACHE.put(cache_key, estimate)
    _LAST_PICK = {"idx": idx, "return": estimate, "meta": {
        "Q": float(Q_t),
        "J": float(J_t),
        "V": float(V),
        "E_bar": float(ctx.get("E_bar", 0.0)),
        "Q_scale": float(ctx.get("Q_scale", 1.0)),
        "J_scale": float(ctx.get("J_scale", 1.0)),
        "A_bits": float(ctx.get("A_bits", 0.0)),
    }}
    return estimate



def reset_state() -> None:
    """Reset cached agent state (useful for tests)."""

    global _AGENT, _CACHE, _LAST_PICK, _CONFIG
    _AGENT = None
    _CACHE = None
    _LAST_PICK = None
    _CONFIG = _Config()


__all__ = ["pick_action_and_estimate", "reset_state"]
