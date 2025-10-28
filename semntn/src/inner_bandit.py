"""Inner bandit algorithms with RA-UCB reject filter.

Implements UCB1, UCB-V, SW-UCB, LinUCB. Public API:
- choose_arm(context: dict) -> int
- update(reward: float, arm_id: int, context: dict | None = None) -> None

Notes:
- Reward should be in [0,1] for stable UCB scaling (we clip).
- RA-UCB filter rejects arms whose LCB < lcb_min before selection.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class InnerBandit:
    def __init__(
        self,
        K: int,
        mode: str = "sw_ucb",
        alpha: float = 1.0,
        window: int = 250,
        lcb_min: float = 0.0,
        eps_rand: float = 0.0,
        linucb_dim: Optional[int] = None,
    ) -> None:
        mode = str(mode).lower()
        if mode not in {"ucb1", "ucbv", "sw_ucb", "linucb"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.K = int(K)
        self.mode = mode
        self.alpha = float(alpha)
        self.window = int(window)
        self.lcb_min = float(np.clip(lcb_min, 0.0, 1.0))
        self.eps = float(np.clip(eps_rand, 0.0, 1.0))

        self.t = 0
        self.n = np.zeros(self.K, dtype=int)
        self.s = np.zeros(self.K, dtype=float)
        self.sq = np.zeros(self.K, dtype=float)

        self.hist: Optional[List[deque]] = None
        if self.window > 0:
            self.hist = [deque(maxlen=self.window) for _ in range(self.K)]

        # LinUCB params
        self.lin_d = int(linucb_dim or 8)
        self.A = np.eye(self.lin_d, dtype=float)
        self.b = np.zeros(self.lin_d, dtype=float)
        self.theta: Optional[np.ndarray] = None

    # ------------ Utilities ------------
    def _mean_var(self, a: int) -> Tuple[float, float]:
        if self.window > 0 and self.hist is not None and len(self.hist[a]) > 0:
            arr = np.asarray(self.hist[a], dtype=float)
            mu = float(arr.mean())
            var = float(arr.var(ddof=0))
            return mu, var
        # full history
        if self.n[a] == 0:
            return 0.0, 1.0
        mu = float(self.s[a] / max(self.n[a], 1))
        var = float(max(self.sq[a] / max(self.n[a], 1) - mu * mu, 0.0))
        return mu, var

    def _ucb_bonus(self, a: int) -> float:
        # standard exploration bonus
        if self.n[a] == 0:
            return 1.0
        if self.mode == "ucbv":
            mu, var = self._mean_var(a)
            # Audibert UCB-V style bonus
            b = np.sqrt(2.0 * var * np.log(max(self.t, 2)) / self.n[a]) + 3.0 * np.log(max(self.t, 2)) / self.n[a]
        elif self.mode == "sw_ucb" and self.window > 0:
            n_eff = len(self.hist[a]) if self.hist is not None else self.n[a]
            b = np.sqrt(2.0 * np.log(max(self.t, 2)) / max(n_eff, 1))
        else:
            b = np.sqrt(2.0 * np.log(max(self.t, 2)) / self.n[a])
        return float(self.alpha * b)

    def _feature(self, context: Dict[str, float], a: int) -> np.ndarray:
        # Simple feature: concatenation of normalized context values and arm one-hot
        vals = [float(v) for v in context.values()]
        x = np.asarray(vals, dtype=float)
        if x.size == 0:
            x = np.zeros(1, dtype=float)
        # normalize to [0,1] robustly
        if x.size > 0:
            lo = np.min(x); hi = np.max(x)
            x = (x - lo) / (hi - lo + 1e-6)
        arm = np.zeros(self.K, dtype=float)
        arm[a] = 1.0
        phi = np.concatenate([x, arm], axis=0)
        # pad/truncate to lin_d
        if phi.size < self.lin_d:
            phi = np.pad(phi, (0, self.lin_d - phi.size), mode="constant")
        else:
            phi = phi[: self.lin_d]
        return phi.astype(float)

    def _linucb_score(self, phi: np.ndarray) -> float:
        A_inv = np.linalg.pinv(self.A)
        theta = A_inv @ self.b
        p = float(theta @ phi + self.alpha * np.sqrt(phi @ A_inv @ phi))
        return p

    # ------------ Public API ------------
    def choose_arm(self, context: Dict[str, float]) -> int:
        self.t += 1
        # epsilon-random for robustness
        if self.eps > 0 and np.random.random() < self.eps:
            return int(np.random.randint(0, self.K))

        # compute indices & RA filter
        indices = np.zeros(self.K, dtype=float)
        lcb = np.zeros(self.K, dtype=float)
        candidates: List[int] = []
        for a in range(self.K):
            if self.mode == "linucb":
                phi = self._feature(context, a)
                idx = self._linucb_score(phi)
                mu, _ = self._mean_var(a)
                bonus = self._ucb_bonus(a)
                indices[a] = idx
                lcb[a] = mu - bonus
            else:
                mu, _ = self._mean_var(a)
                bonus = self._ucb_bonus(a)
                indices[a] = mu + bonus
                lcb[a] = mu - bonus
            if lcb[a] >= self.lcb_min:
                candidates.append(a)

        if len(candidates) == 0:
            # fall back: no RA-safe arm, pick best index among all
            return int(np.argmax(indices))

        # pick among RA-safe arms
        safe_indices = [(indices[a], a) for a in candidates]
        safe_indices.sort(key=lambda x: x[0], reverse=True)
        return int(safe_indices[0][1])

    def update(self, reward: float, arm_id: int, context: Optional[Dict[str, float]] = None) -> None:
        # Optional Lagrangian reward shaping
        r = float(np.clip(reward, 0.0, 1.0))
        if context is not None and bool(context.get("reward_shaping", False)):
            r_mos = float(np.clip(context.get("r_mos", r), 0.0, 1.0))
            E_slot_J = float(context.get("E_slot_J", 0.0))
            E_bar = float(context.get("E_bar", 0.0))
            swer = float(np.clip(context.get("swer", 0.0), 0.0, 1.0))
            I_target = float(context.get("I_target", 0.0))
            q_obs = float(context.get("q_bps_obs", 0.0))
            q_budget = float(context.get("latency_budget_q_bps", 0.0))
            slack_eng = float(context.get("slack_eng", max(E_slot_J - E_bar, 0.0)))
            slack_sem = float(context.get("slack_sem", max(swer - I_target, 0.0)))
            slack_lat = float(context.get("slack_lat", max(q_obs - q_budget, 0.0)))
            # Lagrange multipliers: lambda_k = c * scale_k (c escalates on recent violation)
            scale_Q = float(context.get("scale_Q", 1.0))
            scale_J = float(context.get("scale_J", 1.0))
            scale_S = float(context.get("scale_S", 1.0))
            violated_recent = (slack_eng > 0.0) or (slack_sem > 0.0) or (slack_lat > 0.0)
            c = 2.0 if violated_recent else 1.0
            lam_eng = c * scale_J
            lam_sem = c * scale_S
            lam_lat = c * scale_Q
            r = float(np.clip(r_mos - lam_eng * slack_eng - lam_sem * slack_sem - lam_lat * slack_lat, 0.0, 1.0))
        a = int(arm_id)
        self.n[a] += 1
        self.s[a] += r
        self.sq[a] += r * r
        if self.hist is not None:
            self.hist[a].append(r)
        if self.mode == "linucb" and context is not None:
            phi = self._feature(context, a)
            self.A += np.outer(phi, phi)
            self.b += phi * r