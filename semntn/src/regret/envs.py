from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def map_to_unit_interval(value: float, lower: float, upper: float) -> float:
    """Map ``value`` linearly into the unit interval.

    The helper performs a robust min/max check and clips the result to ``[0, 1]`` to
    guarantee numerical safety.  It acts as the *single source of truth* for reward
    normalisation so that both the bandit algorithms and the pseudo-regret metric
    operate under an identical scale.
    """

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Reward bounds must be finite numbers.")
    if upper <= lower:
        return 0.0
    scaled = (float(value) - lower) / (upper - lower)
    return float(np.clip(scaled, 0.0, 1.0))


@dataclass
class BaseBanditEnv:
    """Common utilities shared by the simple bandit environments in this module."""

    reward_min: float
    reward_max: float

    def map_reward_to_unit_interval(self, reward: float) -> float:
        """Normalise a raw reward sample to ``[0, 1]`` using shared bounds."""

        return map_to_unit_interval(reward, self.reward_min, self.reward_max)

    # --- Interfaces required by the runner/metrics ---------------------------------
    def expected_reward(self, action: int) -> float:
        raise NotImplementedError

    def expected_reward_unit_interval(self, action: int) -> float:
        r"""Return :math:`\mu_{true}(a)` under the same normalisation as the learner."""

        return self.map_reward_to_unit_interval(self.expected_reward(action))

    def enumerate_action_space(self) -> List[int]:
        return list(range(self.K))


class BernoulliBandit(BaseBanditEnv):
    def __init__(self, means: Iterable[float], seed: int | None = None):
        self.means = np.array(list(means), dtype=float)
        if self.means.ndim != 1 or self.means.size == 0:
            raise ValueError("'means' must be a non-empty 1-D array.")
        self.K = int(self.means.size)
        self.rng = np.random.default_rng(seed)
        super().__init__(reward_min=0.0, reward_max=1.0)

    def pull(self, a: int) -> float:
        return float(self.rng.random() < self.means[a])

    def expected_reward(self, action: int) -> float:
        return float(self.means[action])


class GaussianBandit(BaseBanditEnv):
    def __init__(self, mus: Iterable[float], sigma: float = 0.1, seed: int | None = None):
        self.mus = np.array(list(mus), dtype=float)
        if self.mus.ndim != 1 or self.mus.size == 0:
            raise ValueError("'mus' must be a non-empty 1-D array.")
        self.sigma = float(sigma)
        self.K = int(self.mus.size)
        self.rng = np.random.default_rng(seed)

        span = np.max(self.mus) - np.min(self.mus)
        margin = max(1.0, 5.0 * self.sigma)
        reward_min = float(np.min(self.mus) - margin)
        reward_max = float(np.max(self.mus) + margin + max(0.0, span))
        super().__init__(reward_min=reward_min, reward_max=reward_max)

    def pull(self, a: int) -> float:
        return float(self.rng.normal(self.mus[a], self.sigma))

    def expected_reward(self, action: int) -> float:
        return float(self.mus[action])


def gaps_from_means(means: Iterable[float]) -> List[float]:
    arr = np.asarray(list(means), dtype=float)
    m = float(np.max(arr))
    return [m - float(x) for x in arr]
