from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


class BanditPolicy:
    """Interface shared by all baseline policies."""

    def select_action(self, t: int) -> int:
        raise NotImplementedError

    def update(self, action: int, reward01: float) -> None:
        raise NotImplementedError


@dataclass
class UCB1Policy(BanditPolicy):
    """UCB1 policy operating on rewards already mapped to ``[0, 1]``."""

    num_actions: int
    alpha: float = 2.0

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.counts = np.zeros(self.num_actions, dtype=int)
        self.estimates = np.zeros(self.num_actions, dtype=float)

    def select_action(self, t: int) -> int:
        if t <= self.num_actions:
            return t - 1
        logt = math.log(max(2, t))
        bonus = np.array(
            [
                self.alpha * math.sqrt(2.0 * logt / max(1, self.counts[i]))
                for i in range(self.num_actions)
            ],
            dtype=float,
        )
        idx = self.estimates + bonus
        return int(np.argmax(idx))

    def update(self, action: int, reward01: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.estimates[action] += (reward01 - self.estimates[action]) / n


@dataclass
class EpsilonGreedyPolicy(BanditPolicy):
    num_actions: int
    epsilon: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.estimates = np.zeros(self.num_actions, dtype=float)
        self.counts = np.zeros(self.num_actions, dtype=int)
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, t: int) -> int:
        explore = self.rng.random() < self.epsilon
        if explore or t <= self.num_actions:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.estimates))

    def update(self, action: int, reward01: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.estimates[action] += (reward01 - self.estimates[action]) / n


@dataclass
class RandomPolicy(BanditPolicy):
    num_actions: int
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, t: int) -> int:  # noqa: ARG002 (t unused)
        return int(self.rng.integers(self.num_actions))

    def update(self, action: int, reward01: float) -> None:  # noqa: D401, ARG002
        """Random policy ignores feedback."""


@dataclass
class OraclePolicy(BanditPolicy):
    mu_true: Dict[int, float]

    def __post_init__(self) -> None:
        if not self.mu_true:
            raise ValueError("mu_true dictionary must not be empty")
        self.best_action = int(max(self.mu_true, key=self.mu_true.__getitem__))

    def select_action(self, t: int) -> int:  # noqa: ARG002
        return self.best_action

    def update(self, action: int, reward01: float) -> None:  # noqa: D401, ARG002
        """Oracle policy observes rewards but does not adapt."""


# ---------------------------------------------------------------------------
# Legacy helpers retained for compatibility with older experiments.


class UCB1(UCB1Policy):
    def __init__(self, K: int, alpha: float = 2.0):
        super().__init__(num_actions=int(K), alpha=float(alpha))
        self._t = 0

    def select(self) -> int:
        self._t += 1
        return self.select_action(self._t)

    def update(self, a: int, r: float) -> None:
        super().update(a, float(r))


class EpsilonGreedy(EpsilonGreedyPolicy):
    def __init__(self, K: int, epsilon: float = 0.05, seed: Optional[int] = None):
        super().__init__(num_actions=int(K), epsilon=float(epsilon), seed=seed)

    def select(self) -> int:
        return self.select_action(self.counts.sum() + 1)

