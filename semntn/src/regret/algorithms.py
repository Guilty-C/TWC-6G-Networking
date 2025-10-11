import math
import numpy as np


class UCB1:
    def __init__(self, K, alpha=2.0):
        self.K = int(K)
        self.alpha = float(alpha)
        self.n = np.zeros(self.K, dtype=int)
        self.mu = np.zeros(self.K, dtype=float)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        logt = math.log(max(2, self.t))
        bonus = np.array([
            self.alpha * math.sqrt(2.0 * logt / max(1, self.n[i]))
            for i in range(self.K)
        ], dtype=float)
        idx = self.mu + bonus
        return int(np.argmax(idx))

    def update(self, a: int, r: float):
        self.n[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n[a]


class EpsilonGreedy:
    def __init__(self, K, epsilon=0.05):
        self.K = int(K)
        self.eps = float(epsilon)
        self.n = np.zeros(self.K, dtype=int)
        self.mu = np.zeros(self.K, dtype=float)
        self.rng = np.random.default_rng(0)

    def select(self) -> int:
        if self.rng.random() < self.eps or self.n.sum() < self.K:
            return int(self.rng.integers(self.K))
        return int(np.argmax(self.mu))

    def update(self, a: int, r: float):
        self.n[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n[a]


class LinearPredictor:
    """Ultra-light online 'context-free' predictor: arm-wise bias."""
    def __init__(self, K, lr=0.05):
        self.K = int(K)
        self.bias = np.zeros(self.K, dtype=float)
        self.lr = float(lr)

    def predict(self):
        return self.bias.copy()

    def update(self, a: int, r: float):
        err = r - self.bias[a]
        self.bias[a] += self.lr * err


class SemUCB:
    """
    AI-assisted UCB:
      index_i(t) = (1-λ_t)*mu_hat_i + λ_t*mu_ai_i + alpha*sqrt(2 ln t / n_i)
    with λ_t decaying to 0 (log or 1/sqrt), so the regret order remains O(log T).
    """
    def __init__(self, K, alpha=2.0, lambda0=0.5, c_decay=1.0, decay="log",
                 predictor="linear", lr=0.05):
        self.K = int(K)
        self.alpha = float(alpha)
        self.lambda0 = float(lambda0)
        self.c = float(c_decay)
        self.decay = str(decay)
        self.t = 0

        self.ucb = UCB1(K, alpha=alpha)
        self.pred = None if predictor == "none" else LinearPredictor(K, lr=lr)

    def _lambda_t(self) -> float:
        self.t += 1
        if self.decay == "log":
            return min(self.lambda0, self.c / max(1.0, math.log(self.t + 1)))
        return min(self.lambda0, self.c / math.sqrt(max(1.0, self.t)))

    def select(self) -> int:
        lam = self._lambda_t()
        # compute bonus using internal UCB counters (do NOT double-step t)
        logt = math.log(max(2, self.ucb.t + 1))
        bonus = np.array([
            self.ucb.alpha * math.sqrt(2.0 * logt / max(1, self.ucb.n[i]))
            for i in range(self.K)
        ], dtype=float)
        mu_hat = self.ucb.mu
        if self.pred is None:
            self.ucb.t += 1
            idx = mu_hat + bonus
            return int(np.argmax(idx))
        mu_ai = self.pred.predict()
        idx = (1.0 - lam) * mu_hat + lam * mu_ai + bonus
        self.ucb.t += 1
        return int(np.argmax(idx))

    def update(self, a: int, r: float):
        self.ucb.update(a, r)
        if self.pred is not None:
            self.pred.update(a, r)
