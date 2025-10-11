import numpy as np


class BernoulliBandit:
    def __init__(self, means, seed=None):
        self.means = np.array(means, dtype=float)
        self.K = len(self.means)
        self.rng = np.random.default_rng(seed)
        self.opt = float(self.means.max())

    def pull(self, a: int) -> float:
        return float(self.rng.random() < self.means[a])


class GaussianBandit:
    def __init__(self, mus, sigma=0.1, seed=None):
        self.mus = np.array(mus, dtype=float)
        self.sigma = float(sigma)
        self.K = len(self.mus)
        self.rng = np.random.default_rng(seed)
        self.opt = float(self.mus.max())

    def pull(self, a: int) -> float:
        return float(self.rng.normal(self.mus[a], self.sigma))


def gaps_from_means(means):
    m = float(np.max(means))
    return [m - float(x) for x in means]
