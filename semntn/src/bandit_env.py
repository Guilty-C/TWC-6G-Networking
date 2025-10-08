
import numpy as np
class StationaryBernoulliBandit:
    def __init__(self, means, rng):
        self.means=np.array(means, dtype=float); self.K=len(means); self.rng=rng
    def pull(self, a:int)->int:
        return 1 if self.rng.random()<self.means[a] else 0
    @property
    def best_mean(self): return float(self.means.max())
