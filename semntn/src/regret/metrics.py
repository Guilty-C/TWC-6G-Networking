import math
import numpy as np


class RegretMeter:
    """Tracks expected cumulative regret using known optimal mean."""
    def __init__(self, opt_mean: float):
        self.opt = float(opt_mean)
        self.R = 0.0

    def update(self, mu_a: float) -> float:
        self.R += (self.opt - float(mu_a))
        return self.R


def order_check(regrets, gaps, accept_cfg):
    """
    regrets: list[float] of R(t)
    gaps: list[float] of Δ_k = μ* - μ_k (Δ_k>0 for suboptimal arms)
    accept_cfg:
      slope_logT_range: [lo, hi]
      cstar_max: float
    """
    y = np.asarray(regrets, dtype=float)
    T = y.size
    x = np.log(np.arange(1, T + 1, dtype=float))

    # linear fit: y ~ k * log T + b
    k = float(np.polyfit(x, y, deg=1)[0])

    # C* estimate: max_t R(t) / sum_k (ln t / Δ_k)
    eps = 1e-9
    denom = np.zeros(T, dtype=float)
    for g in gaps:
        if g > 0:
            denom += np.log(np.arange(1, T + 1, dtype=float) + eps) / max(eps, g)
    denom = np.maximum(denom, eps)
    cstar = float(np.max(y / denom))

    lo, hi = accept_cfg.get("slope_logT_range", [0.5, 5.0])
    cmax = float(accept_cfg.get("cstar_max", 12.0))
    ok = (lo <= k <= hi) and (cstar <= cmax)

    return {"slope": k, "C*": cstar, "pass": bool(ok)}
