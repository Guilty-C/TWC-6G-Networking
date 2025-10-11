from numpy.random import SeedSequence, default_rng
import numpy as np

def set_all_seeds(seed: int = 2025):
    ss = SeedSequence(seed)
    rng = default_rng(ss.generate_state(1)[0])
    np.random.seed(seed)
    return rng
