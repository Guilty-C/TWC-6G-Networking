import numpy as np
def per_from_snr_db(snr_db, steep=1.2, mid_db=1.0):
    # SNR高 -> PER低；steep越大过渡越陡
    x = (snr_db - mid_db)
    per = 1.0/(1.0 + np.exp(steep * x))
    return float(np.clip(per, 0.0, 1.0))
def wer_from_per(per, sem_weight, base_alpha=1.2):
    wer = base_alpha * per / max(sem_weight, 1e-6)
    return float(np.clip(wer, 0.0, 1.0))
