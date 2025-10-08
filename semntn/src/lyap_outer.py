import numpy as np
from sem_weight_model import SemWeightModel
from utils_seed import set_all_seeds

def _get_inner(use_mock_inner: bool):
    if use_mock_inner:
        import inner_api_mock as inner
    else:
        import inner_api_ucb as inner
    return inner

def _rolling_features(snr_db_seg, win):
    s = np.asarray(snr_db_seg, dtype=float)
    K = min(win, len(s)); head = s[:K]
    energy = np.maximum(head, 0.0); energy_mean = float(energy.mean())
    diff = np.diff(head); sign = np.sign(diff)
    zcr = float((np.abs(np.diff(sign))>0).sum())/max(len(sign),1)
    pos = np.maximum(head, 0.0); idx = np.arange(1, len(pos)+1, dtype=float)
    spec_centroid = float((pos*idx).sum()/(pos.sum()+1e-6))
    snr_mean = float(head.mean())
    keyword_flag = 1.0 if np.max(head) > 8.0 else 0.0
    return np.array([energy_mean, zcr, spec_centroid, snr_mean, keyword_flag], dtype=float)
def run_episode(cfg: dict, snr_db: np.ndarray, V: int, use_mock_inner: bool=True):
    seed = int(cfg.get('seed', 2025))
    set_all_seeds(seed)
    T = int(min(cfg.get('T', len(snr_db)), len(snr_db)))
    K = int(cfg.get('K_head_pkts', 5))
    slot_sec = float(cfg.get('slot_sec', 0.02))

    # 关键：外源到达（与动作无关）
    A_bps = float(cfg.get('arrivals', {}).get('A_bps', 600.0))
    A_bits = A_bps * slot_sec

    # 能量预算（每隙）
    E_bar = float(cfg.get('queues_budget', {}).get('E_per_slot', 0.5))

    # 尺度归一
    Q_scale = float(cfg.get('lyap', {}).get('Q_scale', 1e4))
    J_scale = float(cfg.get('lyap', {}).get('J_scale', 1e2))

    action_space = cfg.get('actions', {})
    model = SemWeightModel(w_min=cfg['sem_weight']['w_min'], w_max=cfg['sem_weight']['w_max'])

    Q = 0.0; J = 0.0
    swwer_hist = []; Q_hist = []; J_hist = []

    inner = _get_inner(use_mock_inner)

    for t in range(T):
        left = max(0, t-K+1)
        feat = _rolling_features(snr_db[left:t+1], K)
        w_sem = model.infer_w_sem(feat)

        ctx = {
            'snr_db': float(snr_db[t]),
            'E_bar': E_bar,
            'slot_sec': slot_sec,
            'A_bits': A_bits
        }

        # ---- 先调用内层得到估计值 ----
        pick = inner.pick_action_and_estimate(ctx, action_space, Q, J, V, w_sem,
                                              Q_scale=Q_scale, J_scale=J_scale)

        # ---- 得到真实观测 ----
        S_bits = float(pick['S_bits'])
        E_hat  = float(pick['E_hat'])
        swwer  = float(pick['SWWER_hat'])

        # ---- 把真实观测写回 ctx，供 UCB update 使用 ----
        ctx['S_bits_obs'] = S_bits
        ctx['E_obs'] = E_hat
        ctx['swwer_obs'] = swwer
        ctx['per_obs'] = float(pick.get('per', 0.0))  # 如果有就传

        # ⚠️ 再次调用 inner，让它用真实观测 update()
        #   （pick_action_and_estimate 内部会优先用 *_obs 更新）
        inner.pick_action_and_estimate(ctx, action_space, Q, J, V, w_sem,
                                       Q_scale=Q_scale, J_scale=J_scale)

        # ---- 队列更新 ----
        Q = max(Q - S_bits, 0.0) + A_bits
        J = max(J + E_hat - E_bar, 0.0)

        swwer_hist.append(swwer); Q_hist.append(Q); J_hist.append(J)

    return {
        'V': int(V),
        'SWWER_mean': float(np.mean(swwer_hist)),
        'Q_mean': float(np.mean(Q_hist)),
        'J_mean': float(np.mean(J_hist))
    }
