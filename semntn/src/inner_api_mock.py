import numpy as np
from wer_proxy import per_from_snr_db, wer_from_per

def _rate_penalty_db(q_bps, q_ref=300.0, c_db=3.0):
    """
    速率越高，需要的SNR门限越高；用一个光滑penalty近似：
    penalty ≈ c_db * log2(q/q_ref + 1)
    q_bps: 300,600,1200,2400...
    """
    return float(c_db * np.log2(q_bps / q_ref + 1.0))

def pick_action_and_estimate(ctx, action_space, Q_t, J_t, V, sem_weight,
                             Q_scale=20000, J_scale=500):
    q_list = action_space.get('q_bps', [300,600,1200,2400])
    p_list = action_space.get('p_grid_w', [0.2,0.4,0.8,1.2,1.6])
    b_list = action_space.get('b_subbands', [1,2,3])

    snr_db   = float(ctx.get('snr_db', 0.0))
    E_bar    = float(ctx.get('E_bar', 0.5))
    slot_sec = float(ctx.get('slot_sec', 0.02))
    A_bits   = float(ctx.get('A_bits', 600.0 * slot_sec))  # 外层传入的“到达量/每隙(比特)”

    best = None
    best_cost = 1e18
    for q in q_list:
        for p in p_list:
            for b in b_list:
                # 等效SNR：功率/子带提升  -  速率惩罚
                snr_eff_db = snr_db + 10.0*np.log10(max(p*b, 1e-6)) - _rate_penalty_db(q)
                per   = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
                swwer = wer_from_per(per, sem_weight)

                # 服务量 = 物理毛速率 * (1-PER) * 时隙
                S_bits = float(q * (1.0 - per) * slot_sec)

                # 能耗近似：仅与发射功率相关（不乘b，保持能耗尺度温和）
                E_hat = float(p)

                # 归一化DPP目标（与外层一致）
                drift_term = (Q_t/Q_scale) * (A_bits - S_bits) + (J_t/J_scale) * (E_hat - E_bar)
                cost = drift_term + V * swwer

                if cost < best_cost:
                    best_cost = cost
                    best = {
                        'action': (q, p, b, 'fixed'),
                        'S_bits': S_bits,
                        'E_hat': E_hat,
                        'SWWER_hat': swwer,
                        'snr_eff_db': float(snr_eff_db),
                        'per': float(per),
                        'A_bits': A_bits
                    }
    return best
