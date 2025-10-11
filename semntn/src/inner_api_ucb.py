import numpy as np
from wer_proxy import per_from_snr_db, wer_from_per

# ---- 全局：单 episode 内有效 ----
_AGENT = None

# 归一化：EWMA 均值/方差（避免 min-max 压平 V 效果）
_NORM_INIT = False
_NORM_MU = 0.0
_NORM_SIG = 1.0
_NORM_ALPHA = 0.01  # EWMA 更新率；越小越稳

# ================================
# UCB 管理器
# ================================
class _UCBArms:
    def __init__(self, actions):
        self.actions = list(actions)
        K = len(self.actions)
        self.n = np.zeros(K, dtype=int)     # 拉动次数
        self.s = np.zeros(K, dtype=float)   # 指数移动平均奖励（已在[0,1]）
        self.t = 0
        self.alpha = 0.1                    # EMA 学习率
        self.eps_rand = 0.10                # 10% 随机探索

    def select(self):
        self.t += 1

        # 1) 未尝试臂优先探索（随机顺序）
        unexplored = np.where(self.n == 0)[0]
        if len(unexplored) > 0:
            i = np.random.choice(unexplored)
            return i, self.actions[i]

        # 2) ε-随机探索
        if np.random.rand() < self.eps_rand:
            i = np.random.randint(len(self.actions))
            return i, self.actions[i]

        # 3) UCB
        bonus = np.sqrt(2.0 * np.log(max(self.t, 2)) / np.maximum(self.n, 1e-9))
        ucb_values = self.s + bonus

        if np.any(np.isnan(ucb_values)) or np.any(np.isinf(ucb_values)):
            i = np.random.randint(len(self.actions))
        else:
            i = int(np.argmax(ucb_values))
        return i, self.actions[i]

    def update(self, idx, r01):
        r01 = float(np.clip(r01, 0.0, 1.0))
        self.n[idx] += 1
        if self.n[idx] == 1:
            self.s[idx] = r01
        else:
            self.s[idx] = (1 - self.alpha) * self.s[idx] + self.alpha * r01

# ================================
# 工具
# ================================
def _init_agent(action_space):
    global _AGENT
    q_list = action_space.get('q_bps', [300, 600, 1200, 2400])
    p_list = action_space.get('p_grid_w', [0.2, 0.4, 0.8, 1.2, 1.6])
    b_list = action_space.get('b_subbands', [1, 2, 3])

    actions = []
    for q in q_list:
        for p in p_list:
            for b in b_list:
                actions.append((float(q), float(p), int(b)))
    _AGENT = _UCBArms(actions)

def _rate_penalty_db(q_bps, q_ref=300.0, c_db=3.5):
    """速率越高门限越高"""
    return float(c_db * np.log2(max(q_bps, 1.0) / q_ref + 1.0))

def _reset_norm():
    global _NORM_INIT, _NORM_MU, _NORM_SIG
    _NORM_INIT = False
    _NORM_MU = 0.0
    _NORM_SIG = 1.0

def _norm_update_and_map01(r_raw):
    """
    用 EWMA 均值/方差做在线标准化，再过 sigmoid，得到 (0,1) 奖励。
    这样不会把不同 V 的尺度差异抹平（比 min-max 更稳、更保真）。
    """
    global _NORM_INIT, _NORM_MU, _NORM_SIG, _NORM_ALPHA
    x = float(r_raw)

    if not _NORM_INIT:
        _NORM_INIT = True
        _NORM_MU = x
        _NORM_SIG = 1.0
    else:
        # EWMA 均值
        _NORM_MU = (1 - _NORM_ALPHA) * _NORM_MU + _NORM_ALPHA * x
        # EWMA 方差近似：用 abs 差做稳健尺度
        abs_dev = abs(x - _NORM_MU)
        _NORM_SIG = (1 - _NORM_ALPHA) * _NORM_SIG + _NORM_ALPHA * max(abs_dev, 1e-6)

    z = (x - _NORM_MU) / max(_NORM_SIG, 1e-6)
    # 温和压缩，保留方向/相对量级
    r01 = 1.0 / (1.0 + np.exp(-0.75 * z))
    return float(np.clip(r01, 0.0, 1.0))

# ================================
# 核心 API
# ================================
def pick_action_and_estimate(ctx, action_space, Q_t, J_t, V, sem_weight,
                             Q_scale=1e4, J_scale=1e2):
    """
    返回：
        {
            'action': (q, p, b, 'ucb'),
            'S_bits': ...,
            'E_hat': ...,
            'SWWER_hat': ...,
            'per': ...,
            'snr_eff_db': ...,
            'debug_info': {...}
        }
    """
    global _AGENT

    if _AGENT is None:
        _init_agent(action_space)

    # 上下文
    snr_db  = float(ctx.get('snr_db', 0.0))
    slot_sec = float(ctx.get('slot_sec', 0.02))

    # 选臂
    idx, (q, p, b) = _AGENT.select()

    # PHY 估计
    snr_eff_db = snr_db + 10.0 * np.log10(max(p * b, 1e-6)) - _rate_penalty_db(q)
    per = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
    swwer = wer_from_per(per, sem_weight)
    S_bits = float(q * (1.0 - per) * slot_sec)
    E_hat  = float(p)  # 能耗只与 p 挂钩（b 提升 SNR 不加能耗）

    # Lyapunov 权重（外层队列规模 → 倾向多服务/少能耗）
    a = Q_t / Q_scale
    b_w = J_t / J_scale

    # V 归一：随 V 提升，逐步加大对能耗与语义错误的惩罚
    V0 = 80.0
    V_norm = float(V) / (float(V) + V0)

    # 关键：双惩罚随 V 增强，导向：
    # - 降低 p（J 下降）
    # - 降低 swwer（通常降 q 或升 b → 吞吐降低，Q 上升）
    kE = 1.0    # 稍弱一点，保留J下降但不压制Q
    kW = 4.0    # 加强语义惩罚，迫使高V时降速率
    kQ = 0.0004 # 新增：轻微速率惩罚系数，让V大时不愿选高q

    r_raw = (
    a * S_bits
    - (b_w + kE * V_norm) * E_hat
    - (kW * V_norm) * (swwer)
    - kQ * V * q           # 关键新增项
)


    # 标准化 → [0,1] 奖励
    r01 = _norm_update_and_map01(r_raw)

    _AGENT.update(idx, r01)

    return {
        'action': (float(q), float(p), int(b), 'ucb'),
        'S_bits': S_bits,
        'E_hat': E_hat,
        'SWWER_hat': float(swwer),
        'per': float(per),
        'snr_eff_db': float(snr_eff_db),
        'debug_info': {
            'selected_index': idx,
            'reward01': float(r01),
            'r_raw': float(r_raw),
            'V_parameter': float(V),
            'V_norm': float(V_norm),
            'kE': kE, 'kW': kW
        }
    }

def reset_agent():
    """在每个 episode（每个 V）开始前调用"""
    global _AGENT
    _AGENT = None
    _reset_norm()
