import os
import numpy as np
from wer_proxy import per_from_snr_db, wer_from_per
from improved_ucb import LRCCUCB, LRCCUCBConfig

# ---- UCB1 管理器（对 (q,p,b) 做逐臂统计） ----
class _UCBArms:
    def __init__(self, actions):
        # actions: 列表 [(q,p,b), ...]
        self.actions = list(actions)
        K = len(self.actions)
        self.n = np.zeros(K, dtype=int)  # 拉动次数
        self.s = np.zeros(K, dtype=float)  # 奖励和（已缩放到[0,1]）
        self.t = 0
    
    def select(self):
        self.t += 1
        # 先把未拉过的臂依次试一次
        for i, a in enumerate(self.actions):
            if self.n[i] == 0:
                return i, a
        mean = self.s / np.maximum(self.n, 1)
        bonus = np.sqrt(2.0 * np.log(self.t) / self.n)
        i = int(np.argmax(mean + bonus))
        return i, self.actions[i]
    
    def update(self, idx, r01):
        # r01 必须在 [0,1] 内
        r01 = float(np.clip(r01, 0.0, 1.0))
        self.n[idx] += 1
        self.s[idx] += r01

# ---- 单例（一个episode内持久） ----
_AGENT = None
_LRCCUCB = None
_ALGO = os.environ.get("INNER_UCB_ALGO", "ucb")
_LAST_PICK = None
_LRCCFG = LRCCUCBConfig()


def configure(algo: str = "ucb", config: dict | None = None) -> None:
    """Configure the inner agent behaviour."""

    global _ALGO, _AGENT, _LRCCUCB, _LRCCFG
    _ALGO = algo
    _AGENT = None
    _LRCCUCB = None
    if config:
        _LRCCFG = LRCCUCBConfig(**config)

def _init_agent(action_space):
    global _AGENT
    q_list = action_space.get('q_bps', [300,600,1200,2400])
    p_list = action_space.get('p_grid_w', [0.2,0.4,0.8,1.2,1.6])
    b_list = action_space.get('b_subbands', [1,2,3])
    actions = []
    for q in q_list:
        for p in p_list:
            for b in b_list:
                actions.append((float(q), float(p), int(b)))
    _AGENT = _UCBArms(actions)


def _get_lrccucb(action_space):
    global _LRCCUCB
    if _LRCCUCB is None:
        _LRCCUCB = LRCCUCB(action_space, config=_LRCCFG)
    return _LRCCUCB

def _rate_penalty_db(q_bps, q_ref=300.0, c_db=3.5):
    # 速率越高门限越高（与 mock 一致但可更陡， 使策略有取舍）
    return float(c_db * np.log2(q_bps / q_ref + 1.0))

def _r01_from_cost_parts(S_bits, E_hat, swwer, A_bits, E_bar, a, b, c,
                        slot_sec, p_min, p_max, q_max):
    """
    将 reward_raw = a*S_bits - b*E_hat - c*swwer 线性映射到 [0,1]
    近似上下界：
    S_bits ∈ [0, q_max*slot_sec], E_hat ∈ [p_min, p_max], swwer ∈ [0,1]
    """
    r_raw = a*S_bits - b*E_hat - c*swwer
    r_min = - b*p_max - c*1.0  # S_bits=0 时
    r_max = a*(q_max*slot_sec) - b*p_min - c*0.0
    if r_max - r_min < 1e-9:
        return 0.5
    r01 = (r_raw - r_min) / (r_max - r_min)
    return float(np.clip(r01, 0.0, 1.0))

def pick_action_and_estimate(ctx, action_space, Q_t, J_t, V, sem_weight,
                           Q_scale=1e4, J_scale=1e2):
    """
    选择一组 (q,p,b) 并给出估计:
    返回 dict: {'action':(q,p,b,'ucb'),'S_bits':..,'E_hat':..,'SWWER_hat':..,'per':..,'snr_eff_db':..}
    """
    global _AGENT, _LAST_PICK
    ctx = dict(ctx)
    ctx.setdefault('Q_scale', Q_scale)
    ctx.setdefault('J_scale', J_scale)
    algo = ctx.get('algo', _ALGO)
    if algo == 'lrc_cucb':
        agent = _get_lrccucb(action_space)
        return agent.step(ctx, action_space, Q_t, J_t, V, sem_weight)

    if 'S_bits_obs' in ctx and _LAST_PICK is not None:
        params = _LAST_PICK['params']
        r_obs = _r01_from_cost_parts(float(ctx.get('S_bits_obs', 0.0)),
                                     float(ctx.get('E_obs', 0.0)),
                                     float(ctx.get('swwer_obs', 0.0)),
                                     params['A_bits'], params['E_bar'],
                                     a=params['a'], b=params['b'], c=params['c'],
                                     slot_sec=params['slot_sec'],
                                     p_min=params['p_min'], p_max=params['p_max'], q_max=params['q_max'])
        _AGENT.update(_LAST_PICK['idx'], r_obs)
        return _LAST_PICK['return']

    if _AGENT is None:
        _init_agent(action_space)
    
    q_list = action_space.get('q_bps', [300,600,1200,2400])
    p_list = action_space.get('p_grid_w', [0.2,0.4,0.8,1.2,1.6])
    b_list = action_space.get('b_subbands', [1,2,3])
    q_max = max(q_list); p_min, p_max = min(p_list), max(p_list)
    
    snr_db = float(ctx.get('snr_db', 0.0))
    E_bar = float(ctx.get('E_bar', 0.5))
    slot_sec = float(ctx.get('slot_sec', 0.02))
    A_bits = float(ctx.get('A_bits', 600.0*slot_sec))
    
    # UCB 选臂
    idx, (q,p,b) = _AGENT.select()
    
    # 物理近似： 功率/带宽抬升 - 速率惩罚
    snr_eff_db = snr_db + 10.0*np.log10(max(p*b, 1e-6)) - _rate_penalty_db(q)
    per = per_from_snr_db(snr_eff_db, steep=1.2, mid_db=1.0)
    swwer = wer_from_per(per, sem_weight)
    S_bits = float(q * (1.0 - per) * slot_sec)
    E_hat = float(p)
    
    # 将外层代价的三个系数转成"最大化奖励"的权重
    a = (Q_t / Q_scale)
    b_ = (J_t / J_scale)
    c = V
    r01 = _r01_from_cost_parts(S_bits, E_hat, swwer, A_bits, E_bar,
                              a=a, b=b_, c=c, slot_sec=slot_sec,
                              p_min=p_min, p_max=p_max, q_max=q_max)

    _LAST_PICK = {
        'idx': idx,
        'reward': r01,
        'params': {
            'slot_sec': slot_sec,
            'A_bits': A_bits,
            'E_bar': E_bar,
            'a': a,
            'b': b_,
            'c': c,
            'p_min': p_min,
            'p_max': p_max,
            'q_max': q_max,
        },
        'return': {
            'action': (float(q), float(p), int(b), 'ucb'),
            'S_bits': S_bits,
            'E_hat': E_hat,
            'SWWER_hat': float(swwer),
            'per': float(per),
            'snr_eff_db': float(snr_eff_db)
        }
    }

    _AGENT.update(idx, r01)
    return _LAST_PICK['return']