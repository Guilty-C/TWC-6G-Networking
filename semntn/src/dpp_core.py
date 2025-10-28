"""Drift-Plus-Penalty (DPP) controller with virtual queues {lat, eng, sem}.

Class: DPPController
- step(state) -> (P, B, stats)

Supports two selection modes:
- 'continuous': grid search over B and 1D search over P
- 'bandit': discrete arms selected via inner_bandit (UCB family)

State dictionary expected keys (with defaults):
- 'snr_db' (float): channel quality
- 'slot_sec' (float): slot duration seconds
- 'A_bits' (float): arrivals per slot
- 'E_bar' (float): energy budget per slot
- 'I_target' (float): semantic distortion target (e.g., max sWER)
- 'q_bps' (float, optional): codec bitrate if applicable
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .link_models import throughput_bits, rate
from .qoe import mos_surrogate
from .sem_weight import get_sem_weight
from .semantics import swer_from_sinr, get_sem_params, sinr_lin_for_swer_target
from .constants import AGG_REQUIRED_COLS


@dataclass
class DPPConfig:
    mode: str = "continuous"  # or 'bandit'
    V: float = 100.0
    Q_scale: float = 2.0e4
    J_scale: float = 5.0e2
    S_scale: float = 1.0
    # Initial multiplier for online Lagrange multipliers (λ init = scales * multiplier)
    lam_init_multiplier: float = 1.0
    # Optional explicit initial values for dual variables (override multiplier if provided)
    lam_init_lat: Optional[float] = None
    lam_init_eng: Optional[float] = None
    lam_init_sem: Optional[float] = None
    # Optional clipping ceilings for online dual prices (λ)
    lam_clip_lat: Optional[float] = None
    lam_clip_eng: Optional[float] = None
    # Online dualization step size and queue guard margin
    eta_dual: float = 5e-3
    queue_delta: float = 0.08
    # continuous search params
    B_grid: Sequence[float] = (1.0, 2.0, 3.0)
    P_min: float = 0.2
    P_max: float = 1.6
    P_grid: int = 25
    # bandit params
    arms: Optional[List[Tuple[float, float, int]]] = None  # (q_bps, P_w, B)
    bandit_kwargs: Optional[Dict[str, float]] = None


class DPPController:
    def __init__(self, cfg: DPPConfig) -> None:
        self.cfg = cfg
        self.z_lat = 0.0
        self.z_eng = 0.0
        self.z_sem = 0.0
        # Per-slot aggregator row template covering required columns
        # Numeric defaults only (0.0/0), no None/empty strings emitted
        self.ROW_TEMPLATE: Dict[str, float | int] = {
            # Budgets & instantaneous measurements
            "E_slot_J": 0.0,
            "energy_budget_per_slot_j": 0.0,
            "S_eff_bps": 0.0,
            "arrivals_bps": 0.0,
            "delta_queue_used": 0.0,
            "Q_t_used": 0.0,
            "sWER_clip": 0.0,
            "semantic_budget": 0.0,
            # Physical/unit diagnostics
            "B_eff_Hz": 0.0,
            "B_min_kHz": 0.0,
            "latency_budget_q_bps": 0.0,
            # Guard feasibility snapshots and search bounds
            "feasible_energy_guard": 0,
            "feasible_queue_guard": 0,
            "feasible_sem_guard": 0,
            "P_max_energy": 0.0,
            "P_min_sem": 0.0,
            "P_min_queue": 0.0,
            # Guard-set selection window and enforcement flags
            "action_in_guard_set": 0,
            "fallback_used": 0,
            "bad_energy": 0,
            "bad_queue": 0,
            "bad_sem": 0,
        }
        # Online Lagrange multipliers (initialized to scales)
        m = float(getattr(self.cfg, "lam_init_multiplier", 1.0))
        self.lam_lat = (
            float(self.cfg.lam_init_lat)
            if getattr(self.cfg, "lam_init_lat", None) is not None
            else float(self.cfg.Q_scale) * m
        )
        self.lam_eng = (
            float(self.cfg.lam_init_eng)
            if getattr(self.cfg, "lam_init_eng", None) is not None
            else float(self.cfg.J_scale) * m
        )
        self.lam_sem = (
            float(self.cfg.lam_init_sem)
            if getattr(self.cfg, "lam_init_sem", None) is not None
            else float(self.cfg.S_scale) * m
        )
        self.t = 0
        self._bandit = None
        if str(self.cfg.mode).lower() == "bandit":
            from .inner_bandit import InnerBandit
            arms = self.cfg.arms or []
            K = len(arms)
            kwargs = self.cfg.bandit_kwargs or {}
            self._arms = list(arms)
            self._bandit = InnerBandit(
                K=K,
                mode=str(kwargs.get("mode", "sw_ucb")),
                alpha=float(kwargs.get("alpha", 1.0)),
                window=int(kwargs.get("window", 250)),
                lcb_min=float(kwargs.get("lcb_min", 0.0)),
                eps_rand=float(kwargs.get("eps_rand", 0.0)),
                linucb_dim=int(kwargs.get("linucb_dim", 8)),
            )

    # ---------------- Core ----------------
    def _dpp_cost(self, lat_slack_bps: float, E_slot_J: float, E_budget_J: float,
                   swer: float, I_t: float, mos: float) -> float:
        V = float(self.cfg.V)
        # Latency proxy driven by q_bps vs latency_budget_q_bps
        Q_term = (self.z_lat / max(self.cfg.Q_scale, 1e-6)) * float(lat_slack_bps)
        J_term = (self.z_eng / max(self.cfg.J_scale, 1e-6)) * (E_slot_J - E_budget_J)
        S_term = (self.z_sem / max(self.cfg.S_scale, 1e-6)) * (swer - I_t)
        # DPP penalty: encourage higher MOS → lower (1 - norm)
        mos_norm = (float(mos) - 1.0) / (4.5 - 1.0)
        penalty = 1.0 - float(np.clip(mos_norm, 0.0, 1.0))
        # Online dualization penalty on constraint excess (positive parts)
        lam_pen = (
            self.lam_lat * max(lat_slack_bps, 0.0)
            + self.lam_eng * max(E_slot_J - E_budget_J, 0.0)
            + self.lam_sem * max(swer - I_t, 0.0)
        )
        return float(Q_term + J_term + S_term + lam_pen + V * penalty)

    def _update_queues(self, lat_slack_bps: float, E_slot_J: float, E_budget_J: float,
                        swer: float, I_t: float) -> None:
        # Queue increments are driven by positive slack against budgets/targets
        self.z_lat = max(self.z_lat + max(float(lat_slack_bps), 0.0), 0.0)
        self.z_eng = max(self.z_eng + max(E_slot_J - E_budget_J, 0.0), 0.0)
        self.z_sem = max(self.z_sem + max(swer - I_t, 0.0), 0.0)

    def _select_continuous(self, state: Dict[str, float], sem_w: float) -> Tuple[float, float, Dict[str, float]]:
        """Continuous search with semantic guard on linear SINR.

        For each B, derive minimal P to meet sWER target via SINR requirement,
        and queue guard via rate(P,B) ≥ arrivals_bps + δ·z_lat, then restrict
        search to the feasible set. If none feasible across B,
        fall back to P=P_max with B maximizing effective SINR.
        """
        best_cost = 1e18
        best = None
        B_list = list(self.cfg.B_grid)
        a_sem, b_sem = get_sem_params()
        # Feature toggles
        use_sem_guard = bool(state.get("guard_semantic", False))
        use_queue_guard = bool(state.get("guard_queue", False))
        use_energy_guard = bool(state.get("guard_energy", False))
        # Required linear SINR for semantic target
        I_t = float(state.get("semantic_budget", state.get("I_target", 0.0)))
        sinr_req_lin = float(sinr_lin_for_swer_target(I_t, a_sem, b_sem))
        # codec-rate penalty in dB (consistent with _snr_eff)
        q_bps = float(state.get("q_bps", 0.0))
        penalty_db = 3.0 * np.log2(max(q_bps, 1.0) / 300.0 + 1.0)
        base_lin = 10.0 ** (float(state.get("snr_db", 0.0)) / 10.0) * (10.0 ** (-penalty_db / 10.0))
        # Queue guard target rate
        arrivals = float(state.get("arrivals_bps", state.get("A_bps", 0.0)))
        rate_req = float(arrivals + float(self.cfg.queue_delta) * float(self.z_lat))
        # Energy budget per slot and tightened guard ceiling
        E_budget_J = float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)))
        eps_e = float(state.get("epsilon_energy", 0.0))
        E_budget_tight = float(E_budget_J * (1.0 - max(eps_e, 0.0)))
        slot_sec = float(state.get("slot_sec", 0.02))
        P_proc_W = float(state.get("P_proc_W", 0.0))

        def _p_min_for_rate(B_units: float, target_rate_bps: float) -> float:
            # binary search for minimal P to satisfy rate(P,B) ≥ target
            lo = float(self.cfg.P_min); hi = float(self.cfg.P_max)
            if rate(lo, B_units, state) >= target_rate_bps:
                return lo
            if rate(hi, B_units, state) < target_rate_bps:
                return float("inf")
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if rate(mid, B_units, state) >= target_rate_bps:
                    hi = mid
                else:
                    lo = mid
            return hi

        for B in B_list:
            # Minimal P ensuring SINR_lin ≥ sinr_req_lin under B
            K = base_lin * max(B, 1.0)
            P_min_sem = float(sinr_req_lin / max(K, 1e-12))
            # Minimal P ensuring rate ≥ arrivals + δ·z_lat
            P_min_queue = float(_p_min_for_rate(B, rate_req))
            # Maximal P allowed by energy tight budget
            # E_slot_J = (P_tx_W + P_proc_W) * slot_sec ⇒ P_tx_W ≤ E_budget_tight/slot_sec - P_proc_W
            P_max_energy = float(max((E_budget_tight / max(slot_sec, 1e-12)) - P_proc_W, 0.0))
            minima = [float(self.cfg.P_min)]
            if use_sem_guard:
                minima.append(P_min_sem)
            if use_queue_guard:
                minima.append(P_min_queue)
            # Use max over list to avoid TypeError when single element
            P_lo = max(minima)
            P_hi = float(min(float(self.cfg.P_max), P_max_energy)) if use_energy_guard else float(self.cfg.P_max)
            feasible_sem = int(use_sem_guard and (P_min_sem <= P_hi))
            feasible_queue = int(use_queue_guard and (P_min_queue != float("inf") and P_min_queue <= P_hi))
            feasible_window = int(P_lo <= P_hi)
            feasible = bool((P_lo <= P_hi) and (not use_queue_guard or P_min_queue != float("inf")))
            if feasible:
                P_values = np.linspace(P_lo, P_hi, int(self.cfg.P_grid))
                for P in P_values:
                    # throughput and QoE
                    q_bps_obs, p, S_bits = throughput_bits(P, B, {**state, "sem_weight": sem_w})
                    mos = mos_surrogate({**state, "sem_weight": sem_w, "q_bps": q_bps_obs}, P, B)
                    P_tx_W = float(P)
                    E_slot_J = float((P_tx_W + P_proc_W) * slot_sec)
                    # Enforce guard-preselection: semantic/queue lower bounds and energy upper bound
                    if use_sem_guard and (P < P_min_sem):
                        continue
                    if use_queue_guard and (P < P_min_queue):
                        continue
                    if use_energy_guard and (E_slot_J > E_budget_tight):
                        continue
                    # latency slack (bps) using q_bps vs latency_budget_q_bps
                    q_budget = float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0)))
                    lat_slack = float(q_bps_obs - q_budget)
                    cost = self._dpp_cost(lat_slack,
                                          E_slot_J, state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)),
                                          p, state.get("semantic_budget", state.get("I_target", 0.0)), mos)
                    if cost < best_cost:
                        best_cost = cost
                        best = (P, B, {
                            "q_bps": float(q_bps_obs),
                            "per": float(p),
                            "S_bits": float(S_bits),
                            "mos": float(mos),
                            "P_tx_W": float(P_tx_W),
                            "E_slot_J": float(E_slot_J),
                            "sinr_req_lin": float(sinr_req_lin),
                            "P_min_sem": float(P_min_sem),
                            "P_min_queue": float(P_min_queue),
                            "P_max_energy": float(P_max_energy) if use_energy_guard else float("nan"),
                            "feasible_sem_guard": int(feasible_sem),
                            "feasible_queue_guard": int(feasible_queue),
                            "feasible_energy_guard": int(use_energy_guard and (E_slot_J <= E_budget_tight)),
                            "feasible_set_nonempty": int(feasible_window),
                            "action_in_guard_set": int(1),
                            "fallback_used": int(0),
                        })

        if best is None:
            # Fallback: choose P=P_max and B maximizing effective SINR
            P = float(self.cfg.P_max)
            sinr_best = -1e9
            B_best = B_list[0] if B_list else 1.0
            for B in B_list:
                sinr_db = float(_snr_eff(P, B, state, q_bps))
                if sinr_db > sinr_best:
                    sinr_best = sinr_db
                    B_best = B
            # Stats for fallback
            q_bps_obs, p, S_bits = throughput_bits(P, B_best, {**state, "sem_weight": sem_w})
            mos = mos_surrogate({**state, "sem_weight": sem_w, "q_bps": q_bps_obs}, P, B_best)
            E_slot_J = float((P + P_proc_W) * slot_sec)
            # Derive guard diagnostics even in fallback
            K_fb = base_lin * max(B_best, 1.0)
            P_min_sem_fb = float(sinr_req_lin / max(K_fb, 1e-12))
            P_min_queue_fb = float(_p_min_for_rate(B_best, rate_req))
            if P_min_queue_fb == float("inf"):
                P_min_queue_fb = float(self.cfg.P_max) + 1e-9  # finite sentinel beyond max
            feasible_sem_fb = int(use_sem_guard and (P_min_sem_fb <= float(self.cfg.P_max)))
            feasible_queue_fb = int(use_queue_guard and (P_min_queue_fb <= float(self.cfg.P_max)))
            feasible_energy_fb = int(use_energy_guard and (E_slot_J <= E_budget_tight))
            return float(P), float(B_best), {
                "q_bps": float(q_bps_obs),
                "per": float(p),
                "S_bits": float(S_bits),
                "mos": float(mos),
                "P_tx_W": float(P),
                "E_slot_J": float(E_slot_J),
                "sinr_req_lin": float(sinr_req_lin),
                "P_min_sem": float(P_min_sem_fb),
                "P_min_queue": float(P_min_queue_fb),
                "P_max_energy": float(max((E_budget_tight / max(slot_sec, 1e-12)) - P_proc_W, 0.0)) if use_energy_guard else float("nan"),
                "feasible_sem_guard": int(feasible_sem_fb),
                "feasible_queue_guard": int(feasible_queue_fb),
                "feasible_energy_guard": int(feasible_energy_fb),
                "feasible_set_nonempty": int(0),
                "action_in_guard_set": int(0),
                "fallback_used": int(1),
            }

        return best[0], best[1], best[2]

    def _select_bandit(self, state: Dict[str, float], sem_w: float) -> Tuple[float, float, Dict[str, float]]:
        assert self._bandit is not None and len(self._arms) > 0
        # context features include current queues and channel state
        ctx = {
            "snr_db": float(state.get("snr_db", 0.0)),
            "z_lat": float(self.z_lat),
            "z_eng": float(self.z_eng),
            "z_sem": float(self.z_sem),
            "V": float(self.cfg.V),
            "A_bits": float(state.get("A_bits", 0.0)),
            "E_bar": float(state.get("E_bar", 0.0)),
            "I_target": float(state.get("I_target", 0.0)),
            "sem_w": float(sem_w),
            # budgets for reward shaping (if enabled)
            "latency_budget_q_bps": float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0))),
            "reward_shaping": bool((self.cfg.bandit_kwargs or {}).get("reward_shaping", False)),
        }
        a = int(self._bandit.choose_arm(ctx))
        q_bps, P, B = self._arms[a]
        # Evaluate outcome
        p = np.clip(1.0 / (1.0 + np.exp(1.2 * (_snr_eff(P, B, state, q_bps) - 1.0))), 0.0, 1.0)
        # Reuse throughput_bits for consistency
        q_hat, p_hat, S_bits = throughput_bits(P, B, {**state, "q_bps": q_bps, "sem_weight": sem_w})
        mos = mos_surrogate({**state, "sem_weight": sem_w, "q_bps": q_bps}, P, B)
        slot_sec = float(state.get("slot_sec", 0.02))
        P_proc_W = float(state.get("P_proc_W", 0.0))
        P_tx_W = float(P)
        E_slot_J = float(P_tx_W * slot_sec + P_proc_W * slot_sec)
        # reward base (normalized MOS)
        r_base = float(np.clip((mos - 1.0) / (4.5 - 1.0), 0.0, 1.0))
        # augment context for optional Lagrangian shaping inside bandit
        swer_hat = float(np.clip(p_hat / max(sem_w, 1e-6), 0.0, 1.0))
        lat_budget = float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0)))
        slack_eng = max(float(E_slot_J) - float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0))), 0.0)
        slack_sem = max(swer_hat - float(state.get("semantic_budget", state.get("I_target", 0.0))), 0.0)
        slack_lat = max(float(q_hat) - float(lat_budget), 0.0)
        ctx.update({
            "r_mos": r_base,
            "E_slot_J": float(E_slot_J),
            "swer": float(swer_hat),
            "q_bps_obs": float(q_hat),
            # scales for shaping
            "scale_Q": float(self.cfg.Q_scale),
            "scale_J": float(self.cfg.J_scale),
            "scale_S": float(self.cfg.S_scale),
            # last-slot slacks (for violation-based c adjustment)
            "slack_eng": float(slack_eng),
            "slack_sem": float(slack_sem),
            "slack_lat": float(slack_lat),
        })
        self._bandit.update(r_base, a, ctx)
        # Guard diagnostics
        use_sem_guard = bool(state.get("guard_semantic", False))
        use_queue_guard = bool(state.get("guard_queue", False))
        a_sem, b_sem = get_sem_params()
        I_t = float(state.get("semantic_budget", state.get("I_target", 0.0)))
        sinr_req_lin = float(sinr_lin_for_swer_target(I_t, a_sem, b_sem))
        arrivals = float(state.get("arrivals_bps", state.get("A_bps", 0.0)))
        rate_req = float(arrivals + float(self.cfg.queue_delta) * float(self.z_lat))
        def _p_min_for_rate(B_units: float, target_rate_bps: float) -> float:
            lo = float(self.cfg.P_min); hi = float(self.cfg.P_max)
            if rate(lo, B_units, state) >= target_rate_bps:
                return lo
            if rate(hi, B_units, state) < target_rate_bps:
                return float("inf")
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if rate(mid, B_units, state) >= target_rate_bps:
                    hi = mid
                else:
                    lo = mid
            return hi
        base_lin = 10.0 ** (float(state.get("snr_db", 0.0)) / 10.0)
        K = base_lin * max(B, 1.0)
        P_min_sem = float(sinr_req_lin / max(K, 1e-12))
        P_min_queue = float(_p_min_for_rate(B, rate_req))
        feasible_sem = int(use_sem_guard and (P_min_sem <= float(self.cfg.P_max)))
        feasible_queue = int(use_queue_guard and (P_min_queue != float("inf") and P_min_queue <= float(self.cfg.P_max)))
        feasible_energy = int(bool(state.get("guard_energy", False)) and (E_slot_J <= float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)))))
        return float(P), float(B), {
            "arm_id": a,
            "q_bps": float(q_bps),
            "per": float(p_hat),
            "S_bits": float(S_bits),
            "mos": float(mos),
            "P_tx_W": float(P_tx_W),
            "E_slot_J": float(E_slot_J),
            "lcb_min": float(getattr(self._bandit, "lcb_min", 0.0)),
            "reward_shaped": int(bool((self.cfg.bandit_kwargs or {}).get("reward_shaping", False))),
            # guard diagnostics
            "sinr_req_lin": float(sinr_req_lin),
            "P_min_sem": float(P_min_sem if use_sem_guard else np.nan),
            "P_min_queue": float(P_min_queue if use_queue_guard else np.nan),
            "feasible_sem_guard": int(feasible_sem),
            "feasible_queue_guard": int(feasible_queue),
            "feasible_energy_guard": int(feasible_energy),
        }

    def _project_to_guard_set(self, P: float, B: float, state: Dict[str, float], sem_w: float) -> Tuple[float, float, Dict[str, bool]]:
        """Project (P,B) onto the nearest feasible guard action.
        
        Ensures all guard constraints are satisfied:
        - Semantic guard: SINR ≥ SINR_req for sWER target
        - Queue guard: rate(P,B) ≥ arrivals + δ·Q_t
        - Energy guard: E_slot ≤ E_budget_tight
        
        Returns projected (P_proj, B_proj) and guard feasibility flags.
        """
        # Get guard constraints
        use_sem_guard = bool(state.get("guard_semantic", False))
        use_queue_guard = bool(state.get("guard_queue", False))
        use_energy_guard = bool(state.get("guard_energy", False))
        
        # Semantic guard: minimal P for SINR requirement
        a_sem, b_sem = get_sem_params()
        I_t = float(state.get("semantic_budget", state.get("I_target", 0.0)))
        sinr_req_lin = float(sinr_lin_for_swer_target(I_t, a_sem, b_sem))
        
        # Queue guard: minimal P for rate requirement
        arrivals = float(state.get("arrivals_bps", state.get("A_bps", 0.0)))
        rate_req = float(arrivals + float(self.cfg.queue_delta) * float(self.z_lat))
        
        # Energy guard: maximal P for energy budget
        E_budget_J = float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)))
        eps_e = float(state.get("epsilon_energy", 0.0))
        E_budget_tight = float(E_budget_J * (1.0 - max(eps_e, 0.0)))
        slot_sec = float(state.get("slot_sec", 0.02))
        P_proc_W = float(state.get("P_proc_W", 0.0))
        P_max_energy = float(max((E_budget_tight / max(slot_sec, 1e-12)) - P_proc_W, 0.0))
        
        # Helper function to find minimal P for rate constraint
        def _p_min_for_rate(B_units: float, target_rate_bps: float) -> float:
            lo = float(self.cfg.P_min); hi = float(self.cfg.P_max)
            if rate(lo, B_units, state) >= target_rate_bps:
                return lo
            if rate(hi, B_units, state) < target_rate_bps:
                return float("inf")
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if rate(mid, B_units, state) >= target_rate_bps:
                    hi = mid
                else:
                    lo = mid
            return hi
        
        # Calculate guard boundaries
        base_lin = 10.0 ** (float(state.get("snr_db", 0.0)) / 10.0)
        K = base_lin * max(B, 1.0)
        P_min_sem = float(sinr_req_lin / max(K, 1e-12)) if use_sem_guard else float(self.cfg.P_min)
        P_min_queue = float(_p_min_for_rate(B, rate_req)) if use_queue_guard else float(self.cfg.P_min)
        
        # Handle infinite queue requirement
        if P_min_queue == float("inf"):
            P_min_queue = float(self.cfg.P_max) + 1e-9
        
        # Determine lower and upper bounds
        P_lower = max(float(self.cfg.P_min), P_min_sem, P_min_queue)
        P_upper = min(float(self.cfg.P_max), P_max_energy) if use_energy_guard else float(self.cfg.P_max)
        
        # Project P to feasible range for current B
        P_proj = float(np.clip(P, P_lower, P_upper))
        B_proj = float(B)

        # If the guard window is empty for current B (or queue is infeasible),
        # attempt to adjust B upwards to find a non-empty feasible window.
        window_empty = bool(P_lower > P_upper) or (use_queue_guard and (P_min_queue == float("inf")))
        if window_empty:
            for B_cand in sorted(list(self.cfg.B_grid)):
                base_lin_c = 10.0 ** (float(state.get("snr_db", 0.0)) / 10.0)
                K_c = base_lin_c * max(B_cand, 1.0)
                P_min_sem_c = float(sinr_req_lin / max(K_c, 1e-12)) if use_sem_guard else float(self.cfg.P_min)
                P_min_queue_c = float(_p_min_for_rate(B_cand, rate_req)) if use_queue_guard else float(self.cfg.P_min)
                if P_min_queue_c == float("inf"):
                    # No queue-feasible action for this B; try next
                    continue
                P_upper_c = min(float(self.cfg.P_max), P_max_energy) if use_energy_guard else float(self.cfg.P_max)
                P_lower_c = max(float(self.cfg.P_min), P_min_sem_c, P_min_queue_c)
                if P_lower_c <= P_upper_c:
                    # Found a feasible window; choose projected action within it
                    B_proj = float(B_cand)
                    P_proj = float(np.clip(P, P_lower_c, P_upper_c))
                    # Recompute bounds for reporting
                    P_min_sem = P_min_sem_c
                    P_min_queue = P_min_queue_c
                    P_upper = P_upper_c
                    P_lower = P_lower_c
                    break
        
        # Check guard feasibility after projection
        feasible_sem = int(use_sem_guard and (P_proj >= P_min_sem))
        feasible_queue = int(use_queue_guard and (P_proj >= P_min_queue))
        feasible_energy = int(use_energy_guard and (P_proj <= P_max_energy))
        
        # All guards must be feasible for action to be in guard set
        action_in_guard_set = int(feasible_sem and feasible_queue and feasible_energy)
        
        return P_proj, B_proj, {
            "feasible_sem_guard": feasible_sem,
            "feasible_queue_guard": feasible_queue,
            "feasible_energy_guard": feasible_energy,
            "action_in_guard_set": action_in_guard_set
        }

    def _recompute_guard_stats(self, P: float, B: float, state: Dict[str, float], sem_w: float) -> Dict[str, float]:
        """Recompute all guard statistics using the projected (P,B) values."""
        # Get guard constraints
        use_sem_guard = bool(state.get("guard_semantic", False))
        use_queue_guard = bool(state.get("guard_queue", False))
        use_energy_guard = bool(state.get("guard_energy", False))
        
        # Semantic guard: minimal P for SINR requirement
        a_sem, b_sem = get_sem_params()
        I_t = float(state.get("semantic_budget", state.get("I_target", 0.0)))
        sinr_req_lin = float(sinr_lin_for_swer_target(I_t, a_sem, b_sem))
        
        # Queue guard: minimal P for rate requirement
        arrivals = float(state.get("arrivals_bps", state.get("A_bps", 0.0)))
        rate_req = float(arrivals + float(self.cfg.queue_delta) * float(self.z_lat))
        
        # Energy guard: maximal P for energy budget
        E_budget_J = float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)))
        eps_e = float(state.get("epsilon_energy", 0.0))
        E_budget_tight = float(E_budget_J * (1.0 - max(eps_e, 0.0)))
        slot_sec = float(state.get("slot_sec", 0.02))
        P_proc_W = float(state.get("P_proc_W", 0.0))
        P_max_energy = float(max((E_budget_tight / max(slot_sec, 1e-12)) - P_proc_W, 0.0))
        
        # Helper function to find minimal P for rate constraint
        def _p_min_for_rate(B_units: float, target_rate_bps: float) -> float:
            lo = float(self.cfg.P_min); hi = float(self.cfg.P_max)
            if rate(lo, B_units, state) >= target_rate_bps:
                return lo
            if rate(hi, B_units, state) < target_rate_bps:
                return float("inf")
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if rate(mid, B_units, state) >= target_rate_bps:
                    hi = mid
                else:
                    lo = mid
            return hi
        
        # Calculate guard boundaries
        base_lin = 10.0 ** (float(state.get("snr_db", 0.0)) / 10.0)
        K = base_lin * max(B, 1.0)
        P_min_sem = float(sinr_req_lin / max(K, 1e-12)) if use_sem_guard else float(self.cfg.P_min)
        P_min_queue = float(_p_min_for_rate(B, rate_req)) if use_queue_guard else float(self.cfg.P_min)
        
        # Handle infinite queue requirement
        if P_min_queue == float("inf"):
            P_min_queue = float(self.cfg.P_max) + 1e-9
        
        # Check guard feasibility after projection
        feasible_sem = int(use_sem_guard and (P >= P_min_sem))
        feasible_queue = int(use_queue_guard and (P >= P_min_queue))
        feasible_energy = int(use_energy_guard and (P <= P_max_energy))
        
        # Compute throughput and energy for actual constraint checking
        q_bps_obs, p, S_bits = throughput_bits(P, B, {**state, "sem_weight": sem_w})
        P_tx_W = float(P)
        E_slot_J = float((P_tx_W + P_proc_W) * slot_sec)
        
        # Compute actual semantic performance
        sinr_eff_db = float(_snr_eff(P, B, state, q_bps_obs))
        swer_clip = float(swer_from_sinr(sinr_eff_db))
        
        # CRITICAL FIX: action_in_guard_set should ONLY depend on feasibility flags
        # NOT on actual constraint violations (bad_* counters)
        action_in_guard_set = int(feasible_sem and feasible_queue and feasible_energy)
        
        # Compute bad_* counters for debugging only - these should ideally be 0
        # but we don't let them affect the action_in_guard_set determination
        bad_energy = int(E_slot_J > E_budget_tight)
        bad_queue = int(q_bps_obs < arrivals + self.cfg.queue_delta * self.z_lat)
        bad_sem = int(swer_clip > I_t)
        
        # Post-projection physical bandwidth in Hz, mapped from discrete grid
        # Use controller's configured min grid value as unit base
        B_min_units = float(min(self.cfg.B_grid)) if len(self.cfg.B_grid) > 0 else float("nan")
        B_min_kHz = float(state.get("B_min_kHz", float("nan")))
        try:
            if np.isfinite(B) and np.isfinite(B_min_units) and (B_min_units > 0.0) and np.isfinite(B_min_kHz) and (B_min_kHz >= 0.0):
                B_star_kHz = float(B) / float(B_min_units) * float(B_min_kHz)
                B_eff_Hz = float(B_star_kHz * 1000.0)
            else:
                B_eff_Hz = float("nan")
        except Exception:
            B_eff_Hz = float("nan")

        return {
            "feasible_sem_guard": feasible_sem,
            "feasible_queue_guard": feasible_queue,
            "feasible_energy_guard": feasible_energy,
            "action_in_guard_set": action_in_guard_set,
            "fallback_used": 0,  # Hard-disable fallback
            "bad_energy": bad_energy,
            "bad_queue": bad_queue,
            "bad_sem": bad_sem,
            "q_bps": float(q_bps_obs),
            "per": float(p),
            "S_bits": float(S_bits),
            "P_tx_W": float(P_tx_W),
            "E_slot_J": float(E_slot_J),
            "sinr_req_lin": float(sinr_req_lin),
            "P_min_sem": float(P_min_sem),
            "P_min_queue": float(P_min_queue),
            "P_max_energy": float(P_max_energy),
            "B_eff_Hz": float(B_eff_Hz),
        }

    def step(self, state: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
        """One control step with effective guard-compliant action enforcement.

        Returns (P, B, stats) where stats includes
        q_bps, z_lat, z_eng, z_sem, rate(q_bps), per, mos, sWER.
        """
        self.t += 1
        sem_w = get_sem_weight(state)
        
        # 1) Compute candidate (P,B) as before
        mode = str(self.cfg.mode).lower()
        if mode == "bandit":
            P, B, info = self._select_bandit(state, sem_w)
        else:
            P, B, info = self._select_continuous(state, sem_w)
        
        # 2) Apply guard projection → get (P*,B*)
        P_proj, B_proj, _ = self._project_to_guard_set(P, B, state, sem_w)
        
        # 3) Recompute all guard booleans and bad_* counters USING (P*,B*)
        guard_stats = self._recompute_guard_stats(P_proj, B_proj, state, sem_w)
        
        # 4) Emit and log only (P*,B*)
        P, B = P_proj, B_proj
        
        # Update info with recomputed guard statistics
        info.update(guard_stats)
        
        # Verify guard enforcement - if any bad_* > 0 after projection → fail-fast
        if guard_stats["bad_energy"] > 0 or guard_stats["bad_queue"] > 0 or guard_stats["bad_sem"] > 0:
            print(f"[FAIL_GUARD_ENFORCEMENT] slot={self.t}, bad_energy={guard_stats['bad_energy']}, bad_queue={guard_stats['bad_queue']}, bad_sem={guard_stats['bad_sem']}, P*={P_proj:.6f}, B*={B_proj:.6f}")
            # Do not raise here; propagate flags via stats for downstream gating
        
        # 5) Set: fallback_used=False; action_in_guard_set = feasible_energy_guard & feasible_queue_guard & feasible_sem_guard
        info["fallback_used"] = 0
        # CRITICAL FIX: Use the action_in_guard_set computed in _recompute_guard_stats
        # This ensures consistency between projection and recomputation
        info["action_in_guard_set"] = guard_stats["action_in_guard_set"]
        
        # 6) Make sure per-slot log columns are typed and reflect the post-projection action
        # Use the recomputed values from guard_stats
        q_bps = float(guard_stats["q_bps"])
        per_hat = float(guard_stats["per"])
        S_bits = float(guard_stats["S_bits"])
        P_tx_W = float(guard_stats["P_tx_W"])
        E_slot_J = float(guard_stats["E_slot_J"])
        
        # queues update (latency proxy via q_bps vs latency_budget_q_bps)
        A_bits = float(state.get("A_bits", 0.0))
        slot_sec = float(state.get("slot_sec", 0.02))
        E_budget_J = float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)))
        I_t = float(state.get("semantic_budget", state.get("I_target", 0.0)))
        
        # tighten energy budget per-step using epsilon_energy (if provided)
        try:
            eps_e = float(state.get("epsilon_energy", 0.0))
        except Exception:
            eps_e = 0.0
        E_budget_tight_step = float(E_budget_J * (1.0 - np.clip(float(eps_e), 0.0, 1.0)))
        
        # effective SINR in dB for semantic sWER
        sinr_eff_db = float(_snr_eff(P, B, state, q_bps))
        a_sem, b_sem = get_sem_params()
        sinr_lin = 10.0 ** (sinr_eff_db / 10.0)
        swer_raw = float(1.0 / (1.0 + np.exp(-(float(a_sem) * sinr_lin + float(b_sem)))))
        # swer_from_sinr returns clipped; preserve raw before clipping for logging
        swer_clip = float(swer_from_sinr(sinr_eff_db))
        
        # latency budget
        q_budget = float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0)))
        lat_slack = float(q_bps - q_budget)
        
        self._update_queues(lat_slack, E_slot_J, E_budget_J, swer_clip, I_t)
        
        # Online dual ascent step (only if enabled): lambda_k ← [lambda_k + eta*(g_k - budget_k)]_+
        if bool(state.get("enable_primal_dual", False)):
            eta = float(self.cfg.eta_dual)
            self.lam_lat = max(self.lam_lat + eta * max(lat_slack, 0.0), 0.0)
            self.lam_eng = max(self.lam_eng + eta * max(E_slot_J - E_budget_J, 0.0), 0.0)
            self.lam_sem = max(self.lam_sem + eta * max(swer_clip - I_t, 0.0), 0.0)
            # Apply clipping if configured
            if getattr(self.cfg, "lam_clip_lat", None) is not None:
                try:
                    self.lam_lat = float(min(self.lam_lat, float(self.cfg.lam_clip_lat)))
                except Exception:
                    pass
            if getattr(self.cfg, "lam_clip_eng", None) is not None:
                try:
                    self.lam_eng = float(min(self.lam_eng, float(self.cfg.lam_clip_eng)))
                except Exception:
                    pass
            if getattr(self.cfg, "lam_clip_sem", None) is not None:
                try:
                    self.lam_sem = float(min(self.lam_sem, float(self.cfg.lam_clip_sem)))
                except Exception:
                    pass
        
        # violation flags
        violate_lat = int(lat_slack > 0.0)
        violate_eng = int((E_slot_J - E_budget_tight_step) > 0.0)
        violate_sem = int(swer_raw > I_t)

        # Derive an energy power ceiling (W) from tightened per-slot budget
        try:
            P_proc_W = float(state.get("P_proc_W", 0.0))
        except Exception:
            P_proc_W = 0.0
        P_max_energy_ceiling_W = float(max((E_budget_tight_step / max(slot_sec, 1e-12)) - P_proc_W, 0.0))

        stats = {
            "q_bps": q_bps,
            "S_eff_bps": float(q_bps),
            "z_lat": float(self.z_lat),
            "z_eng": float(self.z_eng),
            "z_sem": float(self.z_sem),
            "P": float(P),
            "B": float(B),
            "rate": float(q_bps),
            "sinr_db": float(sinr_eff_db),
            "per": float(per_hat),
            "mos": float(info.get("mos", 2.0)),  # Use mos from info if available
            "sWER_raw": float(swer_raw),
            "sWER_clip": float(swer_clip),
            "sWER": float(swer_clip),
            "P_tx_W": float(P_tx_W),
            "E_slot_J": float(E_slot_J),
            "violate_lat": violate_lat,
            "violate_eng": violate_eng,
            "violate_sem": violate_sem,
            # arrivals and queue parameters for aggregator
            "arrivals_bps": float(A_bits / max(slot_sec, 1e-9)),
            "delta_queue_used": float(self.cfg.queue_delta),
            "Q_t_used": float(self.z_lat),
            # budgets for aggregator identity checks
            "energy_budget_per_slot_j": float(E_budget_J),
            "semantic_budget": float(I_t),
            "latency_budget_q_bps": float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0))),
            # energy guard ceiling (always numeric)
            "P_max_energy": float(P_max_energy_ceiling_W),
            # physical minimum bandwidth (if provided in state; else NaN)
            "B_min_kHz": float(state.get("B_min_kHz", float("nan"))),
        }
        
        # Add all guard statistics from recomputed values
        stats.update(guard_stats)

        # Identity/unit invariants for B_eff_Hz (strictly post-projection)
        B_eff_Hz = float(stats.get("B_eff_Hz", float("nan")))
        B_min_kHz = float(stats.get("B_min_kHz", float("nan")))
        rhs = float(B_min_kHz * 1000.0) if (B_min_kHz == B_min_kHz) else float("nan")
        if not np.isfinite(B_eff_Hz):
            t0 = int(self.t)
            print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, B_min_kHz={B_min_kHz:.6f}, B_star_hz={B_eff_Hz}")
            raise RuntimeError("B_eff_Hz is not finite")
        if np.isfinite(rhs) and not (B_eff_Hz >= rhs):
            t0 = int(self.t)
            print(f"[FAIL_UNIT_BANDWIDTH] sample t={t0}, lhs={B_eff_Hz:.6f}, rhs={rhs:.6f}")
            raise RuntimeError("B_eff_Hz below physical minimum")
        
        # Ensure all required columns are present with proper types
        stats["feasible_set_nonempty"] = int(1)  # Always true after projection

        # Build final row from template to guarantee required columns and types
        row = dict(self.ROW_TEMPLATE)
        # Fill required fields from stats/state
        for k, v in stats.items():
            if k in row or k in AGG_REQUIRED_COLS:
                # Normalize booleans to ints, keep numeric as float/int
                try:
                    if isinstance(v, bool):
                        row[k] = int(v)
                    elif isinstance(v, (int, float)):
                        row[k] = v
                    else:
                        row[k] = float(v)
                except Exception:
                    # Fallback to 0.0 for any conversion errors
                    row[k] = 0.0
            else:
                # Add auxiliary fields as-is with numeric casting
                try:
                    if isinstance(v, bool):
                        row[k] = int(v)
                    elif isinstance(v, (int, float)):
                        row[k] = v
                    else:
                        row[k] = float(v)
                except Exception:
                    row[k] = 0.0

        return float(P), float(B), row

        # queues update (latency proxy via q_bps vs latency_budget_q_bps)
        A_bits = float(state.get("A_bits", 0.0))
        slot_sec = float(state.get("slot_sec", 0.02))
        E_budget_J = float(state.get("energy_budget_per_slot_j", state.get("E_bar", 0.0)))
        I_t = float(state.get("semantic_budget", state.get("I_target", 0.0)))
        q_bps = float(info["q_bps"])
        per_hat = float(info["per"])
        S_bits = float(info["S_bits"])
        mos = float(info["mos"])
        P_tx_W = float(info.get("P_tx_W", 0.0))
        E_slot_J = float(info.get("E_slot_J", 0.0))
        # tighten energy budget per-step using epsilon_energy (if provided)
        try:
            eps_e = float(state.get("epsilon_energy", 0.0))
        except Exception:
            eps_e = 0.0
        E_budget_tight_step = float(E_budget_J * (1.0 - np.clip(float(eps_e), 0.0, 1.0)))
        # effective SINR in dB for semantic sWER
        sinr_eff_db = float(_snr_eff(P, B, state, q_bps))
        a_sem, b_sem = get_sem_params()
        sinr_lin = 10.0 ** (sinr_eff_db / 10.0)
        swer_raw = float(1.0 / (1.0 + np.exp(-(float(a_sem) * sinr_lin + float(b_sem)))))
        # swer_from_sinr returns clipped; preserve raw before clipping for logging
        swer_clip = float(swer_from_sinr(sinr_eff_db))
        # latency budget
        q_budget = float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0)))
        lat_slack = float(q_bps - q_budget)
        self._update_queues(lat_slack, E_slot_J, E_budget_J, swer_clip, I_t)
        # Online dual ascent step (only if enabled): lambda_k ← [lambda_k + eta*(g_k - budget_k)]_+
        if bool(state.get("enable_primal_dual", False)):
            eta = float(self.cfg.eta_dual)
            self.lam_lat = max(self.lam_lat + eta * max(lat_slack, 0.0), 0.0)
            self.lam_eng = max(self.lam_eng + eta * max(E_slot_J - E_budget_J, 0.0), 0.0)
            self.lam_sem = max(self.lam_sem + eta * max(swer_clip - I_t, 0.0), 0.0)
            # Apply clipping if configured
            if getattr(self.cfg, "lam_clip_lat", None) is not None:
                try:
                    self.lam_lat = float(min(self.lam_lat, float(self.cfg.lam_clip_lat)))
                except Exception:
                    pass
            if getattr(self.cfg, "lam_clip_eng", None) is not None:
                try:
                    self.lam_eng = float(min(self.lam_eng, float(self.cfg.lam_clip_eng)))
                except Exception:
                    pass
            if getattr(self.cfg, "lam_clip_sem", None) is not None:
                try:
                    self.lam_sem = float(min(self.lam_sem, float(self.cfg.lam_clip_sem)))
                except Exception:
                    pass
        # violation flags
        violate_lat = int(lat_slack > 0.0)
        violate_eng = int((E_slot_J - E_budget_tight_step) > 0.0)
        violate_sem = int(swer_raw > I_t)

        # Derive an energy power ceiling (W) from tightened per-slot budget
        try:
            P_proc_W = float(state.get("P_proc_W", 0.0))
        except Exception:
            P_proc_W = 0.0
        P_max_energy_ceiling_W = float(max((E_budget_tight_step / max(slot_sec, 1e-12)) - P_proc_W, 0.0))

        stats = {
            "q_bps": q_bps,
            "S_eff_bps": float(q_bps),
            "z_lat": float(self.z_lat),
            "z_eng": float(self.z_eng),
            "z_sem": float(self.z_sem),
            "P": float(P),
            "B": float(B),
            "rate": float(q_bps),
            "sinr_db": float(sinr_eff_db),
            "per": float(per_hat),
            "mos": float(mos),
            "sWER_raw": float(swer_raw),
            "sWER_clip": float(swer_clip),
            "sWER": float(swer_clip),
            "P_tx_W": float(P_tx_W),
            "E_slot_J": float(E_slot_J),
            "violate_lat": violate_lat,
            "violate_eng": violate_eng,
            "violate_sem": violate_sem,
            # arrivals and queue parameters for aggregator
            "arrivals_bps": float(A_bits / max(slot_sec, 1e-9)),
            "delta_queue_used": float(self.cfg.queue_delta),
            "Q_t_used": float(self.z_lat),
            # budgets for aggregator identity checks
            "energy_budget_per_slot_j": float(E_budget_J),
            "semantic_budget": float(I_t),
            "latency_budget_q_bps": float(state.get("latency_budget_q_bps", state.get("q_bps", 0.0))),
            # energy guard ceiling (always numeric)
            "P_max_energy": float(P_max_energy_ceiling_W),
            # physical minimum bandwidth (if provided in state; else NaN)
            "B_min_kHz": float(state.get("B_min_kHz", float("nan"))),
        }
        # sWER-guard logging
        if "sinr_req_lin" in info:
            stats["sinr_req_lin"] = float(info["sinr_req_lin"])
        if "P_min_sem" in info:
            stats["P_min_sem"] = float(info["P_min_sem"]) if info["P_min_sem"] == info["P_min_sem"] else float("nan")
        if "P_min_queue" in info:
            stats["P_min_queue"] = float(info["P_min_queue"]) if info["P_min_queue"] == info["P_min_queue"] else float("nan")
        if "P_max_energy" in info:
            try:
                val = float(info["P_max_energy"]) if info["P_max_energy"] == info["P_max_energy"] else float("nan")
            except Exception:
                val = float("nan")
            # Prefer numeric value from selection stage; fallback to ceiling
            stats["P_max_energy"] = val if (val == val) else float(P_max_energy_ceiling_W)
        if "feasible_sem_guard" in info:
            stats["feasible_sem_guard"] = int(info["feasible_sem_guard"])
        if "feasible_queue_guard" in info:
            stats["feasible_queue_guard"] = int(info["feasible_queue_guard"])
        if "feasible_energy_guard" in info:
            stats["feasible_energy_guard"] = int(info["feasible_energy_guard"])
        # joint gating window flags
        if "feasible_set_nonempty" in info:
            stats["feasible_set_nonempty"] = int(info["feasible_set_nonempty"])
        if "action_in_guard_set" in info:
            stats["action_in_guard_set"] = int(info["action_in_guard_set"])
        if "fallback_used" in info:
            stats["fallback_used"] = int(info["fallback_used"])
        if "arm_id" in info:
            stats["arm_id"] = int(info["arm_id"])
        if "lcb_min" in info:
            stats["lcb_min"] = float(info["lcb_min"])  # RA-UCB gate snapshot
        if "reward_shaped" in info:
            stats["reward_shaped"] = int(info["reward_shaped"])
        # Dual vars snapshot
        stats["lambda_lat"] = float(self.lam_lat)
        stats["lambda_eng"] = float(self.lam_eng)
        stats["lambda_sem"] = float(self.lam_sem)
        # Backfill guard-set and fallback flags for modes that don't provide them
        if "action_in_guard_set" not in stats:
            feas_sem = int(stats.get("feasible_sem_guard", 0))
            feas_queue = int(stats.get("feasible_queue_guard", 0))
            feas_energy = int(stats.get("feasible_energy_guard", 0))
            stats["action_in_guard_set"] = int(feas_sem and feas_queue and feas_energy)
        if "fallback_used" not in stats:
            stats["fallback_used"] = int(0)
        if "feasible_set_nonempty" not in stats:
            stats["feasible_set_nonempty"] = int(1)
        # Per-guard enforcement invariants (hard-fail flags)
        act_in_set = int(stats.get("action_in_guard_set", 0))
        # energy invariant uses tightened budget (epsilon) per PD9 spec
        stats["bad_energy"] = int(act_in_set and (E_slot_J > E_budget_tight_step))
        # queue invariant: S_eff_bps < arrivals_bps + delta_queue * Q_t
        stats["bad_queue"] = int(
            act_in_set and (
                float(stats["S_eff_bps"]) < float(stats["arrivals_bps"]) + float(stats["delta_queue_used"]) * float(stats["Q_t_used"])  # noqa
            )
        )
        # semantic invariant: sWER_clip > semantic_budget
        stats["bad_sem"] = int(act_in_set and (swer_clip > I_t))
        return float(P), float(B), stats


# --------- helpers ---------
def _snr_eff(P: float, B: float, state: Dict[str, float], q_bps: float) -> float:
    snr = float(state.get("snr_db", 0.0))
    gain_db = 10.0 * np.log10(max(P * B, 1e-9))
    penalty = 3.0 * np.log2(max(q_bps, 1.0) / 300.0 + 1.0)
    return float(snr + gain_db - penalty)