"""Task1 runner: Lyapunov DPP + optional bandit inner.

Reads configs/task1_basic.yaml and runs an episode over channel trace,
logging CSV with columns:
  t, q_bps, z_lat, z_eng, z_sem, P, B, rate, per, mos, sWER, [arm_id]

Supports modes:
- continuous: grid over B and 1D P search
- bandit: discrete arms with UCB variants and RA-UCB filter
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from .dpp_core import DPPConfig, DPPController
from .utils_seed import set_all_seeds


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(base: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    cand = (base / p).resolve()
    if cand.exists():
        return cand
    # try project root
    return (base.parent / p).resolve()


def _adapt_config(raw: Dict[str, object]) -> Dict[str, object]:
    """Normalize keys to unified schema required by runner/controller.

    Unified keys:
    - T, slot_sec
    - arrivals_bps
    - energy_budget_per_slot_j
    - latency_budget_q_bps
    - semantic_budget
    - V
    - scales: {queue_scale, energy_scale, semantic_scale}
    """
    mode = str(raw.get("mode", "continuous")).lower()
    T = int(raw.get("T", 0))
    slot_sec = float(raw.get("slot_sec", 0.02))
    arrivals_bps = float(raw.get("arrivals_bps", raw.get("A_bps", raw.get("q_bps", 600.0))))
    energy_budget_per_slot_j = float(raw.get("energy_budget_per_slot_j", raw.get("E_per_slot", 0.5)))
    semantic_budget = float(raw.get("semantic_budget", raw.get("I_target", 0.20)))
    # latency proxy in q_bps units
    latency_budget_q_bps = float(raw.get("latency_budget_q_bps", raw.get("q_bps", arrivals_bps)))
    V = float(raw.get("V", 100.0))
    scales = {
        "queue_scale": float((raw.get("scales") or {}).get("queue_scale", raw.get("Q_scale", 2.0e4))),
        "energy_scale": float((raw.get("scales") or {}).get("energy_scale", raw.get("J_scale", 5.0e2))),
        "semantic_scale": float((raw.get("scales") or {}).get("semantic_scale", raw.get("S_scale", 1.0))),
    }
    # --- Unify config keys ---
    # Primal-dual enable flags (compat: enable_pd / pd.enable / use_pd)
    pd_block = (raw.get("pd") or {})
    enable_primal_dual = bool(
        raw.get("enable_primal_dual",
                raw.get("enable_pd",
                        pd_block.get("enable", raw.get("use_pd", False))))
    )
    # Guard flags (compat: use_semantic_guard / use_queue_guard)
    guard_block = (raw.get("guard") or {})
    guard_semantic = bool(guard_block.get("semantic", raw.get("use_semantic_guard", False)))
    guard_queue = bool(guard_block.get("queue", raw.get("use_queue_guard", False)))
    guard_energy = bool(guard_block.get("energy", raw.get("use_energy_guard", False)))
    # PD params (compat: pd_eta / eta, pd_delta_queue / delta_queue)
    pd_eta = float(pd_block.get("eta", raw.get("pd_eta", raw.get("eta", raw.get("eta_dual", 5e-3)))))
    pd_delta_queue = float(pd_block.get("delta_queue", raw.get("pd_delta_queue", raw.get("delta_queue", 0.08))))
    # PD lambda_init (optional): support pd.lambda_init.{lat,eng,sem} and legacy keys
    lam_init_block = (pd_block.get("lambda_init") or raw.get("lambda_init") or {})
    lam_init_lat = lam_init_block.get("lat", lam_init_block.get("latency", raw.get("lam_init_lat", None)))
    lam_init_eng = lam_init_block.get("eng", lam_init_block.get("energy", raw.get("lam_init_eng", None)))
    lam_init_sem = lam_init_block.get("sem", lam_init_block.get("semantic", raw.get("lam_init_sem", None)))
    # PD clip ceilings (optional)
    clip_block = (pd_block.get("clip") or raw.get("clip") or {})
    lam_clip_lat = clip_block.get("lat", clip_block.get("latency", raw.get("lam_clip_lat", None)))
    lam_clip_eng = clip_block.get("eng", clip_block.get("energy", raw.get("lam_clip_eng", None)))
    lam_clip_sem = clip_block.get("sem", clip_block.get("semantic", raw.get("lam_clip_sem", None)))
    # Normalize PD params for controller mapping
    eta_dual = float(pd_eta)
    delta_queue = float(pd_delta_queue)
    return {
        **raw,
        "mode": mode,
        "T": T,
        "slot_sec": slot_sec,
        "arrivals_bps": arrivals_bps,
        "energy_budget_per_slot_j": energy_budget_per_slot_j,
        "semantic_budget": semantic_budget,
        "latency_budget_q_bps": latency_budget_q_bps,
        "V": V,
        "scales": scales,
        "eta_dual": eta_dual,
        "delta_queue": delta_queue,
        "lam_init_lat": float(lam_init_lat) if lam_init_lat is not None else None,
        "lam_init_eng": float(lam_init_eng) if lam_init_eng is not None else None,
        "lam_init_sem": float(lam_init_sem) if lam_init_sem is not None else None,
        "lam_clip_lat": float(lam_clip_lat) if lam_clip_lat is not None else None,
        "lam_clip_eng": float(lam_clip_eng) if lam_clip_eng is not None else None,
        "lam_clip_sem": float(lam_clip_sem) if lam_clip_sem is not None else None,
        # unified keys for downstream
        "enable_primal_dual": enable_primal_dual,
        "pd": {"eta": pd_eta, "delta_queue": pd_delta_queue, "lambda_init": {
            **({"lat": float(lam_init_lat)} if lam_init_lat is not None else {}),
            **({"eng": float(lam_init_eng)} if lam_init_eng is not None else {}),
            **({"sem": float(lam_init_sem)} if lam_init_sem is not None else {}),
        }, "clip": {
            **({"lat": float(lam_clip_lat)} if lam_clip_lat is not None else {}),
            **({"eng": float(lam_clip_eng)} if lam_clip_eng is not None else {}),
            **({"sem": float(lam_clip_sem)} if lam_clip_sem is not None else {}),
        }},
        "guard": {"semantic": guard_semantic, "queue": guard_queue, "energy": guard_energy, "epsilon_energy": float((guard_block.get("epsilon_energy", raw.get("epsilon_energy", 0.0))))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    # Simple key=value overrides, e.g., --override T=300 V=6
    ap.add_argument("--override", type=str, nargs="*", default=[], help="Override config keys (key=value)")
    args = ap.parse_args()

    cfg_raw = _load_yaml(args.config)
    # Apply overrides if provided (flat keys only)
    if args.override:
        for kv in args.override:
            if "=" in kv:
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Try to cast numbers
                try:
                    if "." in v or "e" in v.lower():
                        cfg_raw[k] = float(v)
                    else:
                        cfg_raw[k] = int(v)
                except Exception:
                    # Fallback to string
                    cfg_raw[k] = v
    cfg = _adapt_config(cfg_raw)
    # Seed RNG for reproducibility
    try:
        seed = int(cfg_raw.get("seed", 2025))
    except Exception:
        seed = 2025
    set_all_seeds(seed)
    mode = cfg["mode"]
    outdir = Path(cfg.get("outdir", "outputs"))
    os.makedirs(outdir / "dumps", exist_ok=True)

    # Trace: channel SNR
    trace_path = _resolve_path(args.config.parent, Path(cfg.get("trace", "semntn/data/channel_trace.csv")))
    df_trace = pd.read_csv(trace_path)
    if "snr_db" not in df_trace.columns:
        raise ValueError("trace CSV must contain 'snr_db' column")
    snr_db = df_trace["snr_db"].to_numpy(dtype=float)

    # Episode length & slot
    T = int(cfg.get("T", len(snr_db)))
    slot_sec = float(cfg.get("slot_sec", 0.02))

    # Arrivals (A_t), Energy budget (G_t), Semantic target (I_t)
    arrivals_bps = float(cfg.get("arrivals_bps", 600.0))
    energy_budget_per_slot_j = float(cfg.get("energy_budget_per_slot_j", 0.5))
    semantic_budget = float(cfg.get("semantic_budget", 0.20))  # max sWER target
    A_bits = arrivals_bps * slot_sec

    # DPP config
    scales = cfg.get("scales", {})
    dpp_cfg = DPPConfig(
        mode=mode,
        V=float(cfg.get("V", 100.0)),
        Q_scale=float(scales.get("queue_scale", cfg.get("Q_scale", 2.0e4))),
        J_scale=float(scales.get("energy_scale", cfg.get("J_scale", 5.0e2))),
        S_scale=float(scales.get("semantic_scale", cfg.get("S_scale", 1.0))),
        lam_init_multiplier=float(cfg.get("lam_init_multiplier", 1.0)),
        lam_init_lat=cfg.get("lam_init_lat", None),
        lam_init_eng=cfg.get("lam_init_eng", None),
        lam_init_sem=cfg.get("lam_init_sem", None),
        lam_clip_lat=cfg.get("lam_clip_lat", None),
        lam_clip_eng=cfg.get("lam_clip_eng", None),
        eta_dual=float(cfg.get("eta_dual", 5e-3)),
        queue_delta=float(cfg.get("delta_queue", 0.08)),
        B_grid=tuple(cfg.get("B_grid", [1, 2, 3])),
        P_min=float(cfg.get("P_min", 0.2)),
        P_max=float(cfg.get("P_max", 1.6)),
        P_grid=int(cfg.get("P_grid", 25)),
        # Accept arms as either (q_bps,P,B) or (P,B,q_bps)
        arms=[
            (
                float(a.get("q_bps", a.get("q", a.get("Q", a.get("rate", 0.0))))
                      or float(a.get("q_bps", 0.0))
                ),
                float(a.get("P", a.get("power", 0.0))),
                int(a.get("B", a.get("bw", a.get("backoff", 1))))
            ) for a in cfg.get("arms", [])
        ] or None,
        bandit_kwargs=cfg.get("bandit", {}),
    )
    ctrl = DPPController(dpp_cfg)

    # Fixed codec rate for continuous mode
    fixed_q = float(cfg.get("q_bps", cfg.get("latency_budget_q_bps", 1200.0)))

    # Physical search/grid diagnostics
    B_list = sorted(list(cfg.get("B_grid", list(dpp_cfg.B_grid))))
    B_count = int(len(B_list))
    B_min_val = float(min(B_list)) if B_count > 0 else float("nan")
    P_max_val = float(cfg.get("P_max", dpp_cfg.P_max))

    # Unified physical keys for printing/resolved
    # link.P_max_dBm: if present, +6 dB; else convert from linear P_max (W) to dBm and +6 dB
    link_block = (cfg_raw.get("link") or {})
    pmax_dbm_src = link_block.get("P_max_dBm")
    pmax_lin_src = link_block.get("P_max", cfg_raw.get("P_max", None))
    def _lin_to_dbm(p_w: float) -> float:
        try:
            return float(10.0 * np.log10(max(float(p_w) * 1000.0, 1e-12)))
        except Exception:
            return float("nan")
    def _dbm_to_lin(p_dbm: float) -> float:
        try:
            return float(10.0 ** (float(p_dbm) / 10.0) / 1000.0)
        except Exception:
            return float("nan")
    if pmax_dbm_src is not None:
        # Use provided dBm ceiling directly; convert to linear for controller
        try:
            P_max_dBm = float(pmax_dbm_src)
            P_max_val = float(_dbm_to_lin(P_max_dBm))
        except Exception:
            P_max_dBm = _lin_to_dbm(P_max_val)
    elif pmax_lin_src is not None:
        # Only convert linear Watts to dBm; no auto expansion
        try:
            P_max_val = float(pmax_lin_src)
        except Exception:
            P_max_val = float(cfg.get("P_max", dpp_cfg.P_max))
        P_max_dBm = _lin_to_dbm(P_max_val)
    else:
        # Fallback: derive from current controller value
        P_max_dBm = _lin_to_dbm(P_max_val)

    # search.B_min_kHz: from raw search.B_min (unknown unit) → use 0.5× the value; ensure numeric
    search_block = (cfg_raw.get("search") or {})
    # Prefer explicit kHz; if legacy B_min provided, convert by ×0.5 as requested
    if "B_min_kHz" in search_block:
        try:
            B_min_kHz = float(search_block.get("B_min_kHz"))
        except Exception:
            B_min_kHz = float(B_min_val) if B_count > 0 else float("nan")
    else:
        bmin_src = search_block.get("B_min", None)
        try:
            B_min_kHz = float(bmin_src) * 0.5 if bmin_src is not None else float(B_min_val) if B_count > 0 else float("nan")
        except Exception:
            B_min_kHz = float(B_min_val) if B_count > 0 else float("nan")
    # search.B_grid_k: discrete grid count, at least 16
    bgrid_k_src = search_block.get("B_grid_k", search_block.get("B_grid", None))
    try:
        B_grid_k = int(max(16, int(bgrid_k_src))) if bgrid_k_src is not None else int(max(16, B_count))
    except Exception:
        B_grid_k = int(max(16, B_count))
    # If no explicit B_grid provided, synthesize a simple 1..K list to expand coverage
    if not B_list:
        B_list = list(range(1, int(B_grid_k) + 1))
        B_count = len(B_list)
        B_min_val = float(min(B_list))

    # First-line print of unified physical keys
    print(f"[PHYS] P_max_dBm={P_max_dBm:.2f}, B_min_kHz={B_min_kHz:.3f}, B_grid_k={int(B_grid_k)}")
    # One-line summary of resolved config (units) with unified flags
    print(
        f"[CFG] T={T} slots, slot_sec={slot_sec:.3f}s, arrivals_bps={arrivals_bps:.1f}, "
        f"energy_budget_per_slot_j={energy_budget_per_slot_j:.3f} J/slot, "
        f"semantic_budget={semantic_budget:.3f}, V={cfg.get('V')}, "
        f"scales={{Q:{dpp_cfg.Q_scale:.1f}, J:{dpp_cfg.J_scale:.1f}, S:{dpp_cfg.S_scale:.1f}}}, "
        f"enable_primal_dual={bool(cfg.get('enable_primal_dual', False))}, "
        f"guard.semantic={bool((cfg.get('guard') or {}).get('semantic', False))}, "
        f"guard.queue={bool((cfg.get('guard') or {}).get('queue', False))}, "
        f"guard.energy={bool((cfg.get('guard') or {}).get('energy', False))}, "
        f"pd.eta={float((cfg.get('pd') or {}).get('eta', cfg.get('eta_dual', 5e-3))):.5f}, "
        f"pd.delta_queue={float((cfg.get('pd') or {}).get('delta_queue', cfg.get('delta_queue', 0.08))):.3f}, "
        f"pd.clip.lat={(cfg.get('lam_clip_lat', '[NONE]'))}, pd.clip.eng={(cfg.get('lam_clip_eng', '[NONE]'))}, "
        f"P_max={P_max_val:.3f}, B_min={(B_min_val if B_count>0 else float('nan')):.3f}, |B_grid|={(B_count if B_count>0 else 0)}"
    )
    if mode == "bandit":
        lcb_min = float((cfg.get("bandit") or {}).get("lcb_min", 0.0))
        print(f"[RA-UCB] lcb_min={lcb_min:.3f}")

    # Persist resolved config snapshot alongside logs
    cfg_name = args.config.stem
    resolved = {
        "T": T,
        "slot_sec": slot_sec,
        "arrivals_bps": arrivals_bps,
        "energy_budget_per_slot_j": energy_budget_per_slot_j,
        "latency_budget_q_bps": float(cfg.get("latency_budget_q_bps", fixed_q)),
        "semantic_budget": semantic_budget,
        "V": float(cfg.get("V", 100.0)),
        "scales": {
            "queue_scale": float(scales.get("queue_scale", dpp_cfg.Q_scale)),
            "energy_scale": float(scales.get("energy_scale", dpp_cfg.J_scale)),
            "semantic_scale": float(scales.get("semantic_scale", dpp_cfg.S_scale)),
        },
        "lam_init_multiplier": float(cfg.get("lam_init_multiplier", 1.0)),
        "lam_init_lat": dpp_cfg.lam_init_lat if dpp_cfg.lam_init_lat is not None else "[DEFAULT]",
        "lam_init_eng": dpp_cfg.lam_init_eng if dpp_cfg.lam_init_eng is not None else "[DEFAULT]",
        "lam_init_sem": dpp_cfg.lam_init_sem if dpp_cfg.lam_init_sem is not None else "[DEFAULT]",
        "enable_primal_dual": bool(cfg.get("enable_primal_dual", False)),
        "pd": {
            "eta": float((cfg.get("pd") or {}).get("eta", cfg.get("eta_dual", 5e-3))),
            "delta_queue": float((cfg.get("pd") or {}).get("delta_queue", cfg.get("delta_queue", 0.08))),
            "lambda_init": {
                **({"lat": float(cfg.get("lam_init_lat"))} if cfg.get("lam_init_lat") is not None else {}),
                **({"eng": float(cfg.get("lam_init_eng"))} if cfg.get("lam_init_eng") is not None else {}),
                **({"sem": float(cfg.get("lam_init_sem"))} if cfg.get("lam_init_sem") is not None else {}),
            },
            "clip": {
                **({"lat": float(cfg.get("lam_clip_lat"))} if cfg.get("lam_clip_lat") is not None else {}),
                **({"eng": float(cfg.get("lam_clip_eng"))} if cfg.get("lam_clip_eng") is not None else {}),
                **({"sem": float(cfg.get("lam_clip_sem"))} if cfg.get("lam_clip_sem") is not None else {}),
            },
        },
        "guard": {
            "semantic": bool((cfg.get("guard") or {}).get("semantic", False)),
            "queue": bool((cfg.get("guard") or {}).get("queue", False)),
            "energy": bool((cfg.get("guard") or {}).get("energy", False)),
            "epsilon_energy": float((cfg.get("guard") or {}).get("epsilon_energy", 0.0)),
        },
        # Physical boundaries for acceptance workflows (legacy + unified keys)
        "P_max": float(P_max_val),
        "B_min": float(B_min_val) if B_count > 0 else "[MISSING]",
        "B_grid": list(B_list) if B_count > 0 else [],
        "B_grid_len": int(B_count) if B_count > 0 else "[MISSING]",
        # Unified keys for acceptance/wirecheck
        "P_max_dBm": float(P_max_dBm) if P_max_dBm == P_max_dBm else "[MISSING]",
        "B_min_kHz": float(B_min_kHz) if B_min_kHz == B_min_kHz else "[MISSING]",
        "B_grid_k": int(B_grid_k) if isinstance(B_grid_k, (int, float)) else "[MISSING]",
    }
    resolved_path = outdir / "dumps" / f"{cfg_name}_resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
    print(f"[SAVE] {resolved_path}")
    # Inline RESOLVED summary for acceptance workflows
    print(
        "RESOLVED: "
        f"T={resolved['T']}, slot_sec={resolved['slot_sec']}, arrivals_bps={resolved['arrivals_bps']}, "
        f"energy_budget_per_slot_j={resolved['energy_budget_per_slot_j']}, latency_budget_q_bps={resolved['latency_budget_q_bps']}, "
        f"semantic_budget={resolved['semantic_budget']}, V={resolved['V']}, "
        f"scales={{queue:{resolved['scales']['queue_scale']}, energy:{resolved['scales']['energy_scale']}, semantic:{resolved['scales']['semantic_scale']}}}"
    )

    rows = []
    # Physical queue bits tracker for service measurement
    Q_bits = 0.0
    for t in range(1, T + 1):
        state = {
            "snr_db": float(snr_db[t - 1]),
            "slot_sec": slot_sec,
            "A_bits": A_bits,
            "energy_budget_per_slot_j": energy_budget_per_slot_j,
            "semantic_budget": semantic_budget,
            "latency_budget_q_bps": float(cfg.get("latency_budget_q_bps", fixed_q)),
            # provide codec rate for penalty in link model
            "q_bps": fixed_q if mode == "continuous" else 0.0,
            # propagate physical minimum bandwidth for identity/unit checks
            "B_min_kHz": float(B_min_kHz),
            # feature toggles
            "enable_primal_dual": bool(cfg.get("enable_primal_dual", False)),
            "guard_semantic": bool((cfg.get("guard") or {}).get("semantic", False)),
            "guard_queue": bool((cfg.get("guard") or {}).get("queue", False)),
            "guard_energy": bool((cfg.get("guard") or {}).get("energy", False)),
            "epsilon_energy": float((cfg.get("guard") or {}).get("epsilon_energy", 0.0)),
        }
        P, B, stats = ctrl.step(state)
        # Next physical queue (bits)
        S_bits = float(stats.get("S_bits", 0.0))
        
        # CRITICAL FIX: Use B_eff_Hz from guard stats (post-projection value)
        # This ensures B_eff_Hz is numerically stable and tied to emitted action
        B_eff_Hz = float(stats.get("B_eff_Hz", 0.0))
        
        # Enforce non-NaN via np.nan_to_num as required
        B_eff_Hz = float(np.nan_to_num(B_eff_Hz, nan=B_min_kHz*1000.0))
        # Effective service rate (bps): use model service capacity q_bps (pre-PER)
        S_eff_bps = float(stats.get("q_bps", stats.get("rate", 0.0)))
        Q_next_bits = float(max(Q_bits + A_bits - S_bits, 0.0))
        # Strong validation: PD / Guard columns
        pd_active = 0
        if bool(cfg.get("enable_primal_dual", False)):
            has_pd = all(k in stats for k in ["lambda_lat","lambda_eng","lambda_sem"]) and \
                      all(np.isfinite(float(stats[k])) for k in ["lambda_lat","lambda_eng","lambda_sem"])
            if not has_pd:
                print(f"[ERROR] t={t} PD columns missing/invalid")
            pd_active = int(has_pd)
        guard_sem_active = 0
        guard_queue_active = 0
        guard_energy_active = 0
        if bool((cfg.get("guard") or {}).get("semantic", False)):
            has_sem = all(k in stats for k in ["sinr_req_lin","P_min_sem","feasible_sem_guard"]) and \
                      np.isfinite(float(stats.get("sinr_req_lin", np.nan))) and \
                      np.isfinite(float(stats.get("P_min_sem", np.nan))) and \
                      (int(stats.get("feasible_sem_guard", 0)) in (0,1))
            if not has_sem:
                print(f"[ERROR] t={t} semantic guard columns missing/invalid")
            guard_sem_active = int(has_sem)
        if bool((cfg.get("guard") or {}).get("queue", False)):
            has_queue = all(k in stats for k in ["P_min_queue","feasible_queue_guard"]) and \
                        np.isfinite(float(stats.get("P_min_queue", np.nan))) and \
                        (int(stats.get("feasible_queue_guard", 0)) in (0,1))
            if not has_queue:
                print(f"[ERROR] t={t} queue guard columns missing/invalid")
            guard_queue_active = int(has_queue)
        if bool((cfg.get("guard") or {}).get("energy", False)):
            has_energy = ("feasible_energy_guard" in stats) and (int(stats.get("feasible_energy_guard", 0)) in (0,1))
            if not has_energy:
                print(f"[ERROR] t={t} energy guard columns missing/invalid")
            guard_energy_active = int(has_energy)
        guard_active = int(min(guard_sem_active, guard_queue_active, guard_energy_active))
        # Instantaneous queue violation (alias existing per-slot latency violation)
        queue_violation = int(stats.get("violate_lat", 0))
        row = {
            "t": t,
            **stats,
            "slot_sec": float(slot_sec),
            "A_bits": float(A_bits),
            "energy_budget_per_slot_j": float(energy_budget_per_slot_j),
            "latency_budget_q_bps": float(cfg.get("latency_budget_q_bps", fixed_q)),
            "B_min_kHz": float(B_min_kHz) if B_min_kHz == B_min_kHz else float("nan"),
            "epsilon_energy": float((cfg.get("guard") or {}).get("epsilon_energy", 0.0)),
            "Q_bits": float(Q_bits),
            "Q_next_bits": float(Q_next_bits),
            "B_eff_Hz": float(B_eff_Hz),
            "S_eff_bps": float(S_eff_bps),
            "queue_violation": int(queue_violation),
            "pd_active": int(pd_active),
            "guard_sem_active": int(guard_sem_active),
            "guard_queue_active": int(guard_queue_active),
            "guard_energy_active": int(guard_energy_active),
            "guard_active": int(guard_active),
        }
        rows.append(row)
        Q_bits = Q_next_bits

    log = pd.DataFrame(rows)
    # Name log after config file stem for clarity
    log_path = outdir / "dumps" / f"{cfg_name}_log.csv"
    log.to_csv(log_path, index=False)
    print(f"[SAVE] {log_path}")


if __name__ == "__main__":
    main()