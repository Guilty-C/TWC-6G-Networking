"""Project-wide constants for PD9 aggregation and gating.

Centralizes aggregator schema requirements, versioning, tolerances,
and V-scan gating thresholds to keep runners and reports consistent.
"""
from __future__ import annotations

# Aggregator version tag (bump when schema/logic changes)
AGG_VERSION: str = "agg_pd9_v1"

# Warm-up slots to skip when computing rates (if 't' is available)
AGG_WARMUP_SKIP: int = 50

# Identity check tolerance for mismatch detection
AGG_IDENTITY_TOL: float = 1.0e-3

# V-scan summary minimum rows gate
VSCAN_MIN_ROWS: int = 3

# Common fail codes
FAIL_AGG_REQUIRED_COLUMNS: str = "[FAIL_AGG_REQUIRED_COLUMNS]"
FAIL_AGG_NAN: str = "[FAIL_AGG_NAN]"
FAIL_VSCAN_AGG_MISMATCH: str = "[FAIL_VSCAN_AGG_MISMATCH]"
FAIL_VSCAN_ROWS_LT_MIN: str = "[FAIL_VSCAN_ROWS_LT_MIN]"
FAIL_AGG_VERSION_MISMATCH: str = "[FAIL_AGG_VERSION_MISMATCH]"

# Required columns for aggregation (post warm-up)
AGG_REQUIRED_COLS = {
    # Budgets & instantaneous measurements
    "E_slot_J", "energy_budget_per_slot_j",
    "S_eff_bps", "arrivals_bps", "delta_queue_used", "Q_t_used",
    "sWER_clip", "semantic_budget",
    # Physical/unit diagnostics
    "B_eff_Hz", "B_min_kHz", "latency_budget_q_bps",
    # Guard feasibility snapshots and search bounds
    "feasible_energy_guard", "feasible_queue_guard", "feasible_sem_guard",
    "P_max_energy", "P_min_sem", "P_min_queue",
    # Guard-set selection window and enforcement flags
    "action_in_guard_set", "fallback_used",
    "bad_energy", "bad_queue", "bad_sem",
}