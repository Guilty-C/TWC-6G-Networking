# Task3 V3 Figures Problem Report

## Fig01 – Regret vs Time
- **Spec summary:** cumulative regret over aligned time across algorithms.
- **Implementation summary:** uses per-slot mean QoE, aligns on common integer t, regret vs best at each t, cumulative sum.
- **Status:** SUSPICIOUS
- **Fixes applied:** Added debug export of computed regret series; ensured empty/column checks write diagnostics; retained alignment logic.
- **Remaining risks:** regret definition tied to per-time best not oracle; heterogeneous horizons possible.

## Fig02 – QoE vs V
- **Spec summary:** steady-state QoE (\bar{U}) as function of V for each algorithm.
- **Implementation summary:** filters to known algorithms, enforces shared (window, grid_density) scenario across algorithms, steady-state stats with gentler burn/smoothing; debug includes aggregate and run-level rows.
- **Status:** SUSPICIOUS (data coverage unknown)
- **Fixes applied:** added shared-scenario selector, lighter steady-state smoothing, comprehensive debug CSV.
- **Remaining risks:** still depends on filename metadata; if no common scene, falls back with warning.

## Fig03 – Queue/Energy vs V
- **Spec summary:** steady-state queue length and energy vs V per algorithm under common scenario.
- **Implementation summary:** same shared scenario as Fig02; extracts queue/energy metrics, steady-state stats with debug for both metrics and runs.
- **Status:** SUSPICIOUS
- **Fixes applied:** shared scenario enforcement, debug outputs, steadier smoothing.
- **Remaining risks:** energy/queue inference may mix units; missing columns skip algorithms.

## Fig04 – Parameter Sensitivity
- **Spec summary:** QoE sensitivity to parameters (window, grid, epsilon, L_P, L_B).
- **Implementation summary:** computes max–min QoE delta per parameter; debug CSV records deltas.
- **Status:** BROKEN (conceptual mixing remains)
- **Fixes applied:** diagnostics when data missing; debug export.
- **Remaining risks:** mixes algorithms/scenes; no interaction handling.

## Fig05 – Nonstationary Robustness
- **Spec summary:** regret vs nonstationarity proxy.
- **Implementation summary:** proxies from SNR drift or metadata, regret approximations, debug records per-run proxy/regret.
- **Status:** SUSPICIOUS
- **Fixes applied:** debug CSV, warnings on missing data.
- **Remaining risks:** proxy/regret definitions ad hoc.

## Fig06 – Latency CDF
- **Spec summary:** decision latency CDF per algorithm.
- **Implementation summary:** selects first latency-like column, plots CDF; debug captures latency/CDF pairs.
- **Status:** OK (pending data)
- **Fixes applied:** debug output, better empty handling.
- **Remaining risks:** column inference heuristic.

## Fig07 – SNR–PER Heatmap
- **Spec summary:** PER across bandwidth/SNR grid (phys scan).
- **Implementation summary:** reads phys scan CSV, pivots PER, debug writes grouped means.
- **Status:** SUSPICIOUS (requires external file)
- **Fixes applied:** debug output and missing-file diagnostics.
- **Remaining risks:** file absent; no validation of units.

## Fig08 – P/B Selection Heatmap
- **Spec summary:** frequency heatmap of power-bandwidth choices.
- **Implementation summary:** bins P/B to 0.1, counts occurrences; debug exports grouped counts.
- **Status:** SUSPICIOUS
- **Fixes applied:** debug output, missing-column checks.
- **Remaining risks:** mixes algorithms/scenes; no normalization by time.

## Fig09 – sWER vs SNR
- **Spec summary:** semantic error versus channel SNR per algorithm.
- **Implementation summary:** bins SNR into 40 bins, plots mean sWER; debug captures bin mids and means.
- **Status:** OK (pending data)
- **Fixes applied:** debug output, clearer empty handling.
- **Remaining risks:** bin choice arbitrary.

## Fig10 – QoE–sWER Frontier
- **Spec summary:** Pareto frontier between QoE and sWER across algorithms.
- **Implementation summary:** per-alg mean QoE/sWER, cumulative max frontier; debug exports aggregated stats.
- **Status:** SUSPICIOUS
- **Fixes applied:** debug CSV, missing-data diagnostics.
- **Remaining risks:** mixes scenarios; frontier based on means only.

## Fig11 – Violation over Time
- **Spec summary:** constraint violation probability versus time.
- **Implementation summary:** averages violation flags per integer t, rolling smoothing; debug exports smoothed series.
- **Status:** BROKEN (mixes algorithms/constraints)
- **Fixes applied:** debug output, warnings for missing columns.
- **Remaining risks:** still aggregates across algorithms; smoothing parameters heuristic.

## Fig12 – Energy Efficiency Bar
- **Spec summary:** QoE per unit energy per algorithm.
- **Implementation summary:** derives efficiency from raw data or summary CSV; [instrumentation pending minimal changes].
- **Status:** SUSPICIOUS
- **Fixes applied:** none beyond existing logic (debug TBD).
- **Remaining risks:** energy units inconsistent; no common scenario.

## Fig13 – QoE by Semantic Weight
- **Spec summary:** QoE grouped by semantic weight bins per algorithm.
- **Implementation summary:** quantile bins of sem_weight/sWER fallback; [instrumentation pending minimal changes].
- **Status:** SUSPICIOUS
- **Fixes applied:** none beyond diagnostics noted previously.
- **Remaining risks:** mixes algorithms; semantic proxy may be wrong.

## Fig14 – Ridgeline
- **Spec summary:** MOS distribution across semantic bins.
- **Implementation summary:** KDE per bin; [instrumentation pending minimal changes].
- **Status:** SUSPICIOUS
- **Fixes applied:** none this iteration.
- **Remaining risks:** pooling across algorithms; kernel bandwidth assumptions.

## Fig15 – Semantic Constraint Curve
- **Spec summary:** QoE under semantic violation thresholds.
- **Implementation summary:** MOS averaged under sWER thresholds; [instrumentation pending minimal changes].
- **Status:** BROKEN
- **Fixes applied:** none this iteration.
- **Remaining risks:** ignores algorithm separation; narrow threshold range.

## Fig16 – Recovery Time Bar
- **Spec summary:** recovery time to steady QoE per algorithm.
- **Implementation summary:** rolling MOS recovery time; [instrumentation pending minimal changes].
- **Status:** SUSPICIOUS
- **Fixes applied:** none this iteration.
- **Remaining risks:** relies on monotonic recovery assumption.

## Fig17 – Window Scan
- **Spec summary:** regret versus sliding-window size.
- **Implementation summary:** regret proxy per file, grouped by window; [instrumentation pending minimal changes].
- **Status:** BROKEN
- **Fixes applied:** none this iteration.
- **Remaining risks:** single-alg preference, regret proxy shaky.

## Fig18 – Grid Density vs Regret
- **Spec summary:** regret versus grid density.
- **Implementation summary:** regret proxy per file grouped by grid density; [instrumentation pending minimal changes].
- **Status:** SUSPICIOUS
- **Fixes applied:** none this iteration.
- **Remaining risks:** mixes algorithms; filename-derived metadata fragile.

