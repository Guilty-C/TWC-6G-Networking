import glob
import math
import os
import re
from typing import Iterable, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUT_DIR = os.path.join("outputs", "figures", "task3_v3")
FIG_DPI = 300
MIN_LATENCY_SAMPLES = 20


def _write_debug(df: pd.DataFrame, filename: str, note: str | None = None) -> None:
    """Persist debug data for a figure.

    Always writes a CSV so downstream inspection can see what data were (not)
    used. When ``df`` is empty and a note is provided, a one-row placeholder is
    written instead of omitting the debug file entirely.
    """

    ensure_outdir()
    path = os.path.join(OUT_DIR, filename)
    if df is None:
        df = pd.DataFrame()
    if df.empty and note:
        df = pd.DataFrame({"note": [note]})
    df.to_csv(path, index=False)
    print(f"[DEBUG] wrote {path} (rows={len(df)})")


def ensure_outdir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def infer_tag(filename: str) -> str:
    base = os.path.basename(filename).lower()
    if "raucb_plus" in base or "raucb" in base:
        return "raucb_plus"
    if "safe" in base and "linucb" in base:
        return "safe_linucb"
    if "sw_ucb" in base or "swucb" in base:
        return "sw_ucb"
    if "lagrangian_ppo" in base or "ppo" in base:
        return "lagrangian_ppo"
    if "lyapunov_greedy_oracle" in base or ("oracle" in base and "lyapunov" in base):
        return "lyapunov_greedy_oracle"
    if "oracle" in base:
        return "oracle"
    return "unknown"


def _fill_from_filename(df: pd.DataFrame, col: str, pattern: str, cast) -> pd.DataFrame:
    base = os.path.basename(str(df.attrs.get("_filename", "")))
    m = re.search(pattern, base, re.IGNORECASE)
    if m:
        val = cast(m.group(1))
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    return df


def normalize_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()
    df.attrs["_filename"] = filename
    rename_map = {}
    for col in df.columns:
        low = str(col).lower()
        if low in {"time", "time_slot", "slot", "step", "timestep"}:
            rename_map[col] = "t"
        elif low == "swer":
            rename_map[col] = "sWER"
        elif low in {"pesq", "qoe"}:
            rename_map[col] = "mos"
        elif low in {"bandwidth", "band"}:
            rename_map[col] = "B"
        elif low in {"power", "p_tx"}:
            rename_map[col] = "P"
        elif low in {"lat_ms", "latency", "latency_ms", "decision_ms", "decision_time_ms", "decision_latency_ms"}:
            rename_map[col] = "latency_ms"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "t" not in df.columns:
        df["t"] = np.arange(len(df))
    df["t"] = pd.to_numeric(df["t"], errors="coerce")

    if "file" not in df.columns:
        df["file"] = os.path.basename(filename)
    else:
        df["file"] = df["file"].fillna(os.path.basename(filename))

    df = _fill_from_filename(df, "V", r"V(\d+\.?\d*)", float)
    df = _fill_from_filename(df, "window", r"w(\d+)", int)
    df = _fill_from_filename(df, "grid_density", r"g(\d+)", int)
    df = _fill_from_filename(df, "seed", r"[ _-]s(\d+)", int)

    base = os.path.basename(filename)
    if "algorithm" not in df.columns:
        df["algorithm"] = infer_tag(base)
    else:
        df["algorithm"] = df["algorithm"].fillna(infer_tag(base))

    mos_col = None
    if "mos" in df.columns:
        mos_col = pd.to_numeric(df["mos"], errors="coerce")
        df["mos"] = mos_col
    if "sWER" in df.columns:
        df["sWER"] = pd.to_numeric(df["sWER"], errors="coerce")
    if mos_col is None and "sWER" in df.columns:
        mos_col = 5.0 - 4.0 * df["sWER"]
        df["mos"] = mos_col
    if mos_col is not None:
        df["qoe"] = mos_col.astype(float)
    elif "sWER" in df.columns:
        df["qoe"] = 5.0 - 4.0 * df["sWER"]

    if "Q" not in df.columns and "queue" in df.columns:
        df["Q"] = pd.to_numeric(df["queue"], errors="coerce")
    if "queue" not in df.columns and "Q" in df.columns:
        df["queue"] = df["Q"]
    if "Q" not in df.columns and "q_semantic" in df.columns:
        df["Q"] = pd.to_numeric(df["q_semantic"], errors="coerce")
    if "slot_sec" not in df.columns:
        if "slot_duration_ms" in df.columns:
            df["slot_sec"] = pd.to_numeric(df["slot_duration_ms"], errors="coerce") / 1000.0
        elif "slot_duration" in df.columns:
            df["slot_sec"] = pd.to_numeric(df["slot_duration"], errors="coerce")
    if "E" not in df.columns and "energy" in df.columns:
        df["E"] = pd.to_numeric(df["energy"], errors="coerce")
    if "P" not in df.columns and "power" in df.columns:
        df["P"] = pd.to_numeric(df["power"], errors="coerce")
    if "E" not in df.columns and "energy_cum" in df.columns:
        energy_cum = pd.to_numeric(df["energy_cum"], errors="coerce")
        df["E"] = energy_cum.diff().fillna(energy_cum).clip(lower=0)
    if "E" not in df.columns and "P" in df.columns and "slot_sec" in df.columns:
        df["E"] = pd.to_numeric(df["P"], errors="coerce") * pd.to_numeric(df["slot_sec"], errors="coerce")

    for cand in ["sem_weight", "semantic_weight", "sem_weighting"]:
        if cand in df.columns:
            df["sem_weight"] = pd.to_numeric(df[cand], errors="coerce")
            break

    return df


def load_all_dumps() -> pd.DataFrame:
    patterns = [
        os.path.join("outputs", "dumps", "*.csv"),
        os.path.join("outputs", "dumps", "task3_v3", "**", "*.csv"),
        os.path.join("outputs", "dumps", "task3_v2", "*.csv"),
        os.path.join("task3", "outputs", "dumps", "**", "*.csv"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    if not files:
        print("[WARN] No CSV files found under outputs/dumps/")
        return pd.DataFrame()

    skip_keys = ("summary", "eval", "report", "phys_scan", "resolved", "config")
    dfs = []
    for fp in files:
        base = os.path.basename(fp).lower()
        if any(k in base for k in skip_keys):
            continue
        try:
            df_raw = pd.read_csv(fp)
        except Exception as exc:  # pragma: no cover - defensive for bad files
            print(f"[WARN] Failed to read {fp}: {exc}")
            continue
        df_norm = normalize_columns(df_raw, fp)
        dfs.append(df_norm)
    if not dfs:
        print("[WARN] No valid dumps loaded.")
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(df_all)} rows from {len(dfs)} CSV files.")
    return df_all

def select_best_V_scenario(df_alg: pd.DataFrame) -> pd.DataFrame:
    if df_alg.empty:
        return df_alg
    has_window = "window" in df_alg.columns
    has_grid = "grid_density" in df_alg.columns
    if not has_window and not has_grid:
        alg_name = df_alg["algorithm"].iloc[0] if "algorithm" in df_alg.columns else "unknown"
        coverage = sorted(pd.unique(df_alg["V"].dropna()))
        print(f"[Fig02/03 scenario] alg={alg_name}, picked_scene=all(no scene keys), V_coverage={coverage}")
        return df_alg

    df_alg = df_alg.copy()

    def _is_vsweep_row(row) -> bool:
        if "family" in row and isinstance(row["family"], str) and str(row["family"]).lower() == "vsweep":
            return True
        if "run_tag" in row and isinstance(row["run_tag"], str) and "vsweep" in str(row["run_tag"]).lower():
            return True
        if "file" in row and "vsweep" in str(row["file"]).lower():
            return True
        return False

    df_alg["vsweep_flag"] = df_alg.apply(_is_vsweep_row, axis=1)
    groups = {"non-vsweep": df_alg[~df_alg["vsweep_flag"]], "vsweep": df_alg[df_alg["vsweep_flag"]]}

    def _pick_scene(df_group: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple, List]:
        if df_group.empty:
            return pd.DataFrame(), (), []
        keys = []
        if has_window:
            keys.append("window")
        if has_grid:
            keys.append("grid_density")
        if not keys:
            v_cov = sorted(pd.unique(df_group["V"].dropna()))
            return df_group, (), v_cov
        best = None
        for key_vals, sub in df_group.groupby(keys, dropna=False):
            v_cov = sorted(pd.unique(sub["V"].dropna()))
            entry = (len(v_cov), key_vals, v_cov, sub)
            if best is None or entry[0] > best[0]:
                best = entry
        if best is None:
            return pd.DataFrame(), (), []
        return best[3], best[1], best[2]

    picked = {}
    for label, sub in groups.items():
        scene_df, scene_key, v_cov = _pick_scene(sub)
        picked[label] = (scene_df, scene_key, v_cov)

    cov_non, cov_vs = picked["non-vsweep"][2], picked["vsweep"][2]
    len_non = len(cov_non)
    len_vs = len(cov_vs)
    if len_non >= len_vs:
        scene_df, scene_key, v_cov = picked["non-vsweep"]
        flag = "non-vsweep"
    else:
        scene_df, scene_key, v_cov = picked["vsweep"]
        flag = "vsweep"

    alg_name = df_alg["algorithm"].iloc[0] if "algorithm" in df_alg.columns else "unknown"
    scene_desc = []
    if has_window and scene_key:
        scene_desc.append(f"w={scene_key[0] if has_grid else scene_key}")
    if has_grid and scene_key:
        scene_desc.append(f"g={scene_key[-1]}")
    scene_desc = ",".join(scene_desc) if scene_desc else "all"
    print(f"[Fig02/03 scenario] alg={alg_name}, picked_scene=({scene_desc},{flag}), V_coverage={v_cov}")
    return scene_df.drop(columns=["vsweep_flag"])


def select_common_V_scenario(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Pick a single (window, grid_density) shared across algorithms.

    This enforces a fair cross-algorithm comparison for V sweeps. The chosen
    scenario maximizes the *minimum* V coverage across algorithms; ties are
    broken by the total distinct V count then by alphabetical order of the
    scene key. If no shared scene exists, the function falls back to the input
    dataframe and reports the issue via the metadata dictionary.
    """

    if df.empty:
        return df, {"status": "empty"}
    required = ["algorithm", "V", "window", "grid_density"]
    if any(col not in df.columns for col in required):
        return df, {"status": "missing_scene_keys"}

    work = df.copy()
    work["V"] = pd.to_numeric(work["V"], errors="coerce")
    work = work.dropna(subset=["algorithm", "V", "window", "grid_density"])
    if work.empty:
        return df, {"status": "no_scene_rows"}

    algs = sorted({a for a in work["algorithm"].unique() if str(a).lower() != "unknown"})
    scene_map: dict[str, dict[Tuple[int, int], set]] = {}
    for alg in algs:
        sub = work[work["algorithm"] == alg]
        scenes = {}
        for (w, g), grp in sub.groupby(["window", "grid_density"], dropna=False):
            vset = set(grp["V"].dropna().unique())
            if vset:
                scenes[(int(w), int(g))] = vset
        scene_map[alg] = scenes

    if not scene_map:
        return df, {"status": "no_scene_map"}

    shared_scenes = None
    for scenes in scene_map.values():
        if shared_scenes is None:
            shared_scenes = set(scenes.keys())
        else:
            shared_scenes &= set(scenes.keys())
    if not shared_scenes:
        return df, {"status": "no_shared_scene", "scene_map": {k: list(v.keys()) for k, v in scene_map.items()}}

    def _score_scene(scene: Tuple[int, int]) -> Tuple[int, int, Tuple[int, int]]:
        min_cov = min(len(scene_map[alg].get(scene, set())) for alg in algs)
        total_cov = sum(len(scene_map[alg].get(scene, set())) for alg in algs)
        return (min_cov, total_cov, scene)

    picked_scene = max(shared_scenes, key=_score_scene)
    filtered = work[(work["window"] == picked_scene[0]) & (work["grid_density"] == picked_scene[1])]
    coverage = {
        alg: sorted(scene_map[alg].get(picked_scene, set()))
        for alg in algs
    }
    meta = {"status": "shared", "scene": {"window": picked_scene[0], "grid_density": picked_scene[1]}, "coverage": coverage}
    return filtered, meta


def _steady_state_stats(
    df: pd.DataFrame,
    value_col: str,
    extra_fields: Iterable[str] | None = None,
    burn_ratio: float = 0.2,
    window_frac: float = 0.05,
    min_window: int = 10,
    return_run_details: bool = False,
):
    if df.empty or value_col not in df.columns:
        return (pd.DataFrame(), pd.DataFrame()) if return_run_details else pd.DataFrame()
    extra_fields = list(extra_fields or [])
    cols_needed = ["algorithm", "V", "t", value_col]
    for col in cols_needed:
        if col not in df.columns:
            print(f"[WARN] Missing column {col} for steady-state stats on {value_col}")
            return (pd.DataFrame(), pd.DataFrame()) if return_run_details else pd.DataFrame()

    work = df.copy()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["algorithm", "V", "t", value_col])
    if work.empty:
        return (pd.DataFrame(), pd.DataFrame()) if return_run_details else pd.DataFrame()
    if "seed" not in work.columns:
        work["seed"] = 0
    work["seed"] = work["seed"].fillna(0)
    if "file" not in work.columns:
        work["file"] = "unknown"

    run_cols = ["algorithm", "V", "file", "seed"]
    tmax = work.groupby(run_cols)["t"].max()
    if tmax.empty:
        return (pd.DataFrame(), pd.DataFrame()) if return_run_details else pd.DataFrame()
    tcommon_map = tmax.groupby(level=[0, 1]).min().to_dict()

    records = []
    for (alg, v_val, file, seed), t_max in tmax.items():
        t_common = tcommon_map.get((alg, v_val))
        if t_common is None or not math.isfinite(t_common) or t_common <= 0:
            continue
        burn_start = burn_ratio * t_common
        sub = work[
            (work["algorithm"] == alg)
            & (work["V"] == v_val)
            & (work["file"] == file)
            & (work["seed"] == seed)
        ]
        sub = sub[(sub["t"] >= burn_start) & (sub["t"] <= t_common)]
        sub = sub.sort_values("t")
        vals = sub[value_col].dropna()
        if vals.empty:
            continue
        window = max(min_window, int(window_frac * len(vals)))
        roll = vals.rolling(window=window, min_periods=min_window).mean().dropna()
        if roll.empty:
            continue
        rec = {
            "algorithm": alg,
            "V": float(v_val),
            "mean_seed": roll.mean(),
            "Tcommon": t_common,
            "burn_start": burn_start,
            "window_size": window,
            "file": file,
            "seed": seed,
        }
        for ef in extra_fields:
            if ef in sub.columns:
                rec[ef] = sub[ef].dropna().iloc[0]
        records.append(rec)

    if not records:
        return (pd.DataFrame(), pd.DataFrame()) if return_run_details else pd.DataFrame()
    run_df = pd.DataFrame(records)
    agg = run_df.groupby(["algorithm", "V"]).agg(
        mean=("mean_seed", "mean"),
        std=("mean_seed", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        count=("mean_seed", "count"),
        Tcommon=("Tcommon", "min"),
        burn_start=("burn_start", "min"),
        window_size=("window_size", "max"),
    )
    for ef in extra_fields:
        if ef in run_df.columns:
            agg[ef] = run_df.groupby(["algorithm", "V"])[ef].agg(lambda x: x.dropna().iloc[0])
    agg = agg.reset_index()
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    if return_run_details:
        return agg, run_df
    return agg


def fig_regret_vs_time(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "qoe" not in df.columns or "t" not in df.columns or "algorithm" not in df.columns:
        print("[WARN] fig01: insufficient data for regret_vs_time")
        _write_debug(pd.DataFrame(), "debug_fig01_regret_vs_time.csv", note="missing columns or empty")
        return
    work = df.copy()
    work = work.dropna(subset=["qoe", "t", "algorithm"])
    if work.empty:
        print("[WARN] fig01: no valid rows after dropna")
        _write_debug(pd.DataFrame(), "debug_fig01_regret_vs_time.csv", note="no valid rows after dropna")
        return
    work["t_int"] = pd.to_numeric(work["t"], errors="coerce").astype(int)
    pivot = (
        work.groupby(["t_int", "algorithm"])["qoe"]
        .mean()
        .unstack("algorithm")
        .dropna(how="all")
        .sort_index()
    )
    if pivot.empty or pivot.shape[1] < 1:
        print("[WARN] fig01: pivot empty")
        _write_debug(pivot.reset_index(), "debug_fig01_regret_vs_time.csv", note="pivot empty")
        return
    valid_cols = [c for c in pivot.columns if pivot[c].count() > 10]
    pivot = pivot[valid_cols]
    common_t = pivot.dropna().index
    if len(common_t) == 0:
        print("[WARN] fig01: no common time points to align")
        _write_debug(pivot.reset_index(), "debug_fig01_regret_vs_time.csv", note="no common time points")
        return
    pivot = pivot.loc[common_t]
    best = pivot.max(axis=1)
    plt.figure(figsize=(8, 4))
    debug_rows = []
    for alg in pivot.columns:
        regret = (best - pivot[alg]).clip(lower=0).cumsum()
        plt.plot(pivot.index, regret, label=alg, linewidth=2)
        debug_rows.append(pd.DataFrame({"t": pivot.index, "algorithm": alg, "regret": regret.values}))
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig01_regret_vs_time.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
    _write_debug(debug_df, "debug_fig01_regret_vs_time.csv", note="no regret curves")
    print(f"[INFO] Saved {fig_path}")


def fig_qoe_vs_V(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig02: empty dataframe")
        _write_debug(pd.DataFrame(), "debug_fig02_qoe_vs_V.csv", note="empty dataframe")
        return
    if "qoe" not in df.columns or "V" not in df.columns:
        print("[WARN] fig02: required columns missing (qoe/V)")
        _write_debug(pd.DataFrame(), "debug_fig02_qoe_vs_V.csv", note="missing required columns")
        return
    df = df.dropna(subset=["qoe", "V"])
    df = df[df["algorithm"].notna()]
    df = df[df["algorithm"].str.lower() != "unknown"]
    if df.empty:
        print("[WARN] fig02: no data after filtering algorithms/qoe/V")
        _write_debug(pd.DataFrame(), "debug_fig02_qoe_vs_V.csv", note="no data after filtering")
        return

    plt.figure(figsize=(7, 5))
    debug_rows = []
    plotted = False
    filtered, scene_meta = select_common_V_scenario(df)
    if scene_meta.get("status") != "shared":
        print(f"[WARN] fig02: unable to find shared scenario, status={scene_meta.get('status')}")
    for alg, sub_alg in filtered.groupby("algorithm"):
        stats, run_details = _steady_state_stats(
            sub_alg,
            "qoe",
            extra_fields=["window", "grid_density"],
            return_run_details=True,
        )
        if stats.empty:
            print(f"[WARN] fig02: skip alg={alg} due to empty stats")
            continue
        stats = stats.sort_values("V")
        plt.plot(stats["V"], stats["mean"], marker="o", linewidth=2, label=alg)
        plt.fill_between(
            stats["V"],
            stats["mean"] - stats["ci95"],
            stats["mean"] + stats["ci95"],
            alpha=0.2,
        )
        stats["metric"] = "qoe"
        stats["algorithm"] = alg
        stats["scene_status"] = scene_meta.get("status")
        stats["scene_window"] = scene_meta.get("scene", {}).get("window")
        stats["scene_grid_density"] = scene_meta.get("scene", {}).get("grid_density")
        debug_rows.append(stats)
        if not run_details.empty:
            run_details = run_details.assign(
                scene_status=scene_meta.get("status"),
                scene_window=scene_meta.get("scene", {}).get("window"),
                scene_grid_density=scene_meta.get("scene", {}).get("grid_density"),
            )
            debug_rows.append(run_details.assign(record_type="run_details"))
        plotted = True

    if not plotted:
        print("[WARN] fig02: nothing to plot")
        _write_debug(pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame(), "debug_fig02_qoe_vs_V.csv", note="no plotted data")
        return
    plt.xlabel("V")
    plt.ylabel("QoE (Ubar)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig02_qoe_vs_V.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
    _write_debug(debug_df, "debug_fig02_qoe_vs_V.csv", note="no plotted data")
    print(f"[INFO] Saved {fig_path}")


def _extract_queue_energy(df: pd.DataFrame) -> Tuple[pd.Series | None, pd.Series | None]:
    queue = None
    if "Q" in df.columns:
        queue = pd.to_numeric(df["Q"], errors="coerce")
    elif "queue" in df.columns:
        queue = pd.to_numeric(df["queue"], errors="coerce")
    elif "q_semantic" in df.columns:
        queue = pd.to_numeric(df["q_semantic"], errors="coerce")
    energy = None
    if "E" in df.columns:
        energy = pd.to_numeric(df["E"], errors="coerce")
    elif "energy" in df.columns:
        energy = pd.to_numeric(df["energy"], errors="coerce")
    if energy is None and "energy_cum" in df.columns:
        energy_cum = pd.to_numeric(df["energy_cum"], errors="coerce")
        energy = energy_cum.diff().fillna(energy_cum).clip(lower=0)
    if energy is None and "P" in df.columns:
        slot_sec = None
        if "slot_sec" in df.columns:
            slot_sec = pd.to_numeric(df["slot_sec"], errors="coerce")
        elif "slot_duration_ms" in df.columns:
            slot_sec = pd.to_numeric(df["slot_duration_ms"], errors="coerce") / 1000.0
        if slot_sec is not None:
            energy = pd.to_numeric(df["P"], errors="coerce") * slot_sec
    return queue, energy


def fig_queue_energy_vs_V(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig03: empty dataframe")
        _write_debug(pd.DataFrame(), "debug_fig03_queue_energy_vs_V.csv", note="empty dataframe")
        return
    if "V" not in df.columns or "algorithm" not in df.columns:
        print("[WARN] fig03: required columns missing (V/algorithm)")
        _write_debug(pd.DataFrame(), "debug_fig03_queue_energy_vs_V.csv", note="missing required columns")
        return

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    colors = plt.cm.tab10.colors
    debug_rows = []
    plotted = False
    filtered, scene_meta = select_common_V_scenario(df)
    if scene_meta.get("status") != "shared":
        print(f"[WARN] fig03: unable to find shared scenario, status={scene_meta.get('status')}")
    for idx, (alg, sub_alg) in enumerate(filtered.groupby("algorithm")):
        if str(alg).lower() == "unknown":
            continue
        queue_series, energy_series = _extract_queue_energy(sub_alg)
        if queue_series is not None:
            sub_alg = sub_alg.assign(queue_metric=queue_series)
        if energy_series is not None:
            sub_alg = sub_alg.assign(energy_metric=energy_series)
        if "queue_metric" not in sub_alg.columns or "energy_metric" not in sub_alg.columns:
            print(f"[WARN] fig03: skip alg={alg} due to missing queue/energy columns")
            continue
        stats_q, run_q = _steady_state_stats(
            sub_alg,
            "queue_metric",
            extra_fields=["window", "grid_density"],
            return_run_details=True,
        )
        stats_e, run_e = _steady_state_stats(
            sub_alg,
            "energy_metric",
            extra_fields=["window", "grid_density"],
            return_run_details=True,
        )
        if stats_q.empty and stats_e.empty:
            print(f"[WARN] fig03: skip alg={alg} due to empty stats")
            continue
        color = colors[idx % len(colors)]
        if not stats_q.empty:
            stats_q = stats_q.sort_values("V")
            ax1.plot(stats_q["V"], stats_q["mean"], marker="o", linewidth=2, color=color, label=f"{alg} Qbar")
            ax1.fill_between(
                stats_q["V"],
                stats_q["mean"] - stats_q["ci95"],
                stats_q["mean"] + stats_q["ci95"],
                color=color,
                alpha=0.15,
            )
        if not stats_e.empty:
            stats_e = stats_e.sort_values("V")
            ax2.plot(
                stats_e["V"],
                stats_e["mean"],
                marker="s",
                linewidth=2,
                color=color,
                linestyle="--",
                label=f"{alg} Ebar",
            )
            ax2.fill_between(
                stats_e["V"],
                stats_e["mean"] - stats_e["ci95"],
                stats_e["mean"] + stats_e["ci95"],
                color=color,
                alpha=0.1,
            )
        if not stats_q.empty or not stats_e.empty:
            merged = pd.merge(stats_q, stats_e, on=["algorithm", "V"], how="outer", suffixes=("_queue", "_energy"))
            merged["scene_status"] = scene_meta.get("status")
            merged["scene_window"] = scene_meta.get("scene", {}).get("window")
            merged["scene_grid_density"] = scene_meta.get("scene", {}).get("grid_density")
            debug_rows.append(merged)
            for run_df, tag in [(run_q, "queue_run"), (run_e, "energy_run")]:
                if not run_df.empty:
                    debug_rows.append(
                        run_df.assign(
                            record_type=tag,
                            scene_status=scene_meta.get("status"),
                            scene_window=scene_meta.get("scene", {}).get("window"),
                            scene_grid_density=scene_meta.get("scene", {}).get("grid_density"),
                        )
                    )
            plotted = True

    if not plotted:
        print("[WARN] fig03: nothing plotted")
        return
    ax1.set_xlabel("V")
    ax1.set_ylabel("Qbar")
    ax2.set_ylabel("Ebar")
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="best")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig03_queue_energy_vs_V.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
    _write_debug(debug_df, "debug_fig03_queue_energy_vs_V.csv", note="no plotted data")
    print(f"[INFO] Saved {fig_path}")


def fig_param_sensitivity(summary_csv: str, df_all: pd.DataFrame | None = None) -> None:
    ensure_outdir()
    df_source = df_all if df_all is not None and not df_all.empty else None
    if df_source is None and summary_csv and os.path.exists(summary_csv):
        try:
            df_source = pd.read_csv(summary_csv)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] fig04: cannot read summary {summary_csv}: {exc}")
            return
    if df_source is None or df_source.empty:
        print("[WARN] fig04: no data available for param sensitivity")
        _write_debug(pd.DataFrame(), "debug_fig04_param_sensitivity.csv", note="no data available")
        return
    df_work = df_source.copy()
    if "file" not in df_work.columns and "file" in df_source.columns:
        df_work["file"] = df_source["file"]

    def _maybe_fill(col: str, pattern: str, cast) -> None:
        if col in df_work.columns and df_work[col].notna().any():
            return
        if "file" not in df_work.columns:
            return
        vals = []
        for f in df_work["file"]:
            m = re.search(pattern, os.path.basename(str(f)), re.IGNORECASE)
            vals.append(cast(m.group(1)) if m else np.nan)
        df_work[col] = vals

    _maybe_fill("window", r"w(\d+)", int)
    _maybe_fill("grid_density", r"g(\d+)", int)
    _maybe_fill("epsilon_t", r"eps(?:ilon)?(\d+\.?\d*)", float)
    _maybe_fill("L_P", r"lp(\d+\.?\d*)", float)
    _maybe_fill("L_B", r"lb(\d+\.?\d*)", float)

    if "qoe" not in df_work.columns and "mos_mean" in df_work.columns:
        df_work["qoe"] = df_work["mos_mean"]
    if "qoe" not in df_work.columns and "mos" in df_work.columns:
        df_work["qoe"] = df_work["mos"]

    params = ["window", "grid_density", "epsilon_t", "L_P", "L_B"]
    deltas = {}
    for p in params:
        if p in df_work.columns and df_work[p].notna().sum() > 1:
            grouped = df_work.dropna(subset=[p, "qoe"])
            if grouped.empty:
                continue
            means = grouped.groupby(p)["qoe"].mean()
            if len(means) > 1:
                deltas[p] = means.max() - means.min()
    if not deltas:
        print("[WARN] fig04: no parameter with multiple values to analyze")
        _write_debug(pd.DataFrame(), "debug_fig04_param_sensitivity.csv", note="no parameter with multiple values")
        return

    plt.figure(figsize=(6, 4))
    names = list(deltas.keys())
    vals = [deltas[k] for k in names]
    _write_debug(pd.DataFrame({"parameter": names, "delta": vals}), "debug_fig04_param_sensitivity.csv")
    plt.bar(names, vals, color="#4C72B0")
    plt.ylabel("Delta Ubar")
    plt.xlabel("parameter")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig04_param_sensitivity.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

def fig_nonstationary_robustness(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig05: empty dataframe")
        _write_debug(pd.DataFrame(), "debug_fig05_nonstationary_robustness.csv", note="empty dataframe")
        return
    if "algorithm" not in df.columns:
        print("[WARN] fig05: missing algorithm column")
        _write_debug(pd.DataFrame(), "debug_fig05_nonstationary_robustness.csv", note="missing algorithm column")
        return

    records = []
    seed_series = df["seed"] if "seed" in df.columns else pd.Series([0] * len(df))
    for (alg, file, seed), sub in df.groupby(["algorithm", "file", seed_series]):
        proxy = None
        if "snr_db" in sub.columns:
            snr = pd.to_numeric(sub["snr_db"], errors="coerce").dropna()
            if len(snr) > 1:
                proxy = np.mean(np.abs(np.diff(snr)))
        if proxy is None and "drift_level" in sub.columns:
            drift = pd.to_numeric(sub["drift_level"], errors="coerce").dropna()
            if not drift.empty:
                proxy = drift.max() - drift.min()
        if proxy is None and "V_T" in sub.columns:
            proxy = pd.to_numeric(sub["V_T"], errors="coerce").dropna().mean()
        if proxy is None and "S" in sub.columns:
            proxy = pd.to_numeric(sub["S"], errors="coerce").dropna().mean()
        if proxy is None:
            continue

        regret = None
        if "regret_t" in sub.columns:
            regret = pd.to_numeric(sub["regret_t"], errors="coerce").dropna()
            if not regret.empty:
                regret = regret.iloc[-1]
        if regret is None and "instant_regret" in sub.columns:
            regret = pd.to_numeric(sub["instant_regret"], errors="coerce").dropna().sum()
        if regret is None and "qoe" in sub.columns:
            qoe_vals = pd.to_numeric(sub["qoe"], errors="coerce").dropna()
            if not qoe_vals.empty:
                best = qoe_vals.max()
                regret = (best - qoe_vals).clip(lower=0).sum()
        if regret is None:
            continue
        records.append({"algorithm": alg, "proxy": proxy, "regret": regret})

    if not records:
        print("[WARN] fig05: no records for nonstationary robustness")
        _write_debug(pd.DataFrame(), "debug_fig05_nonstationary_robustness.csv", note="no records")
        return
    df_rec = pd.DataFrame(records)
    plt.figure(figsize=(7, 4))
    for alg, sub in df_rec.groupby("algorithm"):
        sub = sub.sort_values("proxy")
        bins = np.linspace(sub["proxy"].min(), sub["proxy"].max(), 8) if len(sub) > 6 else None
        if bins is not None:
            sub["bin"] = pd.cut(sub["proxy"], bins, include_lowest=True)
            sub_b = sub.groupby("bin")["regret"].mean()
            x = [b.mid for b in sub_b.index]
            y = sub_b.values
        else:
            x = sub["proxy"].values
            y = sub["regret"].values
        plt.plot(x, y, marker="o", linewidth=2, label=alg)
    plt.xlabel("nonstationarity proxy")
    plt.ylabel("R_T")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig05_nonstationary_robustness.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    _write_debug(df_rec, "debug_fig05_nonstationary_robustness.csv")
    print(f"[INFO] Saved {fig_path}")


def fig_latency_cdf(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig06: empty dataframe")
        _write_debug(pd.DataFrame(), "debug_fig06_latency_cdf.csv", note="empty dataframe")
        return
    latency_cols = [c for c in df.columns if ("latency" in c.lower() or "decision" in c.lower()) and "slot" not in c.lower()]
    if not latency_cols:
        print("[WARN] fig06: no latency columns found")
        _write_debug(pd.DataFrame(), "debug_fig06_latency_cdf.csv", note="no latency columns")
        return
    col = latency_cols[0]
    plt.figure(figsize=(7, 4))
    plotted = False
    debug_rows: List[pd.DataFrame] = []
    for alg, sub in df.groupby("algorithm"):
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(vals) < MIN_LATENCY_SAMPLES:
            print(f"[WARN] fig06: skip alg={alg} due to few latency samples")
            continue
        vals = np.sort(vals)
        cdf = np.linspace(0, 1, len(vals))
        plt.plot(vals, cdf, linewidth=2, label=alg)
        debug_rows.append(pd.DataFrame({"algorithm": alg, "latency": vals, "cdf": cdf}))
        plotted = True
    if not plotted:
        print("[WARN] fig06: nothing to plot")
        _write_debug(pd.DataFrame(), "debug_fig06_latency_cdf.csv", note="no plotted algorithms")
        return
    plt.xlabel(f"{col}")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig06_latency_cdf.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
    _write_debug(debug_df, "debug_fig06_latency_cdf.csv", note="no plotted algorithms")
    print(f"[INFO] Saved {fig_path}")


def fig_snr_per_heat() -> None:
    ensure_outdir()
    path = os.path.join("outputs", "dumps", "task3_phys_scan.csv")
    if not os.path.exists(path):
        print(f"[WARN] fig07: missing {path}")
        _write_debug(pd.DataFrame(), "debug_fig07_snr_per_heat.csv", note="missing phys scan file")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] fig07: failed to read phys scan: {exc}")
        _write_debug(pd.DataFrame(), "debug_fig07_snr_per_heat.csv", note="read failure")
        return
    required = {"B", "snr_db", "per"}
    if not required.issubset(set(df.columns)):
        print("[WARN] fig07: required columns missing in phys scan")
        _write_debug(pd.DataFrame(), "debug_fig07_snr_per_heat.csv", note="missing required columns")
        return
    grouped = df.groupby(["B", "snr_db"])["per"].mean().reset_index()
    pivot = grouped.pivot(index="B", columns="snr_db", values="per")
    if pivot.empty:
        print("[WARN] fig07: no data after grouping")
        _write_debug(pd.DataFrame(), "debug_fig07_snr_per_heat.csv", note="empty pivot")
        return
    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", extent=[
        pivot.columns.min(),
        pivot.columns.max(),
        pivot.index.min(),
        pivot.index.max(),
    ])
    plt.colorbar(im, label="PER")
    plt.xlabel("snr_db")
    plt.ylabel("B (kHz)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig07_snr_per_heat.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    _write_debug(grouped, "debug_fig07_snr_per_heat.csv")
    print(f"[INFO] Saved {fig_path}")


def fig_pb_selection_heat(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig08: empty dataframe")
        _write_debug(pd.DataFrame(), "debug_fig08_pb_selection_heat.csv", note="empty dataframe")
        return
    P_col = None
    if "P" in df.columns:
        P_col = "P"
    elif "power" in df.columns:
        P_col = "power"
    B_col = None
    if "B" in df.columns:
        B_col = "B"
    elif "bandwidth" in df.columns:
        B_col = "bandwidth"
    if P_col is None or B_col is None:
        print("[WARN] fig08: missing P/B columns")
        _write_debug(pd.DataFrame(), "debug_fig08_pb_selection_heat.csv", note="missing P/B columns")
        return
    work = df[[P_col, B_col]].dropna()
    if work.empty:
        print("[WARN] fig08: no P/B data")
        _write_debug(pd.DataFrame(), "debug_fig08_pb_selection_heat.csv", note="no P/B data")
        return
    work["P_round"] = pd.to_numeric(work[P_col], errors="coerce").round(1)
    work["B_round"] = pd.to_numeric(work[B_col], errors="coerce").round(1)
    grouped = work.groupby(["P_round", "B_round"]).size().reset_index(name="count")
    pivot = grouped.pivot(index="B_round", columns="P_round", values="count").fillna(0)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    plt.colorbar(im, label="count")
    plt.xlabel("P (dBm)")
    plt.ylabel("B (kHz)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig08_pb_selection_heat.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    _write_debug(grouped, "debug_fig08_pb_selection_heat.csv")
    print(f"[INFO] Saved {fig_path}")


def fig_swer_vs_snr(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "sWER" not in df.columns or "snr_db" not in df.columns:
        print("[WARN] fig09: missing data for sWER vs SNR")
        _write_debug(pd.DataFrame(), "debug_fig09_swer_vs_snr.csv", note="missing columns or empty")
        return
    plt.figure(figsize=(7, 4))
    plotted = False
    debug_rows = []
    for alg, sub in df.groupby("algorithm"):
        vals = sub.dropna(subset=["sWER", "snr_db"])
        if vals.empty:
            continue
        snr_vals = pd.to_numeric(vals["snr_db"], errors="coerce")
        bins = np.linspace(snr_vals.min(), snr_vals.max(), 41)
        vals["snr_bin"] = pd.cut(snr_vals, bins, include_lowest=True)
        grouped = vals.groupby("snr_bin")["sWER"].mean()
        x = [b.mid for b in grouped.index]
        y = grouped.values
        plt.plot(x, y, linewidth=2, marker="o", label=alg)
        debug_rows.append(pd.DataFrame({"algorithm": alg, "snr_mid": x, "sWER_mean": y}))
        plotted = True
    if not plotted:
        print("[WARN] fig09: nothing plotted")
        _write_debug(pd.DataFrame(), "debug_fig09_swer_vs_snr.csv", note="nothing plotted")
        return
    plt.xlabel("SNR (dB)")
    plt.ylabel("sWER")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig09_swer_vs_snr.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
    _write_debug(debug_df, "debug_fig09_swer_vs_snr.csv", note="nothing plotted")
    print(f"[INFO] Saved {fig_path}")


def fig_qoe_swer_frontier(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "qoe" not in df.columns or "sWER" not in df.columns:
        print("[WARN] fig10: missing qoe/sWER data")
        _write_debug(pd.DataFrame(), "debug_fig10_qoe_swer_frontier.csv", note="missing qoe/sWER")
        return
    stats = df.dropna(subset=["qoe", "sWER"])
    if stats.empty:
        print("[WARN] fig10: no data after dropna")
        _write_debug(pd.DataFrame(), "debug_fig10_qoe_swer_frontier.csv", note="no data after dropna")
        return
    agg = stats.groupby("algorithm").agg(Ubar=("qoe", "mean"), Dbar=("sWER", "mean")).reset_index()
    agg = agg.sort_values("Dbar")
    agg["frontier"] = np.maximum.accumulate(agg["Ubar"])
    plt.figure(figsize=(6, 4))
    plt.scatter(agg["Dbar"], agg["Ubar"], c="#4C72B0", label="algorithms")
    plt.plot(agg["Dbar"], agg["frontier"], color="#DD8452", linewidth=2, label="Pareto frontier")
    plt.xlabel("Dbar (sWER)")
    plt.ylabel("Ubar (QoE)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig10_qoe_swer_frontier.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    _write_debug(agg, "debug_fig10_qoe_swer_frontier.csv")
    print(f"[INFO] Saved {fig_path}")


def fig_violation_over_time(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "t" not in df.columns:
        print("[WARN] fig11: missing data/time column")
        _write_debug(pd.DataFrame(), "debug_fig11_violation_over_time.csv", note="missing data/time")
        return
    violate_cols = []
    for cand in ["violate_lat", "violate_eng", "violate_sem", "q_violate_flag", "d_violate_flag", "energy_violate_flag", "violation_flag"]:
        if cand in df.columns:
            violate_cols.append(cand)
    if not violate_cols:
        print("[WARN] fig11: no violation columns found")
        _write_debug(pd.DataFrame(), "debug_fig11_violation_over_time.csv", note="no violation columns")
        return
    plt.figure(figsize=(8, 4))
    plotted = False
    t_int = pd.to_numeric(df["t"], errors="coerce").astype(int)
    for col in violate_cols:
        sub = df.copy()
        sub["t_int"] = t_int
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        grouped = sub.groupby("t_int")[col].mean().dropna()
        if grouped.empty:
            continue
        window = 200 if len(grouped) > 500 else 100
        smoothed = grouped.rolling(window=window, min_periods=max(10, window // 5)).mean()
        plt.plot(smoothed.index, smoothed.values, linewidth=2, label=col)
        _write_debug(smoothed.reset_index().rename(columns={"index": "t", col: "smoothed"}), "debug_fig11_violation_over_time.csv")
        plotted = True
    if not plotted:
        print("[WARN] fig11: nothing plotted")
        _write_debug(pd.DataFrame(), "debug_fig11_violation_over_time.csv", note="nothing plotted")
        return
    plt.xlabel("t")
    plt.ylabel("violation prob")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig11_violation_over_time.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def fig_energy_efficiency_bar(summary_csv: str, df_all: pd.DataFrame) -> None:
    ensure_outdir()
    if (df_all is None or df_all.empty) and (not summary_csv or not os.path.exists(summary_csv)):
        print("[WARN] fig12: no data for energy efficiency")
        return
    alg_records = []
    if df_all is not None and not df_all.empty and "qoe" in df_all.columns:
        for alg, sub in df_all.groupby("algorithm"):
            q_series = pd.to_numeric(sub["qoe"], errors="coerce").dropna()
            queue_series, energy_series = _extract_queue_energy(sub)
            if energy_series is None:
                continue
            energy_series = energy_series.dropna()
            if len(q_series) == 0 or len(energy_series) == 0:
                continue
            t_max = pd.to_numeric(sub["t"], errors="coerce").max() if "t" in sub.columns else None
            burn_start = 0.1 * t_max if t_max and math.isfinite(t_max) else 0
            if "t" in sub.columns:
                mask = pd.to_numeric(sub["t"], errors="coerce") >= burn_start
                q_series = q_series[mask]
                energy_series = energy_series[mask]
            q_mean = q_series.mean()
            e_mean = energy_series.mean()
            if e_mean and e_mean > 0:
                alg_records.append({"algorithm": alg, "eff": q_mean / e_mean})
    if not alg_records and summary_csv and os.path.exists(summary_csv):
        try:
            summary = pd.read_csv(summary_csv)
            if {"qoe_mean", "energy_mean", "algorithm"}.issubset(set(summary.columns)):
                summary = summary.dropna(subset=["qoe_mean", "energy_mean"])
                summary["eff"] = summary["qoe_mean"] / summary["energy_mean"]
                alg_records = summary.groupby("algorithm")["eff"].mean().reset_index().to_dict("records")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] fig12: failed to read summary csv: {exc}")
            return
    if not alg_records:
        print("[WARN] fig12: no valid efficiency records")
        return
    eff_df = pd.DataFrame(alg_records)
    eff_df = eff_df.dropna(subset=["eff"])
    if eff_df.empty:
        print("[WARN] fig12: empty efficiency dataframe")
        return
    plt.figure(figsize=(7, 4))
    plt.bar(eff_df["algorithm"], eff_df["eff"], color="#55A868")
    plt.ylabel("Ubar / Ebar")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig12_energy_efficiency_bar.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def fig_qoe_by_sem_weight(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig13: empty dataframe")
        return
    work = df.copy()
    bins = None
    label = None
    if "sem_weight" in work.columns and work["sem_weight"].notna().sum() > 0:
        bins = pd.qcut(work["sem_weight"], q=5, duplicates="drop")
        label = "sem_weight"
    elif "sWER" in work.columns and work["sWER"].notna().sum() > 0:
        bins = pd.qcut(work["sWER"], q=5, duplicates="drop")
        label = "sWER proxy"
    if bins is None:
        print("[WARN] fig13: no sem_weight or sWER to bin")
        return
    work["bin"] = bins
    grouped = work.dropna(subset=["bin", "mos"]).groupby("bin")["mos"].mean()
    if grouped.empty:
        print("[WARN] fig13: empty grouped data")
        return
    plt.figure(figsize=(7, 4))
    plt.plot([str(b) for b in grouped.index], grouped.values, marker="o")
    plt.xlabel(label)
    plt.ylabel("QoE (mos)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig13_qoe_by_sem_weight.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def fig14_ridgeline(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "mos" not in df.columns:
        print("[WARN] fig14: missing mos data")
        return
    work = df.copy()
    if "sem_weight" in work.columns and work["sem_weight"].notna().sum() > 0:
        bins = pd.qcut(work["sem_weight"], q=5, duplicates="drop")
    elif "sWER" in work.columns and work["sWER"].notna().sum() > 0:
        bins = pd.qcut(work["sWER"], q=5, duplicates="drop")
    else:
        print("[WARN] fig14: no sem_weight/sWER to group")
        return
    work["bin"] = bins
    groups = [(str(b), sub["mos"].dropna().values) for b, sub in work.groupby("bin")]
    groups = [(name, vals) for name, vals in groups if len(vals) >= 10]
    if not groups:
        print("[WARN] fig14: insufficient samples per bin")
        return
    all_vals = np.concatenate([g[1] for g in groups])
    grid = np.linspace(all_vals.min(), all_vals.max(), 200)

    def kde(x: np.ndarray, grid_vals: np.ndarray) -> np.ndarray:
        if len(x) < 2:
            return np.zeros_like(grid_vals)
        bw = 0.3 * np.std(x) if np.std(x) > 1e-6 else 0.1
        diffs = (grid_vals[:, None] - x[None, :]) / bw
        kern = np.exp(-0.5 * diffs**2) / (np.sqrt(2 * np.pi) * bw)
        return kern.mean(axis=1)

    plt.figure(figsize=(8, 6))
    y_offset = 0
    for name, vals in groups:
        density = kde(np.array(vals), grid)
        plt.fill_between(grid, y_offset, y_offset + density, alpha=0.6, label=name)
        plt.plot(grid, y_offset + density, color="k", linewidth=0.8)
        y_offset += density.max() * 1.2
    plt.yticks([])
    plt.xlabel("QoE (mos)")
    plt.legend(title="group")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig14_ridgeline.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def fig_semantic_constraint_curve(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "sWER" not in df.columns or "mos" not in df.columns:
        print("[WARN] fig15: missing sWER/mos data")
        return
    thresholds = np.linspace(0.15, 0.25, 21)
    xs, ys = [], []
    for thr in thresholds:
        sub = df[pd.to_numeric(df["sWER"], errors="coerce") <= thr]
        if sub.empty:
            continue
        xs.append(thr)
        ys.append(pd.to_numeric(sub["mos"], errors="coerce").mean())
    if not xs:
        print("[WARN] fig15: no data after applying thresholds")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Dbar")
    plt.ylabel("Ubar")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig15_semantic_constraint_curve.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")

def fig_recovery_time_bar(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty or "mos" not in df.columns or "t" not in df.columns:
        print("[WARN] fig16: missing mos/t data")
        return
    records = []
    seed_series = df["seed"] if "seed" in df.columns else pd.Series([0] * len(df))
    for (alg, file, seed), sub in df.groupby(["algorithm", "file", seed_series]):
        mos = pd.to_numeric(sub["mos"], errors="coerce").dropna()
        t_vals = pd.to_numeric(sub["t"], errors="coerce").reindex(mos.index)
        if len(mos) < 20:
            continue
        window = max(20, int(0.05 * len(mos)))
        roll = mos.rolling(window=window, min_periods=5).mean().dropna()
        if roll.empty:
            continue
        steady = roll.iloc[-min(len(roll), 200):].mean()
        thresh = 0.9 * steady
        mask = roll >= thresh
        if not mask.any():
            continue
        first_idx = mask[mask].index[0]
        t_rec = t_vals.loc[first_idx] if first_idx in t_vals.index else None
        if t_rec is None or not math.isfinite(t_rec):
            continue
        records.append({"algorithm": alg, "t_rec": t_rec})
    if not records:
        print("[WARN] fig16: no recovery records")
        return
    rec_df = pd.DataFrame(records)
    agg = rec_df.groupby("algorithm")["t_rec"].mean().reset_index()
    plt.figure(figsize=(7, 4))
    plt.bar(agg["algorithm"], agg["t_rec"], color="#C44E52")
    plt.ylabel("recovery time (slots)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig16_recovery_time_bar.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def fig_window_scan(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig17: empty dataframe")
        return
    if "window" not in df.columns:
        df = _fill_from_filename(df.copy(), "window", r"w(\d+)", int)
    if "window" not in df.columns:
        print("[WARN] fig17: window not found")
        return
    if "regret_t" not in df.columns and "instant_regret" not in df.columns and "qoe" not in df.columns:
        print("[WARN] fig17: no regret/qoe data")
        return
    prefer_alg = "sw_ucb"
    algs = [prefer_alg] if prefer_alg in df["algorithm"].unique() else list(df["algorithm"].unique())
    records = []
    for alg in algs:
        sub_alg = df[df["algorithm"] == alg]
        if sub_alg.empty:
            continue
        for file, sub_file in sub_alg.groupby("file"):
            regret = None
            if "regret_t" in sub_file.columns:
                val = pd.to_numeric(sub_file["regret_t"], errors="coerce").dropna()
                regret = val.iloc[-1] if not val.empty else None
            if regret is None and "instant_regret" in sub_file.columns:
                regret = pd.to_numeric(sub_file["instant_regret"], errors="coerce").dropna().sum()
            if regret is None and "qoe" in sub_file.columns:
                q = pd.to_numeric(sub_file["qoe"], errors="coerce").dropna()
                if not q.empty:
                    regret = (q.max() - q).clip(lower=0).sum()
            if regret is None:
                continue
            w_val = sub_file["window"].dropna().iloc[0] if sub_file["window"].notna().any() else np.nan
            records.append({"algorithm": alg, "window": w_val, "regret": regret})
        if records:
            break
    if not records:
        print("[WARN] fig17: no regret records")
        return
    rec_df = pd.DataFrame(records).dropna(subset=["window"])
    agg = rec_df.groupby("window")["regret"].mean().reset_index().sort_values("window")
    plt.figure(figsize=(7, 4))
    plt.plot(agg["window"], agg["regret"], marker="o", linewidth=2)
    plt.xlabel("window size w")
    plt.ylabel("R_T")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig17_window_scan.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def fig_grid_density_vs_regret(df: pd.DataFrame) -> None:
    ensure_outdir()
    if df.empty:
        print("[WARN] fig18: empty dataframe")
        return
    if "grid_density" not in df.columns:
        df = _fill_from_filename(df.copy(), "grid_density", r"g(\d+)", int)
    if "grid_density" not in df.columns:
        print("[WARN] fig18: grid_density not found")
        return
    if "regret_t" not in df.columns and "instant_regret" not in df.columns and "qoe" not in df.columns:
        print("[WARN] fig18: no regret/qoe data")
        return
    records = []
    for file, sub in df.groupby("file"):
        regret = None
        if "regret_t" in sub.columns:
            val = pd.to_numeric(sub["regret_t"], errors="coerce").dropna()
            regret = val.iloc[-1] if not val.empty else None
        if regret is None and "instant_regret" in sub.columns:
            regret = pd.to_numeric(sub["instant_regret"], errors="coerce").dropna().sum()
        if regret is None and "qoe" in sub.columns:
            q = pd.to_numeric(sub["qoe"], errors="coerce").dropna()
            if not q.empty:
                regret = (q.max() - q).clip(lower=0).sum()
        if regret is None:
            continue
        g_val = sub["grid_density"].dropna().iloc[0] if sub["grid_density"].notna().any() else np.nan
        records.append({"grid_density": g_val, "regret": regret})
    if not records:
        print("[WARN] fig18: no regret records")
        return
    rec_df = pd.DataFrame(records).dropna(subset=["grid_density"])
    if rec_df["grid_density"].nunique() <= 1:
        print("[WARN] fig18: only one grid_density value, skip plotting")
        return
    agg = rec_df.groupby("grid_density")["regret"].mean().reset_index().sort_values("grid_density")
    plt.figure(figsize=(7, 4))
    plt.plot(agg["grid_density"], agg["regret"], marker="o", linewidth=2)
    plt.xlabel("grid_density")
    plt.ylabel("R_T")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig18_grid_density_vs_regret.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def main() -> None:
    ensure_outdir()
    df_all = load_all_dumps()
    summary_csv = os.path.join("outputs", "dumps", "task3_summary.csv")

    fig_regret_vs_time(df_all)
    fig_qoe_vs_V(df_all)
    fig_queue_energy_vs_V(df_all)
    fig_param_sensitivity(summary_csv, df_all)
    fig_nonstationary_robustness(df_all)
    fig_latency_cdf(df_all)
    fig_snr_per_heat()
    fig_pb_selection_heat(df_all)
    fig_swer_vs_snr(df_all)
    fig_qoe_swer_frontier(df_all)
    fig_violation_over_time(df_all)
    fig_energy_efficiency_bar(summary_csv, df_all)
    fig_qoe_by_sem_weight(df_all)
    fig14_ridgeline(df_all)
    fig_semantic_constraint_curve(df_all)
    fig_recovery_time_bar(df_all)
    fig_window_scan(df_all)
    fig_grid_density_vs_regret(df_all)


if __name__ == "__main__":
    main()
