import glob
import math
import os
import re
from typing import Iterable, List, Tuple, Optional

import matplotlib
import numpy as np
import pandas as pd

# Use Agg backend to avoid GUI requirements
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Determine Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures", "task3_v3")
FIG_DPI = 300
MIN_LATENCY_SAMPLES = 20

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
        # Remove duplicate columns resulting from rename
        df = df.loc[:, ~df.columns.duplicated()]

    if "t" not in df.columns:
        df["t"] = np.arange(len(df))
    
    # Robust t-column handling
    t_col = df["t"]
    if isinstance(t_col, pd.DataFrame):
        t_col = t_col.iloc[:, 0]
    df["t"] = pd.to_numeric(t_col, errors="coerce")

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
        os.path.join(PROJECT_ROOT, "outputs", "dumps", "*.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "dumps", "task3_v3", "**", "*.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "dumps", "task3_v2", "*.csv"),
        os.path.join(PROJECT_ROOT, "task3", "outputs", "dumps", "**", "*.csv"),
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
        except Exception as exc:
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

def _steady_state_stats(
    df: pd.DataFrame, value_col: str, extra_fields: Iterable[str] | None = None
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    extra_fields = list(extra_fields or [])
    cols_needed = ["algorithm", "V", "t", value_col]
    for col in cols_needed:
        if col not in df.columns:
            print(f"[WARN] Missing column {col} for steady-state stats on {value_col}")
            return pd.DataFrame()

    work = df.copy()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["algorithm", "V", "t", value_col])
    if work.empty:
        return pd.DataFrame()
    if "seed" not in work.columns:
        work["seed"] = 0
    work["seed"] = work["seed"].fillna(0)
    if "file" not in work.columns:
        work["file"] = "unknown"

    run_cols = ["algorithm", "V", "file", "seed"]
    tmax = work.groupby(run_cols)["t"].max()
    if tmax.empty:
        return pd.DataFrame()
    tcommon_map = tmax.groupby(level=[0, 1]).min().to_dict()

    records = []
    for (alg, v_val, file, seed), t_max in tmax.items():
        t_common = tcommon_map.get((alg, v_val))
        if t_common is None or not math.isfinite(t_common) or t_common <= 0:
            continue
        burn_start = 0.5 * t_common
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
        window = max(25, int(0.10 * len(vals)))
        roll = vals.rolling(window=window, min_periods=10).mean().dropna()
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
        return pd.DataFrame()
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
    return agg

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