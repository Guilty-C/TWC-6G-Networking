"""Evaluate semantic weighting under denoising vs. raw pipelines."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from sem_weight_extractor import SemWeightExtractor, SemWeightExtractorConfig
from wer_proxy import per_from_snr_db, wer_from_per

logger = logging.getLogger(__name__)


class SemEvalConfig(BaseModel):
    """Configuration structure for semantic evaluation."""

    seed: int = Field(2025, description="Random seed")
    data_csv: str = Field("data/pesq_measurements.csv", description="Dataset path")
    bucket_count: int = Field(5, description="Number of buckets for w_sem")
    figure_size: List[float] = Field(default_factory=lambda: [10.0, 6.0])
    output_dir: str = Field("outputs/sem_eval", description="Output directory")
    slot_ms: float = Field(20.0, description="Slot duration in milliseconds")
    sample_rate: int = Field(16000, description="Sampling rate")
    per_steep: float = Field(1.2, description="PER curve steepness")
    per_mid_db: float = Field(1.0, description="PER inflection point")
    extractor: Dict[str, Any] = Field(default_factory=dict)

    def build_extractor_config(self) -> SemWeightExtractorConfig:
        payload = {"sample_rate": self.sample_rate, **self.extractor}
        return SemWeightExtractorConfig(**payload)


@dataclass
class SampleResult:
    idx: int
    w_sem_denoised: float
    w_sem_baseline: float
    conf_denoised: float
    conf_baseline: float
    wer_denoised: float
    wer_baseline: float
    swwer_denoised: float
    swwer_baseline: float
    bucket: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _synth_slot(row: pd.Series, cfg: SemEvalConfig, rng: np.random.Generator) -> np.ndarray:
    slot_len = int(round(cfg.sample_rate * cfg.slot_ms / 1000.0))
    t = np.arange(slot_len) / cfg.sample_rate
    amp = np.clip(row.get("pesq", 3.0) / 4.5, 0.25, 1.0)
    speech = amp * (
        0.6 * np.sin(2 * np.pi * 220 * t)
        + 0.4 * np.sin(2 * np.pi * 660 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
    )
    snr_lin = 10 ** (float(row.get("snr_db", 0.0)) / 10.0)
    noise_power = np.mean(speech**2) / max(snr_lin, 1e-3)
    noise = rng.normal(scale=np.sqrt(noise_power), size=slot_len)
    pcm = speech + noise
    loss_rate = float(row.get("packet_loss", 0.0)) / 100.0
    if loss_rate > 0.0:
        burst = rng.random(slot_len) < loss_rate
        pcm = pcm * (1.0 - burst.astype(float))
    return np.clip(pcm, -1.0, 1.0)


def _bucketize(values: np.ndarray, bucket_count: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bucket_count + 1)
    buckets = np.digitize(values, edges, right=False) - 1
    buckets = np.clip(buckets, 0, bucket_count - 1)
    return buckets, edges


def _cdf(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_vals = np.sort(series)
    cdf = np.linspace(0.0, 1.0, len(sorted_vals), endpoint=False)
    return sorted_vals, cdf


def _ensure_dirs(base: str) -> Tuple[str, str]:
    dump_dir = os.path.join(base, "dumps")
    fig_dir = os.path.join(base, "figs")
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return dump_dir, fig_dir


def run(cfg: SemEvalConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    extractor_cfg = cfg.build_extractor_config()
    denoised_extractor = SemWeightExtractor(extractor_cfg)
    baseline_extractor = SemWeightExtractor(extractor_cfg)

    df = pd.read_csv(cfg.data_csv)
    loss_burst_norm = np.clip(df.get("packet_loss", pd.Series(0)).to_numpy(dtype=float) / 10.0, 0.0, 1.0)

    records: List[SampleResult] = []
    w_sem_vals = []
    for idx, row in df.iterrows():
        pcm = _synth_slot(row, cfg, rng)
        loss_burst = float(loss_burst_norm[idx]) if idx < len(loss_burst_norm) else 0.0
        denoise_res = denoised_extractor.process_slot(pcm, loss_burst=loss_burst)
        base_res = baseline_extractor.process_slot(pcm, loss_burst=loss_burst, skip_denoise=True)
        per = per_from_snr_db(float(row.get("snr_db", 0.0)), steep=cfg.per_steep, mid_db=cfg.per_mid_db)
        wer_d = wer_from_per(per, denoise_res["w_sem"])
        wer_b = wer_from_per(per, base_res["w_sem"])
        swwer_d = denoise_res["w_sem"] * wer_d
        swwer_b = base_res["w_sem"] * wer_b
        w_sem_vals.append(denoise_res["w_sem"])
        records.append(
            SampleResult(
                idx=int(idx),
                w_sem_denoised=float(denoise_res["w_sem"]),
                w_sem_baseline=float(base_res["w_sem"]),
                conf_denoised=float(denoise_res["conf_sem"]),
                conf_baseline=float(base_res["conf_sem"]),
                wer_denoised=float(wer_d),
                wer_baseline=float(wer_b),
                swwer_denoised=float(swwer_d),
                swwer_baseline=float(swwer_b),
                bucket=0,
            )
        )

    w_sem_arr = np.asarray(w_sem_vals, dtype=float)
    buckets, edges = _bucketize(w_sem_arr, cfg.bucket_count)
    for rec, bucket in zip(records, buckets):
        rec.bucket = int(bucket)

    dump_dir, fig_dir = _ensure_dirs(cfg.output_dir)

    raw_df = pd.DataFrame([r.to_dict() for r in records])
    raw_path = os.path.join(dump_dir, "sem_eval_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    logger.info("Saved raw evaluation to %s", raw_path)

    cdf_records: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    for bucket, group in raw_df.groupby("bucket"):
        if group.empty:
            continue
        wer_sorted_d, cdf_d = _cdf(group["wer_denoised"].to_numpy())
        wer_sorted_b, cdf_b = _cdf(group["wer_baseline"].to_numpy())
        swwer_sorted_d, s_cdf_d = _cdf(group["swwer_denoised"].to_numpy())
        swwer_sorted_b, s_cdf_b = _cdf(group["swwer_baseline"].to_numpy())
        for x, y in zip(wer_sorted_d, cdf_d):
            cdf_records.append({"bucket": int(bucket), "variant": "denoised", "metric": "wer", "x": float(x), "cdf": float(y)})
        for x, y in zip(wer_sorted_b, cdf_b):
            cdf_records.append({"bucket": int(bucket), "variant": "baseline", "metric": "wer", "x": float(x), "cdf": float(y)})
        for x, y in zip(swwer_sorted_d, s_cdf_d):
            cdf_records.append({"bucket": int(bucket), "variant": "denoised", "metric": "swwer", "x": float(x), "cdf": float(y)})
        for x, y in zip(swwer_sorted_b, s_cdf_b):
            cdf_records.append({"bucket": int(bucket), "variant": "baseline", "metric": "swwer", "x": float(x), "cdf": float(y)})

        var_den = float(np.var(group["w_sem_denoised"].to_numpy()))
        var_bas = float(np.var(group["w_sem_baseline"].to_numpy()))
        ratio = var_den / (var_bas + 1e-9)
        stability_rows.append(
            {
                "bucket": int(bucket),
                "var_denoised": var_den,
                "var_baseline": var_bas,
                "ratio": ratio,
            }
        )
        logger.info(
            "Bucket %d variance ratio (denoised/base) = %.3f", bucket, ratio
        )

    cdf_df = pd.DataFrame(cdf_records)
    cdf_path = os.path.join(dump_dir, "sem_eval_cdf_raw.csv")
    cdf_df.to_csv(cdf_path, index=False)
    logger.info("Saved CDF traces to %s", cdf_path)

    stability_df = pd.DataFrame(stability_rows)
    stability_path = os.path.join(dump_dir, "sem_eval_stability_raw.csv")
    stability_df.to_csv(stability_path, index=False)
    logger.info("Saved stability metrics to %s", stability_path)

    fig = plt.figure(figsize=tuple(cfg.figure_size))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    colors = plt.cm.get_cmap("viridis", cfg.bucket_count)
    for bucket in sorted(raw_df["bucket"].unique()):
        sub = cdf_df[(cdf_df["bucket"] == bucket) & (cdf_df["metric"] == "wer")]
        sub_s = cdf_df[(cdf_df["bucket"] == bucket) & (cdf_df["metric"] == "swwer")]
        col = colors(bucket / max(cfg.bucket_count - 1, 1))
        ax1.plot(
            sub[sub["variant"] == "denoised"]["x"].to_numpy(),
            sub[sub["variant"] == "denoised"]["cdf"].to_numpy(),
            color=col,
            label=f"bucket {bucket} denoised",
        )
        ax1.plot(
            sub[sub["variant"] == "baseline"]["x"].to_numpy(),
            sub[sub["variant"] == "baseline"]["cdf"].to_numpy(),
            linestyle="--",
            color=col,
            label=f"bucket {bucket} baseline",
        )
        ax2.plot(
            sub_s[sub_s["variant"] == "denoised"]["x"].to_numpy(),
            sub_s[sub_s["variant"] == "denoised"]["cdf"].to_numpy(),
            color=col,
            label=f"bucket {bucket} denoised",
        )
        ax2.plot(
            sub_s[sub_s["variant"] == "baseline"]["x"].to_numpy(),
            sub_s[sub_s["variant"] == "baseline"]["cdf"].to_numpy(),
            linestyle="--",
            color=col,
            label=f"bucket {bucket} baseline",
        )
    ax1.set_xlabel("WER")
    ax1.set_ylabel("CDF")
    ax1.set_title("WER CDF by w_sem bucket")
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("SWWER")
    ax2.set_ylabel("CDF")
    ax2.set_title("SWWER CDF by w_sem bucket")
    ax2.grid(True, alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=cfg.bucket_count)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = os.path.join(fig_dir, "sem_eval_cdf.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Saved CDF figure to %s", fig_path)

    plt.figure(figsize=(6, 4))
    plt.plot(stability_df["bucket"], stability_df["var_baseline"], marker="o", label="baseline")
    plt.plot(stability_df["bucket"], stability_df["var_denoised"], marker="s", label="denoised")
    plt.xlabel("Bucket index")
    plt.ylabel("Variance of w_sem")
    plt.title("w_sem stability vs. baseline")
    plt.grid(True, alpha=0.3)
    plt.legend()
    stability_fig = os.path.join(fig_dir, "sem_eval_stability.png")
    plt.tight_layout()
    plt.savefig(stability_fig, dpi=150)
    plt.close()
    logger.info("Saved stability figure to %s", stability_fig)

    summary_path = os.path.join(dump_dir, "sem_eval_summary_raw.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bucket_edges": edges.tolist(),
                "variance_ratio": stability_df.set_index("bucket")["ratio"].to_dict(),
            },
            handle,
            indent=2,
        )
    logger.info("Saved summary to %s", summary_path)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic evaluation runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def _load_config(path: str) -> SemEvalConfig:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return SemEvalConfig(**cfg)


if __name__ == "__main__":
    _setup_logging()
    args = _parse_args()
    config = _load_config(args.config)
    run(config)
