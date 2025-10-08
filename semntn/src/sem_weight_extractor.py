"""Semantic weight extraction aligned with PESQ-driven objectives."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from common.denoise import DenoiseConfig, LogMMSEDenoiser
from common.sem_features import SemFeatureConfig, SemFeatureExtractor

logger = logging.getLogger(__name__)


class MappingConfig(BaseModel):
    """Configuration for the monotonic S-shaped mapping."""

    slope: float = Field(4.0, description="Slope of the logistic mapping")
    bias: float = Field(0.0, description="Bias applied before sigmoid")
    feature_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "energy": 0.8,
            "delta_energy": -0.2,
            "zcr": -0.5,
            "spectral_flux": 0.3,
            "spectral_centroid": 0.1,
            "spectral_flatness": -0.3,
            "hnr": 0.4,
            "loss_burst": -0.4,
        }
    )
    ema_alpha: float = Field(0.6, description="EMA coefficient for w_sem")
    conf_tau: float = Field(0.25, description="Sensitivity of the confidence map")
    conf_floor: float = Field(0.35, description="Lower bound of confidence")
    key_threshold: float = Field(0.65, description="Key-slot trigger threshold")


class SemWeightExtractorConfig(BaseModel):
    """Top-level configuration for the semantic weight extractor."""

    sample_rate: int = Field(16000, description="Sampling rate in Hz")
    denoise: Dict[str, object] = Field(default_factory=dict)
    features: Dict[str, object] = Field(default_factory=dict)
    mapping: MappingConfig = MappingConfig()

    def build_feature_config(self) -> SemFeatureConfig:
        payload = {"sample_rate": self.sample_rate, **self.features}
        return SemFeatureConfig(**payload)


@dataclass
class _ExtractorState:
    prev_w_sem: float = 0.5
    prev_features: Optional[NDArray[np.float64]] = None


class SemWeightExtractor:
    """Encapsulates denoising, feature extraction and mapping to w_sem."""

    def __init__(self, config: SemWeightExtractorConfig):
        self.config = config
        denoise_cfg = self._merge_denoise(config)
        feature_cfg = config.build_feature_config()
        self.denoiser = LogMMSEDenoiser(denoise_cfg)
        self.feature_extractor = SemFeatureExtractor(feature_cfg)
        self.state = _ExtractorState()
        self._feature_order = [
            "energy",
            "delta_energy",
            "zcr",
            "spectral_flux",
            "spectral_centroid",
            "spectral_flatness",
            "hnr",
            "loss_burst",
        ]
        self._feature_scale = {
            "energy": 0.5,
            "delta_energy": 1.0,
            "zcr": 1.0,
            "spectral_flux": 5.0,
            "spectral_centroid": 4000.0,
            "spectral_flatness": 1.0,
            "hnr": 20.0,
            "loss_burst": 1.0,
        }

    def _merge_denoise(self, config: SemWeightExtractorConfig) -> DenoiseConfig:
        base = DenoiseConfig()
        if config.denoise:
            if "bandpass" in config.denoise:
                base.bandpass = base.bandpass.copy(update=config.denoise["bandpass"])
            if "logmmse" in config.denoise:
                base.logmmse = base.logmmse.copy(update=config.denoise["logmmse"])
        base.bandpass.sample_rate = config.sample_rate
        base.logmmse.frame_length_ms = config.features.get(
            "frame_length_ms", base.logmmse.frame_length_ms
        )
        base.logmmse.frame_hop_ms = config.features.get(
            "frame_hop_ms", base.logmmse.frame_hop_ms
        )
        return base

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _prepare_pcm(self, pcm_or_bits) -> NDArray[np.float64]:
        if isinstance(pcm_or_bits, np.ndarray):
            arr = pcm_or_bits.astype(np.float64)
        elif isinstance(pcm_or_bits, (bytes, bytearray)):
            arr = np.frombuffer(pcm_or_bits, dtype=np.int16).astype(np.float64) / 32768.0
        else:
            arr = np.asarray(pcm_or_bits, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("pcm_or_bits must be a 1-D sequence")
        arr = np.clip(arr, -1.0, 1.0)
        return arr

    def _compose_feature_vector(
        self, features: Dict[str, float], loss_burst: float
    ) -> NDArray[np.float64]:
        vec = []
        for key in self._feature_order:
            if key == "loss_burst":
                val = float(loss_burst)
            else:
                val = float(features.get(key, 0.0))
            scale = self._feature_scale.get(key, 1.0)
            vec.append(val / (scale + 1e-9))
        return np.asarray(vec, dtype=np.float64)

    def process_slot(
        self, pcm_or_bits, loss_burst: float = 0.0, skip_denoise: bool = False
    ) -> Dict[str, object]:
        """Process a slot and return semantic weighting information."""

        pcm = self._prepare_pcm(pcm_or_bits)
        enhanced = pcm if skip_denoise else self.denoiser.process(pcm)
        features = self.feature_extractor.extract(enhanced)
        feat_vec = self._compose_feature_vector(features, loss_burst)

        weights = np.array(
            [self.config.mapping.feature_weights.get(k, 0.0) for k in self._feature_order],
            dtype=np.float64,
        )
        score = float(np.dot(weights, feat_vec) + self.config.mapping.bias)
        w_raw = self._sigmoid(self.config.mapping.slope * score)
        ema = self.config.mapping.ema_alpha
        w_sem = ema * self.state.prev_w_sem + (1.0 - ema) * w_raw
        self.state.prev_w_sem = w_sem

        if self.state.prev_features is None:
            delta = 0.0
        else:
            delta = float(np.mean(np.abs(feat_vec - self.state.prev_features)))
        self.state.prev_features = feat_vec
        conf = float(
            np.clip(
                np.exp(-delta / max(self.config.mapping.conf_tau, 1e-6)),
                self.config.mapping.conf_floor,
                1.0,
            )
        )

        is_key = bool(w_sem >= self.config.mapping.key_threshold and conf >= 0.5)

        return {
            "w_sem": float(np.clip(w_sem, 0.0, 1.0)),
            "conf_sem": float(conf),
            "is_key": is_key,
            "features": features,
        }


_DEFAULT_CONFIG = SemWeightExtractorConfig()
_DEFAULT_EXTRACTOR: Optional[SemWeightExtractor] = None


def process_slot(pcm_or_bits, loss_burst: float = 0.0, skip_denoise: bool = False):
    """Module-level convenience wrapper using a singleton extractor."""

    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = SemWeightExtractor(_DEFAULT_CONFIG)
    return _DEFAULT_EXTRACTOR.process_slot(pcm_or_bits, loss_burst, skip_denoise=skip_denoise)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic weight extractor smoke test")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    parser.add_argument("--outdir", type=str, default="outputs/sem_weight")
    return parser


def _load_config(path: Optional[str]) -> SemWeightExtractorConfig:
    if path is None:
        return _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
        else:
            import yaml

            payload = yaml.safe_load(handle)
    return SemWeightExtractorConfig(**payload)


def _smoke(config: SemWeightExtractorConfig, outdir: str) -> None:
    os.makedirs(os.path.join(outdir, "dumps"), exist_ok=True)
    rng = np.random.default_rng(1)
    t = np.arange(0, 0.06, 1.0 / config.sample_rate)
    base = 0.5 * np.sin(2 * np.pi * 520 * t)
    low_noise = base + rng.normal(scale=0.05, size=len(t))
    high_noise = base + rng.normal(scale=0.25, size=len(t))

    extractor = SemWeightExtractor(config)
    res_clean = extractor.process_slot(low_noise, loss_burst=0.0)
    extractor = SemWeightExtractor(config)
    res_noisy = extractor.process_slot(high_noise, loss_burst=0.5)

    assert res_clean["w_sem"] >= res_noisy["w_sem"] - 1e-6, "mapping should be monotonic"

    dump = os.path.join(outdir, "dumps", "sem_weight_smoke_raw.json")
    with open(dump, "w", encoding="utf-8") as handle:
        json.dump({"clean": res_clean, "noisy": res_noisy}, handle, indent=2)
    logger.info("Saved semantic weight smoke results to %s", dump)


if __name__ == "__main__":
    _setup_logging()
    args = _parser().parse_args()
    cfg = _load_config(args.config)
    _smoke(cfg, args.outdir)
