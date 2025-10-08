"""Semantic feature extraction primitives.

The extractor exposes lightweight statistics tailored for on-device semantic
weighting, including energy, zero-crossing rate, spectral descriptors and a
harmonic-to-noise ratio (HNR) proxy.  Outputs are smoothed via EMA to keep the
variance low under harsh satellite channels.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy import signal

logger = logging.getLogger(__name__)


class SemFeatureConfig(BaseModel):
    """Configuration of the feature extractor."""

    sample_rate: int = Field(16000, description="Sampling rate in Hz")
    frame_length_ms: float = Field(20.0, description="Frame length in milliseconds")
    frame_hop_ms: float = Field(10.0, description="Frame hop in milliseconds")
    ema_alpha: float = Field(0.5, description="EMA factor for stabilising features")
    hnr_min_f0: float = Field(80.0, description="Lower bound of pitch search (Hz)")
    hnr_max_f0: float = Field(400.0, description="Upper bound of pitch search (Hz)")


@dataclass
class SemFeatureState:
    """State carried across slots for delta and EMA computations."""

    prev_energy: float = 1e-6
    prev_spec: Optional[NDArray[np.float64]] = None
    ema_features: Optional[NDArray[np.float64]] = None
    history: list = field(default_factory=list)


class SemFeatureExtractor:
    """Compute semantic-aware acoustic descriptors."""

    def __init__(self, config: SemFeatureConfig):
        self.config = config
        self.state = SemFeatureState()
        self._frame_len = int(round(config.frame_length_ms * config.sample_rate / 1000.0))
        self._frame_hop = int(round(config.frame_hop_ms * config.sample_rate / 1000.0))
        self._window = signal.hann(self._frame_len, sym=False)
        self._freqs = np.fft.rfftfreq(self._frame_len, d=1.0 / config.sample_rate)

    def _frame(self, pcm: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(pcm) < self._frame_len:
            pad = self._frame_len - len(pcm)
            pcm = np.pad(pcm, (0, pad), mode="reflect")
        n_frames = 1 + (len(pcm) - self._frame_len) // max(1, self._frame_hop)
        frames = np.lib.stride_tricks.as_strided(
            pcm,
            shape=(n_frames, self._frame_len),
            strides=(pcm.strides[0] * self._frame_hop, pcm.strides[0]),
        ).copy()
        frames *= self._window
        return frames

    def _hnr(self, frame: NDArray[np.float64]) -> float:
        cfg = self.config
        auto = signal.correlate(frame, frame, mode="full")
        mid = len(auto) // 2
        auto = auto[mid:]
        min_lag = max(1, int(cfg.sample_rate / cfg.hnr_max_f0))
        max_lag = int(cfg.sample_rate / max(cfg.hnr_min_f0, 1.0))
        max_lag = min(max_lag, len(auto) - 1)
        if max_lag <= min_lag:
            return 0.0
        segment = auto[min_lag:max_lag]
        peak = np.max(segment)
        denom = auto[0] - peak
        if denom <= 1e-9 or peak <= 0.0:
            return 0.0
        return float(10.0 * np.log10(peak / denom + 1e-9))

    def extract(self, pcm: NDArray[np.float64]) -> Dict[str, float]:
        """Compute smoothed semantic descriptors for a PCM slot."""

        x = np.asarray(pcm, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("pcm must be one-dimensional")

        frames = self._frame(x)
        energy = float(np.mean(x**2))
        delta_energy = float((energy - self.state.prev_energy) / (self.state.prev_energy + 1e-6))
        self.state.prev_energy = 0.9 * self.state.prev_energy + 0.1 * energy

        signs = np.sign(x)
        zcr = float(np.mean(np.abs(np.diff(signs)) > 0))

        spectrum = np.abs(np.fft.rfft(frames, axis=1))
        spec_mean = np.mean(spectrum, axis=0)
        if self.state.prev_spec is None:
            spectral_flux = 0.0
        else:
            prev = self.state.prev_spec / (np.linalg.norm(self.state.prev_spec) + 1e-6)
            curr = spec_mean / (np.linalg.norm(spec_mean) + 1e-6)
            spectral_flux = float(np.maximum(curr - prev, 0.0).sum())
        self.state.prev_spec = spec_mean

        centroid = float(np.sum(self._freqs * spec_mean) / (np.sum(spec_mean) + 1e-6))
        flatness = float(
            np.exp(np.mean(np.log(spec_mean + 1e-6))) / (np.mean(spec_mean) + 1e-6)
        )
        hnr = float(np.mean([self._hnr(f) for f in frames]))

        feats = np.array(
            [energy, delta_energy, zcr, spectral_flux, centroid, flatness, hnr],
            dtype=np.float64,
        )
        alpha = self.config.ema_alpha
        if self.state.ema_features is None:
            self.state.ema_features = feats
        else:
            self.state.ema_features = alpha * self.state.ema_features + (1.0 - alpha) * feats
        smoothed = self.state.ema_features

        names = [
            "energy",
            "delta_energy",
            "zcr",
            "spectral_flux",
            "spectral_centroid",
            "spectral_flatness",
            "hnr",
        ]
        result = {name: float(val) for name, val in zip(names, smoothed)}
        self.state.history.append(result)
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic feature smoke test")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config")
    parser.add_argument("--outdir", type=str, default="outputs/sem_features")
    return parser


def _load_config(path: Optional[str]) -> SemFeatureConfig:
    if path is None:
        return SemFeatureConfig()
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
        else:
            import yaml

            payload = yaml.safe_load(handle)
    return SemFeatureConfig(**payload)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )


def _smoke(outdir: str, config: SemFeatureConfig) -> None:
    os.makedirs(os.path.join(outdir, "dumps"), exist_ok=True)
    rng = np.random.default_rng(0)
    t = np.arange(0, 0.1, 1.0 / config.sample_rate)
    clean = 0.4 * np.sin(2 * np.pi * 440 * t)
    noise = rng.normal(scale=0.2, size=len(t))
    burst = clean + noise

    extractor = SemFeatureExtractor(config)
    feats = extractor.extract(burst)

    dump = os.path.join(outdir, "dumps", "sem_features_raw.json")
    with open(dump, "w", encoding="utf-8") as handle:
        json.dump(feats, handle, indent=2)
    logger.info("Saved features to %s", dump)


if __name__ == "__main__":
    _setup_logging()
    args = _parser().parse_args()
    cfg = _load_config(args.config)
    _smoke(args.outdir, cfg)
