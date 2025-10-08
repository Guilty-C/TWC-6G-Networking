"""Speech denoising utilities tailored for semantic weighting.

The module implements a narrow-band (300–3400 Hz) preprocessing chain with
optional notch filters followed by a logMMSE estimator that leverages an MCRA
(noise floor tracking) backend.  All components are lightweight enough to be
executed within the ~2 ms slot budget on typical mobile chipsets while
remaining fully reproducible.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, validator
from scipy import signal, special

if TYPE_CHECKING:  # pragma: no cover - typing helper for static analyzers
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    ComplexArray = NDArray[np.complex128]
else:  # keep runtime dependencies minimal while retaining rich type hints
    FloatArray = np.ndarray
    ComplexArray = np.ndarray

logger = logging.getLogger(__name__)


class BandpassConfig(BaseModel):
    """Configuration of the pre-filter stage."""

    sample_rate: int = Field(16000, description="Input sampling rate in Hz")
    low_hz: float = Field(300.0, description="Lower cutoff frequency")
    high_hz: float = Field(3400.0, description="Upper cutoff frequency")
    order: int = Field(4, description="Order of the Butterworth filter")
    notch_freqs: List[float] = Field(
        default_factory=list, description="Optional notch frequencies in Hz"
    )
    notch_q: float = Field(25.0, description="Quality factor for notch filters")

    @validator("high_hz")
    def _validate_band(cls, high: float, values: dict[str, object]) -> float:
        sr = values.get("sample_rate", 16000)
        if high >= sr / 2:
            raise ValueError("high_hz must be lower than Nyquist frequency")
        return high


class LogMMSEConfig(BaseModel):
    """Parameters governing the logMMSE + MCRA estimator."""

    frame_length_ms: float = Field(20.0, description="Frame length in ms")
    frame_hop_ms: float = Field(10.0, description="Frame hop in ms")
    noise_init_ms: float = Field(
        60.0, description="Duration used to initialise the noise spectrum"
    )
    beta: float = Field(0.8, description="Smoothing factor for prior SNR")
    mcra_alpha: float = Field(0.92, description="Noise recursion factor")
    mcra_delta: float = Field(0.1, description="Probability smoothing factor")
    xi_min: float = Field(1e-3, description="Minimum a priori SNR")
    gain_floor: float = Field(0.05, description="Minimum gain applied")
    ema_alpha: float = Field(0.6, description="Across-frame EMA for envelopes")


@dataclass
class DenoiseConfig:
    """Convenience wrapper bundling both configs."""

    bandpass: BandpassConfig = field(default_factory=BandpassConfig)
    logmmse: LogMMSEConfig = field(default_factory=LogMMSEConfig)


def _design_bandpass(cfg: BandpassConfig) -> FloatArray:
    nyq = 0.5 * cfg.sample_rate
    low = cfg.low_hz / nyq
    high = cfg.high_hz / nyq
    sos = signal.butter(cfg.order, [low, high], btype="bandpass", output="sos")
    return sos


def _apply_notches(x: FloatArray, cfg: BandpassConfig) -> FloatArray:
    y = x
    for freq in cfg.notch_freqs:
        b, a = signal.iirnotch(w0=freq / (cfg.sample_rate / 2), Q=cfg.notch_q)
        y = signal.lfilter(b, a, y)
    return y


class LogMMSEDenoiser:
    """LogMMSE speech enhancer with MCRA noise tracking."""

    def __init__(self, config: DenoiseConfig):
        self.config = config
        self._bp_sos = _design_bandpass(config.bandpass)
        self._frame_len = int(
            round(
                config.logmmse.frame_length_ms
                * config.bandpass.sample_rate
                / 1000.0
            )
        )
        self._frame_hop = int(
            round(
                config.logmmse.frame_hop_ms * config.bandpass.sample_rate / 1000.0
            )
        )
        self._window = signal.hann(self._frame_len, sym=False)
        self._reset_state()

    def _reset_state(self) -> None:
        self._noise_psd: Optional[FloatArray] = None
        self._snr_post_prev: Optional[FloatArray] = None
        self._gain_prev: Optional[FloatArray] = None
        self._ema_envelope: Optional[float] = None
        self._frame_count = 0

    def _initialise_noise(self, frames: FloatArray) -> None:
        noise_bins = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
        self._noise_psd = noise_bins
        self._snr_post_prev = np.ones_like(noise_bins)
        self._gain_prev = np.ones_like(noise_bins)

    def _estimate(self, spectrum: ComplexArray) -> ComplexArray:
        assert self._noise_psd is not None
        cfg = self.config.logmmse

        power_spec = np.abs(spectrum) ** 2
        snr_post = np.clip(power_spec / (self._noise_psd + 1e-12), 1e-12, 1e6)

        if self._snr_post_prev is None:
            self._snr_post_prev = snr_post.copy()
        if self._gain_prev is None:
            self._gain_prev = np.ones_like(snr_post)

        xi_prior = (
            cfg.beta * (self._gain_prev ** 2) * self._snr_post_prev
            + (1.0 - cfg.beta) * np.maximum(snr_post - 1.0, 0.0)
        )
        xi_prior = np.maximum(xi_prior, cfg.xi_min)
        v = xi_prior * snr_post / (1.0 + xi_prior)
        G = (xi_prior / (1.0 + xi_prior)) * np.exp(0.5 * special.expn(1, v))
        G = np.maximum(G, cfg.gain_floor)

        # Update noise floor using an MCRA rule.
        gamma = snr_post
        q = 1.0 / (1.0 + np.exp(-5.0 * (gamma - 1.0)))
        if self._frame_count == 0:
            prob = q
        else:
            prob = cfg.mcra_delta * q + (1.0 - cfg.mcra_delta) * self._speech_prob
        self._speech_prob = prob
        update_mask = prob < 0.5
        self._noise_psd = np.where(
            update_mask,
            cfg.mcra_alpha * self._noise_psd
            + (1.0 - cfg.mcra_alpha) * power_spec,
            self._noise_psd,
        )

        self._snr_post_prev = snr_post
        self._gain_prev = G

        enhanced = G * spectrum
        return enhanced

    def process(self, pcm: FloatArray) -> FloatArray:
        """Enhance a narrow-band PCM waveform.

        Parameters
        ----------
        pcm:
            1-D float array containing mono PCM samples.
        """

        x = np.asarray(pcm, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("pcm must be a 1-D array")

        bp = signal.sosfilt(self._bp_sos, x)
        if self.config.bandpass.notch_freqs:
            bp = _apply_notches(bp, self.config.bandpass)

        frame_len = self._frame_len
        hop = self._frame_hop
        window = self._window

        n_frames = 1 + (len(bp) - frame_len) // hop if len(bp) >= frame_len else 1
        pad = int(max(0, n_frames * hop + frame_len - len(bp)))
        if pad > 0:
            bp = np.pad(bp, (0, pad), mode="reflect")
            n_frames = 1 + (len(bp) - frame_len) // hop

        frames = np.lib.stride_tricks.as_strided(
            bp,
            shape=(n_frames, frame_len),
            strides=(bp.strides[0] * hop, bp.strides[0]),
        ).copy()
        frames *= window[None, :]

        if self._noise_psd is None:
            init_frames = int(
                max(
                    1,
                    round(
                        self.config.logmmse.noise_init_ms
                        / self.config.logmmse.frame_hop_ms
                    ),
                )
            )
            head = frames[:init_frames]
            self._initialise_noise(head)
            self._speech_prob = np.zeros_like(self._noise_psd)

        out = np.zeros(len(bp), dtype=np.float64)
        ola_norm = np.zeros(len(bp), dtype=np.float64)

        for idx in range(n_frames):
            spec = np.fft.rfft(frames[idx])
            enhanced = self._estimate(spec)
            frame_rec = np.fft.irfft(enhanced)
            left = idx * hop
            out[left : left + frame_len] += frame_rec * window
            ola_norm[left : left + frame_len] += window ** 2
            self._frame_count += 1

        valid = ola_norm > 1e-12
        out[valid] /= ola_norm[valid]

        if self._ema_envelope is None:
            self._ema_envelope = float(np.sqrt(np.mean(out**2) + 1e-12))
        ema = self.config.logmmse.ema_alpha
        env = float(np.sqrt(np.mean(out**2) + 1e-12))
        self._ema_envelope = ema * self._ema_envelope + (1.0 - ema) * env
        out /= max(self._ema_envelope, 1e-6)
        out = np.tanh(out)  # guard rails for clip-less stability

        return out[: len(pcm)]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for denoiser")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config")
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/denoise",
        help="Directory to dump diagnostic artefacts",
    )
    return parser


def _load_config(path: Optional[str]) -> DenoiseConfig:
    if path is None:
        return DenoiseConfig()
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
        else:
            import yaml  # lazy import to keep base deps small

            payload = yaml.safe_load(handle)
    bandpass_cfg = BandpassConfig(**payload.get("bandpass", {}))
    log_cfg = LogMMSEConfig(**payload.get("logmmse", {}))
    return DenoiseConfig(bandpass=bandpass_cfg, logmmse=log_cfg)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )


def _smoke_test(config: DenoiseConfig, outdir: str) -> None:
    os.makedirs(os.path.join(outdir, "dumps"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "figs"), exist_ok=True)

    rng = np.random.default_rng(42)
    t = np.arange(0, 0.12, 1.0 / config.bandpass.sample_rate)
    clean = 0.5 * np.sin(2 * np.pi * 800 * t)
    noise = rng.normal(scale=0.3, size=len(t))
    noisy = clean + noise

    denoiser = LogMMSEDenoiser(config)
    enhanced = denoiser.process(noisy)

    dump_path = os.path.join(outdir, "dumps", "denoise_smoke_raw.json")
    with open(dump_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "rms_in": float(np.sqrt(np.mean(noisy**2))),
                "rms_out": float(np.sqrt(np.mean(enhanced**2))),
                "snr_improv_db": float(
                    10.0
                    * np.log10(
                        (np.mean(clean**2) + 1e-12)
                        / (np.mean((enhanced - clean) ** 2) + 1e-12)
                    )
                ),
            },
            handle,
            indent=2,
        )
    logger.info("Saved smoke test summary to %s", dump_path)

    try:
        import matplotlib.pyplot as plt

        fig = os.path.join(outdir, "figs", "denoise_waveforms.png")
        plt.figure(figsize=(8, 4))
        plt.plot(t, noisy, label="noisy", alpha=0.6)
        plt.plot(t, enhanced, label="enhanced", alpha=0.8)
        plt.plot(t, clean, label="clean", alpha=0.8, linestyle="--")
        plt.xlabel("time [s]"); plt.ylabel("amplitude"); plt.legend(); plt.tight_layout()
        plt.savefig(fig, dpi=150)
        plt.close()
        logger.info("Saved waveform figure to %s", fig)
    except Exception as exc:  # pragma: no cover - optional plotting path
        logger.warning("Matplotlib plotting failed: %s", exc)


if __name__ == "__main__":
    _setup_logging()
    parser = _build_argparser()
    args = parser.parse_args()
    cfg = _load_config(args.config)
    _smoke_test(cfg, args.outdir)
