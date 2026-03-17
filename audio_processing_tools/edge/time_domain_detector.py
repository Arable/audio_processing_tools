from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as spsig
from scipy.stats import kurtosis

@dataclass
class TimeDomainDetectorConfig:
    """Configuration for stage-2 time-domain droplet confirmation."""

    fs: int = 11162
    n_fft: int = 256
    hop: int = 128

    # Stage-2 analysis window: previous 1 hop + current frame.
    # With n_fft=256 and hop=128 this gives [t-128, t+256] => 384 samples.
    prev_context_hops: int = 1
    future_context_hops: int = 0

    # If provided, use these bands and sum their filtered outputs.
    # Otherwise fall back to operating_band.
    mode_bands: Optional[List[Tuple[float, float]]] = None
    operating_band: Tuple[float, float] = (400.0, 3500.0)

    bp_order: int = 4

    envelope_smooth_ms: float = 2.0
    peak_prominence_ratio: float = 0.25
    peak_distance_ms: float = 4.0

    # Initial whole-window confirmation checks
    min_crest_factor: float = 3.0
    min_kurtosis: float = 3.5

    eps: float = 1e-9


def build_time_domain_config(params: Dict[str, Any]) -> TimeDomainDetectorConfig:
    """Build standalone stage-2 config from framework-style params."""
    td = dict(params.get("time_domain", {}) or {})
    det = dict(params.get("detector", {}) or {})

    mode_bands_raw = det.get("mode_bands", None)
    mode_bands: Optional[List[Tuple[float, float]]] = None
    if isinstance(mode_bands_raw, (list, tuple)):
        mode_bands = []
        for bb in mode_bands_raw:
            try:
                lo, hi = float(bb[0]), float(bb[1])
            except Exception:
                continue
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                mode_bands.append((lo, hi))

    return TimeDomainDetectorConfig(
        fs=int(params.get("sample_rate", params.get("fs", 11162))),
        n_fft=int(params.get("n_fft", 256)),
        hop=int(params.get("hop", 128)),
        prev_context_hops=int(td.get("prev_context_hops", 1)),
        future_context_hops=int(td.get("future_context_hops", 0)),
        mode_bands=mode_bands,
        operating_band=tuple(params.get("operating_band", (400.0, 3500.0))),
        bp_order=int(td.get("bp_order", 4)),
        envelope_smooth_ms=float(td.get("envelope_smooth_ms", 2.0)),
        peak_prominence_ratio=float(td.get("peak_prominence_ratio", 0.25)),
        peak_distance_ms=float(td.get("peak_distance_ms", 4.0)),
        min_crest_factor=float(td.get("min_crest_factor", 3.0)),
        min_kurtosis=float(td.get("min_kurtosis", 3.5)),
        eps=float(td.get("eps", 1e-9)),
    )


class TimeDomainRainDetector:
    """
    Stage-2 time-domain confirmation.

    Runs only on frames already selected by stage-1.
    Uses:
      - summed mode-band filtered signal
      - local analysis window around each candidate frame
      - smoothed envelope peak picking
      - whole-window crest factor and kurtosis
    """

    def __init__(self, config: Optional[TimeDomainDetectorConfig] = None):
        self.cfg = config
        self._is_setup = config is not None

    def setup(self, params: Dict[str, Any]) -> None:
        """Standalone setup path, similar to SpectralNoiseProcessor."""
        if self._is_setup:
            return
        self.cfg = build_time_domain_config(params)
        self._is_setup = True

    def _build_mode_signal(self, x: np.ndarray, sr: int) -> np.ndarray:
        cfg = self.cfg

        bands: List[Tuple[float, float]] = []
        if isinstance(cfg.mode_bands, (list, tuple)) and len(cfg.mode_bands) > 0:
            for bb in cfg.mode_bands:
                try:
                    lo, hi = float(bb[0]), float(bb[1])
                except Exception:
                    continue
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    bands.append((lo, hi))

        if not bands:
            op_lo, op_hi = cfg.operating_band
            bands = [(float(op_lo), float(op_hi))]

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.float64)

        nyq = 0.5 * float(sr)
        y_sum = np.zeros_like(x, dtype=np.float64)

        for lo, hi in bands:
            lo_c = float(np.clip(lo, 1e-3, nyq * 0.999))
            hi_c = float(np.clip(hi, lo_c + 1e-3, nyq * 0.999))
            wn = [lo_c / nyq, hi_c / nyq]
            sos = spsig.butter(
                int(cfg.bp_order),
                wn,
                btype="bandpass",
                output="sos",
            )

            # Prefer zero-phase filtering, but fall back to causal filtering for
            # very short signals where filtfilt padding requirements are not met.
            try:
                y_band = spsig.sosfiltfilt(sos, x)
            except ValueError:
                y_band = spsig.sosfilt(sos, x)

            y_sum += y_band

        return y_sum

    def _compute_envelope(self, x: np.ndarray, sr: int) -> np.ndarray:
        cfg = self.cfg
        xa = spsig.hilbert(np.asarray(x, dtype=np.float64))
        env = np.abs(xa)

        smooth_len = max(
            1,
            int(round((float(cfg.envelope_smooth_ms) * 1e-3) * float(sr))),
        )
        if smooth_len > 1:
            kernel = np.ones(smooth_len, dtype=np.float64) / float(smooth_len)
            env = np.convolve(env, kernel, mode="same")

        return env

    def _window_bounds(self, frame_idx: int, n_samples: int) -> Tuple[int, int]:
        """
        Window = previous context hops + current frame + future context hops.

        Default configuration uses previous 1 hop + current frame, i.e.
        [t-128, t+256] for n_fft=256 and hop=128, giving 384 samples.
        """
        cfg = self.cfg
        frame_start = int(frame_idx * cfg.hop)
        prev_context = int(max(0, cfg.prev_context_hops) * cfg.hop)
        future_context = int(max(0, cfg.future_context_hops) * cfg.hop)
        cur_len = int(max(1, cfg.n_fft))

        start = max(0, frame_start - prev_context)
        end = min(int(n_samples), frame_start + cur_len + future_context)
        return start, end

    def _analyze_window(
        self,
        x_mode: np.ndarray,
        sr: int,
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        seg = np.asarray(x_mode[start:end], dtype=np.float64)

        if seg.size == 0:
            return {
                "confirmed": False,
                "confirmed_raindrops": 0,
                "n_candidate_peaks": 0,
                "peak_indices_local": np.zeros(0, dtype=np.int32),
                "crest_factor": 0.0,
                "kurtosis": 0.0,
                "window": (int(start), int(end)),
            }

        env = self._compute_envelope(seg, sr)
        env_max = float(np.max(env)) if env.size else 0.0

        prominence = max(
            float(cfg.eps),
            float(cfg.peak_prominence_ratio) * env_max,
        )
        peak_distance = max(
            1,
            int(round((float(cfg.peak_distance_ms) * 1e-3) * float(sr))),
        )

        peak_idx, _ = spsig.find_peaks(
            env,
            prominence=prominence,
            distance=peak_distance,
        )

        rms = float(np.sqrt(np.mean(seg**2) + float(cfg.eps)))
        peak_abs = float(np.max(np.abs(seg))) if seg.size else 0.0
        crest_factor = peak_abs / max(rms, float(cfg.eps))  

        kurt = float(kurtosis(seg, fisher=False, bias=False)) if seg.size >= 4 else 0.0
        
        if not np.isfinite(kurt):
            kurt = 0.0

        confirmed = bool(
            (peak_idx.size > 0)
            and (crest_factor >= float(cfg.min_crest_factor))
            and (kurt >= float(cfg.min_kurtosis))
        )
        confirmed_raindrops = int(peak_idx.size) if confirmed else 0

        return {
            "confirmed": confirmed,
            "confirmed_raindrops": confirmed_raindrops,
            "n_candidate_peaks": int(peak_idx.size),
            "peak_indices_local": peak_idx.astype(np.int32, copy=False),
            "crest_factor": float(crest_factor),
            "kurtosis": float(kurt),
            "window": (int(start), int(end)),
        }

    def process(
        self,
        x: np.ndarray,
        stage1_is_rain: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.cfg is None:
            self.setup({"sample_rate": sr or 11162})

        cfg = self.cfg
        if sr is None:
            sr = cfg.fs

        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if stage1_is_rain is not None:
            stage1_is_rain = np.asarray(stage1_is_rain, dtype=bool).reshape(-1)
            T = int(stage1_is_rain.shape[0])
            run_mask = stage1_is_rain
        else:
            if x.size < int(cfg.n_fft):
                T = 0
            else:
                T = 1 + (x.size - int(cfg.n_fft)) // int(cfg.hop)
            run_mask = np.ones(T, dtype=bool)
            stage1_is_rain = run_mask.copy()

        confirmed_mask = np.zeros(T, dtype=bool)
        confirmed_counts = np.zeros(T, dtype=np.int32)
        crest_factors = np.zeros(T, dtype=np.float64)
        kurtosis_vals = np.zeros(T, dtype=np.float64)
        candidate_peaks = np.zeros(T, dtype=np.int32)
        details: List[Dict[str, Any]] = []

        x_mode = self._build_mode_signal(x, sr)

        for t in range(T):
            if not bool(run_mask[t]):
                continue

            start, end = self._window_bounds(t, x_mode.size)
            info = self._analyze_window(x_mode, sr, start, end)

            confirmed_mask[t] = bool(info["confirmed"])
            confirmed_counts[t] = int(info["confirmed_raindrops"])
            crest_factors[t] = float(info["crest_factor"])
            kurtosis_vals[t] = float(info["kurtosis"])
            candidate_peaks[t] = int(info["n_candidate_peaks"])

            details.append(
                {
                    "frame_idx": int(t),
                    "window": info["window"],
                    "confirmed": bool(info["confirmed"]),
                    "confirmed_raindrops": int(info["confirmed_raindrops"]),
                    "n_candidate_peaks": int(info["n_candidate_peaks"]),
                    "crest_factor": float(info["crest_factor"]),
                    "kurtosis": float(info["kurtosis"]),
                    "peak_indices_local": info["peak_indices_local"],
                }
            )

        return {
            "confirmed_mask": confirmed_mask,
            "confirmed_counts": confirmed_counts,
            "crest_factor": crest_factors,
            "kurtosis": kurtosis_vals,
            "candidate_peaks": candidate_peaks,
            "details": details,
            "x_mode": x_mode,
            "stage1_is_rain": stage1_is_rain,
            "run_mask": run_mask,
        }