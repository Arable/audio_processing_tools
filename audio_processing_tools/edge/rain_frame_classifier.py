from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Tuple, Optional

import librosa
import numpy as np
import scipy.signal as spsig
from scipy.stats import kurtosis
from scipy.signal import peak_widths


class FrameClass(IntEnum):
    """Frame classification used by the rain detector and downstream suppressor."""

    NOISE = 0
    UNCERTAIN = 1
    RAIN = 2


@dataclass
class TimeDomainSoftLabelConfig:
    fs: int = 11162
    frame_len: int = 256
    hop: int = 128

    operating_band: Tuple[float, float] = (400.0, 3500.0)
    time_flux_band: Optional[Tuple[float, float]] = None
    bp_order: int = 4

    # Subframe-energy dynamics (aligned to the classifier hop by default)
    subframe_len: int = 128
    subframe_hop: int = 128
    baseline_win_sec: float = 0.5
    baseline_min_hist_sec: float = 0.25
    baseline_q: float = 20.0

    # Soft-label thresholds (current working defaults)
    time_flux_score_min: float = 1.5
    baseline_floor: float = 1e-6
    crest_factor_min: float = 4.0
    kurtosis_min: float = 6.0
    min_positive_votes: int = 2

    eps: float = 1e-9


class TimeDomainSoftLabeller:
    """
    Simple time-domain soft labeller for rain-like impulsive frames.

    Current design:
      - bandpass filter with operating band for waveform-shape features
      - optional narrower time-flux band for the subframe-energy / flux path
      - 128-point subframe energy
      - 256-point frame energy formed by summing two adjacent subframes
      - causal rolling low-quantile baseline over ~0.5 s on the subframe-energy stream,
        then mapped to frame level
      - frame-level time-domain flux using only frame-to-(t-2) positive rise
      - 256-point frame-level crest factor and kurtosis
      - soft label from vote count
      - warm-up handling so time-flux scoring is trusted only after sufficient
        causal baseline history is available

    Note:
      - decay-based features are currently disabled and can be reintroduced later if needed.
    """

    def __init__(self, config: Optional[TimeDomainSoftLabelConfig] = None):
        self.cfg = config or TimeDomainSoftLabelConfig()

    def _bandpass(self, x: np.ndarray, band: Optional[Tuple[float, float]] = None) -> np.ndarray:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x.copy()

        use_band = cfg.operating_band if band is None else band
        nyq = 0.5 * float(cfg.fs)
        lo = float(np.clip(use_band[0], 1e-3, nyq * 0.999))
        hi = float(np.clip(use_band[1], lo + 1e-3, nyq * 0.999))
        sos = spsig.butter(
            int(cfg.bp_order),
            [lo / nyq, hi / nyq],
            btype="bandpass",
            output="sos",
        )
        try:
            return spsig.sosfiltfilt(sos, x)
        except ValueError:
            return spsig.sosfilt(sos, x)

    def _frame_view(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < cfg.frame_len:
            return np.empty((0, cfg.frame_len), dtype=np.float64)
        T = 1 + (x.size - cfg.frame_len) // cfg.hop
        stride = x.strides[0]
        # NOTE: uses as_strided (no bounds check) — assumes valid frame geometry
        return np.lib.stride_tricks.as_strided(
            x,
            shape=(T, cfg.frame_len),
            strides=(cfg.hop * stride, stride),
            writeable=False,
        )

    def _subframe_energy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        B = int(max(1, cfg.subframe_len))
        H = int(max(1, cfg.subframe_hop))
        if x.size < B:
            energy = np.array([float(np.mean(x**2))], dtype=np.float64)
            times = np.array([0.0], dtype=np.float64)
            return energy, times

        vals = []
        times = []
        for start in range(0, x.size - B + 1, H):
            seg = x[start : start + B]
            vals.append(float(np.mean(seg**2)))
            times.append(start / float(cfg.fs))
        return np.asarray(vals, dtype=np.float64), np.asarray(times, dtype=np.float64)

    def _rolling_low_quantile_baseline(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        T = x.size
        if T == 0:
            return x.copy(), np.zeros(0, dtype=bool)

        q = float(np.clip(cfg.baseline_q, 0.0, 100.0)) / 100.0
        subframes_per_sec = float(cfg.fs) / max(float(cfg.subframe_hop), 1.0)
        W = max(3, int(round(float(cfg.baseline_win_sec) * subframes_per_sec)))
        min_hist = max(1, int(round(float(cfg.baseline_min_hist_sec) * subframes_per_sec)))

        out = np.empty(T, dtype=np.float64)
        warm_ok = np.zeros(T, dtype=bool)
        for t in range(T):
            i0 = max(0, t - W)
            i1 = t  # causal: exclude current sample from baseline estimate
            if i1 <= i0:
                out[t] = cfg.baseline_floor
                warm_ok[t] = False
            else:
                hist = x[i0:i1]
                out[t] = np.quantile(hist, q)
                warm_ok[t] = hist.size >= min_hist

        out = np.nan_to_num(out, nan=cfg.baseline_floor, posinf=cfg.baseline_floor, neginf=cfg.baseline_floor)
        out = np.maximum(out, cfg.baseline_floor)
        return out, warm_ok

    def _frame_baseline_from_subframes(self, sub_baseline: np.ndarray, n_frames: int) -> np.ndarray:
        sub_baseline = np.asarray(sub_baseline, dtype=np.float64).reshape(-1)
        out = np.zeros(n_frames, dtype=np.float64)
        if n_frames == 0 or sub_baseline.size == 0:
            return out

        for t in range(n_frames):
            b0 = sub_baseline[t] if t < sub_baseline.size else 0.0
            b1 = sub_baseline[t + 1] if (t + 1) < sub_baseline.size else 0.0
            out[t] = float(b0 + b1)
        return out

    def _frame_energy_from_subframes(self, sub_energy: np.ndarray, n_frames: int) -> np.ndarray:
        sub_energy = np.asarray(sub_energy, dtype=np.float64).reshape(-1)
        out = np.zeros(n_frames, dtype=np.float64)
        if n_frames == 0 or sub_energy.size == 0:
            return out

        for t in range(n_frames):
            e0 = sub_energy[t] if t < sub_energy.size else 0.0
            e1 = sub_energy[t + 1] if (t + 1) < sub_energy.size else 0.0
            out[t] = float(e0 + e1)
        return out

    def process(self, x: np.ndarray) -> Dict[str, Any]:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x_bp = self._bandpass(x, cfg.operating_band)
        flux_band = cfg.time_flux_band if cfg.time_flux_band is not None else cfg.operating_band
        x_flux = self._bandpass(x, flux_band)
        frames = self._frame_view(x_bp)
        T = frames.shape[0]

        frame_times = (np.arange(T) * cfg.hop) / float(cfg.fs)
        sub_energy, sub_times = self._subframe_energy(x_flux)
        sub_baseline, sub_warm_ok = self._rolling_low_quantile_baseline(sub_energy)
        frame_energy = self._frame_energy_from_subframes(sub_energy, T)
        frame_baseline = self._frame_baseline_from_subframes(sub_baseline, T)
        frame_warm_ok = np.zeros(T, dtype=bool)
        if sub_warm_ok.size > 0:
            for t in range(T):
                ok0 = bool(sub_warm_ok[t]) if t < sub_warm_ok.size else False
                ok1 = bool(sub_warm_ok[t + 1]) if (t + 1) < sub_warm_ok.size else False
                frame_warm_ok[t] = ok0 and ok1
        baseline_ref = np.maximum(frame_baseline, cfg.baseline_floor)

        frame_flux = np.zeros(T, dtype=np.float64)
        time_flux_score = np.zeros(T, dtype=np.float64)
        crest_factor = np.zeros(T, dtype=np.float64)
        kurt_vals = np.zeros(T, dtype=np.float64)
        vote_count = np.zeros(T, dtype=np.int32)
        soft_score = np.zeros(T, dtype=np.float64)
        soft_label = np.zeros(T, dtype=bool)

        if T > 2:
            frame_flux[2:] = np.maximum(frame_energy[2:] - frame_energy[:-2], 0.0)

        time_flux_score = frame_flux / baseline_ref
        time_flux_score = np.where(frame_warm_ok, time_flux_score, 0.0)

        for t in range(T):
            seg = np.asarray(frames[t], dtype=np.float64)
            rms = float(np.sqrt(np.mean(seg**2) + cfg.eps))
            peak_abs = float(np.max(np.abs(seg))) if seg.size else 0.0
            crest_factor[t] = peak_abs / max(rms, cfg.eps)

            if seg.size >= 4:
                kv = float(kurtosis(seg, fisher=False, bias=False))
                kurt_vals[t] = kv if np.isfinite(kv) else 0.0
            else:
                kurt_vals[t] = 0.0

            votes = 0
            votes += int(time_flux_score[t] >= cfg.time_flux_score_min)
            votes += int(crest_factor[t] >= cfg.crest_factor_min)
            votes += int(kurt_vals[t] >= cfg.kurtosis_min)
            vote_count[t] = votes
            soft_score[t] = votes / 3.0
            soft_label[t] = votes >= int(cfg.min_positive_votes)

        return {
            "x_bp": x_bp,
            "x_flux": x_flux,
            "frame_times": frame_times,
            "sub_energy": sub_energy,
            "sub_times": sub_times,
            "sub_baseline": sub_baseline,
            "sub_warm_ok": sub_warm_ok,
            "frame_energy": frame_energy,
            "frame_baseline": frame_baseline,
            "frame_warm_ok": frame_warm_ok,
            "frame_flux": frame_flux,
            "time_flux_score": time_flux_score,
            "crest_factor": crest_factor,
            "kurtosis": kurt_vals,
            "vote_count": vote_count,
            "soft_score": soft_score,
            "soft_label": soft_label,
            "config": cfg,
        }


class RainFrameClassifierMixin:
    """
    Rain / Noise frame classifier.

    Detector parameters are resolved with precedence:

        1) cfg.detector[name]
        2) getattr(cfg, name)   (legacy flat config)
        3) internal default

    SpectralNoiseProcessor must provide self.cfg.
    """

    # Required detector fields
    REQUIRED_CFG_FIELDS = (
        "mode_bands",
        "primary_mode_idx",
    )

    # ------------------------------------------------------------
    # Detector override helpers
    # ------------------------------------------------------------

    def _dget(self, name: str, default: Any = None) -> Any:
        """Get detector parameter with precedence: cfg.detector -> cfg attr -> default."""
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return default

        det = getattr(cfg, "detector", None)
        if isinstance(det, dict) and name in det:
            return det[name]

        if hasattr(cfg, name):
            return getattr(cfg, name)

        return default

    def _dhas(self, name: str) -> bool:
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return False

        det = getattr(cfg, "detector", None)
        if isinstance(det, dict) and name in det:
            return True

        return hasattr(cfg, name)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def _validate_rain_cfg(self):
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise AttributeError("self.cfg is missing in processor")

        missing = [f for f in self.REQUIRED_CFG_FIELDS if not self._dhas(f)]
        if missing:
            raise AttributeError(
                "RainFrameClassifierMixin missing required detector fields: "
                f"{missing}. Provide them under cfg.detector (preferred) "
                "or as flat cfg attributes."
            )

    # ------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------

    def _detect_rain_over_time(
        self,
        P: np.ndarray,
        freqs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Returns
        -------
        frame_class : np.ndarray
            Per-frame FrameClass values encoded as int8.
        rain_conf : np.ndarray
            Per-frame rain confidence in [0, 1].
        det_debug : Dict[str, Any]
            Detector diagnostics and intermediate signals.
        feature_dump : Dict[str, Any]
            Raw feature arrays needed to replay threshold logic offline.
        """

        self._validate_rain_cfg()

        eps = float(self._dget("eps", 1e-9))

        include_peak_payload = bool(self._dget("feature_dump_include_peak_payload", False))
        include_td_soft_feature_dump = bool(self._dget("feature_dump_include_td_soft", True))

        op_band = self._dget("operating_band", (400.0, 3500.0))
        op_lo, op_hi = float(op_band[0]), float(op_band[1])

        mode_bands = self._dget("mode_bands", None)
        if mode_bands is None:
            raise AttributeError("Missing required detector param: mode_bands")

        mode_bands = tuple((float(a), float(b)) for (a, b) in mode_bands)

        # Optional time-domain soft-labeller configuration.
        td_soft_enable = bool(self._dget("td_soft_enable", False))
        td_soft_bp_order = int(self._dget("td_soft_bp_order", 4))
        td_soft_time_flux_band = self._dget("td_soft_time_flux_band", None)
        if td_soft_time_flux_band is not None:
            td_soft_time_flux_band = (
                float(td_soft_time_flux_band[0]),
                float(td_soft_time_flux_band[1]),
            )
        td_soft_subframe_len = int(self._dget("td_soft_subframe_len", 128))
        td_soft_subframe_hop = int(self._dget("td_soft_subframe_hop", 128))
        td_soft_baseline_win_sec = float(self._dget("td_soft_baseline_win_sec", 0.5))
        td_soft_baseline_q = float(self._dget("td_soft_baseline_q", 20.0))
        td_soft_baseline_min_hist_sec = float(self._dget("td_soft_baseline_min_hist_sec", 0.25))
        td_soft_time_flux_score_min = float(self._dget("td_soft_time_flux_score_min", 1.5))
        td_soft_baseline_floor = float(self._dget("td_soft_baseline_floor", 1e-6))
        td_soft_crest_factor_min = float(self._dget("td_soft_crest_factor_min", 4.0))
        td_soft_kurtosis_min = float(self._dget("td_soft_kurtosis_min", 6.0))
        td_soft_min_positive_votes = int(self._dget("td_soft_min_positive_votes", 2))

        primary_mode_idx = int(self._dget("primary_mode_idx", 0))
        if primary_mode_idx < 0 or primary_mode_idx >= len(mode_bands):
            raise ValueError(
                f"primary_mode_idx ({primary_mode_idx}) out of range for "
                f"mode_bands length ({len(mode_bands)})"
            )

        # noise_hi remains part of NOISE frame assignment.
        noise_hi = float(self._dget("noise_hi", 0.80))


        # New flux-based detector controls.
        # mode_flux_rain_min: main novelty threshold in the mode bands.
        # primary_flux_sanity_min: conservative primary-band sanity check.
        # mode_flux_noise_max: frames below this are eligible for NOISE when peak/rain evidence is weak.
        # The detector score is normalized excess-over-baseline novelty; rain_conf
        # is derived from the rain/noise thresholds directly rather than from a
        # separate arbitrary confidence scale.
        mode_flux_rain_min = float(self._dget("mode_flux_rain_min", 4.0))
        primary_flux_sanity_min = float(self._dget("primary_flux_sanity_min", 3.0))
        mode_flux_noise_max = float(self._dget("mode_flux_noise_max", 1.5))
        mode_flux_rain_min = max(mode_flux_rain_min, 0.0)
        primary_flux_sanity_min = max(primary_flux_sanity_min, 0.0)
        mode_flux_noise_max = max(mode_flux_noise_max, 0.0)

        # Local noise-level normalization for mode flux.
        # Normalize spectral novelty by a rolling low-quantile baseline,
        # which is more stable than mean/std for sparse impulsive rain.
        mode_flux_norm_enable = bool(self._dget("mode_flux_norm_enable", True))
        mode_flux_norm_win_sec = float(self._dget("mode_flux_norm_win_sec", 0.5))
        mode_flux_norm_q = float(self._dget("mode_flux_norm_q", 20.0))
        mode_flux_norm_q = float(np.clip(mode_flux_norm_q, 0.0, 100.0))
        mode_flux_norm_min = float(self._dget("mode_flux_norm_min", 1.0))
        mode_flux_norm_min = max(mode_flux_norm_min, eps)


        # Optional winsorization of mode-flux before z-score normalization.
        # This reduces the leverage of extremely large spikes so nearby moderate
        # raindrops are not made less relevant by a few outliers.
        flux_modes_winsor_enable = bool(self._dget("flux_modes_winsor_enable", False))
        flux_modes_winsor_q = float(self._dget("flux_modes_winsor_q", 99.0))
        flux_modes_winsor_q = float(np.clip(flux_modes_winsor_q, 50.0, 100.0))

        mode_weights = self._dget("mode_weights", None)
        if mode_weights is not None:
            mode_weights = tuple(float(w) for w in mode_weights)
            if len(mode_weights) != len(mode_bands):
                raise ValueError(
                    f"mode_weights length ({len(mode_weights)}) "
                    f"must match mode_bands length ({len(mode_bands)})"
                )

        # Precompute masks once; these do not change across frames.
        F, T = P.shape
        td_soft_debug: Dict[str, Any] = {}
        if td_soft_enable and self._dhas("last_input_audio"):
            try:
                x_in = np.asarray(getattr(self.cfg, "last_input_audio"), dtype=np.float64).reshape(-1)
                td_soft_cfg = TimeDomainSoftLabelConfig(
                    fs=int(self._dget("sample_rate", self._dget("fs", 11162))),
                    frame_len=int(self._dget("n_fft", 256)),
                    hop=int(self._dget("hop", 128)),
                    operating_band=(op_lo, op_hi),
                    time_flux_band=td_soft_time_flux_band,
                    bp_order=td_soft_bp_order,
                    subframe_len=td_soft_subframe_len,
                    subframe_hop=td_soft_subframe_hop,
                    baseline_win_sec=td_soft_baseline_win_sec,
                    baseline_min_hist_sec=td_soft_baseline_min_hist_sec,
                    baseline_q=td_soft_baseline_q,
                    time_flux_score_min=td_soft_time_flux_score_min,
                    baseline_floor=td_soft_baseline_floor,
                    crest_factor_min=td_soft_crest_factor_min,
                    kurtosis_min=td_soft_kurtosis_min,
                    min_positive_votes=td_soft_min_positive_votes,
                    eps=eps,
                )
                td_soft_debug = TimeDomainSoftLabeller(td_soft_cfg).process(x_in)
            except Exception as e:
                td_soft_debug = {"error": str(e)}
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)

        if P.shape[0] != freqs.shape[0]:
            raise ValueError(
                f"P.shape[0] ({P.shape[0]}) must match freqs.shape[0] ({freqs.shape[0]})"
            )
        if not np.any(band_mask):
            raise ValueError(
                f"operating_band {op_band} does not overlap the provided frequency grid"
            )

        P_band = P[band_mask, :]
        freqs_band = freqs[band_mask]

        primary_lo, primary_hi = mode_bands[primary_mode_idx]
        primary_mask = (freqs_band >= primary_lo) & (freqs_band <= primary_hi)
        if not np.any(primary_mask):
            raise ValueError(
                f"primary mode band {(primary_lo, primary_hi)} has no bins inside operating_band {op_band}"
            )

        # Mode bands are expected to be non-overlapping for interpretable weighted flux.
        mode_masks = []
        for lo, hi in mode_bands:
            m_mask = (freqs_band >= lo) & (freqs_band <= hi)
            mode_masks.append(m_mask)
        if not any(np.any(m) for m in mode_masks):
            raise ValueError("No mode band overlaps the operating band")

        peak_top_p = int(self._dget("peak_top_p", 6))
        primary_top_m = int(self._dget("primary_top_m", 3))
        peak_prominence_db = float(self._dget("peak_prominence_db", 3.0))
        peak_min_db_above_floor = float(self._dget("peak_min_db_above_floor", 6.0))
        peak_ratio_min = float(self._dget("peak_ratio_min", 0.50))
        peak_top_p = max(1, peak_top_p)
        primary_top_m = max(1, primary_top_m)
        peak_ratio_min = float(np.clip(peak_ratio_min, 0.0, 1.0))

        flux_primary = np.full(T, np.nan, dtype=np.float64)
        flux_modes = np.full(T, np.nan, dtype=np.float64)

        peak_ratio = np.full(T, np.nan, dtype=np.float64)
        peak_in_mode_count = np.full(T, np.nan, dtype=np.float64)
        peak_in_primary_count = np.full(T, np.nan, dtype=np.float64)
        # Number of peaks retained after truncating to the strongest top-P peaks.
        peak_total_count = np.full(T, np.nan, dtype=np.float64)
        # Diagnostics to help tune the classifier
        peak_detected_count = np.full(T, np.nan, dtype=np.float64)
        primary_ok_dbg = np.full(T, np.nan, dtype=np.float64)
        mode_ok_dbg = np.full(T, np.nan, dtype=np.float64)
        # Binary peak gate debug stream (0.0 fail, 1.0 pass). Kept as a "score"
        # name for possible future soft-scoring extensions.
        peak_gate_score = np.full(T, np.nan, dtype=np.float64)

        # Per-mode raw / normalized flux features for offline threshold tuning.
        mode_flux_by_mode = np.zeros((len(mode_bands), T), dtype=np.float64)
        mode_flux_baseline_by_mode = np.zeros((len(mode_bands), T), dtype=np.float64)
        mode_flux_excess_by_mode = np.zeros((len(mode_bands), T), dtype=np.float64)
        normalized_mode_flux_by_mode = np.zeros((len(mode_bands), T), dtype=np.float64)
        mode_to_primary_ratio = np.zeros((len(mode_bands), T), dtype=np.float64)

        # Detailed per-peak payloads are optional because they are expensive for
        # large batch runs. Keep them disabled by default and compute/store them
        # only when explicitly requested for inspection.
        if include_peak_payload:
            peak_valid_freqs_hz = np.empty(T, dtype=object)
            peak_valid_prominences_db = np.empty(T, dtype=object)
            peak_valid_bandwidths_hz = np.empty(T, dtype=object)
            peak_valid_heights_db = np.empty(T, dtype=object)
            peak_top_p_freqs_hz = np.empty(T, dtype=object)
            peak_top_p_prominences_db = np.empty(T, dtype=object)
            peak_top_p_bandwidths_hz = np.empty(T, dtype=object)
            peak_top_p_heights_db = np.empty(T, dtype=object)
            peak_top_p_in_mode_mask = np.empty(T, dtype=object)
            peak_top_p_in_primary_mask = np.empty(T, dtype=object)
        else:
            peak_valid_freqs_hz = None
            peak_valid_prominences_db = None
            peak_valid_bandwidths_hz = None
            peak_valid_heights_db = None
            peak_top_p_freqs_hz = None
            peak_top_p_prominences_db = None
            peak_top_p_bandwidths_hz = None
            peak_top_p_heights_db = None
            peak_top_p_in_mode_mask = None
            peak_top_p_in_primary_mask = None


        prev_frame_1 = None  # frame at t-1
        prev_frame_2 = None  # frame at t-2

        for t in range(T):
            frame = P_band[:, t]
            if include_peak_payload:
                peak_valid_freqs_hz[t] = np.array([], dtype=np.float64)
                peak_valid_prominences_db[t] = np.array([], dtype=np.float64)
                peak_valid_bandwidths_hz[t] = np.array([], dtype=np.float64)
                peak_valid_heights_db[t] = np.array([], dtype=np.float64)
                peak_top_p_freqs_hz[t] = np.array([], dtype=np.float64)
                peak_top_p_prominences_db[t] = np.array([], dtype=np.float64)
                peak_top_p_bandwidths_hz[t] = np.array([], dtype=np.float64)
                peak_top_p_heights_db[t] = np.array([], dtype=np.float64)
                peak_top_p_in_mode_mask[t] = np.array([], dtype=bool)
                peak_top_p_in_primary_mask[t] = np.array([], dtype=bool)

            if prev_frame_1 is None:
                # First frame: no previous reference available.
                flux_primary[t] = 0.0
                flux_modes[t] = 0.0
                peak_detected_count[t] = 0.0
                peak_total_count[t] = 0.0
                peak_in_mode_count[t] = 0.0
                peak_in_primary_count[t] = 0.0
                peak_ratio[t] = 0.0
                primary_ok_dbg[t] = 0.0
                mode_ok_dbg[t] = 0.0
                peak_gate_score[t] = 0.0
                prev_frame_1 = frame
                continue

            if prev_frame_2 is None:
                # Second frame: still warming up the t-2 reference.
                # Keep flux at zero so all later frames use a consistent delay definition.
                flux = np.zeros_like(frame)
                prev_frame_2 = prev_frame_1
                prev_frame_1 = frame
            else:
                # Use only the non-overlapping t-vs-(t-2) positive rise.
                delta2 = frame - prev_frame_2
                d2_pos = np.maximum(delta2, 0.0)
                flux = d2_pos
                prev_frame_2 = prev_frame_1
                prev_frame_1 = frame

            # Primary mode
            flux_primary[t] = float(np.sum(flux[primary_mask]))

            # All modes
            total_flux_modes = 0.0
            for i, m_mask in enumerate(mode_masks):
                mode_flux_i = float(np.sum(flux[m_mask]))
                mode_flux_by_mode[i, t] = mode_flux_i
                weight = mode_weights[i] if mode_weights is not None else 1.0
                total_flux_modes += weight * mode_flux_i

            flux_modes[t] = total_flux_modes

            # --- Peak structure: among the strongest peaks, require primary-band presence
            #     near the top and enough overall concentration inside the expected mode bands. ---
            spec_db = frame
            floor_db = float(np.median(spec_db))
            height_thresh = floor_db + peak_min_db_above_floor

            peaks, props = spsig.find_peaks(
                spec_db,
                prominence=peak_prominence_db,
                height=height_thresh,
            )
            if peaks.size == 0:
                peak_detected_count[t] = 0.0
                peak_total_count[t] = 0.0
                peak_in_mode_count[t] = 0.0
                peak_in_primary_count[t] = 0.0
                peak_ratio[t] = 0.0
                primary_ok_dbg[t] = 0.0
                mode_ok_dbg[t] = 0.0
                peak_gate_score[t] = 0.0
            else:
                pk_h = np.asarray(props.get("peak_heights", spec_db[peaks]), dtype=np.float64)
                pk_prom = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=np.float64)
                widths_bins, *_ = peak_widths(spec_db, peaks, rel_height=0.5)
                df_hz = float(freqs_band[1] - freqs_band[0]) if freqs_band.size > 1 else 0.0
                pk_bw_hz = np.asarray(widths_bins, dtype=np.float64) * df_hz

                # 1) All peaks satisfying the requested 3-6 dB prominence criterion.
                valid_prom_mask = (pk_prom >= 3.0) & (pk_prom <= 6.0)
                if include_peak_payload and np.any(valid_prom_mask):
                    peak_valid_freqs_hz[t] = freqs_band[peaks[valid_prom_mask]].astype(np.float64)
                    peak_valid_prominences_db[t] = pk_prom[valid_prom_mask].astype(np.float64)
                    peak_valid_bandwidths_hz[t] = pk_bw_hz[valid_prom_mask].astype(np.float64)
                    peak_valid_heights_db[t] = pk_h[valid_prom_mask].astype(np.float64)

                # Strongest top-P peaks for gate computation.
                order = np.argsort(pk_h)[::-1]
                sel = peaks[order[:peak_top_p]]
                sel_h = pk_h[order[:peak_top_p]]
                sel_prom = pk_prom[order[:peak_top_p]]
                sel_bw_hz = pk_bw_hz[order[:peak_top_p]]

                in_primary = primary_mask[sel]
                in_any_mode = np.zeros(sel.size, dtype=bool)
                for m_mask in mode_masks:
                    in_any_mode |= m_mask[sel]

                # 4) top-P peaks that also satisfy the 3-6 dB prominence criterion
                # and are inside the expected mode bands.
                if include_peak_payload:
                    sel_valid_prom = (sel_prom >= 3.0) & (sel_prom <= 6.0)
                    sel_keep = sel_valid_prom & in_any_mode
                    peak_top_p_freqs_hz[t] = freqs_band[sel[sel_keep]].astype(np.float64)
                    peak_top_p_prominences_db[t] = sel_prom[sel_keep].astype(np.float64)
                    peak_top_p_bandwidths_hz[t] = sel_bw_hz[sel_keep].astype(np.float64)
                    peak_top_p_heights_db[t] = sel_h[sel_keep].astype(np.float64)
                    peak_top_p_in_mode_mask[t] = in_any_mode.astype(bool)
                    peak_top_p_in_primary_mask[t] = in_primary.astype(bool)

                top_m = min(primary_top_m, sel.size)
                total = float(sel.size)
                detected_total = float(peaks.size)
                in_m = float(np.sum(in_any_mode))
                in_p = float(np.sum(in_primary))
                ratio = in_m / float(max(1, sel.size))

                primary_ok = float(np.any(in_primary[:top_m]))
                mode_ok = float(ratio >= peak_ratio_min)

                peak_detected_count[t] = detected_total
                peak_total_count[t] = total
                peak_in_mode_count[t] = in_m
                peak_in_primary_count[t] = in_p
                peak_ratio[t] = ratio
                primary_ok_dbg[t] = primary_ok
                mode_ok_dbg[t] = mode_ok
                peak_gate_score[t] = min(primary_ok, mode_ok)


        def rolling_low_quantile_baseline(x: np.ndarray) -> np.ndarray:
            """Rolling low-quantile baseline for sparse impulsive flux signals."""
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            Tloc = x.shape[0]
            if Tloc == 0:
                return x.copy()

            hop = float(self._dget("hop", 128))
            fs = float(self._dget("sample_rate", self._dget("fs", 11162)))
            frames_per_sec = fs / max(hop, 1.0)
            Wsec = max(1e-3, float(mode_flux_norm_win_sec))
            W = max(3, int(round(Wsec * frames_per_sec)))
            if W % 2 == 0:
                W += 1
            half = W // 2

            q = float(mode_flux_norm_q) / 100.0
            out = np.empty(Tloc, dtype=np.float64)
            for t in range(Tloc):
                i0 = max(0, t - half)
                i1 = min(Tloc, t + half + 1)
                out[t] = np.quantile(x[i0:i1], q)

            out = np.nan_to_num(
                out,
                nan=mode_flux_norm_min,
                posinf=mode_flux_norm_min,
                neginf=mode_flux_norm_min,
            )
            out = np.maximum(out, mode_flux_norm_min)
            return out

        flux_modes_raw = flux_modes.copy()
        flux_modes_proc = flux_modes.copy()
        flux_modes_winsor_hi = np.nan

        if flux_modes_winsor_enable:
            finite_mask = np.isfinite(flux_modes_proc)
            if np.any(finite_mask):
                flux_modes_winsor_hi = float(np.percentile(flux_modes_proc[finite_mask], flux_modes_winsor_q))
                flux_modes_proc = np.minimum(flux_modes_proc, flux_modes_winsor_hi)

        # Local flux normalization: convert raw/winsorized mode novelty into an
        # excess-over-baseline score. Using (flux - baseline) in the numerator
        # avoids rewarding tiny absolute novelty just because the baseline is
        # very small, while the denominator keeps the score adaptive.
        mode_flux_baseline = rolling_low_quantile_baseline(flux_modes_proc)
        mode_flux_excess = np.maximum(flux_modes_proc - mode_flux_baseline, 0.0)
        if mode_flux_norm_enable:
            mode_flux_score = mode_flux_excess / (mode_flux_baseline + mode_flux_norm_min)
        else:
            mode_flux_score = mode_flux_excess.copy()

        # 2) normalized_mode_flux for each mode separately.
        for i in range(len(mode_bands)):
            baseline_i = rolling_low_quantile_baseline(mode_flux_by_mode[i])
            excess_i = np.maximum(mode_flux_by_mode[i] - baseline_i, 0.0)
            if mode_flux_norm_enable:
                score_i = excess_i / (baseline_i + mode_flux_norm_min)
            else:
                score_i = excess_i.copy()
            mode_flux_baseline_by_mode[i] = baseline_i
            mode_flux_excess_by_mode[i] = excess_i
            normalized_mode_flux_by_mode[i] = np.nan_to_num(
                score_i,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # 3) Per-mode fraction of the total mode flux.
        # This is more stable than dividing by the primary-mode flux, which can
        # become numerically tiny and create extreme ratios.
        total_mode_flux = np.maximum(np.sum(mode_flux_by_mode, axis=0), eps)
        mode_to_primary_ratio = mode_flux_by_mode / total_mode_flux[np.newaxis, :]
        mode_to_primary_ratio = np.nan_to_num(
            mode_to_primary_ratio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


        # Peak confirmation gate: among the strongest peaks, require at least one
        # primary-band peak in the top-M peaks and enough overall concentration of
        # peaks inside the expected mode bands.
        peak_gate = peak_gate_score >= 1.0

        # Flux-based detector logic.
        # Main decision variable is the normalized mode-band novelty.
        # Primary-band novelty is used only as a simple absolute sanity gate.
        # Frequency structure remains the final spectral confirmation gate.
        mode_flux_score = np.nan_to_num(mode_flux_score, nan=0.0, posinf=0.0, neginf=0.0)
        primary_flux_sanity = np.nan_to_num(flux_primary, nan=0.0, posinf=0.0, neginf=0.0) >= primary_flux_sanity_min
        primary_sanity = primary_flux_sanity
        mode_flux_rain_pass = mode_flux_score >= mode_flux_rain_min

        # Map the detector score into [0,1] for downstream use and plotting.
        # This scaling is not used for the core rain/noise decision, which is
        # driven directly by `mode_flux_rain_min` and `mode_flux_noise_max`.
        conf_denom = max(mode_flux_rain_min - mode_flux_noise_max, eps)
        rain_conf = np.clip(
            (mode_flux_score - mode_flux_noise_max) / conf_denom,
            0.0,
            1.0,
        )
        # (base_rain_conf removed: no longer needed.)

        # Raw rain decision from the classifier. Any persistence / hold behavior
        # should be handled downstream (for example by the noise suppressor), not
        # inside this detector.
        is_rain_raw = mode_flux_rain_pass & peak_gate & primary_sanity
        is_rain = is_rain_raw.copy()

        # Onset frames: first raw-rain frame after a non-rain frame.
        onset_mask = is_rain_raw & (~np.r_[False, is_rain_raw[:-1]])
        onset_indices = np.flatnonzero(onset_mask).astype(np.int32)

        # Noise confidence is now tied to weak mode novelty rather than the z-score map.
        # This keeps the raw detector simple and physically tied to spectral novelty.
        noise_conf = np.clip(1.0 - (mode_flux_score / max(mode_flux_noise_max, eps)), 0.0, 1.0)
        weak_mode_flux = mode_flux_score <= mode_flux_noise_max

        # FrameClass is the canonical raw detector output.
        frame_class = np.full(T, FrameClass.UNCERTAIN, dtype=np.int8)
        frame_class[(noise_conf >= noise_hi) & weak_mode_flux & (~is_rain)] = FrameClass.NOISE
        frame_class[is_rain] = FrameClass.RAIN


        # PSD update gating is intentionally not decided here.
        # Downstream suppressor should derive it from FrameClass.

        det_debug = {
            # Inter-frame evidence
            "flux_primary": flux_primary,
            "flux_modes": flux_modes_raw,
            "flux_modes_proc": flux_modes_proc,
            "flux_modes_winsor_hi": flux_modes_winsor_hi,
            "mode_flux_baseline": mode_flux_baseline,
            "mode_flux_excess": mode_flux_excess,
            "mode_flux_score": mode_flux_score,
            "mode_flux_rain_pass": mode_flux_rain_pass,
            "mode_flux_by_mode": mode_flux_by_mode,
            "mode_flux_baseline_by_mode": mode_flux_baseline_by_mode,
            "mode_flux_excess_by_mode": mode_flux_excess_by_mode,
            "normalized_mode_flux_by_mode": normalized_mode_flux_by_mode,
            # Note: kept historical key name for backward compatibility; values
            # now represent per-mode fraction of total mode flux.
            "mode_to_primary_ratio": mode_to_primary_ratio,
            "rain_conf": rain_conf,
            "primary_flux_sanity": primary_flux_sanity,
            "primary_sanity": primary_sanity,
            "noise_conf": noise_conf,

            # Intra-frame peak structure
            "peak_detected_count": peak_detected_count,
            "peak_total_count": peak_total_count,
            "peak_in_mode_count": peak_in_mode_count,
            "peak_in_primary_count": peak_in_primary_count,
            "peak_ratio": peak_ratio,
            "primary_ok": primary_ok_dbg,
            "mode_ok": mode_ok_dbg,
            "peak_gate_score": peak_gate_score,
            "peak_gate": peak_gate,

            # Final detector decisions
            "is_rain_raw": is_rain_raw,
            "onset_mask": onset_mask,
            "onset_indices": onset_indices,
            "frame_class": frame_class,
            "frame_class_name": np.array(
                [FrameClass(v).name.lower() for v in frame_class], dtype=object
            ),
            "td_soft": td_soft_debug,

            # Thresholds / detector settings useful during tuning
            "tuning_params": {
                "noise_hi": noise_hi,
                "mode_flux_rain_min": mode_flux_rain_min,
                "primary_flux_sanity_min": primary_flux_sanity_min,
                "mode_flux_noise_max": mode_flux_noise_max,
                "rain_conf_threshold_mapped_from_score": True,
                "mode_flux_norm_enable": mode_flux_norm_enable,
                "mode_flux_norm_win_sec": mode_flux_norm_win_sec,
                "mode_flux_norm_q": mode_flux_norm_q,
                "mode_flux_norm_min": mode_flux_norm_min,
                "peak_top_p": peak_top_p,
                "primary_top_m": primary_top_m,
                "peak_ratio_min": peak_ratio_min,
                "peak_prominence_db": peak_prominence_db,
                "peak_min_db_above_floor": peak_min_db_above_floor,
                "flux_modes_winsor_enable": flux_modes_winsor_enable,
                "flux_modes_winsor_q": flux_modes_winsor_q,
                "primary_mode_idx": primary_mode_idx,
                "mode_ratio_definition": "mode_flux_over_total_mode_flux",
                "mode_bands": mode_bands,
                "td_soft_enable": td_soft_enable,
                "td_soft_time_flux_band": td_soft_time_flux_band,
                "td_soft_time_flux_score_min": td_soft_time_flux_score_min,
                "td_soft_crest_factor_min": td_soft_crest_factor_min,
                "td_soft_kurtosis_min": td_soft_kurtosis_min,
                "td_soft_min_positive_votes": td_soft_min_positive_votes,
                "feature_dump_include_peak_payload": include_peak_payload,
                "feature_dump_include_td_soft": include_td_soft_feature_dump,
            },
        }

        if include_peak_payload:
            det_debug.update({
                "peak_valid_freqs_hz": peak_valid_freqs_hz,
                "peak_valid_prominences_db": peak_valid_prominences_db,
                "peak_valid_bandwidths_hz": peak_valid_bandwidths_hz,
                "peak_valid_heights_db": peak_valid_heights_db,
                "peak_top_p_freqs_hz": peak_top_p_freqs_hz,
                "peak_top_p_prominences_db": peak_top_p_prominences_db,
                "peak_top_p_bandwidths_hz": peak_top_p_bandwidths_hz,
                "peak_top_p_heights_db": peak_top_p_heights_db,
                "peak_top_p_in_mode_mask": peak_top_p_in_mode_mask,
                "peak_top_p_in_primary_mask": peak_top_p_in_primary_mask,
            })

        feature_dump_level = int(self._dget("feature_dump_level", 0))
        feature_dump: Dict[str, Any] = {}

        if feature_dump_level > 0:
            feature_dump = {
                "frame_times": (np.arange(T, dtype=np.float64) * float(self._dget("hop", 128)))
                / float(self._dget("sample_rate", self._dget("fs", 11162))),
                "frame_class": frame_class,
                "rain_conf": rain_conf,
                "noise_conf": noise_conf,
                "is_rain_raw": is_rain_raw,
                "onset_mask": onset_mask,
                "onset_indices": onset_indices,
                "flux_primary": flux_primary,
                "flux_modes_raw": flux_modes_raw,
                "flux_modes_proc": flux_modes_proc,
                "flux_modes_winsor_hi": np.asarray([flux_modes_winsor_hi], dtype=np.float64),
                "mode_flux_baseline": mode_flux_baseline,
                "mode_flux_excess": mode_flux_excess,
                "mode_flux_score": mode_flux_score,
                "mode_flux_rain_pass": mode_flux_rain_pass,
                "primary_flux_sanity": primary_flux_sanity.astype(np.int8),
                "primary_sanity": primary_sanity.astype(np.int8),
                "mode_flux_by_mode": mode_flux_by_mode,
                "mode_flux_baseline_by_mode": mode_flux_baseline_by_mode,
                "mode_flux_excess_by_mode": mode_flux_excess_by_mode,
                "normalized_mode_flux_by_mode": normalized_mode_flux_by_mode,
                # Note: kept historical key name for backward compatibility; values
                # now represent per-mode fraction of total mode flux.
                "mode_to_primary_ratio": mode_to_primary_ratio,
                "peak_ratio": peak_ratio,
                "peak_detected_count": peak_detected_count,
                "peak_total_count": peak_total_count,
                "peak_in_mode_count": peak_in_mode_count,
                "peak_in_primary_count": peak_in_primary_count,
                "peak_gate_score": peak_gate_score,
                "peak_gate": peak_gate.astype(np.int8),
            }

            if include_peak_payload:
                feature_dump.update({
                    "peak_valid_freqs_hz": peak_valid_freqs_hz,
                    "peak_valid_prominences_db": peak_valid_prominences_db,
                    "peak_valid_bandwidths_hz": peak_valid_bandwidths_hz,
                    "peak_valid_heights_db": peak_valid_heights_db,
                    "peak_top_p_freqs_hz": peak_top_p_freqs_hz,
                    "peak_top_p_prominences_db": peak_top_p_prominences_db,
                    "peak_top_p_bandwidths_hz": peak_top_p_bandwidths_hz,
                    "peak_top_p_heights_db": peak_top_p_heights_db,
                    "peak_top_p_in_mode_mask": peak_top_p_in_mode_mask,
                    "peak_top_p_in_primary_mask": peak_top_p_in_primary_mask,
                })

            if feature_dump_level > 1:
                feature_dump.update({
                    "frame_class_name": np.array(
                        [FrameClass(v).name.lower() for v in frame_class], dtype=object
                    ),
                    "primary_ok": primary_ok_dbg,
                    "mode_ok": mode_ok_dbg,
                    "tuning_params": {
                        "noise_hi": noise_hi,
                        "mode_flux_rain_min": mode_flux_rain_min,
                        "primary_flux_sanity_min": primary_flux_sanity_min,
                        "mode_flux_noise_max": mode_flux_noise_max,
                        "mode_flux_norm_enable": mode_flux_norm_enable,
                        "mode_flux_norm_win_sec": mode_flux_norm_win_sec,
                        "mode_flux_norm_q": mode_flux_norm_q,
                        "mode_flux_norm_min": mode_flux_norm_min,
                        "peak_top_p": peak_top_p,
                        "primary_top_m": primary_top_m,
                        "peak_ratio_min": peak_ratio_min,
                        "peak_prominence_db": peak_prominence_db,
                        "peak_min_db_above_floor": peak_min_db_above_floor,
                        "flux_modes_winsor_enable": flux_modes_winsor_enable,
                        "flux_modes_winsor_q": flux_modes_winsor_q,
                        "primary_mode_idx": primary_mode_idx,
                        "mode_ratio_definition": "mode_flux_over_total_mode_flux",
                        "mode_bands": mode_bands,
                        "td_soft_enable": td_soft_enable,
                        "td_soft_time_flux_band": td_soft_time_flux_band,
                        "td_soft_time_flux_score_min": td_soft_time_flux_score_min,
                        "td_soft_crest_factor_min": td_soft_crest_factor_min,
                        "td_soft_kurtosis_min": td_soft_kurtosis_min,
                        "td_soft_min_positive_votes": td_soft_min_positive_votes,
                        "feature_dump_include_peak_payload": include_peak_payload,
                        "feature_dump_include_td_soft": include_td_soft_feature_dump,
                    },
                })

                if include_td_soft_feature_dump and td_soft_enable and td_soft_debug:
                    for k in (
                        "frame_energy",
                        "frame_baseline",
                        "frame_warm_ok",
                        "frame_flux",
                        "time_flux_score",
                        "crest_factor",
                        "kurtosis",
                        "vote_count",
                        "soft_score",
                        "soft_label",
                        "frame_times",
                    ):
                        if k in td_soft_debug:
                            feature_dump[f"td_{k}"] = td_soft_debug[k]

        return frame_class, rain_conf, det_debug, feature_dump


class RainFrameClassifierProcessor(RainFrameClassifierMixin):
    """
    Standalone processor wrapper for tuning the rain frame classifier with the
    audio processing framework.

    This lets the classifier be run independently of SpectralNoiseProcessor
    while reusing the same detector configuration structure.
    """

    def __init__(
        self,
        name: str = "rain_frame_classifier",
        config: Optional[Any] = None,
    ):
        self.name = name
        self.cfg = config
        self._is_setup = config is not None

    def setup(self, params: Dict[str, Any]) -> None:
        if self._is_setup:
            return

        # Reuse the stage-1 config builder so detector params are interpreted
        # exactly the same way as in SpectralNoiseProcessor.
        from .spectral_noise_processor import build_noise_config

        sr = int(params.get("sample_rate", params.get("fs", 11162)))
        self.cfg = build_noise_config(sample_rate=sr, params=params)
        self._is_setup = True

    def process(self, x: np.ndarray, sr: Optional[int] = None) -> Dict[str, Any]:
        if self.cfg is None:
            raise RuntimeError(
                "RainFrameClassifierProcessor is not initialized. Call setup(params) before process(...)."
            )

        cfg = self.cfg
        if sr is None:
            sr = int(getattr(cfg, "fs", 11162))

        x = np.asarray(x, dtype=np.float64).reshape(-1)

        setattr(self.cfg, "last_input_audio", x)

        # Match the spectral front-end used by SpectralNoiseProcessor so tuning
        # results transfer directly back into the full stage-1 processor.
        window = getattr(cfg, "window", "hann")
        S = librosa.stft(
            x,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            win_length=cfg.n_fft,
            window=window,
            center=True,
        )
        P = librosa.amplitude_to_db(np.abs(S) + cfg.eps, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
        times = librosa.frames_to_time(
            np.arange(P.shape[1]),
            sr=sr,
            hop_length=cfg.hop,
            n_fft=cfg.n_fft,
        )

        frame_class, rain_conf, det_debug, feature_dump = self._detect_rain_over_time(P, freqs)
        frame_class = np.asarray(frame_class, dtype=np.int8)
        is_rain = frame_class == FrameClass.RAIN
        noise_conf = np.asarray(
            det_debug.get("noise_conf", np.clip(1.0 - rain_conf, 0.0, 1.0)),
            dtype=np.float64,
        )

        feature_dump_level = int(self._dget("feature_dump_level", 0))

        result = {
            "frame_class": frame_class,
            "is_rain": is_rain,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
        }

        if feature_dump_level > 0:
            result["feature_dump"] = feature_dump

        debug_level = int(self._dget("debug_level", 2))
        if debug_level > 0:
            result["times"] = times

        if debug_level > 1:
            result.update({
                "S": S,
                "x_filt": det_debug.get("td_soft", {}).get("x_bp", x),
                "debug": det_debug,
            })

        return result

    def run(self, audio: np.ndarray, params: Dict[str, Any]):
        """
        Framework-compatible wrapper.

        The audio_processing_framework expects each processor to expose:
            run(audio, params) -> (results_dict, state_dict)
        """
        self.setup(params)

        sr = int(params.get("sample_rate", params.get("fs", 11162)))
        results = self.process(audio, sr=sr)

        feature_dump_level = int(params.get("feature_dump_level", 0))
        state = {
            "frame_class": results.get("frame_class"),
            "is_rain": results.get("is_rain"),
            "rain_conf": results.get("rain_conf"),
            "noise_conf": results.get("noise_conf"),
            "debug": results.get("debug"),
        }
        return results, state
