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
    mode_bands: Optional[Tuple[Tuple[float, float], ...]] = None
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

    def _mode_band_comb(
        self,
        x: np.ndarray,
        mode_bands: Optional[Tuple[Tuple[float, float], ...]] = None,
        fallback_band: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Build a simple mode-band comb-filtered signal by summing bandpassed
        outputs across the configured rain mode bands.

        If no mode bands are provided, fall back to a single bandpass.
        """
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x.copy()

        use_mode_bands = cfg.mode_bands if mode_bands is None else mode_bands
        if not use_mode_bands:
            use_band = cfg.operating_band if fallback_band is None else fallback_band
            return self._bandpass(x, use_band)

        y_sum = np.zeros_like(x, dtype=np.float64)
        for band in use_mode_bands:
            y_sum += self._bandpass(x, band)
        return y_sum

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
        # Waveform-shape features use a mode-band comb-filtered signal so the
        # time-domain soft labeller focuses on the known rain resonance bands.
        x_bp = self._mode_band_comb(x, cfg.mode_bands, cfg.operating_band)

        # Time-flux path can optionally use its own dedicated band. If not
        # provided, reuse the same comb-filtered input as the waveform path.
        if cfg.time_flux_band is not None:
            x_flux = self._bandpass(x, cfg.time_flux_band)
        else:
            x_flux = x_bp.copy()
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

    def _build_td_soft_comparison(
        self,
        frame_class: np.ndarray,
        is_rain: np.ndarray,
        td_soft_debug: Dict[str, Any],
        detector_frame_times: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compare detector output against optional time-domain soft labels.

        Notes
        -----
        - Raw detector and soft-label frame grids are expected to share the same
          frame rate / hop.
        - Detector decisions are known to lag soft labels by roughly 2 frames,
          with an acceptable tolerance of +/- 1 frame.
        - We therefore keep both:
            1) strict index-aligned comparison for diagnostics
            2) lag-aware comparison for tuning metrics
        """
        out: Dict[str, Any] = {}
        if not isinstance(td_soft_debug, dict):
            return out

        if "soft_label" not in td_soft_debug:
            return out

        td_soft_label = np.asarray(td_soft_debug.get("soft_label"), dtype=bool).reshape(-1)
        if td_soft_label.size == 0:
            return out

        frame_class = np.asarray(frame_class, dtype=np.int8).reshape(-1)
        is_rain = np.asarray(is_rain, dtype=bool).reshape(-1)
        n = int(min(frame_class.size, is_rain.size, td_soft_label.size))
        if n <= 0:
            return out

        frame_class_cmp = frame_class[:n]
        is_rain_cmp = is_rain[:n]
        td_soft_label_cmp = td_soft_label[:n]

        detector_times_cmp = None
        td_soft_times_cmp = None
        same_grid = False
        max_abs_time_diff = np.nan
        alignment_mode = "shared_prefix_index_alignment"

        if detector_frame_times is not None and "frame_times" in td_soft_debug:
            det_times = np.asarray(detector_frame_times, dtype=np.float64).reshape(-1)
            td_times = np.asarray(td_soft_debug.get("frame_times"), dtype=np.float64).reshape(-1)
            if det_times.size > 0 and td_times.size > 0:
                detector_times_cmp = det_times[:n]
                td_soft_times_cmp = td_times[:n]
                if detector_times_cmp.size == td_soft_times_cmp.size and detector_times_cmp.size > 0:
                    dt = detector_times_cmp - td_soft_times_cmp
                    max_abs_time_diff = float(np.max(np.abs(dt)))
                    same_grid = bool(
                        np.allclose(
                            detector_times_cmp,
                            td_soft_times_cmp,
                            atol=1e-9,
                            rtol=0.0,
                        )
                    )
                    alignment_mode = (
                        "time_aligned_prefix"
                        if same_grid
                        else "time_mismatch_shared_prefix"
                    )

        # Strict index-aligned comparison (diagnostic only)
        agree = is_rain_cmp == td_soft_label_cmp
        tp = is_rain_cmp & td_soft_label_cmp
        fp = is_rain_cmp & (~td_soft_label_cmp)
        fn = (~is_rain_cmp) & td_soft_label_cmp
        tn = (~is_rain_cmp) & (~td_soft_label_cmp)

        # Lag-aware comparison for tuning metrics.
        # Detector rain decisions lag soft labels by ~2 frames, with +/-1 frame tolerance.
        lag_frames = 2
        tolerance_frames = 1
        soft_label_lagged_target = np.zeros(n, dtype=bool)
        for t in range(n):
            center = t - lag_frames
            j0 = max(0, center - tolerance_frames)
            j1 = min(n, center + tolerance_frames + 1)
            if j1 > j0:
                soft_label_lagged_target[t] = bool(np.any(td_soft_label_cmp[j0:j1]))

        agree_lagged = is_rain_cmp == soft_label_lagged_target
        tp_lagged = is_rain_cmp & soft_label_lagged_target
        fp_lagged = is_rain_cmp & (~soft_label_lagged_target)
        fn_lagged = (~is_rain_cmp) & soft_label_lagged_target
        tn_lagged = (~is_rain_cmp) & (~soft_label_lagged_target)

        uncertain_mask = frame_class_cmp == FrameClass.UNCERTAIN
        noise_mask = frame_class_cmp == FrameClass.NOISE
        rain_mask = frame_class_cmp == FrameClass.RAIN

        out = {
            "aligned": True,
            "alignment_mode": alignment_mode,
            "same_grid": same_grid,
            "max_abs_time_diff_sec": max_abs_time_diff,
            "lag_frames": lag_frames,
            "tolerance_frames": tolerance_frames,
            "n_compare": n,
            "frame_class": frame_class_cmp,
            "is_rain": is_rain_cmp,
            "soft_label": td_soft_label_cmp,
            "soft_label_lagged_target": soft_label_lagged_target,
            "agree": agree,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "agree_lagged": agree_lagged,
            "tp_lagged": tp_lagged,
            "fp_lagged": fp_lagged,
            "fn_lagged": fn_lagged,
            "tn_lagged": tn_lagged,
            "uncertain_mask": uncertain_mask,
            "noise_mask": noise_mask,
            "rain_mask": rain_mask,
            "agree_count": int(np.sum(agree)),
            "disagree_count": int(n - np.sum(agree)),
            "tp_count": int(np.sum(tp)),
            "fp_count": int(np.sum(fp)),
            "fn_count": int(np.sum(fn)),
            "tn_count": int(np.sum(tn)),
            "agree_count_lagged": int(np.sum(agree_lagged)),
            "disagree_count_lagged": int(n - np.sum(agree_lagged)),
            "tp_count_lagged": int(np.sum(tp_lagged)),
            "fp_count_lagged": int(np.sum(fp_lagged)),
            "fn_count_lagged": int(np.sum(fn_lagged)),
            "tn_count_lagged": int(np.sum(tn_lagged)),
            "uncertain_count": int(np.sum(uncertain_mask)),
            "noise_count": int(np.sum(noise_mask)),
            "rain_count": int(np.sum(rain_mask)),
        }

        if detector_times_cmp is not None:
            out["detector_frame_times"] = detector_times_cmp
        if td_soft_times_cmp is not None:
            out["soft_frame_times"] = td_soft_times_cmp

        denom = max(n, 1)
        out["agreement_rate"] = float(out["agree_count"]) / float(denom)
        out["agreement_rate_lagged"] = float(out["agree_count_lagged"]) / float(denom)
        out["soft_positive_rate"] = float(np.sum(td_soft_label_cmp)) / float(denom)
        out["soft_positive_rate_lagged_target"] = float(np.sum(soft_label_lagged_target)) / float(denom)
        out["rain_positive_rate"] = float(np.sum(is_rain_cmp)) / float(denom)
        return out

    # ------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------

    def _detect_rain_over_time(
        self,
        P: np.ndarray,
        freqs: np.ndarray,
        detector_frame_times: Optional[np.ndarray] = None,
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
        td_soft_compare: Dict[str, Any] = {}
        if td_soft_enable and self._dhas("last_input_audio"):
            try:
                x_in = np.asarray(getattr(self.cfg, "last_input_audio"), dtype=np.float64).reshape(-1)
                td_soft_cfg = TimeDomainSoftLabelConfig(
                    fs=int(self._dget("sample_rate", self._dget("fs", 11162))),
                    frame_len=int(self._dget("n_fft", 256)),
                    hop=int(self._dget("hop", 128)),
                    operating_band=(op_lo, op_hi),
                    mode_bands=tuple((float(a), float(b)) for (a, b) in mode_bands),
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
                td_soft_full = TimeDomainSoftLabeller(td_soft_cfg).process(x_in)
                td_soft_debug = {
                    "frame_times": td_soft_full.get("frame_times"),
                    "soft_label": td_soft_full.get("soft_label"),
                    "time_flux_score": td_soft_full.get("time_flux_score"),
                    "crest_factor": td_soft_full.get("crest_factor"),
                    "kurtosis": td_soft_full.get("kurtosis"),
                    "vote_count": td_soft_full.get("vote_count"),
                    "soft_score": td_soft_full.get("soft_score"),
                }
            except Exception as e:
                td_soft_debug = {"error": str(e)}
        if detector_frame_times is None:
            detector_frame_times = (
                np.arange(T, dtype=np.float64) * float(self._dget("hop", 128))
            ) / float(self._dget("sample_rate", self._dget("fs", 11162)))
        else:
            detector_frame_times = np.asarray(detector_frame_times, dtype=np.float64).reshape(-1)
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
        normalized_mode_flux_by_mode = np.zeros((len(mode_bands), T), dtype=np.float64)
        peak_count_by_mode = np.zeros((len(mode_bands), T), dtype=np.int32)

        # Optional per-mode representative peak payload for inspection.
        # For each mode band and frame, store at most one valid peak: the tallest
        # peak whose prominence falls in the requested valid range.
        if include_peak_payload:
            peak_valid_freqs_hz = np.empty((len(mode_bands), T), dtype=object)
            peak_valid_prominences_db = np.empty((len(mode_bands), T), dtype=object)
            peak_valid_bandwidths_hz = np.empty((len(mode_bands), T), dtype=object)
        else:
            peak_valid_freqs_hz = None
            peak_valid_prominences_db = None
            peak_valid_bandwidths_hz = None


        prev_frame_1 = None  # frame at t-1
        prev_frame_2 = None  # frame at t-2

        for t in range(T):
            frame = P_band[:, t]
            if include_peak_payload:
                for i in range(len(mode_bands)):
                    peak_valid_freqs_hz[i, t] = np.array([], dtype=np.float64)
                    peak_valid_prominences_db[i, t] = np.array([], dtype=np.float64)
                    peak_valid_bandwidths_hz[i, t] = np.array([], dtype=np.float64)

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
                peak_count_by_mode[:, t] = 0
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
                peak_count_by_mode[:, t] = 0
            else:
                # Valid peaks are those satisfying the requested prominence range.
                pk_h = np.asarray(props.get("peak_heights", spec_db[peaks]), dtype=np.float64)
                pk_prom = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=np.float64)
                widths_bins, *_ = peak_widths(spec_db, peaks, rel_height=0.5)
                df_hz = float(freqs_band[1] - freqs_band[0]) if freqs_band.size > 1 else 0.0
                pk_bw_hz = np.asarray(widths_bins, dtype=np.float64) * df_hz

                valid_prom_mask = (pk_prom >= 3.0) & (pk_prom <= 6.0)
                peaks_valid = peaks[valid_prom_mask]
                pk_h_valid = pk_h[valid_prom_mask]
                pk_prom_valid = pk_prom[valid_prom_mask]
                pk_bw_hz_valid = pk_bw_hz[valid_prom_mask]

                # Per-mode peak counts and optional representative peak payload.
                for i, m_mask in enumerate(mode_masks):
                    if peaks_valid.size == 0:
                        peak_count_by_mode[i, t] = 0
                        continue

                    in_mode_valid = m_mask[peaks_valid]
                    peak_count_by_mode[i, t] = int(np.sum(in_mode_valid))
                    if include_peak_payload and np.any(in_mode_valid):
                        mode_freqs = freqs_band[peaks_valid[in_mode_valid]].astype(np.float64)
                        mode_prom = pk_prom_valid[in_mode_valid].astype(np.float64)
                        mode_bw = pk_bw_hz_valid[in_mode_valid].astype(np.float64)
                        mode_heights = pk_h_valid[in_mode_valid].astype(np.float64)

                        best_idx = int(np.argmax(mode_heights))
                        peak_valid_freqs_hz[i, t] = np.asarray([mode_freqs[best_idx]], dtype=np.float64)
                        peak_valid_prominences_db[i, t] = np.asarray([mode_prom[best_idx]], dtype=np.float64)
                        peak_valid_bandwidths_hz[i, t] = np.asarray([mode_bw[best_idx]], dtype=np.float64)

                # Strongest top-P peaks for gate computation.
                order = np.argsort(pk_h)[::-1]
                sel = peaks[order[:peak_top_p]]
                in_primary = primary_mask[sel]
                in_any_mode = np.zeros(sel.size, dtype=bool)
                for m_mask in mode_masks:
                    in_any_mode |= m_mask[sel]

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

        flux_modes_proc = flux_modes.copy()

        if flux_modes_winsor_enable:
            finite_mask = np.isfinite(flux_modes_proc)
            if np.any(finite_mask):
                winsor_hi = float(np.percentile(flux_modes_proc[finite_mask], flux_modes_winsor_q))
                flux_modes_proc = np.minimum(flux_modes_proc, winsor_hi)

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
            normalized_mode_flux_by_mode[i] = np.nan_to_num(
                score_i,
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


        if td_soft_enable and td_soft_debug and ("error" not in td_soft_debug):
            td_soft_compare = self._build_td_soft_comparison(
                frame_class=frame_class,
                is_rain=is_rain,
                td_soft_debug=td_soft_debug,
                detector_frame_times=detector_frame_times,
            )
        # PSD update gating is intentionally not decided here.
        # Downstream suppressor should derive it from FrameClass.

        det_debug = {
            # Core classifier features
            "flux_primary": flux_primary,
            "mode_flux_score": mode_flux_score,
            "normalized_mode_flux_by_mode": normalized_mode_flux_by_mode,

            # Peak-structure features
            "peak_count_by_mode": peak_count_by_mode,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "is_rain": is_rain,

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
            "td_soft": td_soft_debug,
            "td_soft_compare": td_soft_compare,

            # Thresholds / detector settings useful during tuning
            "tuning_params": {
                "operating_band": (op_lo, op_hi),
                "mode_bands": mode_bands,
                "primary_mode_idx": primary_mode_idx,
                "noise_hi": noise_hi,
                "mode_flux_rain_min": mode_flux_rain_min,
                "primary_flux_sanity_min": primary_flux_sanity_min,
                "mode_flux_noise_max": mode_flux_noise_max,
                "mode_flux_norm_win_sec": mode_flux_norm_win_sec,
                "mode_flux_norm_q": mode_flux_norm_q,
                "mode_flux_norm_min": mode_flux_norm_min,
                "peak_top_p": peak_top_p,
                "primary_top_m": primary_top_m,
                "peak_ratio_min": peak_ratio_min,
                "peak_prominence_db": peak_prominence_db,
                "peak_min_db_above_floor": peak_min_db_above_floor,
                "td_soft_enable": td_soft_enable,
            },
        }

        if include_peak_payload:
            det_debug.update({
                "peak_valid_freqs_hz": peak_valid_freqs_hz,
                "peak_valid_prominences_db": peak_valid_prominences_db,
                "peak_valid_bandwidths_hz": peak_valid_bandwidths_hz,
            })

        feature_dump_level = int(self._dget("feature_dump_level", 0))
        feature_dump: Dict[str, Any] = {}

        if feature_dump_level > 0:
            feature_dump = {
                "frame_times": detector_frame_times,
                "frame_class": frame_class,
                "is_rain": is_rain.astype(np.int8),
                "is_rain_raw": is_rain_raw.astype(np.int8),
                "flux_primary": flux_primary,
                "mode_flux_score": mode_flux_score,
                "normalized_mode_flux_by_mode": normalized_mode_flux_by_mode,
                "peak_ratio": peak_ratio,
                "peak_gate_score": peak_gate_score,
                "peak_gate": peak_gate.astype(np.int8),
                "rain_conf": rain_conf,
                "noise_conf": noise_conf,
            }

            if include_td_soft_feature_dump and td_soft_debug and ("error" not in td_soft_debug):
                feature_dump.update({
                    "td_soft_label": np.asarray(td_soft_debug.get("soft_label"), dtype=np.int8),
                    "td_time_flux_score": td_soft_debug.get("time_flux_score"),
                    "td_crest_factor": td_soft_debug.get("crest_factor"),
                    "td_kurtosis": td_soft_debug.get("kurtosis"),
                    "td_vote_count": td_soft_debug.get("vote_count"),
                    "td_soft_score": td_soft_debug.get("soft_score"),
                })

            if include_peak_payload:
                feature_dump.update({
                    "peak_valid_freqs_hz": peak_valid_freqs_hz,
                    "peak_valid_prominences_db": peak_valid_prominences_db,
                    "peak_valid_bandwidths_hz": peak_valid_bandwidths_hz,
                })

            # Level-2 feature dump block removed for leaner big-run payloads.

        return frame_class, rain_conf, det_debug, feature_dump