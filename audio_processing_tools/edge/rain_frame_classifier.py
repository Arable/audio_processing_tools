from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Tuple, Optional

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as spsig


class FrameClass(IntEnum):
    """Frame classification used by the rain detector and downstream suppressor."""

    NOISE = 0
    UNCERTAIN = 1
    RAIN = 2


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
        P: np.ndarray,          # (F, T) log-compressed power
        freqs: np.ndarray,      # (F,)
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

        self._validate_rain_cfg()

        cfg = self.cfg

        eps = float(self._dget("eps", 1e-9))

        op_band = self._dget("operating_band", (400.0, 3500.0))
        op_lo, op_hi = float(op_band[0]), float(op_band[1])

        mode_bands = self._dget("mode_bands", None)
        if mode_bands is None:
            raise AttributeError("Missing required detector param: mode_bands")

        mode_bands = tuple((float(a), float(b)) for (a, b) in mode_bands)

        primary_mode_idx = int(self._dget("primary_mode_idx", 0))

        rain_hi = float(self._dget("rain_hi", 0.80))
        noise_hi = float(self._dget("noise_hi", 0.85))
        rain_hold_frames = int(self._dget("rain_hold_frames", 1))

        z_win_frames = int(self._dget("z_win_frames", 30))
        if z_win_frames < 3:
            z_win_frames = 3
        if (z_win_frames % 2) == 0:
            z_win_frames += 1

        mode_weights = self._dget("mode_weights", None)
        if mode_weights is not None:
            mode_weights = tuple(float(w) for w in mode_weights)
            if len(mode_weights) != len(mode_bands):
                raise ValueError(
                    f"mode_weights length ({len(mode_weights)}) "
                    f"must match mode_bands length ({len(mode_bands)})"
                )

        F, T = P.shape
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)

        P_band = P[band_mask, :]
        freqs_band = freqs[band_mask]

        peak_top_p = int(self._dget("peak_top_p", 6))
        peaks_in_mode_min = int(self._dget("peaks_in_mode_min", 2))
        peak_prominence_db = float(self._dget("peak_prominence_db", 3.0))
        peak_min_db_above_floor = float(self._dget("peak_min_db_above_floor", 6.0))
        peak_top_p = max(1, peak_top_p)
        peaks_in_mode_min = max(0, peaks_in_mode_min)

        rain_conf = np.zeros(T, dtype=np.float64)
        is_rain = np.zeros(T, dtype=bool)

        flux_primary = np.full(T, np.nan, dtype=np.float64)
        flux_modes = np.full(T, np.nan, dtype=np.float64)
        z_primary = np.full(T, np.nan, dtype=np.float64)
        z_modes = np.full(T, np.nan, dtype=np.float64)

        peak_ratio = np.full(T, np.nan, dtype=np.float64)
        peak_in_mode_count = np.full(T, np.nan, dtype=np.float64)
        peak_total_count = np.full(T, np.nan, dtype=np.float64)

        rain_score_raw = np.full(T, np.nan, dtype=np.float64)

        prev_frame = None

        for t in range(T):
            frame = P_band[:, t]

            if prev_frame is None:
                prev_frame = frame
                flux_primary[t] = 0.0
                flux_modes[t] = 0.0
                peak_total_count[t] = 0.0
                peak_in_mode_count[t] = 0.0
                peak_ratio[t] = 0.0
                continue

            delta = frame - prev_frame
            prev_frame = frame

            flux = np.maximum(delta, 0.0)

            # Primary mode
            f_lo, f_hi = mode_bands[primary_mode_idx]
            m_mask = (freqs_band >= f_lo) & (freqs_band <= f_hi)
            flux_primary[t] = float(np.sum(flux[m_mask]))

            # All modes
            total_flux_modes = 0.0
            for i, (lo, hi) in enumerate(mode_bands):
                m_mask = (freqs_band >= lo) & (freqs_band <= hi)
                weight = mode_weights[i] if mode_weights is not None else 1.0
                total_flux_modes += weight * float(np.sum(flux[m_mask]))

            flux_modes[t] = total_flux_modes

            # --- Peak structure: what fraction of the strongest peaks fall inside mode bands? ---
            spec_db = frame
            floor_db = float(np.median(spec_db))
            height_thresh = floor_db + peak_min_db_above_floor

            peaks, props = spsig.find_peaks(
                spec_db,
                prominence=peak_prominence_db,
                height=height_thresh,
            )
            if peaks.size == 0:
                peak_total_count[t] = 0.0
                peak_in_mode_count[t] = 0.0
                peak_ratio[t] = 0.0
            else:
                pk_h = props.get("peak_heights", spec_db[peaks])
                order = np.argsort(pk_h)[::-1]
                sel = peaks[order[:peak_top_p]]
                sel_freqs = freqs_band[sel]

                in_mode = np.zeros(sel.size, dtype=bool)
                for j, f0 in enumerate(sel_freqs):
                    for (lo, hi) in mode_bands:
                        if (f0 >= lo) and (f0 <= hi):
                            in_mode[j] = True
                            break

                total = float(sel.size)
                in_m = float(np.sum(in_mode))
                peak_total_count[t] = total
                peak_in_mode_count[t] = in_m
                peak_ratio[t] = in_m / float(max(1, sel.size))

        # Rolling robust z-score (median + MAD) to adapt to changing noise floor
        def rolling_robust_z(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)

            # Fill NaNs for filtering, then restore NaNs at the end.
            nan_mask = ~np.isfinite(x)
            if np.all(nan_mask):
                return np.full_like(x, np.nan, dtype=np.float64)

            x_f = x.copy()
            first = float(x_f[np.isfinite(x_f)][0])
            x_f[nan_mask] = first

            # median over time
            med = ndi.median_filter(x_f, size=z_win_frames, mode="nearest")
            # MAD over time
            abs_dev = np.abs(x_f - med)
            mad = ndi.median_filter(abs_dev, size=z_win_frames, mode="nearest")
            scale = 1.4826 * mad

            z = (x_f - med) / (scale + eps)
            z[nan_mask] = np.nan
            return z

        z_primary = rolling_robust_z(flux_primary)
        z_modes = rolling_robust_z(flux_modes)

        # Convert z-scores to [0,1] confidences.
        # Enforce: rain decision requires BOTH primary and multi-mode evidence.
        z_conf_scale_primary = float(self._dget("z_conf_scale_primary", 3.0))
        z_conf_scale_modes = float(self._dget("z_conf_scale_modes", 3.0))
        z_conf_scale_primary = max(z_conf_scale_primary, eps)
        z_conf_scale_modes = max(z_conf_scale_modes, eps)

        rain_conf_primary = np.clip(z_primary / z_conf_scale_primary, 0.0, 1.0)
        rain_conf_modes = np.clip(z_modes / z_conf_scale_modes, 0.0, 1.0)

        rain_conf_primary = np.nan_to_num(rain_conf_primary, nan=0.0, posinf=1.0, neginf=0.0)
        rain_conf_modes = np.nan_to_num(rain_conf_modes, nan=0.0, posinf=1.0, neginf=0.0)

        # Combined confidence: AND-gate behavior in a soft way.
        # Using min() means a frame is only as rain-like as its weaker piece of evidence.
        rain_conf = np.minimum(rain_conf_primary, rain_conf_modes)
        rain_conf = np.nan_to_num(rain_conf, nan=0.0, posinf=1.0, neginf=0.0)
        rain_conf = np.clip(rain_conf, 0.0, 1.0)

        # Peak confirmation gate (raindrop-like spectral structure)
        peak_gate = peak_in_mode_count >= peaks_in_mode_min

        # Raw rain decision before any hold expansion
        is_rain_raw = (rain_conf >= rain_hi) & peak_gate

        # Copy for post-processing (hold expansion may modify it)
        is_rain = is_rain_raw.copy()

        # Onset frames: first raw-rain frame after a non-rain frame.
        # This preserves droplet onset timing before any hold expansion.
        onset_mask = is_rain_raw & (~np.r_[False, is_rain_raw[:-1]])
        onset_indices = np.flatnonzero(onset_mask).astype(np.int32)

        noise_conf = 1.0 - rain_conf

        # Optional: expand rain decision to the RIGHT only (t .. t+w).
        # Rationale: droplet energy can persist into the next few frames (slow decay).
        w = int(rain_hold_frames)
        if w > 0 and np.any(is_rain):
            is_rain_exp = is_rain.copy()
            rain_idx = np.flatnonzero(is_rain)
            for t0 in rain_idx:
                t1 = min(T, t0 + w + 1)
                is_rain_exp[t0:t1] = True
            is_rain = is_rain_exp

        # FrameClass is the canonical detector output.
        # RAIN uses post-hold rain state so short raindrop decay frames are protected
        # from being treated as noise. NOISE is assigned only to confident non-rain
        # frames. All remaining frames are UNCERTAIN.
        frame_class = np.full(T, FrameClass.UNCERTAIN, dtype=np.int8)
        frame_class[noise_conf >= noise_hi] = FrameClass.NOISE
        frame_class[is_rain] = FrameClass.RAIN

        rain_score_raw = rain_conf.copy()

        # PSD update gating is intentionally not decided here.
        # Downstream suppressor should derive it from FrameClass.

        det_debug = {
            # signals you tune against
            "flux_primary": flux_primary,
            "flux_modes": flux_modes,
            "z_primary": z_primary,
            "z_modes": z_modes,
            "peak_ratio": peak_ratio,
            "peak_in_mode_count": peak_in_mode_count,
            "peak_total_count": peak_total_count,

            # decisions
            "rain_score_raw": rain_score_raw,
            "is_rain_raw": is_rain_raw,
            "onset_mask": onset_mask,
            "onset_indices": onset_indices,
            "frame_class": frame_class,
            "frame_class_name": np.array([FrameClass(v).name.lower() for v in frame_class], dtype=object),
        }

        return frame_class, rain_conf, det_debug
