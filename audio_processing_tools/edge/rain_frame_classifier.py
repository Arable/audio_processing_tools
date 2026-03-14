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

        eps = float(self._dget("eps", 1e-9))

        op_band = self._dget("operating_band", (400.0, 3500.0))
        op_lo, op_hi = float(op_band[0]), float(op_band[1])

        mode_bands = self._dget("mode_bands", None)
        if mode_bands is None:
            raise AttributeError("Missing required detector param: mode_bands")

        mode_bands = tuple((float(a), float(b)) for (a, b) in mode_bands)

        primary_mode_idx = int(self._dget("primary_mode_idx", 0))
        if primary_mode_idx < 0 or primary_mode_idx >= len(mode_bands):
            raise ValueError(
                f"primary_mode_idx ({primary_mode_idx}) out of range for "
                f"mode_bands length ({len(mode_bands)})"
            )

        rain_hi = float(self._dget("rain_hi", 0.80))
        noise_hi = float(self._dget("noise_hi", 0.80))
        z_primary_min = float(self._dget("z_primary_min", 1.0))

        z_win_frames = int(self._dget("z_win_frames", 30))
        if z_win_frames < 3:
            z_win_frames = 3
        if (z_win_frames % 2) == 0:
            z_win_frames += 1

        z_min_scale = float(self._dget("z_min_scale", 1e-3))
        z_clip = float(self._dget("z_clip", 10.0))
        z_min_scale = max(z_min_scale, eps)
        z_clip = max(z_clip, 0.0)

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
        flux_from_t1 = np.full(T, np.nan, dtype=np.float64)
        flux_from_t2 = np.full(T, np.nan, dtype=np.float64)
        flux_use_t2 = np.full(T, np.nan, dtype=np.float64)
        z_primary = np.full(T, np.nan, dtype=np.float64)
        z_modes = np.full(T, np.nan, dtype=np.float64)

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

        rain_score_raw = np.full(T, np.nan, dtype=np.float64)

        prev_frame_1 = None  # frame at t-1
        prev_frame_2 = None  # frame at t-2

        for t in range(T):
            frame = P_band[:, t]

            if prev_frame_1 is None:
                # First frame: no previous reference available.
                flux_primary[t] = 0.0
                flux_modes[t] = 0.0
                flux_from_t1[t] = 0.0
                flux_from_t2[t] = 0.0
                flux_use_t2[t] = 0.0
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
                flux_from_t1[t] = 0.0
                flux_from_t2[t] = 0.0
                flux_use_t2[t] = 0.0
                prev_frame_2 = prev_frame_1
                prev_frame_1 = frame
            else:
                # Use only the non-overlapping t-vs-(t-2) positive rise.
                delta2 = frame - prev_frame_2
                d2_pos = np.maximum(delta2, 0.0)
                flux = d2_pos
                flux_from_t1[t] = 0.0
                flux_from_t2[t] = float(np.sum(d2_pos))
                flux_use_t2[t] = 1.0
                prev_frame_2 = prev_frame_1
                prev_frame_1 = frame

            # Primary mode
            flux_primary[t] = float(np.sum(flux[primary_mask]))

            # All modes
            total_flux_modes = 0.0
            for i, m_mask in enumerate(mode_masks):
                weight = mode_weights[i] if mode_weights is not None else 1.0
                total_flux_modes += weight * float(np.sum(flux[m_mask]))

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
                pk_h = props.get("peak_heights", spec_db[peaks])
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
            scale = np.maximum(scale, z_min_scale)

            z = (x_f - med) / (scale + eps)
            if z_clip > 0.0:
                z = np.clip(z, -z_clip, z_clip)

            # Startup guard: first few frames do not have a reliable local baseline yet.
            # Force them to zero novelty instead of letting edge effects drive false spikes.
            n_warm = max(1, z_win_frames // 2)
            z[:n_warm] = 0.0

            z[nan_mask] = np.nan
            return z

        z_primary = rolling_robust_z(flux_primary)

        flux_modes_raw = flux_modes.copy()
        flux_modes_for_z = flux_modes.copy()
        flux_modes_winsor_hi = np.nan

        if flux_modes_winsor_enable:
            finite_mask = np.isfinite(flux_modes_for_z)
            if np.any(finite_mask):
                flux_modes_winsor_hi = float(np.percentile(flux_modes_for_z[finite_mask], flux_modes_winsor_q))
                flux_modes_for_z = np.minimum(flux_modes_for_z, flux_modes_winsor_hi)

        z_modes = rolling_robust_z(flux_modes_for_z)

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

        # Main rain confidence is driven by the multi-mode novelty only.
        # The primary band is retained as a conservative sanity check in the final decision.
        base_rain_conf = rain_conf_modes.copy()
        base_rain_conf = np.nan_to_num(base_rain_conf, nan=0.0, posinf=1.0, neginf=0.0)
        base_rain_conf = np.clip(base_rain_conf, 0.0, 1.0)

        # Peak confirmation gate: among the strongest peaks, require at least one
        # primary-band peak in the top-M peaks and enough overall concentration of
        # peaks inside the expected mode bands.
        peak_gate = peak_gate_score >= 1.0

        # Final rain confidence used by the raw detector decision.
        # The confidence itself is mode-driven; the primary band provides only a
        # conservative sanity check so narrowband primary-only artifacts do not dominate.
        rain_conf = base_rain_conf.copy()
        primary_sanity = np.nan_to_num(z_primary, nan=0.0, posinf=z_clip, neginf=-z_clip) >= z_primary_min

        # Raw rain decision from the classifier. Any persistence / hold behavior
        # should be handled downstream (for example by the noise suppressor), not
        # inside this detector.
        is_rain_raw = (rain_conf >= rain_hi) & peak_gate & primary_sanity
        is_rain = is_rain_raw.copy()

        # Onset frames: first raw-rain frame after a non-rain frame.
        onset_mask = is_rain_raw & (~np.r_[False, is_rain_raw[:-1]])
        onset_indices = np.flatnonzero(onset_mask).astype(np.int32)

        # Intra-frame peak failure blocks rain, but does not by itself force NOISE.
        # Definite noise still requires weak inter-frame rain evidence.
        noise_conf = np.clip(1.0 - base_rain_conf, 0.0, 1.0)

        # FrameClass is the canonical raw detector output.
        # Intra-frame peak failure blocks RAIN, but NOISE is assigned only when
        # inter-frame evidence is also weak. All remaining frames are UNCERTAIN.
        frame_class = np.full(T, FrameClass.UNCERTAIN, dtype=np.int8)
        frame_class[noise_conf >= noise_hi] = FrameClass.NOISE
        frame_class[is_rain] = FrameClass.RAIN

        rain_score_raw = rain_conf.copy()

        # PSD update gating is intentionally not decided here.
        # Downstream suppressor should derive it from FrameClass.

        det_debug = {
            # Inter-frame evidence
            "flux_primary": flux_primary,
            "flux_modes": flux_modes_raw,
            "flux_modes_for_z": flux_modes_for_z,
            "flux_modes_winsor_hi": flux_modes_winsor_hi,
            "flux_from_t1": flux_from_t1,
            "flux_from_t2": flux_from_t2,
            "flux_use_t2": flux_use_t2,
            "z_primary": z_primary,
            "z_modes": z_modes,
            "rain_conf_primary": rain_conf_primary,
            "rain_conf_modes": rain_conf_modes,
            "base_rain_conf": base_rain_conf,
            "rain_conf": rain_conf,
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

            # Thresholds / detector settings useful during tuning
            "tuning_params": {
                "rain_hi": rain_hi,
                "noise_hi": noise_hi,
                "z_primary_min": z_primary_min,
                "peak_top_p": peak_top_p,
                "primary_top_m": primary_top_m,
                "peak_ratio_min": peak_ratio_min,
                "peak_prominence_db": peak_prominence_db,
                "peak_min_db_above_floor": peak_min_db_above_floor,
                "z_win_frames": z_win_frames,
                "z_conf_scale_primary": z_conf_scale_primary,
                "z_conf_scale_modes": z_conf_scale_modes,
                "z_min_scale": z_min_scale,
                "z_clip": z_clip,
                "flux_modes_winsor_enable": flux_modes_winsor_enable,
                "flux_modes_winsor_q": flux_modes_winsor_q,
                "z_startup_zero_frames": max(1, z_win_frames // 2),
                "primary_mode_idx": primary_mode_idx,
                "mode_bands": mode_bands,
            },
        }

        return frame_class, rain_conf, det_debug
