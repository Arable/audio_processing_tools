from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Tuple, Optional

import librosa
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as spsig
from scipy.stats import kurtosis


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
    bp_order: int = 4

    # Envelope from block energy
    env_block_len: int = 32
    env_hop: int = 16
    env_smooth_len: int = 3

    # Soft-label thresholds (initial guesses)
    energy_excess_min: float = 0.20
    crest_factor_min: float = 3.0
    kurtosis_min: float = 3.5
    min_positive_votes: int = 2

    eps: float = 1e-9


class TimeDomainSoftLabeller:
    """
    Simple time-domain soft labeller for rain-like impulsive frames.

    Current design:
      - bandpass filter with operating band
      - block-energy envelope
      - 256-point frames with 128-point hop
      - per-frame features:
          * envelope peak present
          * energy excess over local envelope baseline
          * crest factor
          * kurtosis
      - soft label from vote count
    """

    def __init__(self, config: Optional[TimeDomainSoftLabelConfig] = None):
        self.cfg = config or TimeDomainSoftLabelConfig()

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x.copy()

        nyq = 0.5 * float(cfg.fs)
        lo = float(np.clip(cfg.operating_band[0], 1e-3, nyq * 0.999))
        hi = float(np.clip(cfg.operating_band[1], lo + 1e-3, nyq * 0.999))
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

    def _block_energy_envelope(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        B = int(max(1, cfg.env_block_len))
        H = int(max(1, cfg.env_hop))
        if x.size < B:
            energy = np.array([float(np.mean(x**2))], dtype=np.float64)
            times = np.array([0.0], dtype=np.float64)
        else:
            vals = []
            times = []
            for start in range(0, x.size - B + 1, H):
                seg = x[start : start + B]
                vals.append(float(np.mean(seg**2)))
                times.append((start + 0.5 * B) / float(cfg.fs))
            energy = np.asarray(vals, dtype=np.float64)
            times = np.asarray(times, dtype=np.float64)

        if energy.size > 1 and int(cfg.env_smooth_len) > 1:
            L = int(cfg.env_smooth_len)
            kernel = np.ones(L, dtype=np.float64) / float(L)
            energy = np.convolve(energy, kernel, mode="same")

        return energy, times

    def _frame_view(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < cfg.frame_len:
            return np.empty((0, cfg.frame_len), dtype=np.float64)
        T = 1 + (x.size - cfg.frame_len) // cfg.hop
        stride = x.strides[0]
        return np.lib.stride_tricks.as_strided(
            x,
            shape=(T, cfg.frame_len),
            strides=(cfg.hop * stride, stride),
            writeable=False,
        )

    def process(self, x: np.ndarray) -> Dict[str, Any]:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x_bp = self._bandpass(x)
        frames = self._frame_view(x_bp)
        T = frames.shape[0]

        frame_times = (np.arange(T) * cfg.hop) / float(cfg.fs)
        env, env_times = self._block_energy_envelope(x_bp)

        energy_excess = np.zeros(T, dtype=np.float64)
        crest_factor = np.zeros(T, dtype=np.float64)
        kurt_vals = np.zeros(T, dtype=np.float64)
        env_peak_present = np.zeros(T, dtype=bool)
        vote_count = np.zeros(T, dtype=np.int32)
        soft_score = np.zeros(T, dtype=np.float64)
        soft_label = np.zeros(T, dtype=bool)

        if env.size > 0:
            env_baseline = np.percentile(env, 20.0)
            env_peak_threshold = env_baseline + 0.25 * max(np.max(env) - env_baseline, cfg.eps)
        else:
            env_baseline = 0.0
            env_peak_threshold = np.inf

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

            t0 = frame_times[t]
            t1 = t0 + cfg.frame_len / float(cfg.fs)
            if env.size > 0:
                m = (env_times >= t0) & (env_times < t1)
                if np.any(m):
                    env_seg = env[m]
                    env_peak_present[t] = bool(np.max(env_seg) >= env_peak_threshold)
                    energy_excess[t] = max(float(np.max(env_seg) - env_baseline), 0.0)
                else:
                    env_peak_present[t] = False
                    energy_excess[t] = 0.0
            else:
                env_peak_present[t] = False
                energy_excess[t] = 0.0

            votes = 0
            votes += int(env_peak_present[t])
            votes += int(energy_excess[t] >= cfg.energy_excess_min)
            votes += int(crest_factor[t] >= cfg.crest_factor_min)
            votes += int(kurt_vals[t] >= cfg.kurtosis_min)
            vote_count[t] = votes
            soft_score[t] = votes / 4.0
            soft_label[t] = votes >= int(cfg.min_positive_votes)

        return {
            "x_bp": x_bp,
            "frame_times": frame_times,
            "env": env,
            "env_times": env_times,
            "env_baseline": env_baseline,
            "env_peak_threshold": env_peak_threshold,
            "energy_excess": energy_excess,
            "crest_factor": crest_factor,
            "kurtosis": kurt_vals,
            "env_peak_present": env_peak_present,
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
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns
        -------
        frame_class : np.ndarray
            Per-frame FrameClass values encoded as int8.
        rain_conf : np.ndarray
            Per-frame rain confidence in [0, 1].
        det_debug : Dict[str, Any]
            Detector diagnostics and intermediate signals.
        """

        self._validate_rain_cfg()

        eps = float(self._dget("eps", 1e-9))

        op_band = self._dget("operating_band", (400.0, 3500.0))
        op_lo, op_hi = float(op_band[0]), float(op_band[1])

        mode_bands = self._dget("mode_bands", None)
        if mode_bands is None:
            raise AttributeError("Missing required detector param: mode_bands")

        mode_bands = tuple((float(a), float(b)) for (a, b) in mode_bands)

        # Optional time-domain soft-labeller configuration.
        td_soft_enable = bool(self._dget("td_soft_enable", False))
        td_soft_bp_order = int(self._dget("td_soft_bp_order", 4))
        td_soft_env_block_len = int(self._dget("td_soft_env_block_len", 32))
        td_soft_env_hop = int(self._dget("td_soft_env_hop", 16))
        td_soft_env_smooth_len = int(self._dget("td_soft_env_smooth_len", 3))
        td_soft_energy_excess_min = float(self._dget("td_soft_energy_excess_min", 0.20))
        td_soft_crest_factor_min = float(self._dget("td_soft_crest_factor_min", 3.0))
        td_soft_kurtosis_min = float(self._dget("td_soft_kurtosis_min", 3.5))
        td_soft_min_positive_votes = int(self._dget("td_soft_min_positive_votes", 2))

        primary_mode_idx = int(self._dget("primary_mode_idx", 0))
        if primary_mode_idx < 0 or primary_mode_idx >= len(mode_bands):
            raise ValueError(
                f"primary_mode_idx ({primary_mode_idx}) out of range for "
                f"mode_bands length ({len(mode_bands)})"
            )

        # noise_hi remains part of NOISE frame assignment.
        # The legacy rain_hi threshold is no longer used by the flux-based detector.
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
                    bp_order=td_soft_bp_order,
                    env_block_len=td_soft_env_block_len,
                    env_hop=td_soft_env_hop,
                    env_smooth_len=td_soft_env_smooth_len,
                    energy_excess_min=td_soft_energy_excess_min,
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
        flux_from_t2 = np.full(T, np.nan, dtype=np.float64)

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


        prev_frame_1 = None  # frame at t-1
        prev_frame_2 = None  # frame at t-2

        for t in range(T):
            frame = P_band[:, t]

            if prev_frame_1 is None:
                # First frame: no previous reference available.
                flux_primary[t] = 0.0
                flux_modes[t] = 0.0
                flux_from_t2[t] = 0.0
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
                flux_from_t2[t] = 0.0
                prev_frame_2 = prev_frame_1
                prev_frame_1 = frame
            else:
                # Use only the non-overlapping t-vs-(t-2) positive rise.
                delta2 = frame - prev_frame_2
                d2_pos = np.maximum(delta2, 0.0)
                flux = d2_pos
                flux_from_t2[t] = float(np.sum(d2_pos))
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

            out = np.nan_to_num(out, nan=mode_flux_norm_min, posinf=mode_flux_norm_min, neginf=mode_flux_norm_min)
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
            "flux_from_t2": flux_from_t2,
            "mode_flux_rain_pass": mode_flux_rain_pass,
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
                "mode_bands": mode_bands,
            },
        }

        return frame_class, rain_conf, det_debug


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

        frame_class, rain_conf, det_debug = self._detect_rain_over_time(P, freqs)
        frame_class = np.asarray(frame_class, dtype=np.int8)
        is_rain = frame_class == FrameClass.RAIN
        noise_conf = np.asarray(
            det_debug.get("noise_conf", np.clip(1.0 - rain_conf, 0.0, 1.0)),
            dtype=np.float64,
        )

        return {
            "frame_class": frame_class,
            "is_rain": is_rain,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "freqs": freqs,
            "times": times,
            "S": S,
            "x_filt": x,
            "debug": det_debug,
        }

    def run(self, audio: np.ndarray, params: Dict[str, Any]):
        """
        Framework-compatible wrapper.

        The audio_processing_framework expects each processor to expose:
            run(audio, params) -> (results_dict, state_dict)
        """
        self.setup(params)

        sr = int(params.get("sample_rate", params.get("fs", 11162)))
        results = self.process(audio, sr=sr)

        state = {
            "frame_class": results.get("frame_class"),
            "is_rain": results.get("is_rain"),
            "rain_conf": results.get("rain_conf"),
            "noise_conf": results.get("noise_conf"),
            "debug": results.get("debug"),
        }
        return results, state
