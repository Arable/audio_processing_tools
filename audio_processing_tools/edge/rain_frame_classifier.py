from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Tuple, Optional

import numpy as np
import scipy.signal as spsig
from scipy.stats import kurtosis
from scipy.signal import peak_widths

class FrameClass(IntEnum):
    """Frame classification used by the rain detector and downstream suppressor."""

    NOISE = 0
    UNCERTAIN = 1
    RAIN = 2


def resolve_np_dtype(process_dtype: str):
    dt = str(process_dtype).lower()
    return np.float32 if dt == "float32" else np.float64

def causal_stochastic_low_quantile_baseline(
    x: np.ndarray,
    *,
    q_percent: float,
    samples_per_sec: float,
    win_sec: float,
    min_hist_sec: float = 0.0,
    floor: float = 1e-6,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal stochastic low-quantile baseline tracker.

    The returned baseline at index t is the estimate *before* ingesting x[t], so it is
    causal in the same sense as the previous rolling-quantile implementation.
    """
    x = np.asarray(x, dtype=dtype).reshape(-1)
    T = x.size
    if T == 0:
        return x.copy(), np.zeros(0, dtype=bool)

    q = float(np.clip(q_percent, 0.0, 100.0)) / 100.0
    floor = float(max(floor, 1e-12))
    samples_per_sec = float(max(samples_per_sec, 1e-6))
    W = max(3, int(round(float(win_sec) * samples_per_sec)))
    eta = float(np.clip(2.0 / max(W + 1, 2), 1e-4, 1.0))
    min_hist = max(1, int(round(float(min_hist_sec) * samples_per_sec)))
    scale_alpha = float(np.clip(1.0 - eta, 0.0, 0.9999))

    baseline = float(max(x[0], floor))
    scale = float(max(abs(x[0]), floor))
    out = np.empty(T, dtype=dtype)
    warm_ok = np.zeros(T, dtype=bool)
    hist_count = 0

    for t in range(T):
        # Causal estimate: emit current baseline before ingesting x[t].
        out[t] = baseline
        warm_ok[t] = hist_count >= min_hist

        xt = float(x[t])
        err = xt - baseline
        scale = scale_alpha * scale + (1.0 - scale_alpha) * abs(err)
        step = eta * max(scale, floor)
        delta = q * step if xt >= baseline else -(1.0 - q) * step
        baseline = float(max(baseline + delta, floor))
        hist_count += 1

    out = np.nan_to_num(out, nan=floor, posinf=floor, neginf=floor)
    out = np.maximum(out, floor)

    return out, warm_ok

# --- TD soft label assignment helper ---

def assign_td_soft_label(
    *,
    td_time_flux_score: np.ndarray,
    td_crest_factor: np.ndarray,
    td_kurtosis: np.ndarray,
    time_flux_thr: float,
    crest_thr: float,
    kurt_thr: float,
    min_positive_votes: int = 2,
    eps: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """Assign TD soft label from TD features."""
    td_time_flux_score = np.asarray(td_time_flux_score)
    td_crest_factor = np.asarray(td_crest_factor)
    td_kurtosis = np.asarray(td_kurtosis)

    T = td_time_flux_score.shape[0]

    vote_count = np.zeros(T, dtype=np.int32)
    vote_count += (td_time_flux_score >= float(time_flux_thr)).astype(np.int32)
    vote_count += (td_crest_factor >= float(crest_thr)).astype(np.int32)
    vote_count += (td_kurtosis >= float(kurt_thr)).astype(np.int32)

    soft_score = vote_count.astype(np.float32) / 3.0
    soft_label = vote_count >= int(min_positive_votes)

    return {
        "td_vote_count": vote_count,
        "td_soft_score": soft_score,
        "td_soft_label": soft_label,
    }

# --- Inline TD feature extraction for detector use ---

def extract_td_features_inline(
    *,
    x: np.ndarray,
    fs: int,
    frame_len: int,
    hop: int,
    operating_band: Tuple[float, float],
    mode_bands: Optional[Tuple[Tuple[float, float], ...]],
    td_input_mode: str,
    td_input_band: Optional[Tuple[float, float]],
    time_flux_band: Optional[Tuple[float, float]],
    bp_order: int,
    subframe_len: int,
    subframe_hop: int,
    baseline_win_sec: float,
    baseline_min_hist_sec: float,
    baseline_q: float,
    baseline_floor: float,
    process_dtype: str = "float32",
    eps: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """Inline TD feature extraction for detector use without a labeller-class dependency."""
    dtype = resolve_np_dtype(process_dtype)
    x = np.asarray(x, dtype=dtype).reshape(-1)

    def _bandpass(sig: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        if sig.size == 0:
            return sig.copy()
        nyq = 0.5 * float(fs)
        lo = float(np.clip(band[0], 1e-3, nyq * 0.999))
        hi = float(np.clip(band[1], lo + 1e-3, nyq * 0.999))
        sos = spsig.butter(int(bp_order), [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        try:
            return spsig.sosfiltfilt(sos, sig).astype(dtype, copy=False)
        except ValueError:
            return spsig.sosfilt(sos, sig).astype(dtype, copy=False)

    def _mode_band_comb(sig: np.ndarray) -> np.ndarray:
        if sig.size == 0:
            return sig.copy()
        if not mode_bands:
            return _bandpass(sig, operating_band)
        y_sum = np.zeros_like(sig, dtype=dtype)
        for band in mode_bands:
            y_sum += _bandpass(sig, band)
        return y_sum

    def _frame_view(sig: np.ndarray) -> np.ndarray:
        if sig.size < frame_len:
            return np.empty((0, frame_len), dtype=dtype)
        Tloc = 1 + (sig.size - frame_len) // hop
        stride = sig.strides[0]
        return np.lib.stride_tricks.as_strided(
            sig,
            shape=(Tloc, frame_len),
            strides=(hop * stride, stride),
            writeable=False,
        )

    def _subframe_energy(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if sig.size == 0:
            return np.zeros(0, dtype=dtype), np.zeros(0, dtype=dtype)
        B = int(max(1, subframe_len))
        H = int(max(1, subframe_hop))
        if sig.size < B:
            energy = np.array([float(np.mean(sig**2))], dtype=dtype)
            times = np.array([0.0], dtype=dtype)
            return energy, times

        starts = np.arange(0, sig.size - B + 1, H, dtype=np.int64)
        sig2 = np.asarray(sig, dtype=np.float64) ** 2
        csum = np.empty(sig2.size + 1, dtype=np.float64)
        csum[0] = 0.0
        csum[1:] = np.cumsum(sig2, dtype=np.float64)
        sums = csum[starts + B] - csum[starts]
        energy = (sums / float(B)).astype(dtype, copy=False)
        times = (starts / float(fs)).astype(dtype, copy=False)
        return energy, times

    def _subframe_peak_shape_features(
        sub_vals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        env = np.asarray(sub_vals, dtype=dtype).reshape(-1)
        N = env.size
        if N == 0:
            z = np.zeros(0, dtype=dtype)
            return z, z, z, z, z, z, np.zeros(0, dtype=bool)

        if N >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            env_smooth = np.convolve(env.astype(np.float64), kernel, mode="same")
        else:
            env_smooth = env.astype(np.float64, copy=False)

        rise_time = np.zeros(N, dtype=dtype)
        fall_time = np.zeros(N, dtype=dtype)
        peak_level = np.zeros(N, dtype=dtype)
        peak_mask = np.zeros(N, dtype=bool)
        rise_slope = np.zeros(N, dtype=dtype)
        fall_slope = np.zeros(N, dtype=dtype)
        dt_sec = float(subframe_hop) / float(fs)

        if N >= 3:
            peak_idx = np.flatnonzero(
                (env_smooth[1:-1] >= env_smooth[:-2])
                & (env_smooth[1:-1] > env_smooth[2:])
            ) + 1
        elif N == 2:
            peak_idx = np.array([int(np.argmax(env_smooth))], dtype=np.int64)
        else:
            peak_idx = np.array([0], dtype=np.int64)

        for p in peak_idx:
            peak = float(max(env_smooth[p], eps))
            lo = 0.1 * peak
            hi = 0.9 * peak

            left = env_smooth[: p + 1]
            lo_left = np.where(left <= lo)[0]
            i_lo = int(lo_left[-1]) if lo_left.size else 0
            hi_after = np.where(left[i_lo:] >= hi)[0]
            i_hi = int(i_lo + hi_after[0]) if hi_after.size else int(p)

            right = env_smooth[p:]
            below_hi = np.where(right[1:] <= hi)[0]
            i_hi_fall = int(1 + below_hi[0]) if below_hi.size else 0
            below_lo = np.where(right[i_hi_fall:] <= lo)[0]
            i_lo_fall = int(i_hi_fall + below_lo[0]) if below_lo.size else int(max(right.size - 1, 0))

            rise_dt = float(max(i_hi - i_lo, 0)) * dt_sec
            fall_dt = float(max(i_lo_fall, 0)) * dt_sec
            rise_time[p] = rise_dt
            fall_time[p] = fall_dt

            amp_delta_rise = max(hi - lo, 0.0)
            amp_delta_fall = max(hi - lo, 0.0)
            rise_slope[p] = float(amp_delta_rise / max(rise_dt, dt_sec))
            fall_slope[p] = float(amp_delta_fall / max(fall_dt, dt_sec))

            peak_level[p] = peak
            peak_mask[p] = True

        return (
            np.asarray(env_smooth, dtype=dtype),
            rise_time,
            fall_time,
            rise_slope,
            fall_slope,
            peak_level,
            peak_mask,
        )

    def _frame_max_from_subframes(sub_vals: np.ndarray, n_frames: int) -> np.ndarray:
        sub_vals = np.asarray(sub_vals, dtype=dtype).reshape(-1)
        out = np.zeros(n_frames, dtype=dtype)
        if n_frames == 0 or sub_vals.size == 0:
            return out
        padded = np.zeros(n_frames + 1, dtype=dtype)
        ncopy = min(sub_vals.size, n_frames + 1)
        padded[:ncopy] = sub_vals[:ncopy]
        return np.maximum(padded[:-1], padded[1:])

    def _frame_sum_from_subframes(sub_vals: np.ndarray, n_frames: int) -> np.ndarray:
        sub_vals = np.asarray(sub_vals, dtype=dtype).reshape(-1)
        out = np.zeros(n_frames, dtype=dtype)
        if n_frames == 0 or sub_vals.size == 0:
            return out
        for t in range(n_frames):
            v0 = sub_vals[t] if t < sub_vals.size else 0.0
            v1 = sub_vals[t + 1] if (t + 1) < sub_vals.size else 0.0
            out[t] = float(v0 + v1)
        return out

    td_mode = str(td_input_mode).lower()
    if td_mode == "default":
        # Use the caller-provided waveform as-is. When the caller passes x_proc,
        # this becomes the default operating-band TD frontend.
        x_td = x.copy()
    elif td_mode == "comb_filter":
        x_td = _mode_band_comb(x)
    elif td_mode == "bandpass":
        band = td_input_band if td_input_band is not None else operating_band
        x_td = _bandpass(x, band)
    else:
        raise ValueError(
            f"Unsupported td_input_mode={td_input_mode!r}. "
            "Expected one of {'default', 'comb_filter', 'bandpass'}."
        )

    if time_flux_band is not None:
        x_flux = _bandpass(x, time_flux_band)
    else:
        x_flux = x_td.copy()
    
    frames = _frame_view(x_td)
    Tloc = frames.shape[0]
    frame_times = (np.arange(Tloc, dtype=dtype) * hop) / float(fs)
    sub_energy, sub_times = _subframe_energy(x_flux)
    sub_envelope, sub_rise_time, sub_fall_time, sub_rise_slope, sub_fall_slope, sub_peak_level, sub_peak_mask = _subframe_peak_shape_features(sub_energy)
    sub_baseline, sub_warm_ok = causal_stochastic_low_quantile_baseline(
        sub_energy,
        q_percent=float(baseline_q),
        samples_per_sec=float(fs) / max(float(subframe_hop), 1.0),
        win_sec=float(baseline_win_sec),
        min_hist_sec=float(baseline_min_hist_sec),
        floor=float(baseline_floor),
        dtype=dtype,
    )
    frame_energy = _frame_sum_from_subframes(sub_energy, Tloc)
    frame_envelope = _frame_sum_from_subframes(sub_envelope, Tloc)
    frame_rise_time = _frame_max_from_subframes(sub_rise_time, Tloc)
    frame_fall_time = _frame_max_from_subframes(sub_fall_time, Tloc)
    frame_rise_slope = _frame_max_from_subframes(sub_rise_slope, Tloc)
    frame_fall_slope = _frame_max_from_subframes(sub_fall_slope, Tloc)
    frame_peak_level = _frame_max_from_subframes(sub_peak_level, Tloc)
    frame_baseline = _frame_sum_from_subframes(sub_baseline, Tloc)
    frame_warm_ok = np.zeros(Tloc, dtype=bool)
    if sub_warm_ok.size > 0:
        for t in range(Tloc):
            ok0 = bool(sub_warm_ok[t]) if t < sub_warm_ok.size else False
            ok1 = bool(sub_warm_ok[t + 1]) if (t + 1) < sub_warm_ok.size else False
            frame_warm_ok[t] = ok0 and ok1
    baseline_ref = np.maximum(frame_baseline, float(baseline_floor))

    frame_flux = np.zeros(Tloc, dtype=dtype)
    td_time_flux_score = np.zeros(Tloc, dtype=dtype)
    td_crest_factor = np.zeros(Tloc, dtype=dtype)
    td_kurtosis = np.zeros(Tloc, dtype=dtype)

    if Tloc > 2:
        frame_flux[2:] = np.maximum(frame_energy[2:] - frame_energy[:-2], 0.0)

    td_time_flux_score = frame_flux / baseline_ref
    td_time_flux_score = np.where(frame_warm_ok, td_time_flux_score, 0.0)

    for t in range(Tloc):
        seg = np.asarray(frames[t], dtype=dtype)
        rms = float(np.sqrt(np.mean(seg**2) + eps))
        peak_abs = float(np.max(np.abs(seg))) if seg.size else 0.0
        td_crest_factor[t] = peak_abs / max(rms, eps)
        if seg.size >= 4:
            kv = float(kurtosis(seg, fisher=False, bias=False))
            td_kurtosis[t] = kv if np.isfinite(kv) else 0.0
        else:
            td_kurtosis[t] = 0.0

    return {
        "frame_times": frame_times,
        "time_flux_score": td_time_flux_score,
        "crest_factor": td_crest_factor,
        "kurtosis": td_kurtosis,
        "energy_envelope": frame_envelope,
        "rise_time_sec": frame_rise_time,
        "fall_time_sec": frame_fall_time,
        "rise_slope": frame_rise_slope,
        "fall_slope": frame_fall_slope,
        "peak_energy": frame_peak_level,
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

    def _align_feature_to_frames(
        self,
        values: Any,
        *,
        n_frames: int,
        dtype: Any,
        fill_value: float | int | bool = 0,
    ) -> np.ndarray:
        """Align a 1-D feature array to the detector frame count by truncation / zero-fill."""
        out = np.full(n_frames, fill_value, dtype=dtype)
        if values is None:
            return out
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        ncopy = min(n_frames, arr.size)
        if ncopy > 0:
            out[:ncopy] = arr[:ncopy]
        return out

    def _resolve_detector_frame_times(
        self,
        detector_frame_times: Optional[np.ndarray],
        *,
        n_frames: int,
        dtype: Any,
    ) -> np.ndarray:
        """Resolve detector frame times from the provided array or from hop/fs."""
        if detector_frame_times is None:
            return (
                np.arange(n_frames, dtype=dtype) * float(self._dget("hop", 128))
            ) / float(self._dget("sample_rate", self._dget("fs", 11162)))
        return np.asarray(detector_frame_times, dtype=dtype).reshape(-1)

    def _rain_frame_decision(
        self,
        *,
        td_crest_factor: np.ndarray,
        td_kurtosis: np.ndarray,
        mode_flux_score: np.ndarray,
        c_thr: float,
        k_thr: float,
        m_thr: float,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Frame-level TD+FD rain decision and confidence."""
        td_crest_factor = np.asarray(td_crest_factor)
        td_kurtosis = np.asarray(td_kurtosis)
        mode_flux_score = np.asarray(mode_flux_score)

        is_rain = (
            (td_crest_factor > float(c_thr))
            & (td_kurtosis > float(k_thr))
            & (mode_flux_score >= float(m_thr))
        )

        crest_conf = td_crest_factor / max(float(c_thr), float(eps))
        kurt_conf = td_kurtosis / max(float(k_thr), float(eps))
        mode_conf = mode_flux_score / max(float(m_thr), float(eps))
        rain_conf = np.clip(np.minimum(np.minimum(crest_conf, kurt_conf), mode_conf), 0.0, 1.0)

        return is_rain, rain_conf
    # ------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------

    def _detect_rain_over_time(
        self,
        P: np.ndarray,
        freqs: np.ndarray,
        detector_frame_times: Optional[np.ndarray] = None,
        input_audio: Optional[np.ndarray] = None,
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
        dtype = resolve_np_dtype(self._dget("process_dtype", "float32"))

        include_peak_payload = bool(self._dget("feature_dump_include_peak_payload", False))

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
        td_input_mode = str(self._dget("td_input_mode", "default")).lower()
        td_input_band = self._dget("td_input_band", None)
        if td_input_band is not None:
            td_input_band = (float(td_input_band[0]), float(td_input_band[1]))
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

        # Flux-based detector controls.
        # mode_flux_rain_min: main novelty threshold in the mode bands.
        # mode_flux_noise_max: frames below this are eligible for NOISE when rain evidence is weak.
        # The detector score is normalized excess-over-baseline novelty; rain_conf
        # is derived from the active TD+FD rule rather than from a separate arbitrary scale.
        mode_flux_rain_min = float(self._dget("mode_flux_rain_min", 4.0))
        mode_flux_noise_max = float(self._dget("mode_flux_noise_max", 1.5))
        mode_flux_rain_min = max(mode_flux_rain_min, 0.0)
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

        # Optional winsorization of mode-flux before baseline normalization.
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
        if td_soft_enable and input_audio is not None:
            try:
                x_in = np.asarray(input_audio, dtype=dtype).reshape(-1)
                td_soft_debug = extract_td_features_inline(
                    x=x_in,
                    fs=int(self._dget("sample_rate", self._dget("fs", 11162))),
                    frame_len=int(self._dget("n_fft", 256)),
                    hop=int(self._dget("hop", 128)),
                    operating_band=(op_lo, op_hi),
                    mode_bands=tuple((float(a), float(b)) for (a, b) in mode_bands),
                    td_input_mode=td_input_mode,
                    td_input_band=td_input_band,
                    time_flux_band=td_soft_time_flux_band,
                    bp_order=td_soft_bp_order,
                    subframe_len=td_soft_subframe_len,
                    subframe_hop=td_soft_subframe_hop,
                    baseline_win_sec=td_soft_baseline_win_sec,
                    baseline_min_hist_sec=td_soft_baseline_min_hist_sec,
                    baseline_q=td_soft_baseline_q,
                    baseline_floor=td_soft_baseline_floor,
                    process_dtype=str(self._dget("process_dtype", "float32")),
                    eps=eps,
                )
            except Exception as e:
                td_soft_debug = {"error": str(e)}
        detector_frame_times = self._resolve_detector_frame_times(
            detector_frame_times,
            n_frames=T,
            dtype=dtype,
        )

        td_soft_label = np.zeros(T, dtype=bool)
        td_time_flux_score = np.zeros(T, dtype=dtype)
        td_crest_factor = np.zeros(T, dtype=dtype)
        td_kurtosis = np.zeros(T, dtype=dtype)
        td_rise_time_sec = np.zeros(T, dtype=dtype)
        td_fall_time_sec = np.zeros(T, dtype=dtype)
        td_rise_slope = np.zeros(T, dtype=dtype)
        td_fall_slope = np.zeros(T, dtype=dtype)
        td_energy_envelope = np.zeros(T, dtype=dtype)
        td_peak_energy = np.zeros(T, dtype=dtype)
        td_vote_count = np.zeros(T, dtype=np.int32)
        td_soft_score = np.zeros(T, dtype=dtype)

        if td_soft_debug and ("error" not in td_soft_debug):
            td_time_flux_score = self._align_feature_to_frames(
                td_soft_debug.get("time_flux_score"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_crest_factor = self._align_feature_to_frames(
                td_soft_debug.get("crest_factor"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_kurtosis = self._align_feature_to_frames(
                td_soft_debug.get("kurtosis"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_rise_time_sec = self._align_feature_to_frames(
                td_soft_debug.get("rise_time_sec"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_fall_time_sec = self._align_feature_to_frames(
                td_soft_debug.get("fall_time_sec"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_rise_slope = self._align_feature_to_frames(
                td_soft_debug.get("rise_slope"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_fall_slope = self._align_feature_to_frames(
                td_soft_debug.get("fall_slope"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_energy_envelope = self._align_feature_to_frames(
                td_soft_debug.get("energy_envelope"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )
            td_peak_energy = self._align_feature_to_frames(
                td_soft_debug.get("peak_energy"),
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )

            td_label_out = assign_td_soft_label(
                td_time_flux_score=td_time_flux_score,
                td_crest_factor=td_crest_factor,
                td_kurtosis=td_kurtosis,
                time_flux_thr=td_soft_time_flux_score_min,
                crest_thr=td_soft_crest_factor_min,
                kurt_thr=td_soft_kurtosis_min,
                min_positive_votes=td_soft_min_positive_votes,
                eps=eps,
            )

            td_vote_count = td_label_out["td_vote_count"]
            td_soft_score = td_label_out["td_soft_score"]
            td_soft_label = td_label_out["td_soft_label"]
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
        peak_valid_prom_min_db = float(self._dget("peak_valid_prom_min_db", 3.0))
        peak_valid_prom_max_db = float(self._dget("peak_valid_prom_max_db", 6.0))

        peak_top_p = max(1, peak_top_p)
        primary_top_m = max(1, primary_top_m)
        peak_ratio_min = float(np.clip(peak_ratio_min, 0.0, 1.0))
        peak_valid_prom_max_db = max(peak_valid_prom_min_db, peak_valid_prom_max_db)

        flux_primary = np.full(T, np.nan, dtype=dtype)
        flux_modes = np.full(T, np.nan, dtype=dtype)

        peak_ratio = np.full(T, np.nan, dtype=dtype)
        # Binary peak gate debug stream (0.0 fail, 1.0 pass). Kept as a "score"
        # name for possible future soft-scoring extensions.
        peak_gate_score = np.full(T, np.nan, dtype=dtype)
        peak_valid_count = np.zeros(T, dtype=np.int32)
        peak_count_by_mode = np.zeros((len(mode_bands), T), dtype=np.int32)

        # Per-mode raw / normalized flux features for offline threshold tuning.
        mode_flux_by_mode = np.zeros((len(mode_bands), T), dtype=dtype)
        normalized_mode_flux_by_mode = np.zeros((len(mode_bands), T), dtype=dtype)

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
                    peak_valid_freqs_hz[i, t] = np.array([], dtype=dtype)
                    peak_valid_prominences_db[i, t] = np.array([], dtype=dtype)
                    peak_valid_bandwidths_hz[i, t] = np.array([], dtype=dtype)

            if prev_frame_1 is None:
                # First frame: no previous reference available.
                flux_primary[t] = 0.0
                flux_modes[t] = 0.0
                peak_ratio[t] = 0.0
                peak_gate_score[t] = 0.0
                peak_valid_count[t] = 0
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
                peak_ratio[t] = 0.0
                peak_gate_score[t] = 0.0
                peak_valid_count[t] = 0
                peak_count_by_mode[:, t] = 0
            else:
                # Valid peaks are those satisfying the requested prominence range.
                pk_h = np.asarray(props.get("peak_heights", spec_db[peaks]), dtype=dtype)
                pk_prom = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=dtype)
                widths_bins, *_ = peak_widths(spec_db, peaks, rel_height=0.5)
                df_hz = float(freqs_band[1] - freqs_band[0]) if freqs_band.size > 1 else 0.0
                pk_bw_hz = np.asarray(widths_bins, dtype=dtype) * df_hz

                valid_prom_mask = (
                    (pk_prom >= peak_valid_prom_min_db)
                    & (pk_prom <= peak_valid_prom_max_db)
                )
                peaks_valid = peaks[valid_prom_mask]
                pk_h_valid = pk_h[valid_prom_mask]
                pk_prom_valid = pk_prom[valid_prom_mask]
                pk_bw_hz_valid = pk_bw_hz[valid_prom_mask]
                peak_valid_count[t] = int(peaks_valid.size)

                for i, m_mask in enumerate(mode_masks):
                    if peaks_valid.size > 0:
                        in_mode_valid = m_mask[peaks_valid]
                        peak_count_by_mode[i, t] = int(np.sum(in_mode_valid))
                    else:
                        peak_count_by_mode[i, t] = 0

                # Optional per-mode representative peak payload.
                for i, m_mask in enumerate(mode_masks):
                    if peaks_valid.size == 0:
                        continue

                    in_mode_valid = m_mask[peaks_valid]
                    if include_peak_payload and np.any(in_mode_valid):
                        mode_freqs = freqs_band[peaks_valid[in_mode_valid]].astype(dtype)
                        mode_prom = pk_prom_valid[in_mode_valid].astype(dtype)
                        mode_bw = pk_bw_hz_valid[in_mode_valid].astype(dtype)
                        mode_heights = pk_h_valid[in_mode_valid].astype(dtype)

                        best_idx = int(np.argmax(mode_heights))
                        peak_valid_freqs_hz[i, t] = np.asarray([mode_freqs[best_idx]], dtype=dtype)
                        peak_valid_prominences_db[i, t] = np.asarray([mode_prom[best_idx]], dtype=dtype)
                        peak_valid_bandwidths_hz[i, t] = np.asarray([mode_bw[best_idx]], dtype=dtype)

                if peaks_valid.size == 0:
                    peak_ratio[t] = 0.0
                    peak_gate_score[t] = 0.0
                else:
                    # Strongest top-P valid peaks for gate computation.
                    order = np.argsort(pk_h_valid)[::-1]
                    sel = peaks_valid[order[:peak_top_p]]
                    in_primary = primary_mask[sel]
                    in_any_mode = np.zeros(sel.size, dtype=bool)
                    for m_mask in mode_masks:
                        in_any_mode |= m_mask[sel]

                    top_m = min(primary_top_m, sel.size)
                    ratio = float(np.sum(in_any_mode)) / float(max(1, sel.size))

                    primary_ok = float(np.any(in_primary[:top_m]))
                    mode_ok = float(ratio >= peak_ratio_min)

                    peak_ratio[t] = ratio
                    peak_gate_score[t] = min(primary_ok, mode_ok)

        def rolling_low_quantile_baseline(x: np.ndarray) -> np.ndarray:
            """Causal stochastic low-quantile baseline for sparse impulsive flux signals."""
            hop = float(self._dget("hop", 128))
            fs = float(self._dget("sample_rate", self._dget("fs", 11162)))
            frames_per_sec = fs / max(hop, 1.0)
            baseline, _ = causal_stochastic_low_quantile_baseline(
                x,
                q_percent=float(mode_flux_norm_q),
                samples_per_sec=frames_per_sec,
                win_sec=float(mode_flux_norm_win_sec),
                min_hist_sec=0.0,
                floor=float(mode_flux_norm_min),
                dtype=dtype,
            )
            return baseline

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

        # Peak-structure diagnostics are still computed and exported, although the
        # current simple TD+FD decision rule below does not directly use them.
        peak_gate = peak_gate_score >= 1.0

        # Simple TD+FD decision logic.
        mode_flux_score = np.nan_to_num(mode_flux_score, nan=0.0, posinf=0.0, neginf=0.0)
        td_crest_factor = np.nan_to_num(td_crest_factor, nan=0.0, posinf=0.0, neginf=0.0)
        td_kurtosis = np.nan_to_num(td_kurtosis, nan=0.0, posinf=0.0, neginf=0.0)
        td_soft_score = np.nan_to_num(td_soft_score, nan=0.0, posinf=0.0, neginf=0.0)

        c_thr = float(td_soft_crest_factor_min)
        k_thr = float(td_soft_kurtosis_min)
        m_thr = float(mode_flux_rain_min)

        is_rain, rain_conf = self._rain_frame_decision(
            td_crest_factor=td_crest_factor,
            td_kurtosis=td_kurtosis,
            mode_flux_score=mode_flux_score,
            c_thr=c_thr,
            k_thr=k_thr,
            m_thr=m_thr,
            eps=eps,
        )

        # Noise confidence is the complement of rain confidence, with weak mode flux
        # still used to assign explicit NOISE labels.
        noise_conf = np.clip(1.0 - rain_conf, 0.0, 1.0)
        weak_mode_flux = mode_flux_score <= mode_flux_noise_max

        # FrameClass is the canonical raw detector output.
        frame_class = np.full(T, FrameClass.UNCERTAIN, dtype=np.int8)
        frame_class[(noise_conf >= noise_hi) & weak_mode_flux & (~is_rain)] = FrameClass.NOISE
        frame_class[is_rain] = FrameClass.RAIN

        # PSD update gating is not decided here.
        # Downstream suppressor logic may optionally use FrameClass, depending on configuration.

        det_debug = {
            # Core detector features / outputs still relevant to the current rule
            "mode_flux_score": mode_flux_score,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "frame_class": frame_class,
            "peak_valid_count": peak_valid_count,
            "peak_count_by_mode": peak_count_by_mode,
            # TD first-class features and derived TD labels
            "td_soft_label": td_soft_label,
            "td_time_flux_score": td_time_flux_score,
            "td_crest_factor": td_crest_factor,
            "td_kurtosis": td_kurtosis,
            "td_rise_time_sec": td_rise_time_sec,
            "td_fall_time_sec": td_fall_time_sec,
            "td_rise_slope": td_rise_slope,
            "td_fall_slope": td_fall_slope,
            "td_energy_envelope": td_energy_envelope,
            "td_peak_energy": td_peak_energy,
            "td_vote_count": td_vote_count,
            "td_soft_score": td_soft_score,
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
            # Level 1: compact feature set for offline analysis and replay of the
            # current TD+FD decision logic, plus a few closely related TD shape features.
            feature_dump = {
                "frame_times": detector_frame_times,
                "mode_flux_score": mode_flux_score,
                "td_time_flux_score": td_time_flux_score,
                "td_crest_factor": td_crest_factor,
                "td_kurtosis": td_kurtosis,
            }
            if feature_dump_level == 1:
                feature_dump.update({
                    "normalized_mode_flux_by_mode": normalized_mode_flux_by_mode,
                    "frame_class": frame_class,
                    "td_soft_label": td_soft_label.astype(np.int8),
                    "peak_ratio": peak_ratio,
                    "td_rise_time_sec": td_rise_time_sec,
                    "td_fall_time_sec": td_fall_time_sec,
                    "td_rise_slope": td_rise_slope,
                    "td_fall_slope": td_fall_slope,
                    })

            if feature_dump_level > 1:
                # Level 2: richer diagnostics and detector outputs for analysis.
                feature_dump.update({
                    "peak_ratio": peak_ratio,
                    "flux_primary": flux_primary,
                    "peak_gate_score": peak_gate_score,
                    "peak_gate": peak_gate.astype(np.int8),
                    "peak_valid_count": peak_valid_count,
                    "peak_count_by_mode": peak_count_by_mode,
                    "rain_conf": rain_conf,
                    "noise_conf": noise_conf,
                    "td_soft_label": td_soft_label.astype(np.int8),
                    "td_vote_count": td_vote_count,
                    "td_soft_score": td_soft_score,
                    "td_energy_envelope": td_energy_envelope,
                    "td_peak_energy": td_peak_energy,
                })

                if include_peak_payload:
                    feature_dump.update({
                        "peak_valid_freqs_hz": peak_valid_freqs_hz,
                        "peak_valid_prominences_db": peak_valid_prominences_db,
                        "peak_valid_bandwidths_hz": peak_valid_bandwidths_hz,
                    })

        return frame_class, rain_conf, det_debug, feature_dump