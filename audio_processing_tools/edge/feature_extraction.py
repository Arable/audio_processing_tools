from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.signal as spsig
from scipy.stats import kurtosis


RAIN_FRAME_CLASS = 2

RAW_SPECTRAL_FEATURE_NAMES = (
    "raw_spectral_centroid_hz",
    "raw_spectral_bandwidth_hz",
    "raw_low_freq_ratio",
    "raw_rain_band_ratio",
    "raw_mode_band_ratio_0",
    "raw_mode_band_ratio_1",
    "raw_mode_band_ratio_2",
    "raw_mode_band_ratio_3",
    "raw_mode_band_ratio_4",
    "raw_mode_band_entropy",
    "raw_mode_band_std",
    "raw_mode_band_max_ratio",
    "raw_spectral_flatness",
    "raw_spectral_rolloff_hz",
    "raw_dominant_freq_hz",
    "raw_frame_energy",
    "raw_cepstrum_coeff_0",
    "raw_cepstrum_coeff_1",
    "raw_cepstrum_coeff_2",
    "raw_cepstrum_coeff_3",
    "raw_cepstrum_coeff_4",
)

# --- TD feature name registries ---
TD_CORE_FEATURE_NAMES = (
    "frame_times",
    "td_crest_factor",
    "td_kurtosis",
    "td_block_energy_crest",
    "td_block_peak_width_50",
    "td_block_post_pre_energy_ratio",
)

TD_ENVELOPE_FEATURE_NAMES = (
    "td_energy_envelope",
    "td_rise_time_sec",
    "td_fall_time_sec",
    "td_rise_slope",
    "td_fall_slope",
    "td_peak_energy",
)

TD_FEATURE_NAMES = TD_CORE_FEATURE_NAMES + TD_ENVELOPE_FEATURE_NAMES


def resolve_np_dtype(process_dtype: str) -> Any:
    if process_dtype in ("float32", np.float32):
        return np.float32

    if process_dtype in ("float64", np.float64):
        return np.float64

    return np.dtype(process_dtype).type



# --- Helper for clip-level spectral occupancy ---
def default_spectral_occupancy_bands() -> Tuple[Tuple[str, float, float], ...]:
    """Default semantic frequency bands for clip-level spectral occupancy."""
    return (
        ("dc", 0.0, 43.6015625),
        ("wind_1", 43.6015625, 261.609375),
        ("wind_2", 261.609375, 436.015625),
        ("mode_1", 436.015625, 654.0234375),
        ("inter_1", 654.0234375, 784.828125),
        ("mode_2", 784.828125, 1046.4375),
        ("inter_2a", 1046.4375, 1264.4453125),
        ("inter_2b", 1264.4453125, 1482.453125),
        ("mode_3", 1482.453125, 1787.6640625),
        ("inter_3a", 1787.6640625, 2092.875),
        ("inter_3b", 2092.875, 2354.484375),
        ("mode_4", 2354.484375, 2616.09375),
        ("inter_4a", 2616.09375, 2790.5),
        ("inter_4b", 2790.5, 2964.90625),
        ("inter_4c", 2964.90625, 3139.3125),
        ("mode_5", 3139.3125, 3575.328125),
    )


def compute_clip_spectral_occupancy_stats(
    *,
    raw_power: np.ndarray,
    freqs: np.ndarray,
    frame_class: np.ndarray,
    bands: Optional[Tuple[Tuple[str, float, float], ...]] = None,
    dtype: Any = np.float32,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute compact clip-level spectral occupancy summaries.

    For each semantic frequency band, aggregate band log-power and per-frame
    band power ratio separately over rain and no-rain frames.
    """
    raw_power = np.asarray(raw_power, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    frame_class = np.asarray(frame_class).reshape(-1)

    if raw_power.ndim != 2:
        raise ValueError(f"raw_power must be 2-D, got shape={raw_power.shape}")
    if raw_power.shape[0] != freqs.size:
        raise ValueError(
            f"raw_power.shape[0] ({raw_power.shape[0]}) must match freqs.size ({freqs.size})"
        )
    if raw_power.shape[1] != frame_class.size:
        raise ValueError(
            f"raw_power.shape[1] ({raw_power.shape[1]}) must match frame_class.size ({frame_class.size})"
        )

    if bands is None:
        bands = default_spectral_occupancy_bands()
    bands = tuple((str(name), float(lo), float(hi)) for name, lo, hi in bands)
    n_bands = len(bands)
    n_frames = raw_power.shape[1]

    band_power = np.zeros((n_bands, n_frames), dtype=np.float64)
    for i, (_, lo, hi) in enumerate(bands):
        if i == n_bands - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            band_power[i, :] = np.sum(raw_power[mask, :], axis=0)

    total_band_power = np.sum(band_power, axis=0) + float(eps)
    log_power = np.log1p(np.maximum(band_power, 0.0))
    power_ratio = band_power / total_band_power.reshape(1, -1)

    rain_mask = frame_class == RAIN_FRAME_CLASS
    no_rain_mask = frame_class != RAIN_FRAME_CLASS

    def _empty() -> np.ndarray:
        return np.zeros(n_bands, dtype=dtype)

    def _stats(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
        if arr.shape[1] == 0 or not np.any(mask):
            return {
                f"{prefix}_mean": _empty(),
                f"{prefix}_std": _empty(),
                f"{prefix}_p50": _empty(),
                f"{prefix}_p90": _empty(),
                f"{prefix}_max": _empty(),
            }
        vals = arr[:, mask]
        return {
            f"{prefix}_mean": np.asarray(np.mean(vals, axis=1), dtype=dtype),
            f"{prefix}_std": np.asarray(np.std(vals, axis=1), dtype=dtype),
            f"{prefix}_p50": np.asarray(np.percentile(vals, 50, axis=1), dtype=dtype),
            f"{prefix}_p90": np.asarray(np.percentile(vals, 90, axis=1), dtype=dtype),
            f"{prefix}_max": np.asarray(np.max(vals, axis=1), dtype=dtype),
        }

    out: Dict[str, Any] = {
        "band_names": np.asarray([name for name, _, _ in bands], dtype=object),
        "band_lo_hz": np.asarray([lo for _, lo, _ in bands], dtype=dtype),
        "band_hi_hz": np.asarray([hi for _, _, hi in bands], dtype=dtype),
        "rain_frame_count": int(np.sum(rain_mask)),
        "no_rain_frame_count": int(np.sum(no_rain_mask)),
    }
    out.update(_stats(log_power, rain_mask, "rain_log_power"))
    out.update(_stats(power_ratio, rain_mask, "rain_power_ratio"))
    out.update(_stats(log_power, no_rain_mask, "no_rain_log_power"))
    out.update(_stats(power_ratio, no_rain_mask, "no_rain_power_ratio"))
    return out


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
    bp_order: int,
    subframe_len: int,
    subframe_hop: int,
    block_energy_len: int,
    block_energy_hop: Optional[int],
    block_energy_post_pre_blocks: int,
    block_energy_smooth_enable: bool,
    envelope_features_enable: bool,
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

    def _block_energy_peak_features(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sig.size == 0:
            z = np.zeros(0, dtype=dtype)
            return z, z, z

        B = int(max(1, block_energy_len))
        H = int(block_energy_hop) if block_energy_hop is not None else B
        H = max(1, H)

        if sig.size < B:
            z = np.zeros(0, dtype=dtype)
            return z, z, z

        starts = np.arange(0, sig.size - B + 1, H, dtype=np.int64)

        sig2 = np.asarray(sig, dtype=np.float64) ** 2
        csum = np.empty(sig2.size + 1, dtype=np.float64)
        csum[0] = 0.0
        csum[1:] = np.cumsum(sig2, dtype=np.float64)

        sums = csum[starts + B] - csum[starts]

        # Convert block mean-energy back to RMS-amplitude envelope before
        # computing crest-like features. This keeps td_block_energy_crest on a
        # dimensionless amplitude-crest scale comparable to td_crest_factor.
        env = np.sqrt(np.maximum(sums / float(B), 0.0))

        if block_energy_smooth_enable and env.size >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            env = np.convolve(env, kernel, mode="same")

        n_frames = max(0, 1 + (sig.size - frame_len) // hop)

        crest_out = np.zeros(n_frames, dtype=dtype)
        width_out = np.zeros(n_frames, dtype=dtype)
        ratio_out = np.zeros(n_frames, dtype=dtype)

        blocks_per_frame = max(1, int(np.ceil(frame_len / H)))
        post_pre = max(1, int(block_energy_post_pre_blocks))

        for t in range(n_frames):
            b0 = t * max(1, int(np.round(hop / H)))
            b1 = min(env.size, b0 + blocks_per_frame)

            if b1 <= b0:
                continue

            frame_env = env[b0:b1]

            # Crest factor of the block-energy envelope.
            # Do not add the waveform-domain eps inside sqrt here: block energies
            # can be very small, and sqrt(eps) can dominate the denominator,
            # incorrectly driving the crest factor toward ~0.
            rms_env = float(np.sqrt(np.mean(frame_env ** 2)))
            peak_idx_local = int(np.argmax(frame_env))
            peak_val = float(frame_env[peak_idx_local])

            crest_out[t] = peak_val / max(rms_env, eps)

            if peak_val > eps:
                try:
                    # Avoid scipy PeakPropertyWarning for flat / degenerate peaks.
                    # Require a real local peak with non-zero prominence.
                    if (
                        frame_env.size >= 3
                        and peak_idx_local > 0
                        and peak_idx_local < (frame_env.size - 1)
                    ):
                        left_v = float(frame_env[peak_idx_local - 1])
                        center_v = float(frame_env[peak_idx_local])
                        right_v = float(frame_env[peak_idx_local + 1])

                        prominence_est = center_v - max(left_v, right_v)

                        if prominence_est > eps:
                            widths, _, _, _ = spsig.peak_widths(
                                frame_env,
                                [peak_idx_local],
                                rel_height=0.5,
                            )

                            width_val = float(widths[0]) if len(widths) > 0 else 0.0

                            if np.isfinite(width_val) and width_val > 0.0:
                                width_out[t] = width_val
                            else:
                                width_out[t] = 0.0
                        else:
                            width_out[t] = 0.0
                    else:
                        width_out[t] = 0.0

                except Exception:
                    width_out[t] = 0.0

            peak_idx = b0 + peak_idx_local

            pre0 = max(0, peak_idx - post_pre)
            pre1 = peak_idx
            post0 = peak_idx + 1
            post1 = min(env.size, peak_idx + 1 + post_pre)

            pre_energy = float(np.mean(env[pre0:pre1])) if pre1 > pre0 else 0.0
            post_energy = float(np.mean(env[post0:post1])) if post1 > post0 else 0.0

            # Symmetric pre/post energy comparison around the pulse.
            # Values:
            #   ~0   -> balanced
            #   >0   -> longer post-tail / decay
            #   <0   -> sharper decay than rise
            ratio = (post_energy + eps) / (pre_energy + eps)
            ratio_out[t] = np.log(ratio)

        return crest_out, width_out, ratio_out

    def _subframe_peak_shape_features(
        sub_vals: np.ndarray,
        *,
        enable_envelope_features: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        env = np.asarray(sub_vals, dtype=dtype).reshape(-1)
        N = env.size
        if N == 0:
            z = np.zeros(0, dtype=dtype)
            return z, z, z, z, z, z, np.zeros(0, dtype=bool)

        if not enable_envelope_features:
            z = np.zeros(N, dtype=dtype)
            return z, z, z, z, z, z, np.zeros(N, dtype=bool)

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
        # Number of new subframes per STFT hop and total subframes per STFT frame.
        # When frame_len == hop (streaming), subs_per_hop == subs_per_frame == k.
        # When hop < frame_len (overlapping STFT), subs_per_hop < subs_per_frame.
        subs_per_hop = max(1, int(round(float(hop) / float(subframe_hop))))
        subs_per_frame = max(1, int(round(float(frame_len) / float(subframe_hop))))
        for t in range(n_frames):
            s0 = t * subs_per_hop
            s1 = min(sub_vals.size, s0 + subs_per_frame)
            if s1 > s0:
                out[t] = float(np.max(sub_vals[s0:s1]))
        return out

    def _frame_sum_from_subframes(sub_vals: np.ndarray, n_frames: int) -> np.ndarray:
        sub_vals = np.asarray(sub_vals, dtype=dtype).reshape(-1)
        out = np.zeros(n_frames, dtype=dtype)
        if n_frames == 0 or sub_vals.size == 0:
            return out
        subs_per_hop = max(1, int(round(float(hop) / float(subframe_hop))))
        subs_per_frame = max(1, int(round(float(frame_len) / float(subframe_hop))))
        for t in range(n_frames):
            s0 = t * subs_per_hop
            s1 = min(sub_vals.size, s0 + subs_per_frame)
            if s1 > s0:
                out[t] = float(np.sum(sub_vals[s0:s1]))
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

    frames = _frame_view(x_td)
    Tloc = frames.shape[0]
    frame_times = (np.arange(Tloc, dtype=dtype) * hop) / float(fs)
    sub_energy, _ = _subframe_energy(x_td)
    (
        block_energy_crest,
        block_peak_width_50,
        block_post_pre_energy_ratio,
    ) = _block_energy_peak_features(x_td)
    (
        sub_envelope,
        sub_rise_time,
        sub_fall_time,
        sub_rise_slope,
        sub_fall_slope,
        sub_peak_level,
        _,
    ) = _subframe_peak_shape_features(
        sub_energy,
        enable_envelope_features=bool(envelope_features_enable),
    )
    frame_envelope = _frame_sum_from_subframes(sub_envelope, Tloc)
    frame_rise_time = _frame_max_from_subframes(sub_rise_time, Tloc)
    frame_fall_time = _frame_max_from_subframes(sub_fall_time, Tloc)
    frame_rise_slope = _frame_max_from_subframes(sub_rise_slope, Tloc)
    frame_fall_slope = _frame_max_from_subframes(sub_fall_slope, Tloc)
    frame_peak_level = _frame_max_from_subframes(sub_peak_level, Tloc)
    td_crest_factor = np.zeros(Tloc, dtype=dtype)
    td_kurtosis = np.zeros(Tloc, dtype=dtype)

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
        "td_crest_factor": td_crest_factor,
        "td_kurtosis": td_kurtosis,
        "td_block_energy_crest": block_energy_crest,
        "td_block_peak_width_50": block_peak_width_50,
        "td_block_post_pre_energy_ratio": block_post_pre_energy_ratio,
        "td_energy_envelope": frame_envelope,
        "td_rise_time_sec": frame_rise_time,
        "td_fall_time_sec": frame_fall_time,
        "td_rise_slope": frame_rise_slope,
        "td_fall_slope": frame_fall_slope,
        "td_peak_energy": frame_peak_level,
    }


def extract_td_features_causal_frame_inline(
    *,
    x: np.ndarray,
    n_frames: int,
    fs: int,
    frame_len: int,
    hop: int,
    operating_band: Tuple[float, float],
    mode_bands: Optional[Tuple[Tuple[float, float], ...]],
    td_input_mode: str,
    td_input_band: Optional[Tuple[float, float]],
    bp_order: int,
    subframe_len: int,
    subframe_hop: int,
    block_energy_len: int,
    block_energy_hop: Optional[int],
    block_energy_post_pre_blocks: int,
    block_energy_smooth_enable: bool,
    envelope_features_enable: bool,
    process_dtype: str = "float32",
    eps: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """
    Causal frame-aligned TD feature extraction.

    For frame t, compute TD features only from samples in the causal FFT
    analysis window:

        x[t * hop : t * hop + frame_len]

    This mirrors embedded streaming behavior where frame decisions are emitted
    only after the current FFT window has fully arrived.

    Unlike extract_td_features_inline(), this avoids retrospective full-clip
    TD extraction timing.

    The caller is responsible for applying any desired causal prefilter before
    calling this function. This helper only controls the TD frame timing.
    """
    dtype = resolve_np_dtype(process_dtype)
    x = np.asarray(x, dtype=dtype).reshape(-1)

    out: Dict[str, list[float]] = {name: [] for name in TD_FEATURE_NAMES}

    for t in range(int(n_frames)):
        start = int(t * hop)
        end = int(start + frame_len)

        frame_x = x[start:end]

        # Zero-pad final partial frame if needed
        if frame_x.size < frame_len:
            frame_pad = np.zeros(frame_len, dtype=dtype)

            if frame_x.size > 0:
                frame_pad[: frame_x.size] = frame_x

            frame_x = frame_pad

        td_one = extract_td_features_inline(
            x=frame_x,
            fs=fs,
            frame_len=frame_len,
            hop=hop,
            operating_band=operating_band,
            mode_bands=mode_bands,
            td_input_mode=td_input_mode,
            td_input_band=td_input_band,
            bp_order=bp_order,
            subframe_len=subframe_len,
            subframe_hop=subframe_hop,
            block_energy_len=block_energy_len,
            block_energy_hop=block_energy_hop,
            block_energy_post_pre_blocks=block_energy_post_pre_blocks,
            block_energy_smooth_enable=block_energy_smooth_enable,
            envelope_features_enable=envelope_features_enable,
            process_dtype=process_dtype,
            eps=eps,
        )

        for name in TD_FEATURE_NAMES:
            vals = np.asarray(td_one.get(name, []), dtype=dtype).reshape(-1)
            out[name].append(float(vals[-1]) if vals.size else 0.0)

    return {name: np.asarray(values, dtype=dtype) for name, values in out.items()}


# --- Raw spectral-shape features for diagnostics ---
def extract_raw_spectral_shape_features_inline(
    *,
    fs: int,
    n_fft: int,
    hop: int,
    operating_band: Tuple[float, float],
    rain_band: Tuple[float, float] = (400.0, 800.0),
    low_band: Tuple[float, float] = (0.0, 200.0),
    mode_bands: Optional[Tuple[Tuple[float, float], ...]] = None,
    rolloff_fraction: float = 0.85,
    process_dtype: str = "float32",
    eps: float = 1e-12,
    raw_power: Optional[np.ndarray] = None,
    freqs: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract spectral-shape features from the raw linear power spectrum.

    raw_power (F, T) and freqs (F,) must be provided by the caller.
    These features intentionally use raw linear power, not the detector input P.
    The detector P may already be log(P(t)) - log(N_lag(t)), which is useful for
    novelty/flux detection but not ideal for answering where the original energy
    is distributed across frequency.
    """
    dtype = resolve_np_dtype(process_dtype)

    def _empty_raw_spectral_features() -> Dict[str, np.ndarray]:
        z = np.zeros(0, dtype=dtype)
        return {k: z for k in RAW_SPECTRAL_FEATURE_NAMES}

    if raw_power is None or freqs is None:
        return _empty_raw_spectral_features()

    power = np.asarray(raw_power, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if power.ndim != 2:
        raise ValueError(f"raw_power must be 2-D, got shape={power.shape}")
    if power.shape[0] != freqs.size:
        raise ValueError(
            f"raw_power.shape[0] ({power.shape[0]}) must match freqs.size ({freqs.size})"
        )

    if power.size == 0 or power.shape[1] == 0:
        return _empty_raw_spectral_features()

    total_power = np.sum(power, axis=0) + eps

    # Exclude DC from ratio normalization. DC / very-low-frequency energy is
    # not meaningful for rain/noise spectral occupancy analysis and can dilute
    # the ratios.
    non_dc_mask = freqs > 0.0
    total_power_no_dc = (
        np.sum(power[non_dc_mask, :], axis=0) + eps
        if np.any(non_dc_mask)
        else total_power
    )

    low_lo, low_hi = float(low_band[0]), float(low_band[1])
    rain_lo, rain_hi = float(rain_band[0]), float(rain_band[1])
    op_lo, op_hi = float(operating_band[0]), float(operating_band[1])

    low_mask = (freqs >= max(low_lo, eps)) & (freqs < low_hi)
    rain_mask = (freqs >= rain_lo) & (freqs <= rain_hi)
    op_mask = (freqs >= op_lo) & (freqs <= op_hi)

    op_power = (
        np.sum(power[op_mask, :], axis=0) + eps
        if np.any(op_mask)
        else total_power
    )

    # Spectral centroid, bandwidth, rolloff, flatness, and cepstrum are computed
    # over the operating band so they describe the spectral shape seen by the
    # rain detector.
    shape_power = power[op_mask, :] if np.any(op_mask) else power[non_dc_mask, :]
    shape_freqs = freqs[op_mask] if np.any(op_mask) else freqs[non_dc_mask]
    if shape_power.size == 0 or shape_power.shape[0] == 0:
        shape_power = power
        shape_freqs = freqs

    shape_total_power = np.sum(shape_power, axis=0) + eps
    shape_freq_col = shape_freqs.reshape(-1, 1).astype(np.float64)

    centroid = np.sum(shape_freq_col * shape_power, axis=0) / shape_total_power
    bandwidth = np.sqrt(
        np.sum(
            ((shape_freq_col - centroid.reshape(1, -1)) ** 2) * shape_power,
            axis=0,
        )
        / shape_total_power
    )

    low_ratio = (
        np.sum(power[low_mask, :], axis=0) / total_power_no_dc
        if np.any(low_mask)
        else np.zeros(power.shape[1], dtype=np.float64)
    )
    rain_ratio = (
        np.sum(power[rain_mask, :], axis=0) / total_power_no_dc
        if np.any(rain_mask)
        else np.zeros(power.shape[1], dtype=np.float64)
    )

    # Mode-band occupancy features. These summarize how raw spectral energy is
    # distributed across configured rain resonance bands.
    if mode_bands is None:
        mode_bands = (
            (450.0, 650.0),
            (800.0, 1050.0),
            (1500.0, 1800.0),
            (2350.0, 2550.0),
            (3150.0, 3350.0),
        )
    mode_bands = tuple((float(lo), float(hi)) for lo, hi in mode_bands)

    mode_band_power = []
    for lo, hi in mode_bands:
        m = (freqs >= float(lo)) & (freqs <= float(hi))
        if np.any(m):
            mode_band_power.append(np.sum(power[m, :], axis=0))
        else:
            mode_band_power.append(np.zeros(power.shape[1], dtype=np.float64))

    mode_band_power = np.asarray(mode_band_power, dtype=np.float64)
    mode_band_total = np.sum(mode_band_power, axis=0) + eps
    mode_band_ratio = mode_band_power / mode_band_total.reshape(1, -1)
    mode_band_entropy = -np.sum(mode_band_ratio * np.log(mode_band_ratio + eps), axis=0)
    mode_band_std = np.std(mode_band_ratio, axis=0)
    mode_band_max_ratio = np.max(mode_band_ratio, axis=0)

    flat_power = power[op_mask, :] if np.any(op_mask) else power
    spectral_flatness = np.exp(np.mean(np.log(flat_power + eps), axis=0)) / (
        np.mean(flat_power + eps, axis=0) + eps
    )

    cumsum_power = np.cumsum(shape_power, axis=0)
    rolloff_threshold = float(np.clip(rolloff_fraction, 0.0, 1.0)) * shape_total_power
    rolloff_idx = np.argmax(cumsum_power >= rolloff_threshold.reshape(1, -1), axis=0)
    spectral_rolloff = shape_freqs[np.clip(rolloff_idx, 0, shape_freqs.size - 1)]

    dominant_idx = np.argmax(shape_power, axis=0)
    dominant_freq = shape_freqs[np.clip(dominant_idx, 0, shape_freqs.size - 1)]

    frame_energy = op_power

    # Real cepstrum computed from the operating-band log-power spectrum.
    # coeff_0 mostly tracks broad log-energy level while coeff_1..4 capture
    # compact spectral-envelope / resonance-shape structure.
    cepstrum_input = np.log(np.maximum(shape_power, eps))
    cepstrum = np.fft.irfft(cepstrum_input, axis=0)
    cep_coeffs = np.zeros((5, shape_power.shape[1]), dtype=np.float64)
    n_cep = min(cep_coeffs.shape[0], cepstrum.shape[0])
    if n_cep > 0:
        cep_coeffs[:n_cep, :] = cepstrum[:n_cep, :]

    def _mode_ratio_or_zero(index: int) -> np.ndarray:
        if mode_band_ratio.shape[0] > index:
            return np.asarray(mode_band_ratio[index], dtype=dtype)
        return np.zeros(power.shape[1], dtype=dtype)

    return {
        "raw_spectral_centroid_hz": np.asarray(centroid, dtype=dtype),
        "raw_spectral_bandwidth_hz": np.asarray(bandwidth, dtype=dtype),
        "raw_low_freq_ratio": np.asarray(low_ratio, dtype=dtype),
        "raw_rain_band_ratio": np.asarray(rain_ratio, dtype=dtype),
        "raw_mode_band_ratio_0": _mode_ratio_or_zero(0),
        "raw_mode_band_ratio_1": _mode_ratio_or_zero(1),
        "raw_mode_band_ratio_2": _mode_ratio_or_zero(2),
        "raw_mode_band_ratio_3": _mode_ratio_or_zero(3),
        "raw_mode_band_ratio_4": _mode_ratio_or_zero(4),
        "raw_mode_band_entropy": np.asarray(mode_band_entropy, dtype=dtype),
        "raw_mode_band_std": np.asarray(mode_band_std, dtype=dtype),
        "raw_mode_band_max_ratio": np.asarray(mode_band_max_ratio, dtype=dtype),
        "raw_spectral_flatness": np.asarray(spectral_flatness, dtype=dtype),
        "raw_spectral_rolloff_hz": np.asarray(spectral_rolloff, dtype=dtype),
        "raw_dominant_freq_hz": np.asarray(dominant_freq, dtype=dtype),
        "raw_frame_energy": np.asarray(frame_energy, dtype=dtype),
        "raw_cepstrum_coeff_0": np.asarray(cep_coeffs[0], dtype=dtype),
        "raw_cepstrum_coeff_1": np.asarray(cep_coeffs[1], dtype=dtype),
        "raw_cepstrum_coeff_2": np.asarray(cep_coeffs[2], dtype=dtype),
        "raw_cepstrum_coeff_3": np.asarray(cep_coeffs[3], dtype=dtype),
        "raw_cepstrum_coeff_4": np.asarray(cep_coeffs[4], dtype=dtype),
    }
