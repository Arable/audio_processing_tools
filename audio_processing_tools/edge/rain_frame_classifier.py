from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
from collections.abc import Sequence

import numpy as np
import scipy.signal as spsig
from audio_processing_tools.edge.feature_extraction import (
    RAW_SPECTRAL_FEATURE_NAMES,
    TD_FEATURE_NAMES,
    compute_clip_spectral_occupancy_stats,
    extract_raw_spectral_shape_features_inline,
    extract_td_features_inline,
)


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


def assign_td_soft_label(
    *,
    td_crest_factor: np.ndarray,
    td_kurtosis: np.ndarray,
    crest_thr: float,
    kurt_thr: float,
    min_positive_votes: int = 2,
) -> Dict[str, np.ndarray]:
    """Assign TD soft label from TD impulse features."""
    td_crest_factor = np.asarray(td_crest_factor)
    td_kurtosis = np.asarray(td_kurtosis)

    T = td_crest_factor.shape[0]

    vote_count = np.zeros(T, dtype=np.int32)
    vote_count += (td_crest_factor >= float(crest_thr)).astype(np.int32)
    vote_count += (td_kurtosis >= float(kurt_thr)).astype(np.int32)

    soft_score = vote_count.astype(np.float32) / 2.0
    soft_label = vote_count >= int(min_positive_votes)

    return {
        "td_vote_count": vote_count,
        "td_soft_score": soft_score,
        "td_soft_label": soft_label,
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
    REQUIRED_CFG_FIELDS = ("mode_bands",)

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

    def _align_feature_dict_to_frames(
        self,
        feature_dict: Dict[str, Any],
        feature_names: Sequence[str],
        *,
        n_frames: int,
        dtype: Any,
        fill_value: float | int | bool = 0,
    ) -> Dict[str, np.ndarray]:
        """Align a dictionary of 1-D feature arrays to the detector frame count."""
        return {
            name: self._align_feature_to_frames(
                feature_dict.get(name),
                n_frames=n_frames,
                dtype=dtype,
                fill_value=fill_value,
            )
            for name in feature_names
        }

    def _resolve_detector_frame_times(
        self,
        detector_frame_times: Optional[np.ndarray],
        *,
        n_frames: int,
        dtype: Any,
    ) -> np.ndarray:
        """Resolve detector frame times from the provided array or from hop/fs."""
        if detector_frame_times is None:
            return (np.arange(n_frames, dtype=dtype) * float(self._dget("hop", 128))) / float(
                self._dget("sample_rate", self._dget("fs", 11162))
            )
        return np.asarray(detector_frame_times, dtype=dtype).reshape(-1)

    def _rain_frame_decision(
        self,
        *,
        primary_mode_flux: np.ndarray,
        support_mode_flux_1: np.ndarray,
        support_mode_flux_2: np.ndarray,
        support_mode_flux_3: np.ndarray,
        primary_flux_min: float,
        mode1_flux_min: float,
        mode2_flux_min: float,
        mode3_flux_min: float,
        min_support_count: int,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frame-level fixed-band FD rain decision using the gated normalized per-mode flux.

        Thresholds are applied to log1p-transformed gated mode-flux values so runtime
        matches the offline threshold search used during tuning.

        Rule:
            primary_ok = primary_mode_flux >= primary_flux_min
            support_hits = (support_mode_flux_1 >= mode1_flux_min)
                         + (support_mode_flux_2 >= mode2_flux_min)
                         + (support_mode_flux_3 >= mode3_flux_min)
            is_rain = primary_ok & (support_hits >= min_support_count)

        This intentionally matches the offline replay logic used for clip counts.
        rain_conf is returned as a binary float so downstream frame-count aggregation
        remains aligned with the hard frame decision.
        """

        # Apply clip + log1p in one step (matches offline threshold search space)
        f0 = np.log1p(np.clip(np.asarray(primary_mode_flux), 0.0, None))
        f1 = np.log1p(np.clip(np.asarray(support_mode_flux_1), 0.0, None))
        f2 = np.log1p(np.clip(np.asarray(support_mode_flux_2), 0.0, None))
        f3 = np.log1p(np.clip(np.asarray(support_mode_flux_3), 0.0, None))

        primary_flux_min = float(primary_flux_min)
        mode1_flux_min = float(mode1_flux_min)
        mode2_flux_min = float(mode2_flux_min)
        mode3_flux_min = float(mode3_flux_min)
        min_support_count = int(max(1, min_support_count))

        primary_ok = f0 >= primary_flux_min
        support_hits = (
            (f1 >= mode1_flux_min).astype(np.int32)
            + (f2 >= mode2_flux_min).astype(np.int32)
            + (f3 >= mode3_flux_min).astype(np.int32)
        )

        is_rain = primary_ok & (support_hits >= min_support_count)
        rain_conf = is_rain.astype(np.float32, copy=False)

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
        raw_power: Optional[np.ndarray] = None,
        work_dtype: Optional[Any] = None,
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
        if work_dtype is not None:
            dtype = work_dtype

        peak_features_enable = bool(self._dget("peak_features_enable", False))
        include_peak_payload = peak_features_enable and bool(self._dget("feature_dump_include_peak_payload", False))
        feature_dump_include_frame_class = bool(self._dget("feature_dump_include_frame_class", True))

        # Feature-dump size controls. These only affect what is persisted in
        # feature_dump; detector internals and det_debug remain available during
        # runtime for debugging.
        feature_dump_include_mode_flux_score = bool(self._dget("feature_dump_include_mode_flux_score", False))
        feature_dump_include_raw_spectral_basic = bool(self._dget("feature_dump_include_raw_spectral_basic", False))
        feature_dump_include_raw_spectral_frame_features = bool(
            self._dget("feature_dump_include_raw_spectral_frame_features", True)
        )
        feature_dump_include_td_soft = bool(self._dget("feature_dump_include_td_soft", False))
        feature_dump_include_td_envelope = bool(self._dget("feature_dump_include_td_envelope", False))
        feature_dump_include_peak_summary = bool(self._dget("feature_dump_include_peak_summary", False))
        feature_dump_dense_enable = bool(self._dget("feature_dump_dense_enable", True))
        feature_dump_sparse_enable = bool(self._dget("feature_dump_sparse_enable", False))
        feature_dump_clip_summary_enable = bool(self._dget("feature_dump_clip_summary_enable", False))
        feature_dump_sparse_gate_feature = (
            str(self._dget("feature_dump_sparse_gate_feature", "td_block_energy_crest")).strip().lower()
        )
        feature_dump_sparse_gate_threshold = float(self._dget("feature_dump_sparse_gate_threshold", 3.5))

        clip_spectral_occupancy_enable = bool(self._dget("clip_spectral_occupancy_enable", False))
        clip_spectral_occupancy_dtype = resolve_np_dtype(self._dget("clip_spectral_occupancy_dtype", "float32"))

        op_band = self._dget("operating_band", (400.0, 3500.0))
        op_lo, op_hi = float(op_band[0]), float(op_band[1])

        mode_bands = self._dget("mode_bands", None)
        if mode_bands is None:
            raise AttributeError("Missing required detector param: mode_bands")

        mode_bands = tuple((float(a), float(b)) for (a, b) in mode_bands)
        if len(mode_bands) < 4:
            raise ValueError(
                "Fixed-band rain decision requires at least 4 mode bands: "
                "mode 0 as primary and modes 1, 2, 3 as support"
            )

        # Raw spectral-shape diagnostics computed from the original waveform
        # spectrum rather than the detector novelty spectrum P.
        raw_spectral_shape_enable = bool(self._dget("raw_spectral_shape_enable", True))
        raw_spectral_rain_band = self._dget("raw_spectral_rain_band", (400.0, 800.0))
        raw_spectral_low_band = self._dget("raw_spectral_low_band", (50.0, 200.0))
        raw_spectral_rolloff_fraction = float(self._dget("raw_spectral_rolloff_fraction", 0.85))

        # TD features should match the previous pre-filtered detector path even
        # when input_audio is now the raw waveform. Prefer the processor's
        # existing _build_prefilter_sos() implementation when available.
        td_apply_input_prefilter = bool(self._dget("td_apply_input_prefilter", True))
        td_prefilter_mode = str(
            self._dget(
                "td_prefilter_mode",
                self._dget("pre_filter_mode", "none"),
            )
        ).lower()

        # TD features are always extracted when input_audio is available; soft labels remain optional.
        td_soft_enable = bool(self._dget("td_soft_enable", False))
        td_soft_bp_order = int(self._dget("td_soft_bp_order", 4))
        td_soft_subframe_len = int(self._dget("td_soft_subframe_len", 128))
        td_soft_subframe_hop = int(self._dget("td_soft_subframe_hop", 128))
        td_block_energy_len = int(self._dget("td_block_energy_len", 8))
        td_block_energy_hop_raw = self._dget("td_block_energy_hop", None)
        td_block_energy_hop = None if td_block_energy_hop_raw is None else int(td_block_energy_hop_raw)
        td_block_energy_post_pre_blocks = int(self._dget("td_block_energy_post_pre_blocks", 4))
        td_block_energy_smooth_enable = bool(self._dget("td_block_energy_smooth_enable", True))
        td_input_mode = str(self._dget("td_input_mode", "default")).lower()
        td_input_band = self._dget("td_input_band", None)
        if td_input_band is not None:
            td_input_band = (float(td_input_band[0]), float(td_input_band[1]))
        td_soft_crest_factor_min = float(self._dget("td_soft_crest_factor_min", 4.0))
        td_soft_kurtosis_min = float(self._dget("td_soft_kurtosis_min", 6.0))
        td_soft_min_positive_votes = int(self._dget("td_soft_min_positive_votes", 2))
        td_envelope_features_enable = bool(self._dget("td_envelope_features_enable", False))

        # noise_hi remains part of NOISE frame assignment.
        noise_hi = float(self._dget("noise_hi", 0.80))

        # mode_flux_score is still used only for weak/noise assignment below.
        mode_flux_noise_max = float(self._dget("mode_flux_noise_max", 1.5))
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
                    f"mode_weights length ({len(mode_weights)}) must match mode_bands length ({len(mode_bands)})"
                )

        # Precompute masks once; these do not change across frames.
        F, T = P.shape
        td_soft_debug: Dict[str, Any] = {}
        raw_spectral_debug: Dict[str, Any] = {}
        if input_audio is not None:
            try:
                x_in = np.asarray(input_audio, dtype=dtype).reshape(-1)
                fs_local = int(self._dget("sample_rate", self._dget("fs", 11162)))
                n_fft_local = int(self._dget("n_fft", 256))
                hop_local = int(self._dget("hop", 128))

                x_td_in = x_in
                if td_apply_input_prefilter and td_prefilter_mode not in {"", "none"}:
                    build_prefilter = getattr(self, "_build_prefilter_sos", None)
                    if callable(build_prefilter):
                        sos = build_prefilter(fs_local, td_prefilter_mode)
                        if sos is not None:
                            try:
                                x_td_in = spsig.sosfiltfilt(sos, x_in).astype(dtype, copy=False)
                            except ValueError:
                                x_td_in = spsig.sosfilt(sos, x_in).astype(dtype, copy=False)

                td_soft_debug = extract_td_features_inline(
                    x=x_td_in,
                    fs=fs_local,
                    frame_len=n_fft_local,
                    hop=hop_local,
                    operating_band=(op_lo, op_hi),
                    mode_bands=tuple((float(a), float(b)) for (a, b) in mode_bands),
                    td_input_mode=td_input_mode,
                    td_input_band=td_input_band,
                    bp_order=td_soft_bp_order,
                    subframe_len=td_soft_subframe_len,
                    subframe_hop=td_soft_subframe_hop,
                    envelope_features_enable=td_envelope_features_enable,
                    block_energy_len=td_block_energy_len,
                    block_energy_hop=td_block_energy_hop,
                    block_energy_post_pre_blocks=td_block_energy_post_pre_blocks,
                    block_energy_smooth_enable=td_block_energy_smooth_enable,
                    process_dtype=str(self._dget("process_dtype", "float32")),
                    eps=eps,
                )

                if raw_spectral_shape_enable:
                    raw_spectral_debug = extract_raw_spectral_shape_features_inline(
                        x=x_in,
                        fs=fs_local,
                        n_fft=n_fft_local,
                        hop=hop_local,
                        operating_band=(op_lo, op_hi),
                        rain_band=(
                            float(raw_spectral_rain_band[0]),
                            float(raw_spectral_rain_band[1]),
                        ),
                        low_band=(
                            float(raw_spectral_low_band[0]),
                            float(raw_spectral_low_band[1]),
                        ),
                        mode_bands=mode_bands,
                        rolloff_fraction=raw_spectral_rolloff_fraction,
                        process_dtype=str(self._dget("process_dtype", "float32")),
                        eps=eps,
                        raw_power=raw_power,
                        freqs=freqs,
                    )
            except Exception as e:
                td_soft_debug = {"error": str(e)}
                raw_spectral_debug = {"error": str(e)}

        detector_frame_times = self._resolve_detector_frame_times(
            detector_frame_times,
            n_frames=T,
            dtype=dtype,
        )

        td_soft_label = np.zeros(T, dtype=bool)
        td_crest_factor = np.zeros(T, dtype=dtype)
        td_kurtosis = np.zeros(T, dtype=dtype)
        td_rise_time_sec = np.zeros(T, dtype=dtype)
        td_fall_time_sec = np.zeros(T, dtype=dtype)
        td_rise_slope = np.zeros(T, dtype=dtype)
        td_fall_slope = np.zeros(T, dtype=dtype)
        td_energy_envelope = np.zeros(T, dtype=dtype)
        td_peak_energy = np.zeros(T, dtype=dtype)
        td_block_energy_crest = np.zeros(T, dtype=dtype)
        td_block_peak_width_50 = np.zeros(T, dtype=dtype)
        td_block_post_pre_energy_ratio = np.zeros(T, dtype=dtype)
        td_vote_count = np.zeros(T, dtype=np.int32)
        td_soft_score = np.zeros(T, dtype=dtype)

        aligned_raw_spectral = {name: np.zeros(T, dtype=dtype) for name in RAW_SPECTRAL_FEATURE_NAMES}

        # Registry-driven TD alignment block
        aligned_td = {name: np.zeros(T, dtype=dtype) for name in TD_FEATURE_NAMES}

        if td_soft_debug and ("error" not in td_soft_debug):
            aligned_td = self._align_feature_dict_to_frames(
                td_soft_debug,
                TD_FEATURE_NAMES,
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )

            # Sanity check: registry names must match extraction output keys.
            # Envelope features are optional and only expected when enabled.
            expected_td_features = {
                "td_crest_factor",
                "td_kurtosis",
                "td_block_energy_crest",
                "td_block_peak_width_50",
                "td_block_post_pre_energy_ratio",
            }

            if td_envelope_features_enable:
                expected_td_features.update(
                    {
                        "td_rise_time_sec",
                        "td_fall_time_sec",
                        "td_rise_slope",
                        "td_fall_slope",
                        "td_energy_envelope",
                        "td_peak_energy",
                    }
                )

            missing_td_features = sorted(name for name in expected_td_features if name not in td_soft_debug)

            if missing_td_features:
                det_missing = ", ".join(missing_td_features)
                raise KeyError(f"TD feature extraction mismatch. Missing TD features: {det_missing}")

            td_crest_factor = aligned_td["td_crest_factor"]
            td_kurtosis = aligned_td["td_kurtosis"]
            td_block_energy_crest = aligned_td["td_block_energy_crest"]
            td_block_peak_width_50 = aligned_td["td_block_peak_width_50"]
            td_block_post_pre_energy_ratio = aligned_td["td_block_post_pre_energy_ratio"]

            if td_envelope_features_enable:
                td_rise_time_sec = aligned_td["td_rise_time_sec"]
                td_fall_time_sec = aligned_td["td_fall_time_sec"]
                td_rise_slope = aligned_td["td_rise_slope"]
                td_fall_slope = aligned_td["td_fall_slope"]
                td_energy_envelope = aligned_td["td_energy_envelope"]
                td_peak_energy = aligned_td["td_peak_energy"]

            if td_soft_enable:
                td_label_out = assign_td_soft_label(
                    td_crest_factor=td_crest_factor,
                    td_kurtosis=td_kurtosis,
                    crest_thr=td_soft_crest_factor_min,
                    kurt_thr=td_soft_kurtosis_min,
                    min_positive_votes=td_soft_min_positive_votes,
                )

                td_vote_count = td_label_out["td_vote_count"]
                td_soft_score = td_label_out["td_soft_score"]
                td_soft_label = td_label_out["td_soft_label"]

        if raw_spectral_debug and ("error" not in raw_spectral_debug):
            aligned_raw_spectral = self._align_feature_dict_to_frames(
                raw_spectral_debug,
                RAW_SPECTRAL_FEATURE_NAMES,
                n_frames=T,
                dtype=dtype,
                fill_value=0,
            )

        band_mask = (freqs >= op_lo) & (freqs <= op_hi)

        if P.shape[0] != freqs.shape[0]:
            raise ValueError(f"P.shape[0] ({P.shape[0]}) must match freqs.shape[0] ({freqs.shape[0]})")
        if not np.any(band_mask):
            raise ValueError(f"operating_band {op_band} does not overlap the provided frequency grid")

        P_band = P[band_mask, :]
        freqs_band = freqs[band_mask]

        primary_lo, primary_hi = mode_bands[0]
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

        if peak_features_enable:
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
                flux_primary[t] = 0.0
                flux_modes[t] = 0.0
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

            if peak_features_enable:
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
                    widths_bins, *_ = spsig.peak_widths(
                        spec_db,
                        peaks,
                        rel_height=0.5,
                    )
                    df_hz = float(freqs_band[1] - freqs_band[0]) if freqs_band.size > 1 else 0.0
                    pk_bw_hz = np.asarray(widths_bins, dtype=dtype) * df_hz

                    valid_prom_mask = (pk_prom >= peak_valid_prom_min_db) & (pk_prom <= peak_valid_prom_max_db)
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

        peak_gate = peak_gate_score >= 1.0

        # Fixed-band FD rain decision.
        # Primary rain band is intentionally fixed to mode 0.
        # Support bands are intentionally fixed to modes 1, 2, and 3.
        mode_flux_score = np.nan_to_num(mode_flux_score, nan=0.0, posinf=0.0, neginf=0.0)
        td_crest_factor = np.nan_to_num(td_crest_factor, nan=0.0, posinf=0.0, neginf=0.0)
        td_kurtosis = np.nan_to_num(td_kurtosis, nan=0.0, posinf=0.0, neginf=0.0)
        td_block_energy_crest = np.nan_to_num(td_block_energy_crest, nan=0.0, posinf=0.0, neginf=0.0)
        td_block_peak_width_50 = np.nan_to_num(td_block_peak_width_50, nan=0.0, posinf=0.0, neginf=0.0)
        td_block_post_pre_energy_ratio = np.nan_to_num(
            td_block_post_pre_energy_ratio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        td_soft_score = np.nan_to_num(td_soft_score, nan=0.0, posinf=0.0, neginf=0.0)
        # FD rain decision thresholds are applied in log1p space.
        # min_support_count refers to support bands {1,2,3}.
        primary_flux_min = float(self._dget("new_rain_primary_flux_min", 1.8))
        legacy_mode12_flux_min = float(self._dget("new_rain_mode12_flux_min", 2.6))
        mode1_flux_min = float(self._dget("new_rain_mode1_flux_min", legacy_mode12_flux_min))
        mode2_flux_min = float(self._dget("new_rain_mode2_flux_min", legacy_mode12_flux_min))
        mode3_flux_min = float(self._dget("new_rain_mode3_flux_min", 3.0))
        min_support_count = int(self._dget("new_rain_min_support_count", 2))

        primary_mode_flux = np.nan_to_num(normalized_mode_flux_by_mode[0], nan=0.0, posinf=0.0, neginf=0.0)
        support_mode_flux_1 = np.nan_to_num(normalized_mode_flux_by_mode[1], nan=0.0, posinf=0.0, neginf=0.0)
        support_mode_flux_2 = np.nan_to_num(normalized_mode_flux_by_mode[2], nan=0.0, posinf=0.0, neginf=0.0)
        support_mode_flux_3 = np.nan_to_num(normalized_mode_flux_by_mode[3], nan=0.0, posinf=0.0, neginf=0.0)
        if normalized_mode_flux_by_mode.shape[0] > 4:
            support_mode_flux_4 = np.nan_to_num(
                normalized_mode_flux_by_mode[4],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            support_mode_flux_4 = np.zeros_like(primary_mode_flux)

        # TD gate: require minimum crest factor and optionally reject overly spiky
        # frames using an upper threshold on kurtosis.
        td_gate_threshold = float(self._dget("td_gate_threshold", 2.5))
        td_kurtosis_upper_threshold = self._dget("td_kurtosis_upper_threshold", None)
        td_gate_value = td_crest_factor
        td_gate_mask = td_gate_value > td_gate_threshold
        if td_kurtosis_upper_threshold is not None:
            td_kurtosis_upper_threshold = float(td_kurtosis_upper_threshold)
            td_gate_mask = td_gate_mask & (td_kurtosis <= td_kurtosis_upper_threshold)

        if feature_dump_sparse_gate_feature == "td_block_energy_crest":
            sparse_gate_source = td_block_energy_crest
        elif feature_dump_sparse_gate_feature == "td_crest_factor":
            sparse_gate_source = td_crest_factor
        else:
            sparse_gate_source = td_block_energy_crest

        sparse_gate_source_safe = np.nan_to_num(
            sparse_gate_source,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if feature_dump_sparse_enable:
            raw_spectral_dump_mask = sparse_gate_source_safe > feature_dump_sparse_gate_threshold
        else:
            # Dense-only mode keeps all frames.
            raw_spectral_dump_mask = np.ones(T, dtype=bool)

        sparse_frame_idx = np.flatnonzero(raw_spectral_dump_mask).astype(np.int32)

        gate_scale = td_gate_mask.astype(dtype)

        primary_mode_flux_gated = primary_mode_flux * gate_scale
        support_mode_flux_1_gated = support_mode_flux_1 * gate_scale
        support_mode_flux_2_gated = support_mode_flux_2 * gate_scale
        support_mode_flux_3_gated = support_mode_flux_3 * gate_scale

        is_rain, rain_conf = self._rain_frame_decision(
            primary_mode_flux=primary_mode_flux_gated,
            support_mode_flux_1=support_mode_flux_1_gated,
            support_mode_flux_2=support_mode_flux_2_gated,
            support_mode_flux_3=support_mode_flux_3_gated,
            primary_flux_min=primary_flux_min,
            mode1_flux_min=mode1_flux_min,
            mode2_flux_min=mode2_flux_min,
            mode3_flux_min=mode3_flux_min,
            min_support_count=min_support_count,
            eps=eps,
        )

        # Noise confidence is the complement of rain confidence, with weak mode flux
        # still used to assign explicit NOISE labels.
        noise_conf = np.clip(1.0 - rain_conf, 0.0, 1.0)
        mode_flux_score_gated = mode_flux_score * gate_scale
        weak_mode_flux = mode_flux_score_gated <= mode_flux_noise_max

        # FrameClass is the canonical detector output used by downstream logic
        frame_class = np.full(T, FrameClass.UNCERTAIN, dtype=np.int8)
        frame_class[(noise_conf >= noise_hi) & weak_mode_flux & (~is_rain)] = FrameClass.NOISE
        frame_class[is_rain] = FrameClass.RAIN

        det_debug = {
            "mode_flux_score": mode_flux_score,
            "mode_flux_score_gated": mode_flux_score_gated,
            "primary_mode_flux": primary_mode_flux,
            "support_mode_flux_1": support_mode_flux_1,
            "support_mode_flux_2": support_mode_flux_2,
            "support_mode_flux_3": support_mode_flux_3,
            "support_mode_flux_4": support_mode_flux_4,
            "primary_mode_flux_gated": primary_mode_flux_gated,
            "support_mode_flux_1_gated": support_mode_flux_1_gated,
            "support_mode_flux_2_gated": support_mode_flux_2_gated,
            "support_mode_flux_3_gated": support_mode_flux_3_gated,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "frame_class": frame_class,
            "td_soft_label": td_soft_label,
            "td_crest_factor": td_crest_factor,
            "td_kurtosis": td_kurtosis,
            "td_block_energy_crest": td_block_energy_crest,
            "td_block_peak_width_50": td_block_peak_width_50,
            "td_block_post_pre_energy_ratio": td_block_post_pre_energy_ratio,
            "td_block_energy_len": td_block_energy_len,
            "td_block_energy_hop": td_block_energy_hop,
            "td_block_energy_post_pre_blocks": td_block_energy_post_pre_blocks,
            "td_block_energy_smooth_enable": td_block_energy_smooth_enable,
            "td_gate_threshold": td_gate_threshold,
            "td_kurtosis_upper_threshold": td_kurtosis_upper_threshold,
            "td_gate_mask": td_gate_mask,
            "raw_spectral_dump_mask": raw_spectral_dump_mask,
            "raw_spectral_dump_mask_fraction": (
                float(np.mean(raw_spectral_dump_mask.astype(np.float32))) if T > 0 else 0.0
            ),
            "td_vote_count": td_vote_count,
            "td_soft_score": td_soft_score,
            "sparse_frame_idx": sparse_frame_idx,
            "feature_dump_dense_enable": feature_dump_dense_enable,
            "feature_dump_sparse_enable": feature_dump_sparse_enable,
            "feature_dump_clip_summary_enable": feature_dump_clip_summary_enable,
            "feature_dump_sparse_gate_feature": feature_dump_sparse_gate_feature,
            "feature_dump_sparse_gate_threshold": feature_dump_sparse_gate_threshold,
            "raw_spectral_shape_enable": raw_spectral_shape_enable,
            "raw_spectral_uses_raw_power": raw_power is not None,
            "td_apply_input_prefilter": td_apply_input_prefilter,
            "td_prefilter_mode": td_prefilter_mode,
            "clip_spectral_occupancy_enable": clip_spectral_occupancy_enable,
        }

        # Registry-driven raw spectral debug wiring.
        det_debug.update(aligned_raw_spectral)

        if td_envelope_features_enable:
            det_debug.update(
                {
                    "td_rise_time_sec": td_rise_time_sec,
                    "td_fall_time_sec": td_fall_time_sec,
                    "td_rise_slope": td_rise_slope,
                    "td_fall_slope": td_fall_slope,
                    "td_energy_envelope": td_energy_envelope,
                    "td_peak_energy": td_peak_energy,
                }
            )

        if peak_features_enable:
            det_debug.update(
                {
                    "peak_ratio": peak_ratio,
                    "peak_gate_score": peak_gate_score,
                    "peak_valid_count": peak_valid_count,
                    "peak_count_by_mode": peak_count_by_mode,
                }
            )

        if include_peak_payload:
            det_debug.update(
                {
                    "peak_valid_freqs_hz": peak_valid_freqs_hz,
                    "peak_valid_prominences_db": peak_valid_prominences_db,
                    "peak_valid_bandwidths_hz": peak_valid_bandwidths_hz,
                }
            )

        clip_spectral_occupancy: Dict[str, Any] = {}
        if clip_spectral_occupancy_enable:
            if raw_power is not None:
                try:
                    clip_spectral_occupancy = compute_clip_spectral_occupancy_stats(
                        raw_power=raw_power,
                        freqs=freqs,
                        frame_class=frame_class,
                        bands=self._dget("clip_spectral_occupancy_bands", None),
                        dtype=clip_spectral_occupancy_dtype,
                        eps=eps,
                    )
                    det_debug["clip_spectral_occupancy"] = clip_spectral_occupancy
                except Exception as e:
                    det_debug["clip_spectral_occupancy_error"] = str(e)
            else:
                det_debug["clip_spectral_occupancy_error"] = (
                    "raw_power is required for clip_spectral_occupancy_enable=True"
                )

        feature_dump_level = int(self._dget("feature_dump_level", 0))
        feature_dump: Dict[str, Any] = {}

        if feature_dump_level > 0:
            fd_dense = {}
            fd_sparse = {}
            fd_clip_summary = {}

            if feature_dump_dense_enable:
                fd_dense.update(
                    {
                        "primary_mode_flux": primary_mode_flux,
                        "support_mode_flux_1": support_mode_flux_1,
                        "support_mode_flux_2": support_mode_flux_2,
                        "support_mode_flux_3": support_mode_flux_3,
                        "support_mode_flux_4": support_mode_flux_4,
                        "td_block_energy_crest": td_block_energy_crest,
                        "td_block_peak_width_50": td_block_peak_width_50,
                        "td_block_post_pre_energy_ratio": td_block_post_pre_energy_ratio,
                        "td_gate_mask": td_gate_mask,
                    }
                )

                if feature_dump_include_frame_class:
                    fd_dense["frame_class"] = frame_class

                if feature_dump_include_td_soft:
                    fd_dense.update(
                        {
                            "td_crest_factor": td_crest_factor,
                            "td_kurtosis": td_kurtosis,
                            "td_vote_count": td_vote_count,
                            "td_soft_score": td_soft_score,
                        }
                    )

                if feature_dump_include_mode_flux_score:
                    fd_dense["mode_flux_score"] = mode_flux_score
                    fd_dense["mode_flux_score_gated"] = mode_flux_score_gated

            if feature_dump_sparse_enable:
                fd_sparse["sparse_frame_idx"] = sparse_frame_idx

                raw_spectral_basic_names = {
                    "raw_spectral_centroid_hz",
                    "raw_rain_band_ratio",
                    "raw_spectral_rolloff_hz",
                }

                if feature_dump_include_raw_spectral_frame_features:
                    for name in RAW_SPECTRAL_FEATURE_NAMES:
                        if name in raw_spectral_basic_names and not feature_dump_include_raw_spectral_basic:
                            continue

                        fd_sparse[f"sparse_{name}"] = aligned_raw_spectral[name][sparse_frame_idx]
                elif feature_dump_include_raw_spectral_basic:
                    for name in raw_spectral_basic_names:
                        fd_sparse[f"sparse_{name}"] = aligned_raw_spectral[name][sparse_frame_idx]

            if feature_dump_clip_summary_enable and clip_spectral_occupancy:
                fd_clip_summary["clip_spectral_occupancy"] = clip_spectral_occupancy

            # Keep backward-compatible flat feature_dump structure.
            # The downstream flattening loader supports both flat and 3-tier formats.
            feature_dump.update(fd_dense)
            feature_dump.update(fd_sparse)
            feature_dump.update(fd_clip_summary)

        det_debug["peak_features_enable"] = peak_features_enable

        if raw_spectral_debug and "error" in raw_spectral_debug:
            det_debug["raw_spectral_shape_error"] = raw_spectral_debug["error"]

        return frame_class, rain_conf, det_debug, feature_dump


# ---------------------------------------------------------------------------
# Stateful frame-level processing
# ---------------------------------------------------------------------------


class CausalLowQuantileTracker:
    """
    Stateful per-frame equivalent of causal_stochastic_low_quantile_baseline.

    Call push(x) once per frame to obtain the causal baseline *before* ingesting
    x — identical semantics to the batch function.
    """

    def __init__(
        self,
        *,
        q_percent: float,
        frames_per_sec: float,
        win_sec: float,
        min_hist_sec: float = 0.0,
        floor: float = 1e-6,
        dtype=np.float32,
    ):
        self._q = float(np.clip(q_percent, 0.0, 100.0)) / 100.0
        self._floor = float(max(floor, 1e-12))
        self._dtype = dtype
        W = max(3, int(round(float(win_sec) * float(frames_per_sec))))
        self._eta = float(np.clip(2.0 / max(W + 1, 2), 1e-4, 1.0))
        self._min_hist = max(1, int(round(float(min_hist_sec) * float(frames_per_sec))))
        self._scale_alpha = float(np.clip(1.0 - self._eta, 0.0, 0.9999))
        self._baseline: Optional[float] = None
        self._scale: float = 1.0
        self._hist_count: int = 0

    def reset(self) -> None:
        self._baseline = None
        self._scale = 1.0
        self._hist_count = 0

    def push(self, x: float) -> float:
        """Update with x; return the causal baseline before ingesting x."""
        x = float(x)
        floor = self._floor
        if self._baseline is None:
            self._baseline = max(x, floor)
            self._scale = max(abs(x), floor)
        out = max(self._baseline, floor)
        err = x - self._baseline
        self._scale = self._scale_alpha * self._scale + (1.0 - self._scale_alpha) * abs(err)
        step = self._eta * max(self._scale, floor)
        delta = self._q * step if x >= self._baseline else -(1.0 - self._q) * step
        self._baseline = max(self._baseline + delta, floor)
        self._hist_count += 1
        return out


class RainFrameClassifierState:
    """
    Stateful, frame-by-frame rain classifier.

    This class provides the foundation for causal / embedded rain detection
    while preserving numerical parity with the existing clip-level detector.

    There are currently two execution modes:

    1) replay_clip()
       Offline parity-validation mode.
       Replays a full clip frame-by-frame while precomputing TD and
       raw-spectral features from the entire waveform. This path is intended
       for regression testing against
       RainFrameClassifierMixin._detect_rain_over_time and should achieve
       near-identical results.

    2) process_frame()
       True streaming / causal mode.
       Processes one frame at a time using only past and present state.
       This path is intended for future CM7 embedded deployment. Exact
       parity with replay_clip() is not guaranteed because streaming mode
       cannot use future context, filtfilt(), or clip-global winsorization.

    Maintains causal state between frames:
        - spectrogram history for t vs (t-2) flux computation
        - per-mode low-quantile baseline trackers
        - rolling audio buffer for TD / raw-spectral extraction

    Typical usage::

        state = RainFrameClassifierState(freqs=freqs, mode_bands=cfg.mode_bands)
        for frame_P, frame_audio in stream:
            result = state.process_frame(frame_P, frame_audio=frame_audio)

    For regression testing against the clip-level mixin use replay_clip() or
    build via the from_mixin() factory.
    """

    def __init__(
        self,
        *,
        freqs: np.ndarray,
        mode_bands,
        operating_band=(400.0, 3500.0),
        fs: int = 11162,
        n_fft: int = 256,
        hop: int = 128,
        process_dtype: str = "float32",
        eps: float = 1e-9,
        # Mode flux normalization
        mode_flux_norm_enable: bool = True,
        mode_flux_norm_win_sec: float = 0.5,
        mode_flux_norm_q: float = 20.0,
        mode_flux_norm_min: float = 1.0,
        mode_weights=None,
        # TD gate
        td_gate_threshold: float = 2.5,
        td_kurtosis_upper_threshold=None,
        # FD decision thresholds
        new_rain_primary_flux_min: float = 1.8,
        new_rain_mode12_flux_min: float = 2.6,
        new_rain_mode1_flux_min: Optional[float] = None,
        new_rain_mode2_flux_min: Optional[float] = None,
        new_rain_mode3_flux_min: float = 3.0,
        new_rain_min_support_count: int = 2,
        # Noise label
        noise_hi: float = 0.80,
        mode_flux_noise_max: float = 1.5,
        # TD extraction params
        td_input_mode: str = "default",
        td_input_band=None,
        td_soft_bp_order: int = 4,
        td_soft_subframe_len: int = 128,
        td_soft_subframe_hop: int = 128,
        td_block_energy_len: int = 8,
        td_block_energy_hop=None,
        td_block_energy_post_pre_blocks: int = 4,
        td_block_energy_smooth_enable: bool = True,
        td_apply_input_prefilter: bool = True,
        td_prefilter_mode: str = "none",
        td_envelope_features_enable: bool = False,
        td_soft_enable: bool = False,
        td_soft_crest_factor_min: float = 4.0,
        td_soft_kurtosis_min: float = 6.0,
        td_soft_min_positive_votes: int = 2,
        # Raw spectral
        raw_spectral_shape_enable: bool = True,
        raw_spectral_rain_band=(400.0, 800.0),
        raw_spectral_low_band=(50.0, 200.0),
        raw_spectral_rolloff_fraction: float = 0.85,
        # Highpass prefilter params (used by replay_clip to match clip-level path)
        hp_cutoff_hz: float = 350.0,
        hp_order: int = 4,
        # Winsorization of combined flux (mirrors flux_modes_winsor_* in the batch path)
        flux_modes_winsor_enable: bool = False,
        flux_modes_winsor_q: float = 99.0,
    ):
        dtype = resolve_np_dtype(process_dtype)
        self._dtype = dtype
        self._eps = float(eps)
        self._fs = int(fs)
        self._n_fft = int(n_fft)
        self._hop = int(hop)
        self._process_dtype_str = str(process_dtype)

        # Frequency grid and band masks (computed once)
        freqs_arr = np.asarray(freqs, dtype=dtype)
        self._freqs = freqs_arr
        op_lo, op_hi = float(operating_band[0]), float(operating_band[1])
        self._op_band = (op_lo, op_hi)
        band_mask = (freqs_arr >= op_lo) & (freqs_arr <= op_hi)
        if not np.any(band_mask):
            raise ValueError(f"operating_band {operating_band} does not overlap the provided frequency grid")
        self._band_mask = band_mask
        self._freqs_band = freqs_arr[band_mask]

        mode_bands = tuple((float(a), float(b)) for a, b in mode_bands)
        if len(mode_bands) < 4:
            raise ValueError(
                "Frame-level rain decision requires at least 4 mode bands (mode 0 as primary, modes 1-3 as support)"
            )
        self._mode_bands = mode_bands
        self._n_modes = len(mode_bands)

        primary_lo, primary_hi = mode_bands[0]
        self._primary_mask = (self._freqs_band >= primary_lo) & (self._freqs_band <= primary_hi)
        if not np.any(self._primary_mask):
            raise ValueError(f"Primary mode band {(primary_lo, primary_hi)} has no bins inside operating_band")
        self._mode_masks = [(self._freqs_band >= lo) & (self._freqs_band <= hi) for lo, hi in mode_bands]

        if mode_weights is not None:
            mode_weights = tuple(float(w) for w in mode_weights)
            if len(mode_weights) != len(mode_bands):
                raise ValueError("mode_weights length must match mode_bands length")
        self._mode_weights = mode_weights

        # Decision thresholds
        legacy_mode12 = float(new_rain_mode12_flux_min)
        self._primary_flux_min = float(new_rain_primary_flux_min)
        self._mode1_flux_min = float(new_rain_mode1_flux_min) if new_rain_mode1_flux_min is not None else legacy_mode12
        self._mode2_flux_min = float(new_rain_mode2_flux_min) if new_rain_mode2_flux_min is not None else legacy_mode12
        self._mode3_flux_min = float(new_rain_mode3_flux_min)
        self._min_support_count = int(max(1, new_rain_min_support_count))
        self._noise_hi = float(noise_hi)
        self._mode_flux_noise_max = float(max(mode_flux_noise_max, 0.0))
        self._td_gate_threshold = float(td_gate_threshold)
        self._td_kurtosis_upper_threshold = (
            float(td_kurtosis_upper_threshold) if td_kurtosis_upper_threshold is not None else None
        )

        # Mode flux normalization
        self._mode_flux_norm_enable = bool(mode_flux_norm_enable)
        self._mode_flux_norm_min = float(max(mode_flux_norm_min, eps))

        # TD feature extraction params
        self._td_input_mode = str(td_input_mode).lower()
        self._td_input_band = (float(td_input_band[0]), float(td_input_band[1])) if td_input_band is not None else None
        self._td_soft_bp_order = int(td_soft_bp_order)
        self._td_soft_subframe_len = int(td_soft_subframe_len)
        self._td_soft_subframe_hop = int(td_soft_subframe_hop)
        self._td_block_energy_len = int(td_block_energy_len)
        self._td_block_energy_hop = None if td_block_energy_hop is None else int(td_block_energy_hop)
        self._td_block_energy_post_pre_blocks = int(td_block_energy_post_pre_blocks)
        self._td_block_energy_smooth_enable = bool(td_block_energy_smooth_enable)
        self._td_apply_input_prefilter = bool(td_apply_input_prefilter)
        self._td_prefilter_mode = str(td_prefilter_mode).lower()
        self._td_envelope_features_enable = bool(td_envelope_features_enable)
        self._td_soft_enable = bool(td_soft_enable)
        self._td_soft_crest_factor_min = float(td_soft_crest_factor_min)
        self._td_soft_kurtosis_min = float(td_soft_kurtosis_min)
        self._td_soft_min_positive_votes = int(td_soft_min_positive_votes)

        # Raw spectral params
        self._raw_spectral_shape_enable = bool(raw_spectral_shape_enable)
        self._raw_spectral_rain_band = (
            float(raw_spectral_rain_band[0]),
            float(raw_spectral_rain_band[1]),
        )
        self._raw_spectral_low_band = (
            float(raw_spectral_low_band[0]),
            float(raw_spectral_low_band[1]),
        )
        self._raw_spectral_rolloff_fraction = float(raw_spectral_rolloff_fraction)

        # Prefilter params for replay_clip
        self._hp_cutoff_hz = float(hp_cutoff_hz)
        self._hp_order = int(hp_order)

        # Winsorization params (used by replay_clip only; not applicable to true streaming)
        self._flux_modes_winsor_enable = bool(flux_modes_winsor_enable)
        self._flux_modes_winsor_q = float(np.clip(flux_modes_winsor_q, 50.0, 100.0))
        self._total_flux_cap: Optional[float] = None

        # Causal baseline trackers: one per mode + one for combined flux
        frames_per_sec = float(fs) / float(max(hop, 1))
        tracker_kwargs: Dict[str, Any] = dict(
            q_percent=float(np.clip(mode_flux_norm_q, 0.0, 100.0)),
            frames_per_sec=frames_per_sec,
            win_sec=float(mode_flux_norm_win_sec),
            min_hist_sec=0.0,
            floor=float(mode_flux_norm_min),
            dtype=dtype,
        )
        self._mode_trackers = [CausalLowQuantileTracker(**tracker_kwargs) for _ in mode_bands]
        self._combined_tracker = CausalLowQuantileTracker(**tracker_kwargs)

        # Rolling spectrogram frame history for flux delta (t vs t-2)
        self._prev_frame_1: Optional[np.ndarray] = None
        self._prev_frame_2: Optional[np.ndarray] = None

        # Audio ring buffer: n_fft + hop samples so extract_td_features_inline
        # returns 2 frames and we take the last one.
        self._audio_buf: np.ndarray = np.zeros(n_fft + hop, dtype=dtype)

        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all streaming state. Call before processing a new clip."""
        self._prev_frame_1 = None
        self._prev_frame_2 = None
        self._audio_buf[:] = 0.0
        self._frame_idx = 0
        self._total_flux_cap = None
        self._combined_tracker.reset()
        for tracker in self._mode_trackers:
            tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sos(self, mode: str):
        """Build SOS prefilter for the given mode ("highpass" or "bandpass")."""
        nyq = 0.5 * self._fs
        if mode == "highpass" and self._hp_cutoff_hz > 0:
            norm_cut = float(np.clip(self._hp_cutoff_hz / nyq, 1e-4, 0.9999))
            return spsig.butter(self._hp_order, norm_cut, btype="highpass", output="sos")
        if mode == "bandpass":
            op_lo, op_hi = self._op_band
            lo = float(np.clip(op_lo, 1e-3, nyq * 0.999))
            hi = float(np.clip(op_hi, lo + 1e-3, nyq * 0.999))
            return spsig.butter(self._hp_order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        return None

    def _decide_frame(
        self,
        frame_spectrum: np.ndarray,
        td_features: Dict[str, float],
        raw_spectral_features: Dict[str, float],
        frame_time: Optional[float],
    ) -> Dict[str, Any]:
        """
        Core flux + gate + decision logic for one frame.

        Called by both process_frame (streaming buffer path) and replay_clip
        (pre-computed full-audio path).
        """
        dtype = self._dtype

        frame = np.asarray(frame_spectrum, dtype=dtype).reshape(-1)[self._band_mask]

        # ---- Flux computation (t vs t-2, causal) -------------------------
        if self._prev_frame_1 is None:
            flux = np.zeros(frame.shape[0], dtype=dtype)
            self._prev_frame_1 = frame.copy()
        elif self._prev_frame_2 is None:
            flux = np.zeros(frame.shape[0], dtype=dtype)
            self._prev_frame_2 = self._prev_frame_1
            self._prev_frame_1 = frame.copy()
        else:
            flux = np.maximum(frame - self._prev_frame_2, 0.0)
            self._prev_frame_2 = self._prev_frame_1
            self._prev_frame_1 = frame.copy()

        # Per-mode raw flux
        mode_flux_raw = np.zeros(self._n_modes, dtype=dtype)
        total_flux = 0.0
        for i, m_mask in enumerate(self._mode_masks):
            mf = float(np.sum(flux[m_mask]))
            mode_flux_raw[i] = mf
            w = self._mode_weights[i] if self._mode_weights is not None else 1.0
            total_flux += w * mf

        # Apply per-clip winsor cap set by replay_clip (not used in true streaming)
        if self._total_flux_cap is not None:
            total_flux = min(total_flux, self._total_flux_cap)

        # Combined flux normalization
        combined_baseline = self._combined_tracker.push(total_flux)
        combined_excess = max(total_flux - combined_baseline, 0.0)
        if self._mode_flux_norm_enable:
            mode_flux_score = combined_excess / (combined_baseline + self._mode_flux_norm_min)
        else:
            mode_flux_score = combined_excess
        mode_flux_score = float(np.nan_to_num(mode_flux_score))

        # Per-mode normalized flux
        normalized_mode_flux = np.zeros(self._n_modes, dtype=dtype)
        for i in range(self._n_modes):
            baseline_i = self._mode_trackers[i].push(float(mode_flux_raw[i]))
            excess_i = max(float(mode_flux_raw[i]) - baseline_i, 0.0)
            if self._mode_flux_norm_enable:
                score_i = excess_i / (baseline_i + self._mode_flux_norm_min)
            else:
                score_i = excess_i
            normalized_mode_flux[i] = float(np.nan_to_num(score_i))

        primary_mode_flux = float(normalized_mode_flux[0])
        support_mode_flux_1 = float(normalized_mode_flux[1])
        support_mode_flux_2 = float(normalized_mode_flux[2])
        support_mode_flux_3 = float(normalized_mode_flux[3])
        support_mode_flux_4 = float(normalized_mode_flux[4]) if self._n_modes > 4 else 0.0

        # ---- TD gate ----------------------------------------------------
        td_crest_factor = float(np.nan_to_num(td_features.get("td_crest_factor", 0.0)))
        td_kurtosis = float(np.nan_to_num(td_features.get("td_kurtosis", 0.0)))
        td_block_energy_crest = float(np.nan_to_num(td_features.get("td_block_energy_crest", 0.0)))
        td_block_peak_width_50 = float(np.nan_to_num(td_features.get("td_block_peak_width_50", 0.0)))
        td_block_post_pre_energy_ratio = float(np.nan_to_num(td_features.get("td_block_post_pre_energy_ratio", 0.0)))

        td_gate_mask = bool(td_crest_factor > self._td_gate_threshold)
        if self._td_kurtosis_upper_threshold is not None:
            td_gate_mask = td_gate_mask and (td_kurtosis <= self._td_kurtosis_upper_threshold)

        gate_scale = float(td_gate_mask)

        pmf_gated = primary_mode_flux * gate_scale
        smf1_gated = support_mode_flux_1 * gate_scale
        smf2_gated = support_mode_flux_2 * gate_scale
        smf3_gated = support_mode_flux_3 * gate_scale

        # ---- FD rain decision -------------------------------------------
        f0 = float(np.log1p(max(pmf_gated, 0.0)))
        f1 = float(np.log1p(max(smf1_gated, 0.0)))
        f2 = float(np.log1p(max(smf2_gated, 0.0)))
        f3 = float(np.log1p(max(smf3_gated, 0.0)))

        primary_ok = f0 >= self._primary_flux_min
        support_hits = (
            int(f1 >= self._mode1_flux_min) + int(f2 >= self._mode2_flux_min) + int(f3 >= self._mode3_flux_min)
        )
        is_rain = primary_ok and (support_hits >= self._min_support_count)

        rain_conf = 1.0 if is_rain else 0.0
        noise_conf = 1.0 - rain_conf

        # ---- Frame class assignment -------------------------------------
        mfs_gated = mode_flux_score * gate_scale
        weak_mode_flux = mfs_gated <= self._mode_flux_noise_max

        if is_rain:
            frame_class = int(FrameClass.RAIN)
        elif noise_conf >= self._noise_hi and weak_mode_flux:
            frame_class = int(FrameClass.NOISE)
        else:
            frame_class = int(FrameClass.UNCERTAIN)

        # ---- Build result -----------------------------------------------
        t_idx = self._frame_idx
        t_sec = float(frame_time) if frame_time is not None else float(t_idx * self._hop) / float(self._fs)
        self._frame_idx += 1

        result: Dict[str, Any] = {
            "frame_class": frame_class,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "frame_idx": t_idx,
            "frame_time": t_sec,
            "primary_mode_flux": primary_mode_flux,
            "support_mode_flux_1": support_mode_flux_1,
            "support_mode_flux_2": support_mode_flux_2,
            "support_mode_flux_3": support_mode_flux_3,
            "support_mode_flux_4": support_mode_flux_4,
            "td_crest_factor": td_crest_factor,
            "td_kurtosis": td_kurtosis,
            "td_block_energy_crest": td_block_energy_crest,
            "td_block_peak_width_50": td_block_peak_width_50,
            "td_block_post_pre_energy_ratio": td_block_post_pre_energy_ratio,
            "td_gate_mask": td_gate_mask,
            "mode_flux_score": mode_flux_score,
        }

        if self._td_soft_enable:
            td_vote_count = int(td_crest_factor >= self._td_soft_crest_factor_min) + int(
                td_kurtosis >= self._td_soft_kurtosis_min
            )
            result["td_vote_count"] = td_vote_count
            result["td_soft_score"] = float(td_vote_count) / 2.0

        for name in RAW_SPECTRAL_FEATURE_NAMES:
            result[name] = float(raw_spectral_features.get(name, 0.0))

        return result

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_spectrum: np.ndarray,
        frame_audio: Optional[np.ndarray] = None,
        frame_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classify one spectral frame using the true streaming path.

        This method is intended for causal / embedded execution and is not
        expected to be bitwise-equivalent to replay_clip(), because it uses
        only the current rolling audio buffer and cannot use future context.
        Use replay_clip() for offline golden-regression parity checks.

        Parameters
        ----------
        frame_spectrum : array, shape (F,)
            One column of the novelty/power spectrogram (same F axis as freqs).
        frame_audio : array, shape (~hop,), optional
            Raw audio samples for this frame. Used for TD and raw-spectral
            feature extraction. Omitting disables those features.
        frame_time : float, optional
            Absolute time offset in seconds. Defaults to frame_idx * hop / fs.

        Returns
        -------
        dict
            frame_class (int), rain_conf, noise_conf, per-mode flux scalars,
            TD feature scalars, raw-spectral feature scalars, frame_idx, frame_time.
        """
        dtype = self._dtype
        eps = self._eps

        frame_spectrum = np.asarray(frame_spectrum, dtype=dtype).reshape(-1)
        if frame_spectrum.shape[0] != self._freqs.shape[0]:
            raise ValueError(f"frame_spectrum has {frame_spectrum.shape[0]} bins; expected {self._freqs.shape[0]}")

        # ---- Audio buffer update + streaming TD/RS extraction -----------
        td_features: Dict[str, float] = {}
        raw_spectral_features: Dict[str, float] = {}

        if frame_audio is not None:
            chunk = np.asarray(frame_audio, dtype=dtype).reshape(-1)
            shift = min(chunk.size, self._audio_buf.size)
            self._audio_buf = np.roll(self._audio_buf, -shift)
            self._audio_buf[-shift:] = chunk[-shift:]

            # Skip frame 0: buffer only has the current hop, no prior context.
            if self._frame_idx > 0:
                try:
                    td_raw = extract_td_features_inline(
                        x=self._audio_buf,
                        fs=self._fs,
                        frame_len=self._n_fft,
                        hop=self._hop,
                        operating_band=self._op_band,
                        mode_bands=self._mode_bands,
                        td_input_mode=self._td_input_mode,
                        td_input_band=self._td_input_band,
                        bp_order=self._td_soft_bp_order,
                        subframe_len=self._td_soft_subframe_len,
                        subframe_hop=self._td_soft_subframe_hop,
                        envelope_features_enable=self._td_envelope_features_enable,
                        block_energy_len=self._td_block_energy_len,
                        block_energy_hop=self._td_block_energy_hop,
                        block_energy_post_pre_blocks=self._td_block_energy_post_pre_blocks,
                        block_energy_smooth_enable=self._td_block_energy_smooth_enable,
                        process_dtype=self._process_dtype_str,
                        eps=eps,
                    )
                    # Buffer holds 2 frames; take the last (current frame).
                    td_features = {k: float(np.asarray(v).reshape(-1)[-1]) for k, v in td_raw.items() if np.ndim(v) > 0}
                except Exception:  # noqa: BLE001
                    td_features = {}

                if self._raw_spectral_shape_enable:
                    try:
                        rs_raw = extract_raw_spectral_shape_features_inline(
                            x=self._audio_buf,
                            fs=self._fs,
                            n_fft=self._n_fft,
                            hop=self._hop,
                            operating_band=self._op_band,
                            rain_band=self._raw_spectral_rain_band,
                            low_band=self._raw_spectral_low_band,
                            mode_bands=self._mode_bands,
                            rolloff_fraction=self._raw_spectral_rolloff_fraction,
                            process_dtype=self._process_dtype_str,
                            eps=eps,
                            raw_power=None,
                            freqs=self._freqs,
                        )
                        raw_spectral_features = {
                            k: float(np.asarray(v).reshape(-1)[-1]) for k, v in rs_raw.items() if np.ndim(v) > 0
                        }
                    except Exception:  # noqa: BLE001
                        raw_spectral_features = {}

        return self._decide_frame(frame_spectrum, td_features, raw_spectral_features, frame_time)

    # ------------------------------------------------------------------
    # Convenience: replay a full clip (for regression testing)
    # ------------------------------------------------------------------

    def replay_clip(
        self,
        P: np.ndarray,
        audio: Optional[np.ndarray] = None,
        frame_times: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Replay a full clip through _decide_frame, matching the clip-level batch path.

        Pre-computes all TD and raw-spectral features from the full audio in one
        call (identical to _detect_rain_over_time), applying the same highpass
        prefilter. This eliminates the 1-frame timing offset and missing-prefilter
        differences that arise in the streaming buffer path.

        This is the canonical path for golden-regression parity validation.
        The process_frame() path should be validated separately as the causal
        streaming implementation matures.

        Calls reset() first so each replay starts from a clean state.

        Parameters
        ----------
        P : array, shape (F, T)
        audio : array, shape (N,), optional
        frame_times : array, shape (T,), optional

        Returns
        -------
        dict of 1-D arrays, length T.
        """
        self.reset()
        dtype = self._dtype
        eps = self._eps

        P = np.asarray(P, dtype=dtype)
        if P.ndim != 2:
            raise ValueError(f"P must be a 2-D array with shape (F, T); got shape {P.shape}")
        if P.shape[0] != self._freqs.shape[0]:
            raise ValueError(f"P has {P.shape[0]} frequency bins; expected {self._freqs.shape[0]}")

        T = P.shape[1]
        if frame_times is not None:
            frame_times = np.asarray(frame_times, dtype=dtype).reshape(-1)
            if frame_times.size < T:
                raise ValueError(f"frame_times has {frame_times.size} entries; expected at least {T}")

        # Pre-compute all TD and RS features from full audio
        all_td: Optional[Dict[str, np.ndarray]] = None
        all_rs: Optional[Dict[str, np.ndarray]] = None

        if audio is not None:
            x_in = np.asarray(audio, dtype=dtype).reshape(-1)

            # Apply prefilter to audio for TD features — mirrors _detect_rain_over_time
            x_td = x_in
            if self._td_apply_input_prefilter and self._td_prefilter_mode not in ("", "none"):
                sos = self._build_sos(self._td_prefilter_mode)
                if sos is not None:
                    try:
                        x_td = spsig.sosfiltfilt(sos, x_in).astype(dtype, copy=False)
                    except ValueError:
                        x_td = spsig.sosfilt(sos, x_in).astype(dtype, copy=False)

            try:
                all_td = extract_td_features_inline(
                    x=x_td,
                    fs=self._fs,
                    frame_len=self._n_fft,
                    hop=self._hop,
                    operating_band=self._op_band,
                    mode_bands=self._mode_bands,
                    td_input_mode=self._td_input_mode,
                    td_input_band=self._td_input_band,
                    bp_order=self._td_soft_bp_order,
                    subframe_len=self._td_soft_subframe_len,
                    subframe_hop=self._td_soft_subframe_hop,
                    envelope_features_enable=self._td_envelope_features_enable,
                    block_energy_len=self._td_block_energy_len,
                    block_energy_hop=self._td_block_energy_hop,
                    block_energy_post_pre_blocks=self._td_block_energy_post_pre_blocks,
                    block_energy_smooth_enable=self._td_block_energy_smooth_enable,
                    process_dtype=self._process_dtype_str,
                    eps=eps,
                )
            except Exception:  # noqa: BLE001
                all_td = None

            if self._raw_spectral_shape_enable:
                try:
                    all_rs = extract_raw_spectral_shape_features_inline(
                        x=x_in,
                        fs=self._fs,
                        n_fft=self._n_fft,
                        hop=self._hop,
                        operating_band=self._op_band,
                        rain_band=self._raw_spectral_rain_band,
                        low_band=self._raw_spectral_low_band,
                        mode_bands=self._mode_bands,
                        rolloff_fraction=self._raw_spectral_rolloff_fraction,
                        process_dtype=self._process_dtype_str,
                        eps=eps,
                        raw_power=None,
                        freqs=self._freqs,
                    )
                except Exception:  # noqa: BLE001
                    all_rs = None

        # Pre-compute per-clip winsor cap, matching the batch path's flux_modes_winsor_enable.
        # True streaming cannot do this (future frames unknown), but replay_clip has all frames.
        if self._flux_modes_winsor_enable:
            raw_total_flux = np.empty(T, dtype=float)
            _pf1: Optional[np.ndarray] = None
            _pf2: Optional[np.ndarray] = None
            for t in range(T):
                _frame = np.asarray(P[:, t], dtype=dtype).reshape(-1)[self._band_mask]
                if _pf1 is None or _pf2 is None:
                    _tf = 0.0
                else:
                    _flux = np.maximum(_frame - _pf2, 0.0)
                    _tf = sum(
                        (self._mode_weights[i] if self._mode_weights is not None else 1.0)
                        * float(np.sum(_flux[m_mask]))
                        for i, m_mask in enumerate(self._mode_masks)
                    )
                _pf2 = _pf1
                _pf1 = _frame.copy()
                raw_total_flux[t] = _tf
            _finite = np.isfinite(raw_total_flux)
            if np.any(_finite):
                self._total_flux_cap = float(np.percentile(raw_total_flux[_finite], self._flux_modes_winsor_q))

        frames = []
        for t in range(T):
            # Slice per-frame TD features from pre-computed arrays
            td_features: Dict[str, float] = {}
            if all_td is not None:
                for k, v in all_td.items():
                    arr = np.asarray(v).reshape(-1)
                    if t < len(arr):
                        td_features[k] = float(arr[t])

            raw_spectral_features: Dict[str, float] = {}
            if all_rs is not None:
                for k, v in all_rs.items():
                    arr = np.asarray(v).reshape(-1)
                    if t < len(arr):
                        raw_spectral_features[k] = float(arr[t])

            ft = float(frame_times[t]) if frame_times is not None else None
            frames.append(self._decide_frame(P[:, t], td_features, raw_spectral_features, ft))

        keys = list(frames[0].keys())
        return {k: np.array([f[k] for f in frames]) for k in keys}

    # ------------------------------------------------------------------
    # Factory: build from an existing mixin processor
    # ------------------------------------------------------------------

    @classmethod
    def from_mixin(
        cls,
        mixin: "RainFrameClassifierMixin",
        freqs: np.ndarray,
    ) -> "RainFrameClassifierState":
        """
        Construct RainFrameClassifierState from any processor that uses
        RainFrameClassifierMixin, mirroring its config exactly.
        """
        dget = mixin._dget
        legacy_mode12 = float(dget("new_rain_mode12_flux_min", 2.6))
        return cls(
            freqs=freqs,
            mode_bands=dget("mode_bands"),
            operating_band=dget("operating_band", (400.0, 3500.0)),
            fs=int(dget("sample_rate", dget("fs", 11162))),
            n_fft=int(dget("n_fft", 256)),
            hop=int(dget("hop", 128)),
            process_dtype=str(dget("process_dtype", "float32")),
            eps=float(dget("eps", 1e-9)),
            mode_flux_norm_enable=bool(dget("mode_flux_norm_enable", True)),
            mode_flux_norm_win_sec=float(dget("mode_flux_norm_win_sec", 0.5)),
            mode_flux_norm_q=float(np.clip(dget("mode_flux_norm_q", 20.0), 0.0, 100.0)),
            mode_flux_norm_min=float(dget("mode_flux_norm_min", 1.0)),
            mode_weights=dget("mode_weights", None),
            td_gate_threshold=float(dget("td_gate_threshold", 2.5)),
            td_kurtosis_upper_threshold=dget("td_kurtosis_upper_threshold", None),
            new_rain_primary_flux_min=float(dget("new_rain_primary_flux_min", 1.8)),
            new_rain_mode12_flux_min=legacy_mode12,
            new_rain_mode1_flux_min=float(dget("new_rain_mode1_flux_min", legacy_mode12)),
            new_rain_mode2_flux_min=float(dget("new_rain_mode2_flux_min", legacy_mode12)),
            new_rain_mode3_flux_min=float(dget("new_rain_mode3_flux_min", 3.0)),
            new_rain_min_support_count=int(dget("new_rain_min_support_count", 2)),
            noise_hi=float(dget("noise_hi", 0.80)),
            mode_flux_noise_max=float(dget("mode_flux_noise_max", 1.5)),
            td_input_mode=str(dget("td_input_mode", "default")).lower(),
            td_input_band=dget("td_input_band", None),
            td_soft_bp_order=int(dget("td_soft_bp_order", 4)),
            td_soft_subframe_len=int(dget("td_soft_subframe_len", 128)),
            td_soft_subframe_hop=int(dget("td_soft_subframe_hop", 128)),
            td_block_energy_len=int(dget("td_block_energy_len", 8)),
            td_block_energy_hop=dget("td_block_energy_hop", None),
            td_block_energy_post_pre_blocks=int(dget("td_block_energy_post_pre_blocks", 4)),
            td_block_energy_smooth_enable=bool(dget("td_block_energy_smooth_enable", True)),
            td_apply_input_prefilter=bool(dget("td_apply_input_prefilter", True)),
            td_prefilter_mode=str(dget("td_prefilter_mode", dget("pre_filter_mode", "none"))).lower(),
            td_envelope_features_enable=bool(dget("td_envelope_features_enable", False)),
            td_soft_enable=bool(dget("td_soft_enable", False)),
            td_soft_crest_factor_min=float(dget("td_soft_crest_factor_min", 4.0)),
            td_soft_kurtosis_min=float(dget("td_soft_kurtosis_min", 6.0)),
            td_soft_min_positive_votes=int(dget("td_soft_min_positive_votes", 2)),
            raw_spectral_shape_enable=bool(dget("raw_spectral_shape_enable", True)),
            raw_spectral_rain_band=dget("raw_spectral_rain_band", (400.0, 800.0)),
            raw_spectral_low_band=dget("raw_spectral_low_band", (50.0, 200.0)),
            raw_spectral_rolloff_fraction=float(dget("raw_spectral_rolloff_fraction", 0.85)),
            hp_cutoff_hz=float(dget("hp_cutoff_hz", 350.0)),
            hp_order=int(dget("hp_order", 4)),
            flux_modes_winsor_enable=bool(dget("flux_modes_winsor_enable", False)),
            flux_modes_winsor_q=float(np.clip(dget("flux_modes_winsor_q", 99.0), 50.0, 100.0)),
        )
