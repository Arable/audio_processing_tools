# edge/spectral_noise_processor.py



from dataclasses import dataclass, fields, field

from typing import Any, Dict, Optional, Tuple
import json

import numpy as np
import scipy.signal as spsig
import scipy.ndimage as ndi
import librosa

from audio_processing_tools.processors import BaseProcessor
from audio_processing_tools.edge.rain_frame_classifier import RainFrameClassifierMixin, FrameClass



@dataclass
class NoiseProcessorConfig:
    """
    Configuration for SpectralNoiseProcessor.

    Includes:
      - STFT parameters
      - rain-frame detection bands + thresholds
      - quantile-based noise PSD tracking
      - adaptive-confidence noise suppression
      - high-pass filtering
    """

    # -----------------------------------------------------------
    # Core
    # -----------------------------------------------------------
    fs: int = 11162
    n_fft: int = 256
    hop: int = 128

    # -----------------------------------------------------------
    # High-pass filter
    # -----------------------------------------------------------
    hp_cutoff_hz: float = 350.0
    hp_order: int = 4

    # Pre-filter stage (applied before STFT)
    #   - "highpass": current behavior using hp_cutoff_hz/hp_order
    #   - "bandpass": use cfg.operating_band with bp_order
    #   - "none": no pre-filter
    pre_filter_mode: str = "highpass"  # "highpass" | "bandpass" | "none"
    bp_order: int = 4

    # -----------------------------------------------------------
    # Operating band (noise estimation + gain applied here)
    # -----------------------------------------------------------
    # NOTE: `operating_band` is the canonical band used everywhere.
    # If callers still pass legacy `fmin`/`fmax`, they are translated into `operating_band`
    # inside `build_noise_config()`.
    operating_band: Tuple[float, float] = (400.0, 3500.0)

    # -----------------------------------------------------------
    # Noise tracking (quantile/min-stats)
    # -----------------------------------------------------------
    q: float = 0.25        # target low-quantile level for causal stochastic baseline tracking
    win_sec: float = 0.5   # effective adaptation horizon (seconds) for the stochastic tracker


    median_frames: int = 0  # optional median filter over time; 0 disables

    # Numerical stability
    eps: float = 1e-9

    # Hard safety clamp for PSD: N <= noise_psd_max_ratio * P (per-bin, per-frame).
    # 1.0 means "never exceed instantaneous power"; values like 0.8–0.95 add extra safety.
    noise_psd_max_ratio: float = 1.0

    # If True, use noise PSD from the previous frame when computing gain for the current frame.
    # This avoids current-frame leakage into its own suppression (more causal / stable).
    use_lagged_noise_psd: bool = False

    # -----------------------------------------------------------
    # Adaptive oversubtraction
    # oversub = oversub_base + noise_conf * (oversub_max - oversub_base)
    # -----------------------------------------------------------
    oversub_base: float = 1.0   # mild suppression when rain likely
    oversub_max: float = 3.0    # strong suppression when noise likely. 3.0 as default

    # Gain clamp to avoid instability / musical noise
    gain_floor: float = 0.0
    gain_ceil: float = 1.0

    # select between sqrt_sub and wiener
    gain_mode: str = "sqrt_sub"      # or "wiener"
    gain_smooth_alpha: float = 0.7   # EMA for temporal smoothing
    # Temporary debug switch: if False, disable frame-class/confidence-driven
    # gain adaptation and use uniform attenuation / smoothing.
    adaptive_gain_enable: bool = True
    # Optional frequency-domain smoothing of gain (helps reduce musical noise)
    gain_freq_smooth_enable: bool = True
    gain_freq_kernel: Tuple[float, ...] = (0.2, 0.6, 0.2)

    # New / refined PSD tracking params
    pre_smooth_frames: int = 0      # try 3–5, 0 disables
    ema_up: float = 0.6             # fast when noise increases
    ema_down: float = 0.95          # slow when noise decreases

    # -----------------------------------------------------------
    # Spectral SNR gating (optional)
    # -----------------------------------------------------------
    # If enabled, reduce suppression on frames where the estimated SNR in the
    # mode bands is high (helps preserve raindrop peaks even if the classifier
    # is briefly uncertain).
    snr_gating_enable: bool = False
    # SNR reference where gate ~= 0.5 (dimensionless). 1.0 means P==N.
    snr_gating_snr1: float = 1.0
    # Exponent to sharpen/soften the gate. 1.0 = linear.
    snr_gating_power: float = 1.0
    # If True, compute SNR only over the union of detector mode_bands.
    # If mode_bands are missing/empty, falls back to the full operating band.
    snr_gating_use_mode_bands: bool = True

    # -----------------------------------------------------------
    # Detector input normalization (optional)
    # -----------------------------------------------------------
    # If enabled, build a coarse/bootstrapped noise PSD first, then classify frames
    # using the lagged PSD (t-1) before the final PSD estimation pass.
    detector_use_noise_norm: bool = True
    # "log_sub" => 10*log10(P) - 10*log10(N_lag)
    # "ratio_db" => 10*log10(P / N_lag)
    detector_noise_norm_mode: str = "log_sub"

    # -----------------------------------------------------------
    # Suppressor bypass for offline feature extraction / tuning
    # -----------------------------------------------------------
    # When True, keep the detector / soft-label path active but disable
    # noise-PSD estimation, gain computation, and spectral suppression.
    # This is useful when we want to extract detector features on raw clips
    # and compare them directly against soft frame labels.
    disable_suppression: bool = False

    # Classifier-only mode: run the exact same detector frontend and frame
    # classifier path as stage-1, but return early before final PSD update,
    # gain computation, and ISTFT. This lets SpectralNoiseProcessor act as the
    # tuning / analysis processor without requiring a separate
    # RainFrameClassifierProcessor wrapper.
    classifier_only_mode: bool = False

    # -----------------------------------------------------------
    # Debug / tuning
    # -----------------------------------------------------------
    debug_enable: bool = False
    debug_frame_decim: int = 1  # store every Nth FrameFeatures; 1 = all

    # Separate feature payload for offline training / plotting.
    # When enabled, the processor exports a compact `features` dict containing
    # both soft-label / detector outputs and frequency-domain tuning features.
    dump_features: bool = False
    feature_decim: int = 1  # keep every Nth frame in exported features; 1 = all

    # -----------------------------------------------------------
    # Runtime / performance
    # -----------------------------------------------------------
    # Use float32 by default to better match CM7/single-precision behavior.
    process_dtype: str = "float32"   # "float32" | "float64"
    # Only reconstruct time-domain output audio when explicitly requested.
    compute_output_audio: bool = False

    # -----------------------------------------------------------
    # Nested detector configuration (framework-friendly)
    # -----------------------------------------------------------
    suppressor: Dict[str, Any] = field(default_factory=dict)
    detector: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------
# Config builder (dataclass defaults + params overrides)
#
# Notes:
#   - suppressor params may be supplied under params["suppressor"] and are also
#     flattened into top-level dataclass fields for stage-1 use.
#   - detector params are stored under cfg.detector and are consumed by
#     RainFrameClassifierMixin from that nested dict.
#   - detector-specific flat keys that are not dataclass fields are not copied
#     onto cfg as top-level attributes.
# -----------------------------------------------------------
def build_noise_config(sample_rate: int, params: Dict[str, Any]) -> "NoiseProcessorConfig":
    cfg = NoiseProcessorConfig(fs=int(sample_rate))

    cfg_fields = {f.name for f in fields(NoiseProcessorConfig)}

    # --- NEW: allow nested suppressor/detector dicts (framework-friendly) ---
    # Callers may pass either a flat dict of overrides, or a structured dict:
    #   {
    #     "suppressor": {...},
    #     "detector": {...},
    #     ...optional flat overrides...
    #   }
    # Precedence (highest wins):
    #   flat overrides > suppressor/detector nested overrides > dataclass defaults
    params = dict(params)  # avoid mutating caller dict

    sup = params.get("suppressor", None)
    if isinstance(sup, dict):
        cfg.suppressor = dict(sup)
        # nested suppressor provides defaults, but explicit flat keys win
        params = {**sup, **params}

    det = params.get("detector", None)
    if isinstance(det, dict):
        cfg.detector = dict(det)
    # Detector params intentionally remain nested under cfg.detector.
    # They are read by RainFrameClassifierMixin via cfg.detector[...] rather than
    # being flattened into top-level NoiseProcessorConfig attributes.

    # Legacy support: allow callers to pass fmin/fmax, but always normalize into operating_band.
    if "operating_band" not in params:
        fmin = params.get("fmin", None)
        fmax = params.get("fmax", None)
        if fmin is not None and fmax is not None:
            params["operating_band"] = (float(fmin), float(fmax))

    # apply overrides (only dataclass fields)
    for k, v in params.items():
        if k not in cfg_fields:
            continue

        if k == "operating_band":
            if isinstance(v, (list, tuple)) and len(v) == 2:
                v = (float(v[0]), float(v[1]))

        if k == "gain_freq_kernel":
            # allow list/tuple -> tuple[float]
            v = tuple(float(x) for x in v)

        setattr(cfg, k, v)

    op_lo, op_hi = cfg.operating_band
    cfg.operating_band = (float(op_lo), float(op_hi))
    return cfg

class SpectralNoiseProcessor(RainFrameClassifierMixin):
    """
    Noise processor:
      - STFT via librosa
      - Raindrop frame detection (via RainFrameClassifierMixin)
      - Optional suppressor bypass for offline feature extraction
      - Noise PSD estimation (FFT)
      - Adaptive, confidence-weighted suppression
      - ISTFT to get noise-suppressed waveform

    This class is intended as the "engine" used by the higher-level
    NoiseProcessor in the audio_processing_framework.
    """

    def __init__(self, config: Optional[NoiseProcessorConfig] = None):
        self.cfg = config
        self._is_setup = config is not None
        if self._is_setup:
            # Validate rain-frame classifier requirements once
            self._validate_rain_cfg()
            self._validate_suppressor_cfg()

    def setup(self, params: Dict[str, Any]):
        """
        Build NoiseProcessorConfig from framework params.
        Called once per parameter change / file batch.
        """
        if self._is_setup:
            # already configured; do not overwrite an explicit config
            return

        sr = int(params.get("sample_rate", params.get("fs", 11162)))

        self.cfg = build_noise_config(
            sample_rate=sr,
            params=params,
        )
        if self.cfg is None:
            raise RuntimeError("Failed to build NoiseProcessorConfig")
        # Validate rain-frame classifier requirements
        self._validate_rain_cfg()
        self._validate_suppressor_cfg()
        self._is_setup = True

    def _validate_suppressor_cfg(self) -> None:
        """Validate suppressor-related config (independent from detector validation)."""
        cfg = self.cfg
        if cfg is None:
            raise AttributeError("self.cfg is missing in processor")

        # Operating band sanity
        if not hasattr(cfg, "operating_band"):
            raise AttributeError("NoiseProcessorConfig missing operating_band")
        op_lo, op_hi = cfg.operating_band
        if not (np.isfinite(op_lo) and np.isfinite(op_hi) and 0.0 < float(op_lo) < float(op_hi)):
            raise ValueError(f"Invalid operating_band: {cfg.operating_band!r}")

        # STFT sanity
        if int(cfg.n_fft) <= 0 or int(cfg.hop) <= 0:
            raise ValueError(f"Invalid STFT params n_fft={cfg.n_fft}, hop={cfg.hop}")
        if int(cfg.hop) > int(cfg.n_fft):
            # Allowed, but usually unintended in this pipeline
            raise ValueError(f"hop ({cfg.hop}) should not exceed n_fft ({cfg.n_fft})")

        # Gain bounds
        if not (0.0 <= float(cfg.gain_floor) <= float(cfg.gain_ceil) <= 1.0):
            raise ValueError(f"Invalid gain bounds: floor={cfg.gain_floor}, ceil={cfg.gain_ceil}")

        # Oversubtraction sanity
        if float(cfg.oversub_base) <= 0.0 or float(cfg.oversub_max) <= 0.0:
            raise ValueError(f"Invalid oversub params: base={cfg.oversub_base}, max={cfg.oversub_max}")
        if float(cfg.oversub_max) < float(cfg.oversub_base):
            raise ValueError(f"oversub_max ({cfg.oversub_max}) must be >= oversub_base ({cfg.oversub_base})")

        # Smoothing alphas
        if not (0.0 <= float(cfg.gain_smooth_alpha) <= 1.0):
            raise ValueError(f"Invalid gain_smooth_alpha: {cfg.gain_smooth_alpha}")

    def _work_dtype(self):
        cfg = self.cfg
        dt = str(getattr(cfg, "process_dtype", "float32")).lower()
        return np.float32 if dt == "float32" else np.float64
    def _detector_param(self, name: str, default: Any = None) -> Any:
        """Resolve detector params using the same precedence as RainFrameClassifierMixin."""
        return self._dget(name, default)

    def _detector_flag(self, name: str, default: bool = False) -> bool:
        """Bool helper for detector params resolved via cfg.detector -> cfg attr -> default."""
        return bool(self._detector_param(name, default))

    def _build_prefilter_sos(self, sr: int, mode: str):
        cfg = self.cfg
        nyq = 0.5 * sr

        if mode == "bandpass":
            op_lo, op_hi = cfg.operating_band
            lo = float(op_lo)
            hi = float(op_hi)
            lo = np.clip(lo, 1e-3, nyq * 0.999)
            hi = np.clip(hi, lo + 1e-3, nyq * 0.999)
            wn = [lo / nyq, hi / nyq]
            return spsig.butter(int(getattr(cfg, "bp_order", cfg.hp_order)), wn, btype="bandpass", output="sos")

        if mode == "highpass" and cfg.hp_cutoff_hz > 0:
            norm_cut = np.clip(cfg.hp_cutoff_hz / nyq, 1e-4, 0.9999)
            return spsig.butter(cfg.hp_order, norm_cut, btype="highpass", output="sos")

        return None

    def _time_smooth(self, X: np.ndarray, L: int) -> np.ndarray:
        if L <= 1:
            return X
        dtype = self._work_dtype()
        B, T = X.shape
        Y = np.empty_like(X, dtype=dtype)
        csum = np.cumsum(X, axis=1, dtype=dtype)
        for t in range(T):
            t0 = max(0, t - L + 1)
            if t0 == 0:
                Y[:, t] = csum[:, t] / (t + 1)
            else:
                Y[:, t] = (csum[:, t] - csum[:, t0 - 1]) / (t - t0 + 1)
        return Y

    # ----------------------- Gain computation -----------------------

    def _compute_gain(
        self,
        P_band: np.ndarray,      # (K, T)
        N_band: np.ndarray,      # (K, T)
        noise_conf: np.ndarray,  # (T,)
        snr_gate: Optional[np.ndarray] = None,  # (T,) in [0,1]; 1 => protect (less suppression)
        debug_out: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Compute smoothed gain G_band for the operating band.

        P_band    : signal power in operating band  (K, T)
        N_band    : noise PSD estimate in band      (K, T)
        noise_conf: per-frame noise confidence      (T,) in [0, 1]
                    0 = definitely rain, 1 = definitely noise
        snr_gate  : optional (T,) array in [0,1]; 1 = protect (reduce suppression)
        """
        cfg = self.cfg
        eps = cfg.eps
        dtype = self._work_dtype()

        K, T = P_band.shape
        assert N_band.shape == (K, T)
        assert noise_conf.shape[0] == T

        noise_conf = np.clip(noise_conf, 0.0, 1.0)
        adaptive_gain_enable = bool(getattr(cfg, "adaptive_gain_enable", True))
        th = 0.7  # rain-like threshold for reducing smoothing/suppression (tune via nested cfg.suppressor if desired)
        denom = max(1e-9, 1.0 - th)

        if adaptive_gain_enable:
            # Effective "noise-ness" only above threshold
            eff_noise = np.clip((noise_conf - th) / denom, 0.0, 1.0)  # (T,)
            oversub = cfg.oversub_base + eff_noise * (cfg.oversub_max - cfg.oversub_base)

            # Optional: spectral SNR gating (frame-level). High SNR => reduce oversubtraction.
            if snr_gate is not None:
                sg = np.asarray(snr_gate, dtype=dtype).reshape(-1)
                if sg.shape[0] == T:
                    sg = np.clip(sg, 0.0, 1.0)
                    oversub = oversub * (1.0 - sg)
        else:
            # Debug mode for isolating PSD-driven modulation:
            # use a time-uniform oversubtraction independent of frame class / confidence.
            eff_noise = np.zeros(T, dtype=dtype)
            oversub = np.full(T, float(cfg.oversub_base), dtype=dtype)

        if debug_out is not None:
            debug_out["th"] = float(th)
            debug_out["adaptive_gain_enable"] = adaptive_gain_enable
            debug_out["eff_noise_t"] = eff_noise.astype(dtype, copy=False)
            debug_out["oversub_t"] = np.asarray(oversub, dtype=dtype)

        oversub_2d = oversub[None, :]  # (1, T) → broadcast to (K, T)

        if cfg.gain_mode.lower() == "wiener":
            # Wiener-like: G = max(P - alpha N, 0) / (P + eps)
            P_clean = np.maximum(P_band - oversub_2d * N_band, 0.0)
            G_raw = P_clean / (P_band + eps)
        else:
            # Default: sqrt spectral subtraction
            # G = 1 - alpha * sqrt(N / (P + eps))
            ratio = N_band / (P_band + eps)
            if debug_out is not None:
                # before clipping
                debug_out["ratio_median_t"] = np.median(ratio, axis=0)
                debug_out["ratio_p90_t"] = np.percentile(ratio, 90, axis=0)
                debug_out["ratio_max_t"] = np.max(ratio, axis=0)
            # Gain safety: cap ratio so sqrt_sub can't go hugely negative when N>P
            # (Even with PSD clamps, transient bins can still spike.)
            ratio = np.clip(ratio, 0.0, 1.0)

            G_raw = 1.0 - oversub_2d * np.sqrt(ratio)

        # Bound gain
        G_raw = np.clip(G_raw, cfg.gain_floor, cfg.gain_ceil)
        if debug_out is not None:
            debug_out["G_raw_median_t"] = np.median(G_raw, axis=0)
            debug_out["G_raw_p10_t"] = np.percentile(G_raw, 10, axis=0)
            debug_out["G_raw_min_t"] = np.min(G_raw, axis=0)

        # ---------- frequency smoothing (except on rain-like frames) ----------

        gain_freq_smooth_enable = bool(getattr(cfg, "gain_freq_smooth_enable", True))
        kernel_cfg = getattr(cfg, "gain_freq_kernel", (0.2, 0.6, 0.2))
        kernel = np.asarray(kernel_cfg, dtype=dtype).reshape(-1)
        if kernel.size < 1:
            kernel = np.array([1.0], dtype=dtype)
        kernel = kernel / (kernel.sum() + 1e-12)

        G_freq = np.empty_like(G_raw)
        for t in range(T):
            if (not gain_freq_smooth_enable) or (kernel.size == 1):
                G_freq[:, t] = G_raw[:, t]
            elif (not adaptive_gain_enable) or (noise_conf[t] >= th):
                G_freq[:, t] = np.convolve(G_raw[:, t], kernel, mode="same")
            else:
                G_freq[:, t] = G_raw[:, t]
        if debug_out is not None:
            debug_out["G_freq_median_t"] = np.median(G_freq, axis=0)
            debug_out["G_freq_min_t"] = np.min(G_freq, axis=0)

        # ---------- temporal smoothing (reduced on rain-like frames) ----------

        alpha_base = float(np.clip(cfg.gain_smooth_alpha, 0.0, 1.0))

        G_time = np.empty_like(G_freq)
        G_time[:, 0] = G_freq[:, 0]

        for t in range(1, T):
            if adaptive_gain_enable:
                nc = noise_conf[t]

                if nc < th:
                    alpha_t = 0.0  # no temporal smoothing on rain-like frames
                else:
                    eff_nc = (nc - th) / denom
                    alpha_t = alpha_base * eff_nc

                G_time[:, t] = alpha_t * G_time[:, t - 1] + (1.0 - alpha_t) * G_freq[:, t]

                # Ensure smoothing never reduces gain on rain-like frames
                if nc < th:
                    G_time[:, t] = np.maximum(G_time[:, t], G_freq[:, t])
            else:
                alpha_t = alpha_base
                G_time[:, t] = alpha_t * G_time[:, t - 1] + (1.0 - alpha_t) * G_freq[:, t]

        G_time = np.clip(G_time, cfg.gain_floor, cfg.gain_ceil)
        if debug_out is not None:
            debug_out["G_time_median_t"] = np.median(G_time, axis=0)
            debug_out["G_time_p10_t"] = np.percentile(G_time, 10, axis=0)
            debug_out["G_time_min_t"] = np.min(G_time, axis=0)
        return G_time
    def _mode_union_mask(self, freqs_band: np.ndarray, mode_bands: Any) -> np.ndarray:
        """Return boolean mask over `freqs_band` selecting the union of mode bands."""
        dtype = self._work_dtype()
        fb = np.asarray(freqs_band, dtype=dtype).reshape(-1)
        mask = np.zeros(fb.shape[0], dtype=bool)
        if not isinstance(mode_bands, (list, tuple)):
            return mask
        for bb in mode_bands:
            try:
                lo, hi = float(bb[0]), float(bb[1])
            except Exception:
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            if hi <= lo:
                continue
            mask |= (fb >= lo) & (fb <= hi)
        return mask

    def _compute_gain_noise_only(
        self,
        P_band: np.ndarray,   # (K, T)
        N_band: np.ndarray,   # (K, T)
    ) -> np.ndarray:
        """
        Gain for pure noise case using square-root spectral subtraction
        or Wiener-like rule, with simple smoothing.

        Assumes every frame is noise.
        """
        cfg = self.cfg
        eps = cfg.eps
        dtype = self._work_dtype()

        K, T = P_band.shape
        assert N_band.shape == (K, T)

        alpha = cfg.oversub_max  # use max oversub for noise-only tuning

        if cfg.gain_mode.lower() == "wiener":
            # G = max(P - alpha N, 0) / (P + eps)
            P_clean = np.maximum(P_band - alpha * N_band, 0.0)
            G_raw = P_clean / (P_band + eps)
        else:
            # sqrt spectral subtraction:
            # G = 1 - alpha * sqrt(N / (P + eps))
            ratio = N_band / (P_band + eps)
            ratio = np.clip(ratio, 0.0, 1.0)
            G_raw = 1.0 - alpha * np.sqrt(ratio)

        # Bound
        G_raw = np.clip(G_raw, cfg.gain_floor, cfg.gain_ceil)

        # -------- frequency smoothing --------
        kernel = np.array([0.25, 0.5, 0.25], dtype=dtype)
        G_freq = np.empty_like(G_raw)
        for t in range(T):
            G_freq[:, t] = np.convolve(G_raw[:, t], kernel, mode="same")

        # -------- temporal smoothing --------
        alpha_g = float(np.clip(cfg.gain_smooth_alpha, 0.0, 1.0))
        G_time = np.empty_like(G_freq)
        G_time[:, 0] = G_freq[:, 0]

        for t in range(1, T):
            G_time[:, t] = alpha_g * G_time[:, t-1] + (1 - alpha_g) * G_freq[:, t]

        return np.clip(G_time, cfg.gain_floor, cfg.gain_ceil)


  
    def _estimate_noise_psd_fft(
        self,
        P: np.ndarray,         # (F, T)
        freqs: np.ndarray,    # (F,)
        is_rain: np.ndarray,  # (T,)
        sr: Optional[int] = None,
    ) -> np.ndarray:

        cfg = self.cfg
        _, T = P.shape
        eps = cfg.eps

        op_lo, op_hi = cfg.operating_band
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)
        if sr is None:
            sr = cfg.fs
        frames_per_sec = float(sr) / float(cfg.hop)
        W = max(10, int(cfg.win_sec * frames_per_sec))
        K = int(band_mask.sum())

        # Extract band power
        P_band_all = P[band_mask, :]  # (K, T)

        # Optional pre-smoothing over time
        L = int(getattr(cfg, "pre_smooth_frames", 0))
        if L and L > 1:
            P_band_all = self._time_smooth(P_band_all, L)

        dtype = self._work_dtype()
        noise_psd = np.zeros_like(P, dtype=dtype)

        # Causal stochastic low-quantile tracker.
        # q controls the target lower quantile; win_sec sets the effective adaptation horizon.
        # This replaces the much more expensive rolling-buffer quantile computation.
        eta = float(2.0 / max(W + 1, 2))
        eta = float(np.clip(eta, 1e-4, 1.0))
        scale_alpha = float(cfg.ema_down)
        step_floor = float(max(cfg.eps, 1e-9))

        tracker = np.maximum(P_band_all[:, 0].astype(dtype, copy=True), 0.0)
        tracker_scale = np.maximum(np.abs(P_band_all[:, 0]).astype(dtype, copy=True), step_floor)
        warmup_count = 0
        ema_up = float(cfg.ema_up)
        ema_down = float(cfg.ema_down)

        for t in range(T):
            P_band = P_band_all[:, t]

            # Warmup protection: during the initial phase keep adapting from all frames.
            warmup_need = max(10, W // 2)
            allow_update = (warmup_count < warmup_need) or (not is_rain[t])

            if t == 0:
                raw_q = tracker
                if allow_update:
                    warmup_count += 1
            else:
                err = P_band - tracker
                tracker_scale = scale_alpha * tracker_scale + (1.0 - scale_alpha) * np.abs(err)
                step = eta * np.maximum(tracker_scale, step_floor)

                # Stochastic low-quantile update:
                #  - above baseline: move up slowly by q * step
                #  - below baseline: move down by (1-q) * step
                delta = np.where(P_band >= tracker, cfg.q * step, -(1.0 - cfg.q) * step)
                candidate = tracker + delta
                candidate = np.maximum(candidate, 0.0)

                if allow_update:
                    tracker = candidate
                    warmup_count += 1

                raw_q = tracker

            # Optional asymmetric EMA smoothing on top of the stochastic baseline.
            if t == 0:
                N_band = raw_q
            else:
                prev = noise_psd[band_mask, t - 1]
                up = raw_q > prev
                lam = np.where(up, ema_up, ema_down)
                N_band = lam * prev + (1.0 - lam) * raw_q

            # Clamp: PSD estimate should not exceed instantaneous band power (prevents runaway)
            maxr = float(getattr(cfg, "noise_psd_max_ratio", 1.0))
            maxr = 1.0 if (not np.isfinite(maxr)) else float(np.clip(maxr, 0.0, 1.0))
            N_band = np.minimum(N_band, maxr * P_band_all[:, t])
            noise_psd[band_mask, t] = np.maximum(N_band, 0.0)

        # Optional median smoothing over time (per frequency) after stochastic tracking
        m = int(getattr(cfg, "median_frames", 0))
        if m and m > 1:
            if m % 2 == 0:
                m += 1
            noise_psd = ndi.median_filter(noise_psd, size=(1, m))

        return noise_psd

    def _decimate_feature_value(self, value: Any, step: int) -> Any:
        """Decimate feature arrays/lists by frame step when possible."""
        if step <= 1:
            return value
        if value is None:
            return None
        try:
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return value
                return value[..., ::step]
            if isinstance(value, list):
                return value[::step]
            if isinstance(value, tuple):
                return value[::step]
        except Exception:
            return value
        return value

    def _build_features_payload(
        self,
        *,
        times_s: np.ndarray,
        frame_class: np.ndarray,
        is_rain: np.ndarray,
        rain_conf: np.ndarray,
        noise_conf: np.ndarray,
        det_debug: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """
        Build a compact exported feature payload for offline threshold tuning.

        The payload is intentionally separate from `debug` so training / plotting
        code can consume a stable structure without carrying all suppressor state.
        """
        dtype = self._work_dtype()
        features: Dict[str, Any] = {
            "frame_times": self._decimate_feature_value(np.asarray(times_s, dtype=dtype), step),
            "frame_class": self._decimate_feature_value(np.asarray(frame_class), step),
            "is_rain": self._decimate_feature_value(np.asarray(is_rain), step),
            "rain_conf": self._decimate_feature_value(np.asarray(rain_conf, dtype=dtype), step),
            "noise_conf": self._decimate_feature_value(np.asarray(noise_conf, dtype=dtype), step),
        }

        if not isinstance(det_debug, dict):
            return features

        # Prefer the detector-side feature dump if available.
        feature_dump = det_debug.get("feature_dump", None)
        if isinstance(feature_dump, dict):
            for k, v in feature_dump.items():
                features[k] = self._decimate_feature_value(v, step)
            return features

        # Otherwise pass through whatever the detector exported, decimating where possible.
        for k, v in det_debug.items():
            if k == "feature_dump":
                continue
            features[k] = self._decimate_feature_value(v, step)

        return features

    # ----------------------- Main API -----------------------

    def process(
        self,
        x: np.ndarray,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Lazy setup if constructed without a config
        if self.cfg is None:
            self.setup({"sample_rate": sr or 11162})

        cfg = self.cfg
        if sr is None:
            sr = cfg.fs

        work_dtype = self._work_dtype()
        compute_output_audio = bool(getattr(cfg, "compute_output_audio", False))

        # Ensure x is configured precision and 1-D
        x = np.asarray(x, dtype=work_dtype).reshape(-1)
        x_proc = x
        mode = str(getattr(cfg, "pre_filter_mode", "highpass")).lower()

        if mode not in ("highpass", "bandpass", "none"):
            mode = "highpass"

        if mode != "none":
            sos = self._build_prefilter_sos(sr, mode)
            if sos is not None:
                x_proc = spsig.sosfiltfilt(sos, x_proc).astype(work_dtype, copy=False)

        # 1) STFT
        S = librosa.stft(
            x_proc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
        )
        P = (np.abs(S).astype(work_dtype, copy=False)) ** 2
        freqs = np.asarray(librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft), dtype=work_dtype)
        times = np.asarray(librosa.frames_to_time(np.arange(P.shape[1]), sr=sr, hop_length=cfg.hop), dtype=work_dtype)

        F, T = P.shape

        op_lo, op_hi = cfg.operating_band
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)
        disable_suppression = bool(getattr(cfg, "disable_suppression", False))
        classifier_only_mode = bool(getattr(cfg, "classifier_only_mode", False))

        # 2) Frame classification (optionally using lagged/bootstrapped noise PSD)
        bypass_classifier = self._detector_flag("bypass_classifier", False)
        detector_use_noise_norm = self._detector_flag("detector_use_noise_norm", True)
        detector_noise_norm_mode = str(getattr(cfg, "detector_noise_norm_mode", "log_sub")).lower()

        bootstrap_noise_psd = None
        detector_noise_psd_lag = None

        if bypass_classifier:
            # Treat all frames as noise so we can inspect suppressor behavior.
            frame_class = np.full(T, FrameClass.NOISE, dtype=np.int8)
            rain_conf = np.zeros(T, dtype=work_dtype)
            det_debug = {
                "frame_class": frame_class,
                "frame_class_name": np.array(["noise"] * T, dtype=object),
                "rain_score_raw": np.zeros(T, dtype=work_dtype),
                "is_rain_raw": np.zeros(T, dtype=bool),
                "onset_mask": np.zeros(T, dtype=bool),
                "onset_indices": np.zeros(0, dtype=np.int32),
            }
            feature_dump = None
        else:
            P_for_detection = P.copy()
            P_for_detection[~band_mask, :] = 0.0

            if detector_use_noise_norm:
                # Bootstrap/coarse PSD pass using all frames as candidate noise frames.
                # This gives a lagged baseline for the detector without creating a circular dependency
                # with the final classifier-gated PSD estimate.
                bootstrap_is_rain_for_psd = np.zeros(T, dtype=bool)
                bootstrap_noise_psd = self._estimate_noise_psd_fft(
                    P,
                    freqs,
                    bootstrap_is_rain_for_psd,
                    sr=sr,
                )

                detector_noise_psd_lag = bootstrap_noise_psd.copy()
                if detector_noise_psd_lag.shape[1] > 1:
                    detector_noise_psd_lag = np.roll(detector_noise_psd_lag, shift=1, axis=1)
                    detector_noise_psd_lag[:, 0] = bootstrap_noise_psd[:, 0]

                # Safety clamp for lagged detector PSD as well.
                maxr_det = float(getattr(cfg, "noise_psd_max_ratio", 1.0))
                maxr_det = 1.0 if (not np.isfinite(maxr_det)) else float(np.clip(maxr_det, 0.0, 1.0))
                detector_noise_psd_lag = np.minimum(detector_noise_psd_lag, maxr_det * P)

                if detector_noise_norm_mode == "ratio_db":
                    P_for_detection = 10.0 * np.log10(P_for_detection / (detector_noise_psd_lag + cfg.eps) + cfg.eps)
                else:
                    # default: log-subtracted spectrum ~= dB above noise floor
                    P_for_detection = 10.0 * np.log10(P_for_detection + cfg.eps) - 10.0 * np.log10(detector_noise_psd_lag + cfg.eps)
            else:
                # legacy detector input: absolute spectrum in dB
                P_for_detection = 10.0 * np.log10(P_for_detection + cfg.eps)

            frame_class, rain_conf, det_debug, feature_dump = self._detect_rain_over_time(
                P_for_detection,
                freqs,
                detector_frame_times=np.asarray(times, dtype=work_dtype),
                input_audio=x,
            )
            if isinstance(det_debug, dict) and isinstance(feature_dump, dict):
                det_debug["feature_dump"] = feature_dump

        frame_class = np.asarray(frame_class, dtype=np.int8)
        is_rain = frame_class == FrameClass.RAIN
        is_noise = frame_class == FrameClass.NOISE
        noise_conf = np.asarray(
            det_debug.get("noise_conf", np.clip(1.0 - rain_conf, 0.0, 1.0)),
            dtype=work_dtype,
        )

        # Ensure we have a time axis in seconds for tuning/plots
        times_s = np.asarray(times, dtype=work_dtype)

        # If the detector provides FrameFeatures, optionally decimate to reduce payload
        if isinstance(det_debug, dict) and "frame_features" in det_debug:
            ff = det_debug.get("frame_features")
            decim = int(getattr(cfg, "debug_frame_decim", 1))
            if decim > 1 and ff is not None:
                try:
                    det_debug["frame_features"] = ff[::decim]
                except Exception:
                    pass

        if "feature_dump" not in locals():
            feature_dump = None

        features = None
        if bool(getattr(cfg, "dump_features", False)):
            feature_step = max(1, int(getattr(cfg, "feature_decim", 1)))
            features = self._build_features_payload(
                times_s=times_s,
                frame_class=frame_class,
                is_rain=is_rain,
                rain_conf=rain_conf,
                noise_conf=noise_conf,
                det_debug=det_debug,
                step=feature_step,
            )

        # Keep large tensors and verbose debug only when explicitly requested.
        keep_debug = bool(getattr(cfg, "debug_enable", False))
        keep_detector_debug = keep_debug
        keep_spectra = keep_debug
        keep_noise_psd = bool(getattr(cfg, "keep_noise_psd", False)) or (not classifier_only_mode)
        keep_filtered_audio = keep_debug
        keep_gain_debug = keep_debug

        if classifier_only_mode:
            debug = None
            if keep_debug:
                debug = {
                    "detector_params": dict(getattr(cfg, "detector", {}) or {}),
                    "suppressor_params": dict(getattr(cfg, "suppressor", {}) or {}),
                    "times_s": times_s,
                    "freqs": freqs,
                    "bootstrap_noise_psd": bootstrap_noise_psd,
                    "detector_noise_psd_lag": detector_noise_psd_lag,
                    "detector_use_noise_norm": detector_use_noise_norm,
                    "td_soft_enable": self._detector_flag("td_soft_enable", False),
                    "bypass_classifier": bypass_classifier,
                    "detector_noise_norm_mode": detector_noise_norm_mode,
                    "disable_suppression": disable_suppression,
                    "classifier_only_mode": True,
                    "operating_band": (float(op_lo), float(op_hi)),
                    "band_mask": band_mask,
                    "pre_filter_mode": mode,
                    "pre_filter_band": (float(cfg.operating_band[0]), float(cfg.operating_band[1])),
                    "features_available": features is not None,
                }

            result = {
                "frame_class": frame_class,
                "rain_conf": rain_conf,
                "noise_conf": noise_conf,
                "times": times,
                "freqs": freqs,
            }
            if features is not None:
                result["features"] = features
            if keep_detector_debug:
                result["det_debug"] = det_debug
            if debug is not None:
                result["debug"] = debug
            if keep_filtered_audio:
                result["x_filt"] = x_proc
                result["y"] = x_proc
            if keep_spectra:
                result["S"] = S
                result["S_hat"] = S
            if keep_noise_psd:
                result["noise_psd"] = np.zeros_like(P, dtype=work_dtype)

            return result


        # PSD update gating is derived from the canonical frame class.
        # Only confident NOISE frames are used to update the final noise PSD.
        use_for_noise_psd = np.asarray(is_noise, dtype=bool)
        is_rain_for_psd = ~use_for_noise_psd

        # 3) Noise PSD estimation / suppression.
        # For offline feature extraction we may want the detector path active
        # while completely bypassing the suppressor.
        P_band_all = P[band_mask]
        mel_filter = None
        snr_gate = None
        snr_mode = None
        gain_dbg: Dict[str, Any] = {}

        if disable_suppression:
            noise_psd = np.zeros_like(P, dtype=work_dtype)
            N_band_all = noise_psd[band_mask]
            N_band_eff = N_band_all
            ratio_med_t = np.zeros(T, dtype=work_dtype)
            G = np.ones_like(P)
            G_band = np.ones_like(P_band_all)
            S_hat = S.copy()
            y_hat = x_proc.copy() if compute_output_audio else None
            y_out = x_proc.copy() if compute_output_audio else None
            gain_dbg["suppression_disabled"] = True
        else:
            noise_psd = self._estimate_noise_psd_fft(P, freqs, is_rain_for_psd, sr=sr)
            mel_filter = None

            # 4) Gain computation (confidence-weighted)
            N_band_all = noise_psd[band_mask]

            # Optional: use lagged noise PSD for gain computation (N(t-1) applied to S(t))
            if bool(getattr(cfg, "use_lagged_noise_psd", False)) and N_band_all.shape[1] > 1:
                N_band_lag = np.roll(N_band_all, shift=1, axis=1)
                # Bootstrap first frame with its own estimate
                N_band_lag[:, 0] = N_band_all[:, 0]
            else:
                N_band_lag = N_band_all

            # IMPORTANT: lagged PSD can exceed current-frame power when the signal dips.
            # Clamp the effective PSD to current power to prevent ratio->1 everywhere and gain collapse.
            maxr = float(getattr(cfg, "noise_psd_max_ratio", 1.0))
            maxr = 1.0 if (not np.isfinite(maxr)) else float(np.clip(maxr, 0.0, 1.0))
            N_band_eff = np.minimum(N_band_lag, maxr * P_band_all)

            ratio_med_t = np.median(N_band_eff / (P_band_all + cfg.eps), axis=0)  # (T,)

            # ---- Optional spectral SNR gating (computed from mode bands) ----
            if bool(getattr(cfg, "snr_gating_enable", False)):
                # Use detector mode bands by default (falls back to whole operating band).
                det = getattr(cfg, "detector", {}) or {}
                mode_bands = det.get("mode_bands", None) if bool(getattr(cfg, "snr_gating_use_mode_bands", True)) else None

                freqs_band = freqs[band_mask]
                if mode_bands is not None:
                    mode_mask = self._mode_union_mask(freqs_band, mode_bands)
                else:
                    mode_mask = np.ones(freqs_band.shape[0], dtype=bool)

                if not np.any(mode_mask):
                    mode_mask = np.ones(freqs_band.shape[0], dtype=bool)

                # Frame-level SNR in the selected region
                Pm = np.sum(P_band_all[mode_mask, :], axis=0)
                Nm = np.sum(N_band_eff[mode_mask, :], axis=0)
                snr_mode = Pm / (Nm + cfg.eps)

                snr1 = float(getattr(cfg, "snr_gating_snr1", 1.0))
                snr1 = max(1e-9, snr1)
                gate = snr_mode / (snr_mode + snr1)

                pwr = float(getattr(cfg, "snr_gating_power", 1.0))
                if pwr != 1.0 and np.isfinite(pwr) and pwr > 0.0:
                    gate = np.power(np.clip(gate, 0.0, 1.0), pwr)

                snr_gate = np.clip(gate, 0.0, 1.0)

            if getattr(cfg, "debug_enable", False):
                ratio_med = float(np.median(N_band_all / (P_band_all + 1e-12)))
                ratio_p90 = float(np.percentile(N_band_all / (P_band_all + 1e-12), 90))
                print("median N/P:", ratio_med, "p90 N/P:", ratio_p90)

            # Gain computation (confidence-weighted)
            G_band = self._compute_gain(
                P_band_all,
                N_band_eff,
                noise_conf=noise_conf,
                snr_gate=snr_gate,
                debug_out=gain_dbg,
            )

            G = np.ones_like(P)
            G[band_mask] = G_band

            if getattr(cfg, "debug_enable", False):
                try:
                    gmed = float(np.median(G_band))
                    gmin = float(np.min(G_band))
                    gp10 = float(np.percentile(G_band, 10))
                    nconf_med = float(np.median(noise_conf))
                    print(
                        "[SpectralNoiseProcessor] gain stats:",
                        f"median={gmed:.3f}",
                        f"p10={gp10:.3f}",
                        f"min={gmin:.3f}",
                        f"noise_conf_median={nconf_med:.3f}",
                    )
                except Exception:
                    pass

            # 5) Apply gain & ISTFT
            S_hat = G * S
            if compute_output_audio:
                y_hat = librosa.istft(
                    S_hat,
                    hop_length=cfg.hop,
                    win_length=cfg.n_fft,
                    window="hann",
                    center=True,
                    length=len(x),
                ).astype(work_dtype, copy=False)
                # Choose output waveform:
                # Always output the normal noise-suppressed waveform
                y_out = y_hat
            else:
                y_hat = None
                y_out = None

        # 6) Debug
        op_lo, op_hi = cfg.operating_band

        # Keep detector debug under a dedicated key to avoid collisions and make plots simpler.
        debug = None
        if keep_debug:
            debug = {
                "detector_params": dict(getattr(cfg, "detector", {}) or {}),
                "suppressor_params": dict(getattr(cfg, "suppressor", {}) or {}),
                # SNR gating (optional)
                "snr_mode": snr_mode,
                "snr_gate": snr_gate,

                # Common time axis and frequency axis
                "times_s": times_s,
                "freqs": freqs,
                "bootstrap_noise_psd": bootstrap_noise_psd,
                "detector_noise_psd_lag": detector_noise_psd_lag,
                "detector_use_noise_norm": detector_use_noise_norm,
                "detector_noise_norm_mode": detector_noise_norm_mode,
                "td_soft_enable": self._detector_flag("td_soft_enable", False),
                "bypass_classifier": bypass_classifier,
                "disable_suppression": disable_suppression,
                "classifier_only_mode": classifier_only_mode,

                # PSD update / suppressor-side signals
                "use_for_noise_psd": use_for_noise_psd,
                "is_rain_for_psd": is_rain_for_psd,
                "G": G,
                "noise_psd": noise_psd,
                "use_lagged_noise_psd": bool(getattr(cfg, "use_lagged_noise_psd", False)),
                "gain_dbg": gain_dbg if keep_gain_debug else None,

                # Band metadata
                "operating_band": (float(op_lo), float(op_hi)),
                "band_mask": band_mask,

                # Pre-filter info for debug/tuning/plots
                "pre_filter_mode": mode,
                "pre_filter_band": (float(cfg.operating_band[0]), float(cfg.operating_band[1])),
                "np_ratio_median_t": ratio_med_t,
                "noise_psd_max_ratio": float(getattr(cfg, "noise_psd_max_ratio", 1.0)),
                "features_available": features is not None,
            }

        result = {
            "frame_class": frame_class,
            "freqs": freqs,
            "times": times,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
        }
        if features is not None:
            result["features"] = features
        if keep_detector_debug:
            result["det_debug"] = det_debug
        if debug is not None:
            result["debug"] = debug
        if keep_filtered_audio:
            result["x_filt"] = x_proc
            result["y"] = y_out
            result["y_suppressed"] = y_hat
        if keep_spectra:
            result["S"] = S
            result["S_hat"] = S_hat
        if keep_noise_psd:
            result["noise_psd"] = noise_psd

        return result


# -----------------------------------------------------------
# RainDetectorProcessor: framework-facing processor for rain-frame detection
# -----------------------------------------------------------

class RainDetectorProcessor(BaseProcessor):
    """
    Framework-facing processor for rain-frame detection.

    This is a thin adapter over SpectralNoiseProcessor so the orchestrator can
    call a single processor that owns:
      - frequency-domain feature extraction
      - time-domain feature extraction / soft labels
      - noise PSD estimation
      - optional suppression / reconstruction
      - rain frame detection

    The core algorithm remains inside SpectralNoiseProcessor.
    """

    def __init__(self, name: str = "rain_detector"):
        self.name = name
        self._proc_cache: Dict[str, SpectralNoiseProcessor] = {}

    def _params_cache_key(self, params: Dict[str, Any]) -> str:
        try:
            return json.dumps(params, sort_keys=True, default=str)
        except Exception:
            return repr(sorted(params.items(), key=lambda kv: kv[0]))

    def run(
        self,
        audio_data: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Basic input validation from the shared processor base.
        self._validate_audio(audio_data, params)

        params_local = dict(params)
        keep_state_audio = bool(params_local.get("keep_state_audio", False))
        params_local.setdefault("compute_output_audio", keep_state_audio)

        sample_rate = int(params_local.get("sample_rate", 11162))
        cache_key = self._params_cache_key(params_local)
        proc = self._proc_cache.get(cache_key)
        if proc is None:
            proc = SpectralNoiseProcessor()
            proc.setup(params_local)
            self._proc_cache[cache_key] = proc
        cfg = proc.cfg
        out, latency = self._with_timing(proc.process, audio_data, sr=sample_rate)

        frame_class = np.asarray(out.get("frame_class", []), dtype=np.int8)
        is_rain = frame_class == FrameClass.RAIN
        clip_rain_min_frames = int(params_local.get("clip_rain_min_frames", 1))
        clip_rain_min_frames = max(1, clip_rain_min_frames)
        rain_frame_count = int(np.sum(is_rain))
        clip_rain_fraction = float(np.mean(is_rain)) if is_rain.size else 0.0
        clip_is_rain = bool(rain_frame_count >= clip_rain_min_frames)
        rain_conf_arr = np.asarray(out.get("rain_conf", []), dtype=np.float32).reshape(-1)
        if rain_frame_count > 0 and rain_conf_arr.size == is_rain.size:
            median_rain_conf = float(np.median(rain_conf_arr[is_rain]))
        else:
            median_rain_conf = 0.0

        # Promote clip confidence toward 1.0 when rain is sustained well beyond the
        # minimum frame threshold required to call the clip rainy.
        abundance_ref = max(2 * clip_rain_min_frames, 1)
        abundance_conf = float(np.clip(rain_frame_count / float(abundance_ref), 0.0, 1.0))
        clip_rain_conf = float(max(median_rain_conf, abundance_conf))
        freqs = out.get("freqs", None)
        noise_psd = out.get("noise_psd", None)

        metrics: Dict[str, Any] = {
            "rain_frame_fraction": clip_rain_fraction,  # backward-compatible name
            "clip_rain_fraction": clip_rain_fraction,
            "rain_frame_count": rain_frame_count,
            "clip_is_rain": clip_is_rain,
            "clip_rain_conf": clip_rain_conf,
            "median_rain_conf": median_rain_conf,
            "clip_rain_min_frames": clip_rain_min_frames,
            "latency_s": latency,
        }

        if (
            noise_psd is not None
            and freqs is not None
            and isinstance(noise_psd, np.ndarray)
            and isinstance(freqs, np.ndarray)
        ):
            f_lo, f_hi = cfg.operating_band
            band_mask = (freqs >= f_lo) & (freqs <= f_hi)
            if np.any(band_mask):
                noise_band = noise_psd[band_mask]
                noise_db = 10.0 * np.log10(noise_band + cfg.eps)
                metrics["mean_noise_floor_db"] = float(np.mean(noise_db))
                metrics["median_noise_floor_db"] = float(np.median(noise_db))

        keep_state_spectra = bool(params_local.get("keep_state_spectra", False))
        keep_state_debug = bool(params_local.get("keep_state_debug", False))
        keep_state_config = bool(params_local.get("keep_state_config", False))
        keep_state_features = bool(params_local.get("keep_state_features", True))

        state: Dict[str, Any] = {
            "frame_class": out.get("frame_class"),
            "times": out.get("times"),
            "rain_conf": out.get("rain_conf"),
            "noise_conf": out.get("noise_conf"),
            "rain_frame_count": rain_frame_count,
            "clip_rain_fraction": clip_rain_fraction,
            "clip_is_rain": clip_is_rain,
            "clip_rain_conf": clip_rain_conf,
            "median_rain_conf": median_rain_conf,
            "clip_rain_min_frames": clip_rain_min_frames,
            "latency_s": latency,
            "processor": self.name,
        }

        if keep_state_features:
            state["features"] = out.get("features")

        if keep_state_debug:
            state["debug"] = out.get("debug")
            state["det_debug"] = out.get("det_debug")
            state["freqs"] = out.get("freqs")
            state["noise_psd"] = out.get("noise_psd")

        if keep_state_spectra:
            state["S"] = out.get("S")
            state["S_hat"] = out.get("S_hat")

        if keep_state_audio:
            state["input_audio"] = audio_data
            state["filtered_audio"] = out.get("x_filt")
            state["output_audio"] = out.get("y")

        if keep_state_config:
            state["config"] = cfg

        return metrics, state
