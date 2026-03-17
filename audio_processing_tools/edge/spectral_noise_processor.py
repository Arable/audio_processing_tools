# edge/spectral_noise_processor.py

from __future__ import annotations

from dataclasses import dataclass, fields, field

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal as spsig
import scipy.ndimage as ndi
import librosa


from .rain_frame_classifier import RainFrameClassifierMixin, FrameClass


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
    q: float = 0.25        # quantile level. 0.4 as default
    win_sec: float = 0.5 # rolling window size (seconds). 1.0 as default

    # Smoothing for PSD evolution
    ema_feedback: float = 0.80  # fallback if ema_up/ema_down not provided

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
    # number of mel bands
    n_mels: int = 24  # or 32; number of mel bands

    # select between fft and mel
    noise_psd_mode: str = "fft"  # "fft" or "mel"

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
    detector_use_noise_norm: bool = False
    # "log_sub" => 10*log10(P) - 10*log10(N_lag)
    # "ratio_db" => 10*log10(P / N_lag)
    detector_noise_norm_mode: str = "log_sub"

    # -----------------------------------------------------------
    # Debug / tuning
    # -----------------------------------------------------------
    debug_enable: bool = False
    debug_frame_decim: int = 1  # store every Nth FrameFeatures; 1 = all

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
      - Noise PSD estimation (FFT or mel)
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

    def _time_smooth(self, X: np.ndarray, L: int) -> np.ndarray:
        if L <= 1:
            return X
        B, T = X.shape
        Y = np.empty_like(X, dtype=np.float64)
        csum = np.cumsum(X, axis=1, dtype=np.float64)
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
                sg = np.asarray(snr_gate, dtype=np.float64).reshape(-1)
                if sg.shape[0] == T:
                    sg = np.clip(sg, 0.0, 1.0)
                    oversub = oversub * (1.0 - sg)
        else:
            # Debug mode for isolating PSD-driven modulation:
            # use a time-uniform oversubtraction independent of frame class / confidence.
            eff_noise = np.zeros(T, dtype=np.float64)
            oversub = np.full(T, float(cfg.oversub_base), dtype=np.float64)

        if debug_out is not None:
            debug_out["th"] = float(th)
            debug_out["adaptive_gain_enable"] = adaptive_gain_enable
            debug_out["eff_noise_t"] = eff_noise.astype(np.float64, copy=False)
            debug_out["oversub_t"] = np.asarray(oversub, dtype=np.float64)

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
        kernel = np.asarray(kernel_cfg, dtype=np.float64).reshape(-1)
        if kernel.size < 1:
            kernel = np.array([1.0], dtype=np.float64)
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
        fb = np.asarray(freqs_band, dtype=np.float64).reshape(-1)
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
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
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

    def _estimate_noise_psd_mel(
        self,
        P: np.ndarray,
        freqs: np.ndarray,
        is_rain: np.ndarray,
        sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate noise PSD using mel-band min-stats then expand back to FFT bins.

        Returns
        -------
        noise_psd_fft : (F, T)
        W_mel         : (B, F) mel filterbank used (for debug/inspection)
        """
        cfg = self.cfg
        F, T = P.shape
        eps = cfg.eps

        if sr is None:
            sr = cfg.fs

        op_lo, op_hi = cfg.operating_band
        W_mel = librosa.filters.mel(
            sr=int(sr),
            n_fft=cfg.n_fft,
            n_mels=int(getattr(cfg, "n_mels", 24)),
            fmin=float(op_lo),
            fmax=float(op_hi),
        ).astype(np.float64)  # (B, F)

        # Mel power
        P_mel = W_mel @ P  # (B, T)

        # Optional pre-smoothing over time
        L = int(getattr(cfg, "pre_smooth_frames", 0))
        if L and L > 1:
            P_mel = self._time_smooth(P_mel, L)

        B = P_mel.shape[0]

        frames_per_sec = float(sr) / float(cfg.hop)
        W_frames = max(10, int(cfg.win_sec * frames_per_sec))

        # Ring buffer for noise-only frames
        buffer = np.full((W_frames, B), np.nan, dtype=np.float64)
        buf_idx = 0
        buf_count = 0

        noise_mel = np.zeros((B, T), dtype=np.float64)

        ema_up = float(getattr(cfg, "ema_up", cfg.ema_feedback))
        ema_down = float(getattr(cfg, "ema_down", cfg.ema_feedback))

        for t in range(T):
            Pm = P_mel[:, t]

            warmup_need = max(10, W_frames // 2)
            if buf_count < warmup_need or (not is_rain[t]):
                buffer[buf_idx] = Pm
                buf_idx = (buf_idx + 1) % W_frames
                buf_count = min(buf_count + 1, W_frames)
            if buf_count == 0:
                raw_q = Pm
            else:
                valid = buffer[:buf_count]
                raw_q = np.nanquantile(valid, cfg.q, axis=0)

            if t == 0:
                Nm = raw_q
            else:
                prev = noise_mel[:, t - 1]
                lam = np.where(raw_q > prev, ema_up, ema_down)
                Nm = lam * prev + (1.0 - lam) * raw_q

            # Clamp: noise estimate should not exceed instantaneous mel power
            maxr = float(getattr(cfg, "noise_psd_max_ratio", 1.0))
            maxr = 1.0 if (not np.isfinite(maxr)) else float(np.clip(maxr, 0.0, 1.0))
            Nm = np.minimum(Nm, maxr * Pm)
            noise_mel[:, t] = np.maximum(Nm, 0.0)

        # Optional median filter over time
        m = int(getattr(cfg, "median_frames", 0))
        if m and m > 1:
            if m % 2 == 0:
                m += 1
            noise_mel = ndi.median_filter(noise_mel, size=(1, m))

        # Expand mel -> FFT. Distribute mel energy back to FFT bins and normalize by filter sum.
        den = (W_mel.sum(axis=0, keepdims=True) + eps)  # (1, F)
        
        W_norm = W_mel / (W_mel.sum(axis=1, keepdims=True) + eps)
        noise_psd_fft = W_norm.T @ noise_mel

        # Zero outside operating band
        band_mask = (freqs >= float(op_lo)) & (freqs <= float(op_hi))
        noise_psd_fft[~band_mask, :] = 0.0

        return noise_psd_fft, W_mel

  
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

        # Ring buffer of noise-only frames for quantile stats
        buffer = np.full((W, K), np.nan, dtype=np.float64)
        buf_idx = 0
        buf_count = 0

        noise_psd = np.zeros_like(P, dtype=np.float64)

        ema_up = float(getattr(cfg, "ema_up", cfg.ema_feedback))
        ema_down = float(getattr(cfg, "ema_down", cfg.ema_feedback))

        for t in range(T):
            P_band = P_band_all[:, t]

            # Warmup protection: until buffer is reasonably filled, keep learning from *all* frames.
            # Otherwise, early "rain" decisions can prevent any buffer fill → buf_count stays 0 → PSD diverges.
            warmup_need = max(10, W // 2)  # fill at least half the window (tunable)
            if buf_count < warmup_need or (not is_rain[t]):
                buffer[buf_idx] = P_band
                buf_idx = (buf_idx + 1) % W
                buf_count = min(buf_count + 1, W)

            # Robust min-stats via quantile over buffer
            if buf_count == 0:
                raw_q = P_band
            else:
                valid = buffer[:buf_count]
                raw_q = np.nanquantile(valid, cfg.q, axis=0)

            # EMA with asymmetric rates (fast-up / slow-down)
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

        # Optional median smoothing over time (per frequency)
        m = int(getattr(cfg, "median_frames", 0))
        if m and m > 1:
            if m % 2 == 0:
                m += 1
            noise_psd = ndi.median_filter(noise_psd, size=(1, m))

        return noise_psd
 
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

        # Ensure x is float64 and 1-D
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x_proc = x
        mode = str(getattr(cfg, "pre_filter_mode", "highpass")).lower()

        if mode not in ("highpass", "bandpass", "none"):
            mode = "highpass"

        if mode != "none":
            nyq = 0.5 * sr

            if mode == "bandpass":
                op_lo, op_hi = cfg.operating_band
                lo = float(op_lo)
                hi = float(op_hi)

                # Clamp to valid digital band limits
                lo = np.clip(lo, 1e-3, nyq * 0.999)
                hi = np.clip(hi, lo + 1e-3, nyq * 0.999)

                wn = [lo / nyq, hi / nyq]
                sos = spsig.butter(int(getattr(cfg, "bp_order", cfg.hp_order)), wn, btype="bandpass", output="sos")
                x_proc = spsig.sosfiltfilt(sos, x_proc)

            else:
                # highpass
                if cfg.hp_cutoff_hz > 0:
                    norm_cut = np.clip(cfg.hp_cutoff_hz / nyq, 1e-4, 0.9999)
                    sos = spsig.butter(cfg.hp_order, norm_cut, btype="highpass", output="sos")
                    x_proc = spsig.sosfiltfilt(sos, x_proc)

        # 1) STFT
        S = librosa.stft(
            x_proc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
        )
        P = np.abs(S) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
        times = librosa.frames_to_time(np.arange(P.shape[1]), sr=sr, hop_length=cfg.hop)

        F, T = P.shape

        op_lo, op_hi = cfg.operating_band
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)

        # 2) Frame classification (optionally using lagged/bootstrapped noise PSD)
        bypass_classifier = bool(getattr(cfg, "detector", {}).get("bypass_classifier", False))
        detector_use_noise_norm = bool(getattr(cfg, "detector_use_noise_norm", False))
        detector_noise_norm_mode = str(getattr(cfg, "detector_noise_norm_mode", "log_sub")).lower()

        bootstrap_noise_psd = None
        detector_noise_psd_lag = None

        if bypass_classifier:
            # Treat all frames as noise so we can inspect suppressor behavior.
            frame_class = np.full(T, FrameClass.NOISE, dtype=np.int8)
            rain_conf = np.zeros(T, dtype=np.float64)
            det_debug = {
                "frame_class": frame_class,
                "frame_class_name": np.array(["noise"] * T, dtype=object),
                "rain_score_raw": np.zeros(T, dtype=np.float64),
                "is_rain_raw": np.zeros(T, dtype=bool),
                "onset_mask": np.zeros(T, dtype=bool),
                "onset_indices": np.zeros(0, dtype=np.int32),
            }
        else:
            P_for_detection = P.copy()
            P_for_detection[~band_mask, :] = 0.0

            if detector_use_noise_norm:
                # Bootstrap/coarse PSD pass using all frames as candidate noise frames.
                # This gives a lagged baseline for the detector without creating a circular dependency
                # with the final classifier-gated PSD estimate.
                bootstrap_is_rain_for_psd = np.zeros(T, dtype=bool)
                if cfg.noise_psd_mode == "fft":
                    bootstrap_noise_psd = self._estimate_noise_psd_fft(
                        P,
                        freqs,
                        bootstrap_is_rain_for_psd,
                        sr=sr,
                    )
                elif cfg.noise_psd_mode == "mel":
                    bootstrap_noise_psd, _ = self._estimate_noise_psd_mel(
                        P,
                        freqs,
                        bootstrap_is_rain_for_psd,
                        sr=sr,
                    )
                else:
                    raise ValueError(f"Unknown PSD mode: {cfg.noise_psd_mode}")

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

            frame_class, rain_conf, det_debug = self._detect_rain_over_time(
                P_for_detection,
                freqs,
            )

        frame_class = np.asarray(frame_class, dtype=np.int8)
        is_rain = frame_class == FrameClass.RAIN
        is_noise = frame_class == FrameClass.NOISE
        noise_conf = np.asarray(
            det_debug.get("noise_conf", np.clip(1.0 - rain_conf, 0.0, 1.0)),
            dtype=np.float64,
        )

        # Ensure we have a time axis in seconds for tuning/plots
        times_s = np.asarray(times, dtype=np.float64)

        # If the detector provides FrameFeatures, optionally decimate to reduce payload
        if isinstance(det_debug, dict) and "frame_features" in det_debug:
            ff = det_debug.get("frame_features")
            decim = int(getattr(cfg, "debug_frame_decim", 1))
            if decim > 1 and ff is not None:
                try:
                    det_debug["frame_features"] = ff[::decim]
                except Exception:
                    pass

        # PSD update gating is derived from the canonical frame class.
        # Only confident NOISE frames are used to update the final noise PSD.
        use_for_noise_psd = np.asarray(is_noise, dtype=bool)
        is_rain_for_psd = ~use_for_noise_psd

        # 3) Noise PSD estimation (MCRA / quantile)
        if cfg.noise_psd_mode == "fft":
            noise_psd = self._estimate_noise_psd_fft(P, freqs, is_rain_for_psd, sr=sr)
            mel_filter = None
        elif cfg.noise_psd_mode == "mel":
            noise_psd, mel_filter = self._estimate_noise_psd_mel(P, freqs, is_rain_for_psd, sr=sr)
        else:
            raise ValueError(f"Unknown PSD mode: {cfg.noise_psd_mode}")

        # 4) Gain computation (confidence-weighted)
        P_band_all = P[band_mask]
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
        snr_gate = None
        snr_mode = None
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
        gain_dbg: Dict[str, Any] = {}
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
        y_hat = librosa.istft(
            S_hat,
            hop_length=cfg.hop,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
            length=len(x),
        )

        # Choose output waveform:
        # Always output the normal noise-suppressed waveform
        y_out = y_hat

        # 6) Debug
        op_lo, op_hi = cfg.operating_band

        # Keep detector debug under a dedicated key to avoid collisions and make plots simpler.
        debug = {
            "detector": det_debug,
            "detector_params": dict(getattr(cfg, "detector", {}) or {}),
            "suppressor_params": dict(getattr(cfg, "suppressor", {}) or {}),
            # Extract useful detector tuning signals if present
            "z_primary": (det_debug.get("z_primary") if isinstance(det_debug, dict) else None),
            "z_modes": (det_debug.get("z_modes") if isinstance(det_debug, dict) else None),
            "peak_ratio": (det_debug.get("peak_ratio") if isinstance(det_debug, dict) else None),
            "flux_primary": (det_debug.get("flux_primary") if isinstance(det_debug, dict) else None),
            "flux_modes": (det_debug.get("flux_modes") if isinstance(det_debug, dict) else None),
            "rain_score_raw": (det_debug.get("rain_score_raw") if isinstance(det_debug, dict) else None),
            "frame_class": (det_debug.get("frame_class") if isinstance(det_debug, dict) else None),
            "frame_class_name": (det_debug.get("frame_class_name") if isinstance(det_debug, dict) else None),
            "is_rain_raw": (det_debug.get("is_rain_raw") if isinstance(det_debug, dict) else None),
            "onset_mask": (det_debug.get("onset_mask") if isinstance(det_debug, dict) else None),
            "onset_indices": (det_debug.get("onset_indices") if isinstance(det_debug, dict) else None),

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

            # Classifier outputs
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "is_rain": is_rain,
            "is_noise": is_noise,
            "use_for_noise_psd": use_for_noise_psd,
            "is_rain_for_psd": is_rain_for_psd,

            # Suppression / PSD
            "G": G,
            "noise_psd": noise_psd,
            "use_lagged_noise_psd": bool(getattr(cfg, "use_lagged_noise_psd", False)),
            # Gain diagnostics (where suppression can collapse)
            "gain_dbg": gain_dbg,

            # Band metadata
            "operating_band": (float(op_lo), float(op_hi)),
            "band_mask": band_mask,

            # Pre-filter info for debug/tuning/plots
            "pre_filter_mode": mode,
            "pre_filter_band": (float(cfg.operating_band[0]), float(cfg.operating_band[1])),
            "np_ratio_median_t": ratio_med_t,
            "noise_psd_max_ratio": float(getattr(cfg, "noise_psd_max_ratio", 1.0)),
        }

        # Optional: include mel filter if present
        if mel_filter is not None:
            debug["mel_filter"] = mel_filter

        return {
            "y": y_out,            # what your harness will treat as "denoised_audio"
            "y_suppressed": y_hat, # always available for A/B
            "S": S,
            "S_hat": S_hat,
            "noise_psd": noise_psd,
            "frame_class": frame_class,
            "is_rain": is_rain,
            "freqs": freqs,
            "times": times,
            # Back-compat convenience: top-level detector keys (if present)
            "det_debug": det_debug,
            "debug": debug,
            "x_hp": x_proc,   # back-compat name
            "x_filt": x_proc, # clearer name (HPF/BPF/none)
            # Added explicit outputs for harness convenience
            "use_for_noise_psd": use_for_noise_psd,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
        }