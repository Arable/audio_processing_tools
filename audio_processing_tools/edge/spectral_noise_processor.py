# edge/spectral_noise_processor.py

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal as spsig
import scipy.ndimage as ndi
import librosa


from .rain_frame_classifier import RainFrameClassifierMixin

# Helper to safely convert numpy scalars/arrays for debug, and optionally gate printing.
def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _mode_isolation_mask(
    freqs: np.ndarray,
    mode_bands: Tuple[Tuple[float, float], ...],
    attn_db: float,
    smooth_bins: int,
) -> np.ndarray:
    """
    Build a 1D amplitude mask g[f] that is 1.0 inside any mode band,
    and attenuated elsewhere by attn_db (amplitude dB). Optionally smooth in bins.
    """
    attn = 10.0 ** (float(attn_db) / 20.0)  # amplitude scaling
    g = np.full_like(freqs, attn, dtype=np.float64)

    for (a, b) in mode_bands:
        g[(freqs >= float(a)) & (freqs <= float(b))] = 1.0

    sb = int(smooth_bins)
    if sb and sb > 0:
        ker = np.ones(2 * sb + 1, dtype=np.float64)
        ker /= ker.sum()
        g = np.convolve(g, ker, mode="same")
        g = np.clip(g, 0.0, 1.0)

    return g

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
    rain_like_thresh: float = 0.7
    # number of mel bands
    n_mels: int = 24  # or 32; number of mel bands

    # select between fft and mel
    noise_psd_mode: str = "fft"  # "fft" or "mel"

    # New / refined PSD tracking params
    pre_smooth_frames: int = 0      # try 3–5, 0 disables
    ema_up: float = 0.6             # fast when noise increases
    ema_down: float = 0.95          # slow when noise decreases

    # ---------------- mode-frequency evidence (new) ----------------
    peak_top_p: int = 6
    peak_primary_q: int = 3
    peaks_in_mode_min: int = 2

    mode_bands: Tuple[Tuple[float, float], ...] = (
        (450.0, 650.0),
        (750.0, 950.0),
        (1500.0, 1800.0),
        (2350.0, 2600.0),
        (3150.0, 3350.0),
        (3750.0, 3950.0),
    )
    mode_weights: Tuple[float, ...] = (1.0, 0.8, 0.7, 0.6, 0.5, 0.4)

    mode_rise_db_low: float = 2.0
    mode_rise_db_high: float = 6.0
    mode_peak_db_low: float = 2.0
    mode_peak_db_high: float = 8.0

    mode_w_rise: float = 0.8
    mode_w_peak: float = 0.2

    primary_mode_idx: int = 0
    primary_min_score: float = 0.5
    primary_rise_db_thresh: float = 6.0
    mode_overall_margin_db_low: float = 1.5
    mode_overall_margin_db_high: float = 6.0

    # Probabilistic evidence weights (must sum to ~1.0; tune as needed)
    prob_w_mode_rise: float = 0.55
    prob_w_mode_peak: float = 0.20
    prob_w_margin: float = 0.25

    # ---------------- 3-state classification (new) ----------------
    rain_hi: float = 0.80
    noise_hi: float = 0.85
    noise_update_thresh: float = 0.85

    rain_hold_frames: int = 1
    noise_confirm_frames: int = 2

    # -----------------------------------------------------------
    # Debug / tuning
    # -----------------------------------------------------------
    debug_enable: bool = False
    debug_frame_decim: int = 1  # store every Nth FrameFeatures; 1 = all

     # -----------------------------------------------------------
    # TEMP experiment: mode-only reconstruction (for listening tests)
    # -----------------------------------------------------------
    mode_isolation_enable: bool = False
    mode_isolation_attn_db: float = -60.0     # attenuation outside mode bands (dB, amplitude)
    mode_isolation_smooth_bins: int = 2       # 0 disables; small smoothing reduces ringing
    mode_isolation_affects_classifier: bool = False
    mode_isolation_only: bool = True

# -----------------------------------------------------------
# Config builder (dataclass defaults + params overrides)
# -----------------------------------------------------------

def build_noise_config(sample_rate: int, params: Dict[str, Any]) -> "NoiseProcessorConfig":
    """
    Single-source-of-truth builder:
      - defaults come from NoiseProcessorConfig dataclass
      - overrides come from params dict
    """
    cfg = NoiseProcessorConfig(fs=int(sample_rate))

    # set of dataclass field names
    cfg_fields = {f.name for f in fields(NoiseProcessorConfig)}

    # Legacy support: allow callers to pass fmin/fmax, but always normalize into operating_band.
    # operating_band wins if both are provided.
    if "operating_band" not in params:
        fmin = params.get("fmin", None)
        fmax = params.get("fmax", None)
        if fmin is not None and fmax is not None:
            params = dict(params)  # avoid mutating caller dict
            params["operating_band"] = (float(fmin), float(fmax))

    # apply overrides
    for k, v in params.items():
        if k not in cfg_fields:
            continue

        if k == "mode_bands":
            # accept list[list[2]] / list[tuple[2]] / tuple[tuple[2]]
            v = tuple((float(a), float(b)) for (a, b) in v)

        if k == "mode_weights":
            v = tuple(float(x) for x in v)

        if k == "operating_band":
            if isinstance(v, (list, tuple)) and len(v) == 2:
                v = (float(v[0]), float(v[1]))

        setattr(cfg, k, v)

    # Final normalization
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
        self._is_setup = True

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
    ) -> np.ndarray:
        """
        Compute smoothed gain G_band for the operating band.

        P_band    : signal power in operating band  (K, T)
        N_band    : noise PSD estimate in band      (K, T)
        noise_conf: per-frame noise confidence      (T,) in [0, 1]
                    0 = definitely rain, 1 = definitely noise
        """
        cfg = self.cfg
        eps = cfg.eps

        K, T = P_band.shape
        assert N_band.shape == (K, T)
        assert noise_conf.shape[0] == T

        noise_conf = np.clip(noise_conf, 0.0, 1.0)
        th = getattr(cfg, "rain_like_thresh", 0.7)
        denom = max(1e-9, 1.0 - th)

        # Effective "noise-ness" only above threshold
        eff_noise = np.clip((noise_conf - th) / denom, 0.0, 1.0)  # (T,)

        oversub = cfg.oversub_base + eff_noise * (cfg.oversub_max - cfg.oversub_base)
        oversub_2d = oversub[None, :]  # (1, T) → broadcast to (K, T)

        if cfg.gain_mode.lower() == "wiener":
            # Wiener-like: G = max(P - alpha N, 0) / (P + eps)
            P_clean = np.maximum(P_band - oversub_2d * N_band, 0.0)
            G_raw = P_clean / (P_band + eps)
        else:
            # Default: sqrt spectral subtraction
            # G = 1 - alpha * sqrt(N / (P + eps))
            ratio = N_band / (P_band + eps)
            ratio = np.clip(ratio, 0.0, 1.0)
            G_raw = 1.0 - oversub_2d * np.sqrt(ratio)

        # Bound gain
        G_raw = np.clip(G_raw, cfg.gain_floor, cfg.gain_ceil)

        # ---------- frequency smoothing (except on rain-like frames) ----------

        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        G_freq = np.empty_like(G_raw)

        for t in range(T):
            if noise_conf[t] < th:
                G_freq[:, t] = G_raw[:, t]
            else:
                G_freq[:, t] = np.convolve(G_raw[:, t], kernel, mode="same")

        # ---------- temporal smoothing (reduced on rain-like frames) ----------

        alpha_base = float(np.clip(cfg.gain_smooth_alpha, 0.0, 1.0))

        G_time = np.empty_like(G_freq)
        G_time[:, 0] = G_freq[:, 0]

        for t in range(1, T):
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

        G_time = np.clip(G_time, cfg.gain_floor, cfg.gain_ceil)
        return G_time

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
        P: np.ndarray,            # (F,T)
        freqs: np.ndarray,        # (F,)
        is_rain: np.ndarray,      # (T,)
        sr: Optional[int] = None, # <-- add this if you can
    ) -> Tuple[np.ndarray, np.ndarray]:

        cfg = self.cfg
        _, T = P.shape
        eps = cfg.eps
        n_mels = getattr(cfg, "n_mels", 24)

        if sr is None:
            sr = cfg.fs  # but ideally sr should match the STFT sr

        op_lo, op_hi = cfg.operating_band
        W_mel = librosa.filters.mel(
            sr=sr,
            n_fft=cfg.n_fft,
            n_mels=n_mels,
            fmin=float(op_lo),
            fmax=float(op_hi),
        ).astype(np.float64)  # (B,F)

        P_mel = W_mel @ P  # (B,T)

        # --- optional time smoothing ---
        L = int(getattr(cfg, "pre_smooth_frames", 0))
        if L and L > 1:
            P_mel = self._time_smooth(P_mel, L)

        B = P_mel.shape[0]

        frames_per_sec = sr / cfg.hop
        W_frames = max(10, int(cfg.win_sec * frames_per_sec))

        buffer = np.full((W_frames, B), np.nan, dtype=np.float64)
        buf_idx = 0
        buf_count = 0

        noise_mel = np.zeros_like(P_mel)

        ema_up = float(getattr(cfg, "ema_up", cfg.ema_feedback))
        ema_down = float(getattr(cfg, "ema_down", cfg.ema_feedback))

        for t in range(T):
            Pm = P_mel[:, t]

            if not is_rain[t]:
                buffer[buf_idx] = Pm
                buf_idx = (buf_idx + 1) % W_frames
                buf_count = min(buf_count + 1, W_frames)

            if buf_count == 0:
                raw_q = Pm
            else:
                valid = buffer[:buf_count]
                raw_q = np.nanquantile(valid, cfg.q, axis=0)

            if t == 0:
                N_mel = raw_q
            else:
                prev = noise_mel[:, t - 1]
                lam = np.where(raw_q > prev, ema_up, ema_down)
                N_mel = lam * prev + (1.0 - lam) * raw_q

            noise_mel[:, t] = N_mel

        # optional median filter
        m = int(getattr(cfg, "median_frames", 0))
        if m and m > 1:
            if m % 2 == 0:
                m += 1
            noise_mel = ndi.median_filter(noise_mel, size=(1, m))

        # --- expand mel → FFT (normalize per FFT bin = column sum of W_mel) ---
        den = (W_mel.sum(axis=0, keepdims=True).T + eps)  # (F,1)
        noise_psd_fft = (W_mel.T @ noise_mel) / den       # (F,T)

        op_lo, op_hi = cfg.operating_band
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)
        noise_psd_fft[~band_mask, :] = 0.0

        # debug ratios (now consistent with smoothing)
        if getattr(cfg, "debug_enable", False):
            ratio_mel = noise_mel / (P_mel + eps)
            print("median N/P mel:", float(np.median(ratio_mel)), "p90:", float(np.percentile(ratio_mel, 90)))

        return noise_psd_fft, W_mel

  
    def _estimate_noise_psd_fft(
        self,
        P: np.ndarray,         # (F, T)
        freqs: np.ndarray,    # (F,)
        is_rain: np.ndarray,  # (T,)
    ) -> np.ndarray:

        cfg = self.cfg
        _, T = P.shape
        eps = cfg.eps

        op_lo, op_hi = cfg.operating_band
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)
        K = int(band_mask.sum())

        frames_per_sec = cfg.fs / cfg.hop
        W = max(10, int(cfg.win_sec * frames_per_sec))

        # --- extract band power ---
        P_band_all = P[band_mask, :]  # (K, T)

        # --- optional time smoothing ---
        L = int(getattr(cfg, "pre_smooth_frames", 0))
        if L and L > 1:
            P_band_all = self._time_smooth(P_band_all, L)

        buffer = np.full((W, K), np.nan, dtype=np.float64)
        buf_idx = 0
        buf_count = 0

        noise_psd = np.zeros_like(P)

        ema_up = float(getattr(cfg, "ema_up", cfg.ema_feedback))
        ema_down = float(getattr(cfg, "ema_down", cfg.ema_feedback))

        for t in range(T):
            P_band = P_band_all[:, t]

            if not is_rain[t]:
                buffer[buf_idx] = P_band
                buf_idx = (buf_idx + 1) % W
                buf_count = min(buf_count + 1, W)

            if buf_count == 0:
                raw_q = P_band
            else:
                valid = buffer[:buf_count]
                raw_q = np.nanquantile(valid, cfg.q, axis=0)

            if t == 0:
                N_band = raw_q
            else:
                prev = noise_psd[band_mask, t - 1]
                up = raw_q > prev
                lam = np.where(up, ema_up, ema_down)
                N_band = lam * prev + (1.0 - lam) * raw_q

            noise_psd[band_mask, t] = N_band

        # optional median smoothing
        m = getattr(cfg, "median_frames", 0)
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

        # 0) Pre-filter (before STFT)
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

                # 1b) TEMP EXPERIMENT: mode-only reconstruction for listening/inspection
        mode_iso_enable = bool(getattr(cfg, "mode_isolation_enable", False))
        y_mode = None
        S_mode = None
        mode_mask = None

        if mode_iso_enable:
            mode_mask = _mode_isolation_mask(
                freqs=freqs,
                mode_bands=tuple(getattr(cfg, "mode_bands", ())),
                attn_db=float(getattr(cfg, "mode_isolation_attn_db", -60.0)),
                smooth_bins=int(getattr(cfg, "mode_isolation_smooth_bins", 2)),
            )
            S_mode = (mode_mask[:, None]) * S
            y_mode = librosa.istft(
                S_mode,
                hop_length=cfg.hop,
                win_length=cfg.n_fft,
                window="hann",
                center=True,
                length=len(x),
            )
        op_lo, op_hi = cfg.operating_band
        band_mask = (freqs >= op_lo) & (freqs <= op_hi)
        # 2) Rain detection
        P_for_detection = P
        if mode_iso_enable and cfg.mode_isolation_affects_classifier:
            P_for_detection = np.abs(S_mode) ** 2

        is_rain, det_debug, rain_conf, noise_conf = self._detect_rain_over_time(P_for_detection, freqs)

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

        # Prefer explicit PSD gating from the classifier if provided
        if isinstance(det_debug, dict) and "use_for_noise_psd" in det_debug:
            use_for_noise_psd = np.asarray(det_debug["use_for_noise_psd"], dtype=bool)
        else:
            # Conservative fallback: treat high rain_conf as rain-ish (so NOT used for PSD)
            th = float(getattr(cfg, "rain_like_thresh", 0.8))
            use_for_noise_psd = ~(rain_conf >= th)

        is_rain_for_psd = ~use_for_noise_psd

        # Optional hold: exclude a couple frames after rain spikes
        hold = int(getattr(cfg, "rain_hold_frames", 0))
        if hold > 0 and is_rain_for_psd.any():
            is_rain_for_psd = ndi.binary_dilation(is_rain_for_psd, iterations=hold)

        # 3) Noise PSD estimation (MCRA / quantile)
        if cfg.noise_psd_mode == "fft":
            noise_psd = self._estimate_noise_psd_fft(P, freqs, is_rain_for_psd)
            mel_filter = None
        elif cfg.noise_psd_mode == "mel":
            noise_psd, mel_filter = self._estimate_noise_psd_mel(P, freqs, is_rain_for_psd, sr=sr)
        else:
            raise ValueError(f"Unknown PSD mode: {cfg.noise_psd_mode}")

        # 4) Gain computation (confidence-weighted)
        P_band_all = P[band_mask]
        N_band_all = noise_psd[band_mask]

        if getattr(cfg, "debug_enable", False):
            ratio_med = float(np.median(N_band_all / (P_band_all + 1e-12)))
            ratio_p90 = float(np.percentile(N_band_all / (P_band_all + 1e-12), 90))
            print("median N/P:", ratio_med, "p90 N/P:", ratio_p90)

        # Gain computation (confidence-weighted)
        G_band = self._compute_gain(P_band_all, N_band_all, noise_conf=noise_conf)

        G = np.ones_like(P)
        G[band_mask] = G_band

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
        # - if mode isolation experiment enabled, output mode-only reconstruction
        # - otherwise output the normal noise-suppressed waveform
        y_out = y_mode if (mode_iso_enable and y_mode is not None) else y_hat
        # if mode_iso_enable and cfg.mode_isolation_only:
        #     y_out = y_mode
        # else:
        #     y_out = y_hat

        # 6) Debug
        op_lo, op_hi = cfg.operating_band

        # Keep detector debug under a dedicated key to avoid collisions and make plots simpler.
        debug = {
            "detector": det_debug,

            # Common time axis and frequency axis
            "times_s": times_s,
            "freqs": freqs,

            # Classifier outputs
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "use_for_noise_psd": use_for_noise_psd,
            "is_rain_for_psd": is_rain_for_psd,

            # Suppression / PSD
            "G": G,
            "noise_psd": noise_psd,

            # Band metadata
            "operating_band": (float(op_lo), float(op_hi)),
            "band_mask": band_mask,

            # Pre-filter info for debug/tuning/plots
            "pre_filter_mode": mode,
            "pre_filter_band": (float(cfg.operating_band[0]), float(cfg.operating_band[1])),

            # Key thresholds for plotting reference
            "rain_hi": _as_float(getattr(cfg, "rain_hi", np.nan)),
            "noise_hi": _as_float(getattr(cfg, "noise_hi", np.nan)),
            "noise_update_thresh": _as_float(getattr(cfg, "noise_update_thresh", np.nan)),
            "primary_rise_db_thresh": _as_float(getattr(cfg, "primary_rise_db_thresh", np.nan)),
            "peaks_in_mode_min": int(getattr(cfg, "peaks_in_mode_min", 0)),
            
            # TEMP experiment: mode-only reconstruction
            "mode_isolation_enable": mode_iso_enable,
            "mode_isolation_attn_db": _as_float(getattr(cfg, "mode_isolation_attn_db", np.nan)),
            "mode_isolation_smooth_bins": int(getattr(cfg, "mode_isolation_smooth_bins", 0)),
            "mode_isolation_mask": mode_mask,
            "S_mode": S_mode,

        }

        # Optional: include mel filter if present
        if mel_filter is not None:
            debug["mel_filter"] = mel_filter

        return {
            "y": y_out,            # what your harness will treat as "denoised_audio"
            "y_suppressed": y_hat, # always available for A/B
            "y_mode": y_mode,      # mode-only listening output
            "S": S,
            "S_hat": S_hat,
            "noise_psd": noise_psd,
            "is_rain": is_rain,
            "freqs": freqs,
            "times": times,
            # Back-compat convenience: top-level detector keys (if present)
            "det_debug": det_debug,
            "debug": debug,
            "x_hp": x_proc,   # back-compat name
            "x_filt": x_proc, # clearer name (HPF/BPF/none)
        }