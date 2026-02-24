# edge/spectral_noise_processor.py

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal as spsig
import scipy.ndimage as ndi
import librosa

from .rain_frame_classifier import RainFrameClassifierMixin

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

    # -----------------------------------------------------------
    # Operating band (noise estimation + gain applied here)
    # -----------------------------------------------------------
    # NOTE: `operating_band` is the canonical band used everywhere.
    # `fmin`/`fmax` are kept ONLY for backward-compatibility with older params.
    operating_band: Tuple[float, float] = (400.0, 3500.0)

    # Back-compat only (deprecated): if provided, they will be reconciled into operating_band
    fmin: float = 400.0
    fmax: float = 3500.0

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

    # Reconcile back-compat fmin/fmax with operating_band
    # Priority: explicit operating_band wins; otherwise derive from fmin/fmax.
    if getattr(cfg, "operating_band", None) is not None:
        op_lo, op_hi = cfg.operating_band
        cfg.fmin = float(op_lo)
        cfg.fmax = float(op_hi)
    else:
        cfg.operating_band = (float(cfg.fmin), float(cfg.fmax))

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
        ratio_mel = noise_mel / (P_mel + eps)
        print("median N/P mel:", np.median(ratio_mel), "p90:", np.percentile(ratio_mel, 90))

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

        # 0) High-pass filter
        x_proc = x
        if cfg.hp_cutoff_hz > 0:
            nyq = 0.5 * sr
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
        # 2) Rain detection
        is_rain, det_debug, rain_conf, noise_conf = self._detect_rain_over_time(P, freqs)

        # Prefer explicit PSD gating from the classifier if provided
        if isinstance(det_debug, dict) and "use_for_noise_psd" in det_debug:
            use_for_noise_psd = np.asarray(det_debug["use_for_noise_psd"], dtype=bool)
            is_rain_for_psd = ~use_for_noise_psd
        else:
            # Conservative fallback: treat high rain_conf as rain-ish
            th = float(getattr(cfg, "rain_like_thresh", 0.8))
            is_rain_for_psd = (rain_conf >= th)

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

        ratio_med = np.median(N_band_all / (P_band_all + 1e-12))
        ratio_p90 = np.percentile(N_band_all / (P_band_all  + 1e-12), 90)
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
        )

        # 6) Debug
        debug = {
            **det_debug,
            "rain_conf": rain_conf,
            "noise_conf": noise_conf,
            "G": G,
            "noise_psd": noise_psd,
            "is_rain_for_psd": is_rain_for_psd,
        }
        if mel_filter is not None:
            debug["mel_filter"] = mel_filter

        return {
            "y": y_hat,
            "S": S,
            "S_hat": S_hat,
            "noise_psd": noise_psd,
            "is_rain": is_rain,
            "freqs": freqs,
            "times": times,
            "debug": debug,
            "x_hp": x_proc,
        }