# noise_processor.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import librosa
import scipy.ndimage as ndi
from scipy import signal as spsig


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
    fmin: float = 400.0
    fmax: float = 3500.0

    # -----------------------------------------------------------
    # Rain detection bands
    # -----------------------------------------------------------
    rain_band: Tuple[float, float] = (400.0, 600.0)
    adj_band: Tuple[float, float] = (600.0, 800.0)

    # Rain detection thresholds:
    #   dAB = L_A - L_B
    #   drise = L_A(t) - L_A(t-1)
    N_db: float = 6.0      # dAB ≥ N_db
    M_db: float = 6.0      # drise ≥ M_db

    # -----------------------------------------------------------
    # Noise tracking (quantile/min-stats)
    # -----------------------------------------------------------
    q: float = 0.25        # quantile level
    win_sec: float = 0.125 # rolling window size (seconds)

    # Smoothing for PSD evolution
    ema_lambda: float = 0.80

    # Numerical stability
    eps: float = 1e-9

    # -----------------------------------------------------------
    # Confidence threshold bands for soft scoring
    # These define the "low" and "high" regions for soft sigmoid weights.
    # -----------------------------------------------------------
    dab_conf_low: float = 3.0
    dab_conf_high: float = 6.0
    drise_conf_low: float = 3.0
    drise_conf_high: float = 6.0

    # Weights for spectral + temporal cues
    rain_conf_w_spec: float = 0.4
    rain_conf_w_rise: float = 0.6

    # EMA smoothing of confidence to stabilize the adaptive gain
    rain_conf_smooth_alpha: float = 0.8

    # -----------------------------------------------------------
    # Adaptive oversubtraction
    # oversub = oversub_base + noise_conf * (oversub_max - oversub_base)
    # -----------------------------------------------------------
    oversub_base: float = 1.0   # mild suppression when rain likely
    oversub_max: float = 10.0    # strong suppression when noise likely

    # Gain clamp to avoid instability / musical noise
    gain_floor: float = 0.0
    gain_ceil: float = 1.0
    gain_smooth_alpha: float = 0.7

    # select between sqrt_sub and wiener
    gain_mode: str = "sqrt_sub"      # or "wiener"
    gain_smooth_alpha: float = 0.7   # EMA for temporal smoothing
    rain_like_thresh: float = 0.7

class SpectralNoiseProcessor:
    """
    Noise processor:
      - STFT via librosa
      - Raindrop frame detection using primary band 400–600 vs 600–800 Hz
      - Noise PSD estimation for 400–3500 Hz using quantile tracking
        (skipping frames likely containing raindrops)
      - Adaptive, confidence-weighted Wiener-style suppression
      - ISTFT to get noise-suppressed waveform

    Intended as a plug-in "noise processor" in your audio-processing framework.
    """

    def __init__(self, config: NoiseProcessorConfig):
        self.cfg = config

    # ------------- helpers -------------

    @staticmethod
    def _band_db(P_frame: np.ndarray, freqs: np.ndarray, band: tuple, eps: float) -> float:
        """Compute dB power in a [f_lo, f_hi) band."""
        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return -np.inf
        val = P_frame[mask].sum()
        return 10.0 * np.log10(val + eps)

    def _detect_rain_frame(
        self,
        P_frame: np.ndarray,
        freqs: np.ndarray,
        prev_LA: Optional[float],
    ) -> Dict[str, Any]:
        """
        Basic rain-frame detector based on the primary band (rain_band)
        versus an adjacent band (adj_band).

        Returns LA, LB, dAB, drise and a hard boolean 'is_rain' gate.
        """
        cfg = self.cfg
        LA = self._band_db(P_frame, freqs, cfg.rain_band, cfg.eps)
        LB = self._band_db(P_frame, freqs, cfg.adj_band, cfg.eps)

        if prev_LA is None or np.isneginf(LA) or np.isneginf(LB):
            return {
                "is_rain": False,
                "LA": LA,
                "LB": LB,
                "dAB": 0.0,
                "drise": 0.0,
            }

        dAB = LA - LB
        drise = LA - prev_LA

        # Hard gate (can be used for noise PSD update masking)
        is_rain = (dAB >= cfg.N_db) and (drise >= cfg.M_db)

        return {
            "is_rain": is_rain,
            "LA": LA,
            "LB": LB,
            "dAB": dAB,
            "drise": drise,
        }

    @staticmethod
    def _soft_score(value: float, low: float, high: float) -> float:
        """
        Map a scalar value into [0,1] using a linear ramp between [low, high].

        value <= low  -> 0
        value >= high -> 1
        in between    -> (value - low) / (high - low)
        """
        if not np.isfinite(value):
            return 0.0
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    def _compute_rain_noise_confidence(
        self,
        dAB: float,
        drise: float,
        prev_rain_conf: Optional[float],
    ) -> (float, float):
        """
        Compute smoothed rain_conf and noise_conf in [0,1].

        rain_conf is high when:
          - dAB (LA-LB) is large
          - drise (LA(t)-LA(t-1)) is large

        noise_conf = 1 - rain_conf.
        """
        cfg = self.cfg

        s_spec = self._soft_score(dAB, cfg.dab_conf_low,  cfg.dab_conf_high)
        s_rise = self._soft_score(drise, cfg.drise_conf_low, cfg.drise_conf_high)

        raw_rain_conf = (
            cfg.rain_conf_w_spec * s_spec +
            cfg.rain_conf_w_rise * s_rise
        )

        if prev_rain_conf is None:
            rain_conf = raw_rain_conf
        else:
            a = cfg.rain_conf_smooth_alpha
            rain_conf = a * prev_rain_conf + (1.0 - a) * raw_rain_conf

        rain_conf = float(np.clip(rain_conf, 0.0, 1.0))
        noise_conf = 1.0 - rain_conf
        return rain_conf, noise_conf

    def _compute_gain(
        self,
        P_band: np.ndarray,     # (K, T)
        N_band: np.ndarray,     # (K, T)
        noise_conf: np.ndarray, # (T,)
    ) -> np.ndarray:
        """
        Compute smoothed gain G_band for the operating band.

        P_band    : signal power in operating band  (K, T)
        N_band    : noise PSD estimate in band      (K, T)
        noise_conf: per-frame noise confidence      (T,), in [0, 1]
                    0 = definitely rain, 1 = definitely noise
        """
        cfg = self.cfg
        eps = cfg.eps

        K, T = P_band.shape
        assert N_band.shape == (K, T)
        assert noise_conf.shape[0] == T

        noise_conf = np.clip(noise_conf, 0.0, 1.0)
        th = getattr(cfg, "rain_like_thresh", 0.7)

        # -------------------------------------------------
        # 1) Base gain (mode-dependent) + per-frame oversub
        # -------------------------------------------------
        # Effective "noise-ness" only above threshold:
        #   noise_conf <= th → eff_noise = 0
        #   noise_conf  > th → eff_noise in (0,1]
        denom = max(1e-9, 1.0 - th)
        eff_noise = np.clip((noise_conf - th) / denom, 0.0, 1.0)  # (T,)

        oversub = cfg.oversub_base + eff_noise * (cfg.oversub_max - cfg.oversub_base)
        oversub_2d = oversub[None, :]   # (1, T) → (K, T) broadcast

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

        # Clip to [gain_floor, gain_ceil]
        G_raw = np.clip(G_raw, cfg.gain_floor, cfg.gain_ceil)

        # -------------------------------------------------
        # 2) Frequency smoothing (3-tap kernel; none on rain-like frames)
        # -------------------------------------------------
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        G_freq = np.empty_like(G_raw)

        for t in range(T):
            if noise_conf[t] < th:
                # rain-like frame: do NOT blur peaks across frequency
                G_freq[:, t] = G_raw[:, t]
            else:
                # noise-like: smooth over neighbouring bins
                G_freq[:, t] = np.convolve(G_raw[:, t], kernel, mode="same")

        # -------------------------------------------------
        # 3) Temporal smoothing (reduced / off for rain-like frames)
        # -------------------------------------------------
        alpha_base = float(np.clip(cfg.gain_smooth_alpha, 0.0, 1.0))

        G_time = np.empty_like(G_freq)
        G_time[:, 0] = G_freq[:, 0]

        for t in range(1, T):
            nc = noise_conf[t]

            if nc < th:
                # rain-like: no temporal smoothing
                alpha_t = 0.0
            else:
                # map nc ∈ [th, 1] → eff_nc ∈ [0, 1], then scale by alpha_base
                eff_nc = (nc - th) / denom
                alpha_t = alpha_base * eff_nc

            G_time[:, t] = alpha_t * G_time[:, t - 1] + (1.0 - alpha_t) * G_freq[:, t]

            # ensure smoothing never *reduces* gain on rain-like frames
            if nc < th:
                G_time[:, t] = np.maximum(G_time[:, t], G_freq[:, t])

        # Final safety clip
        G_time = np.clip(G_time, cfg.gain_floor, cfg.gain_ceil)
        return G_time
    
    # ------------- main API -------------

    def process(
        self,
        x: np.ndarray,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main API entry point for the noise processor.

        Args:
            x: 1D numpy array of audio samples (time-domain).
            sr: Sampling rate. If None, uses config.fs (11162 Hz).

        Returns:
            dict with:
              - 'y': noise-suppressed waveform
              - 'S': original complex STFT (of high-passed signal)
              - 'S_hat': noise-suppressed complex STFT
              - 'noise_psd': [F, T] noise PSD
              - 'is_rain': boolean [T] rain-frame mask
              - 'freqs': frequency bins (Hz)
              - 'times': frame times (sec)
              - 'debug': dict with LA, LB, dAB, drise, G, rain_conf, noise_conf
              - 'x_hp': high-passed input
        """
        cfg = self.cfg
        if sr is None:
            sr = cfg.fs
        else:
            if sr != cfg.fs:
                # In production, you might resample here. For now we assume they match.
                pass

        # -------------------------------
        # 0) Optional high-pass filtering
        # -------------------------------
        x_proc = x
        if getattr(cfg, "hp_cutoff_hz", None) is not None and cfg.hp_cutoff_hz > 0:
            nyq = 0.5 * sr
            norm_cut = cfg.hp_cutoff_hz / nyq
            norm_cut = min(max(norm_cut, 1e-4), 0.9999)
            sos = spsig.butter(cfg.hp_order, norm_cut, btype="highpass", output="sos")
            x_proc = spsig.sosfiltfilt(sos, x_proc)

        # -------------------------------
        # 1) STFT (librosa) on high-passed signal
        # -------------------------------
        S = librosa.stft(
            x_proc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
        )
        P = np.abs(S) ** 2  # power spectrogram
        freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
        times = librosa.frames_to_time(np.arange(P.shape[1]), sr=sr, hop_length=cfg.hop)

        F, T = P.shape
        band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
        K = int(band_mask.sum())

        # history length (frames) for quantile tracking
        frames_per_sec = sr / cfg.hop
        W = max(10, int(cfg.win_sec * frames_per_sec))

        # noise history buffer for operating band
        buffer = np.full((W, K), np.nan, dtype=np.float64)
        buf_idx = 0
        buf_count = 0

        noise_psd = np.zeros_like(P)
        is_rain = np.zeros(T, dtype=bool)

        debug_LA = np.full(T, np.nan)
        debug_LB = np.full(T, np.nan)
        debug_dAB = np.full(T, np.nan)
        debug_drise = np.full(T, np.nan)
        debug_rain_conf = np.full(T, np.nan)
        debug_noise_conf = np.full(T, np.nan)

        prev_LA: Optional[float] = None
        prev_rain_conf: Optional[float] = None

        # -------------------------------
        # 2) Per-frame noise tracking + confidence
        # -------------------------------
        for t_idx in range(T):
            P_frame = P[:, t_idx]

            # ---- Rain detection ----
            det = self._detect_rain_frame(P_frame, freqs, prev_LA)
            is_rain_frame = det["is_rain"]
            LA = det["LA"]
            LB = det["LB"]
            dAB = det["dAB"]
            drise = det["drise"]

            prev_LA = LA

            # ---- Confidence (rain & noise) ----
            rain_conf, noise_conf = self._compute_rain_noise_confidence(
                dAB, drise, prev_rain_conf
            )
            prev_rain_conf = rain_conf

            is_rain[t_idx] = is_rain_frame
            debug_LA[t_idx] = LA
            debug_LB[t_idx] = LB
            debug_dAB[t_idx] = dAB
            debug_drise[t_idx] = drise
            debug_rain_conf[t_idx] = rain_conf
            debug_noise_conf[t_idx] = noise_conf

            # ---- Noise update (on operating band only) ----
            P_band = P_frame[band_mask]

            # Optionally gate noise tracking by is_rain_frame
            if not is_rain_frame:
                buffer[buf_idx] = P_band
                buf_idx = (buf_idx + 1) % W
                buf_count = min(buf_count + 1, W)

            if buf_count == 0:
                raw_q = P_band.copy()
            else:
                valid = buffer[:buf_count]
                raw_q = np.nanquantile(valid, cfg.q, axis=0)

            if t_idx == 0:
                N_band = raw_q
            else:
                prev_band = noise_psd[band_mask, t_idx - 1]
                N_band = cfg.ema_lambda * prev_band + (1.0 - cfg.ema_lambda) * raw_q

            noise_psd[band_mask, t_idx] = N_band

        # No additional median filter: noise_psd_sm = noise_psd
        noise_psd_sm = noise_psd

        # -------------------------------
        # 3) Gain on operating band
        # -------------------------------
        G = np.ones_like(P)
        P_band_all = P[band_mask]              # (K, T)
        N_band_all = noise_psd_sm[band_mask]   # (K, T)

        noise_conf_vec = debug_noise_conf      # length T
        G_band = self._compute_gain(P_band_all, N_band_all, noise_conf_vec)
        G[band_mask] = G_band

        # -------------------------------
        # 4) Apply gain & ISTFT
        # -------------------------------
        S_hat = G * S
        y_hat = librosa.istft(
            S_hat,
            hop_length=cfg.hop,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
        )

        debug = {
            "LA": debug_LA,
            "LB": debug_LB,
            "dAB": debug_dAB,
            "drise": debug_drise,
            "G": G,
            "rain_conf": debug_rain_conf,
            "noise_conf": debug_noise_conf,
        }

        return {
            "y": y_hat,
            "S": S,
            "S_hat": S_hat,
            "noise_psd": noise_psd_sm,
            "is_rain": is_rain,
            "freqs": freqs,
            "times": times,
            "debug": debug,
            "x_hp": x_proc,
        }

