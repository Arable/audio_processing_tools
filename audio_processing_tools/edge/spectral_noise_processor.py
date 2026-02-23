# edge/spectral_noise_processor.py

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
# spectral_noise_processor.py

from dataclasses import dataclass


import numpy as np
import scipy.signal as spsig
import scipy.ndimage as ndi
import librosa

from .noise_defaults import NoiseProcessorConfig, build_noise_config
from .rain_frame_classifier import RainFrameClassifierMixin  # if in separate module
# or: from .spectral_noise_processor import RainFrameClassifierMixin
#      if you keep the mixin in the same file above the class.


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
        self._is_setup = False

    def setup(self, params: Dict[str, Any]):
        """
        Build NoiseProcessorConfig from framework params.
        Called once per parameter change / file batch.
        """
        if self._is_setup:
            return

        sr = int(params.get("sample_rate", params.get("fs", 11162)))

        self.cfg = build_noise_config(
            sample_rate=sr,
            params=params,
        )

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
        F, T = P.shape
        eps = cfg.eps
        n_mels = getattr(cfg, "n_mels", 24)

        if sr is None:
            sr = cfg.fs  # but ideally sr should match the STFT sr

        W_mel = librosa.filters.mel(
            sr=sr,
            n_fft=cfg.n_fft,
            n_mels=n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
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

        ema_up = float(getattr(cfg, "ema_up", cfg.ema_lambda))
        ema_down = float(getattr(cfg, "ema_down", cfg.ema_lambda))

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

        band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
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
        F, T = P.shape
        eps = cfg.eps

        band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
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

        ema_up = float(getattr(cfg, "ema_up", cfg.ema_lambda))
        ema_down = float(getattr(cfg, "ema_down", cfg.ema_lambda))

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
        cfg = self.cfg
        if cfg is None:
            raise ValueError(
                "SpectralNoiseProcessor.cfg is None. Pass a NoiseProcessorConfig or call setup(params) before process()."
            )

        if sr is None:
            sr = cfg.fs

        # Ensure rain-frame classifier requirements are present on cfg
        if not getattr(self, "_is_setup", False):
            self._validate_rain_cfg()
            self._is_setup = True

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
        band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
        # 2) Rain detection
        is_rain, det_debug, rain_conf, noise_conf = self._detect_rain_over_time(P, freqs)

        th = float(getattr(cfg, "rain_like_thresh", 0.8))

        # Prefer the classifier’s own PSD-gating signal when present (new 3-state path)
        if isinstance(det_debug, dict) and ("use_for_noise_psd" in det_debug):
            use_for_noise_psd = np.asarray(det_debug["use_for_noise_psd"], dtype=bool)
            # is_rain_for_psd = frames we EXCLUDE from PSD updates
            is_rain_for_psd = ~use_for_noise_psd
        else:
            # Legacy fallback: treat high rain_conf as rain-ish
            is_rain_for_psd = (np.asarray(rain_conf) >= th)

            # Optional hold: exclude a couple frames after rain spikes
            hold = int(getattr(cfg, "rain_hold_frames", 0))
            if hold > 0 and is_rain_for_psd.any():
                # "dilate" rain mask forward/backward a bit
                is_rain_for_psd = ndi.binary_dilation(is_rain_for_psd, iterations=hold)
        
        # 3) Noise PSD estimation (MCRA / quantile)
        if cfg.noise_psd_mode == "fft":
            noise_psd = self._estimate_noise_psd_fft(P, freqs, is_rain_for_psd)
            mel_filter = None
        elif cfg.noise_psd_mode == "mel":
            noise_psd, mel_filter = self._estimate_noise_psd_mel(P, freqs, is_rain_for_psd, sr=sr)
        
        # 4) Gain computation (confidence-weighted, with mode-band protection)
        P_band_all = P[band_mask]
        N_band_all = noise_psd[band_mask]

        ratio_med = np.median(N_band_all / (P_band_all + 1e-12))
        ratio_p90 = np.percentile(N_band_all / (P_band_all  + 1e-12), 90)
        print("median N/P:", ratio_med, "p90 N/P:", ratio_p90)

        # Confidence-weighted gain (reduce suppression on rain-like frames)
        G_band = self._compute_gain(P_band_all, N_band_all, noise_conf=noise_conf)

        G = np.ones_like(P)
        G[band_mask] = G_band

        # --- Mode-band protection on rain-like frames ---
        # When rain is likely, avoid suppressing the rain mode bands so drops remain measurable.
        enable_protect = bool(getattr(cfg, "mode_protect_enable", True))
        if enable_protect and hasattr(cfg, "mode_bands"):
            mode_mask = np.zeros_like(freqs, dtype=bool)
            for (lo, hi) in cfg.mode_bands:
                mode_mask |= (freqs >= float(lo)) & (freqs <= float(hi))

            # Define "rain-like" frames for protection
            protect_thresh = float(getattr(cfg, "mode_protect_rain_thresh", th))
            rain_like = (np.asarray(rain_conf) >= protect_thresh)

            protect_gain = float(getattr(cfg, "mode_protect_gain", 1.0))
            protect_gain = float(np.clip(protect_gain, cfg.gain_floor, cfg.gain_ceil))

            if rain_like.any() and mode_mask.any():
                # Ensure gain in mode bands is not driven too low on rain-like frames
                G[np.ix_(mode_mask, rain_like)] = np.maximum(G[np.ix_(mode_mask, rain_like)], protect_gain)

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
            "is_rain_for_psd": is_rain_for_psd,
            "G": G,
            "noise_psd": noise_psd,
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

  # def _estimate_noise_psd_mel(
    #     self,
    #     P: np.ndarray,            # (F,T)
    #     freqs: np.ndarray,        # (F,)
    #     is_rain: np.ndarray,      # (T,)
    #     sr: Optional[int] = None, # <-- add this if you can
    # ) -> Tuple[np.ndarray, np.ndarray]:

    #     cfg = self.cfg
    #     F, T = P.shape
    #     eps = cfg.eps
    #     n_mels = getattr(cfg, "n_mels", 24)

    #     if sr is None:
    #         sr = cfg.fs  # but ideally sr should match the STFT sr

    #     W_mel = librosa.filters.mel(
    #         sr=sr,
    #         n_fft=cfg.n_fft,
    #         n_mels=n_mels,
    #         fmin=cfg.fmin,
    #         fmax=cfg.fmax,
    #     ).astype(np.float64)  # (B,F)

    #     P_mel = W_mel @ P  # (B,T)

    #     # --- optional time smoothing ---
    #     L = int(getattr(cfg, "pre_smooth_frames", 0))
    #     if L and L > 1:
    #         P_mel = self._time_smooth(P_mel, L)

    #     B = P_mel.shape[0]

    #     frames_per_sec = sr / cfg.hop
    #     W_frames = max(10, int(cfg.win_sec * frames_per_sec))

    #     buffer = np.full((W_frames, B), np.nan, dtype=np.float64)
    #     buf_idx = 0
    #     buf_count = 0

    #     noise_mel = np.zeros_like(P_mel)

    #     ema_up = float(getattr(cfg, "ema_up", cfg.ema_lambda))
    #     ema_down = float(getattr(cfg, "ema_down", cfg.ema_lambda))

    #     for t in range(T):
    #         Pm = P_mel[:, t]

    #         if not is_rain[t]:
    #             buffer[buf_idx] = Pm
    #             buf_idx = (buf_idx + 1) % W_frames
    #             buf_count = min(buf_count + 1, W_frames)

    #         if buf_count == 0:
    #             raw_q = Pm
    #         else:
    #             valid = buffer[:buf_count]
    #             raw_q = np.nanquantile(valid, cfg.q, axis=0)

    #         if t == 0:
    #             N_mel = raw_q
    #         else:
    #             prev = noise_mel[:, t - 1]
    #             lam = np.where(raw_q > prev, ema_up, ema_down)
    #             N_mel = lam * prev + (1.0 - lam) * raw_q

    #         noise_mel[:, t] = N_mel

    #     # optional median filter
    #     m = int(getattr(cfg, "median_frames", 0))
    #     if m and m > 1:
    #         if m % 2 == 0:
    #             m += 1
    #         noise_mel = ndi.median_filter(noise_mel, size=(1, m))

    #     # --- expand mel → FFT (normalize per FFT bin = column sum of W_mel) ---
    #     den = (W_mel.sum(axis=0, keepdims=True).T + eps)  # (F,1)
    #     noise_psd_fft = (W_mel.T @ noise_mel) / den       # (F,T)

    #     band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
    #     noise_psd_fft[~band_mask, :] = 0.0

    #     # debug ratios (now consistent with smoothing)
    #     ratio_mel = noise_mel / (P_mel + eps)
    #     print("median N/P mel:", np.median(ratio_mel), "p90:", np.percentile(ratio_mel, 90))
       
    #     ratio_fft_from_mel = noise_psd_fft[band_mask] / (P[band_mask] + eps)
    #     print("median N/P fft(from mel):", np.median(ratio_fft_from_mel),
    #     "p90:", np.percentile(ratio_fft_from_mel, 90))

    #     return noise_psd_fft, W_mel
   # # ----------------------- Noise PSD (mel-based) -----------------------

    # def _estimate_noise_psd_mel(
    #     self,
    #     P: np.ndarray,         # (F, T)
    #     freqs: np.ndarray,     # (F,)  (unused here but kept for symmetry)
    #     is_rain: np.ndarray,   # (T,)
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Estimate noise PSD via mel bands:

    #       1) Project P to mel bands → P_mel (B, T)
    #       2) Run min-stats / quantile tracking in mel domain
    #       3) Expand noise PSD back to FFT bins using the mel filterbank

    #     Returns:
    #         noise_psd_fft : (F, T)
    #         mel_filter    : (B, F)
    #     """
    #     cfg = self.cfg
    #     F, T = P.shape
    #     eps = cfg.eps

    #     n_mels = getattr(cfg, "n_mels", 24)

    #     mel_filter = librosa.filters.mel(
    #         sr=cfg.fs,
    #         n_fft=cfg.n_fft,
    #         n_mels=n_mels,
    #         fmin=cfg.fmin,
    #         fmax=cfg.fmax,
    #     )  # (B, F)

    #     P_mel = mel_filter @ P  # (B, T)
    #     B = P_mel.shape[0]

    #     frames_per_sec = cfg.fs / cfg.hop
    #     W = max(10, int(cfg.win_sec * frames_per_sec))

    #     buffer_mel = np.full((W, B), np.nan, dtype=np.float64)
    #     buf_idx = 0
    #     buf_count = 0

    #     noise_mel = np.zeros_like(P_mel)

    #     for t in range(T):
    #         Pm_frame = P_mel[:, t]

    #         if not is_rain[t]:
    #             buffer_mel[buf_idx] = Pm_frame
    #             buf_idx = (buf_idx + 1) % W
    #             buf_count = min(buf_count + 1, W)

    #         if buf_count == 0:
    #             raw_q = Pm_frame.copy()
    #         else:
    #             valid = buffer_mel[:buf_count]
    #             raw_q = np.nanquantile(valid, cfg.q, axis=0)

    #         if t == 0:
    #             N_mel = raw_q
    #         else:
    #             prev_mel = noise_mel[:, t - 1]
    #             N_mel = cfg.ema_lambda * prev_mel + (1.0 - cfg.ema_lambda) * raw_q

    #         noise_mel[:, t] = N_mel

    #     m = getattr(cfg, "median_frames", 0)
    #     if m and m > 1:
    #         if m % 2 == 0:
    #             m += 1
    #         noise_mel = ndi.median_filter(noise_mel, size=(1, m))

    #     # Expand mel → FFT
    #     weights = mel_filter.T  # (F, B)

    #     num = weights @ noise_mel          # (F, T)
    #     den = np.sum(weights, axis=1, keepdims=True) + eps

    #     noise_psd_fft = num / den
    #     return noise_psd_fft, mel_filter

    # # ----------------------- Noise PSD (FFT-based) -----------------------

    # def _estimate_noise_psd_fft(
    #     self,
    #     P: np.ndarray,         # (F, T)
    #     freqs: np.ndarray,     # (F,)
    #     is_rain: np.ndarray,   # (T,)
    # ) -> np.ndarray:
    #     """
    #     Estimate noise PSD directly per FFT bin in the operating band
    #     via min-stats / quantile tracking over non-rain frames.
    #     """
    #     cfg = self.cfg
    #     F, T = P.shape
    #     eps = cfg.eps

    #     band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
    #     K = int(band_mask.sum())

    #     frames_per_sec = cfg.fs / cfg.hop
    #     W = max(10, int(cfg.win_sec * frames_per_sec))

    #     buffer = np.full((W, K), np.nan, dtype=np.float64)
    #     buf_idx = 0
    #     buf_count = 0

    #     noise_psd = np.zeros_like(P)

    #     for t in range(T):
    #         P_frame = P[:, t]
    #         P_band = P_frame[band_mask]

    #         if not is_rain[t]:
    #             buffer[buf_idx] = P_band
    #             buf_idx = (buf_idx + 1) % W
    #             buf_count = min(buf_count + 1, W)

    #         if buf_count == 0:
    #             raw_q = P_band.copy()
    #         else:
    #             valid = buffer[:buf_count]
    #             raw_q = np.nanquantile(valid, cfg.q, axis=0)

    #         if t == 0:
    #             N_band = raw_q
    #         else:
    #             prev_band = noise_psd[band_mask, t - 1]
    #             N_band = cfg.ema_lambda * prev_band + (1.0 - cfg.ema_lambda) * raw_q

    #         noise_psd[band_mask, t] = N_band

    #     m = getattr(cfg, "median_frames", 0)
    #     if m and m > 1:
    #         if m % 2 == 0:
    #             m += 1
    #         noise_psd = ndi.median_filter(noise_psd, size=(1, m))

    #     return noise_psd
