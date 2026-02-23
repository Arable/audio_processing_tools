"""
noise_defaults.py

Defines:
    - DEFAULT_NOISE_PROCESSOR_PARAMS: default settings for the spectral noise processor
    - build_noise_config(): utility to merge defaults with parameter overrides
"""

from typing import Dict, Any


from dataclasses import dataclass
from typing import Dict, Any, Tuple


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
    q: float = 0.25        # quantile level. 0.4 as default
    win_sec: float = 0.5 # rolling window size (seconds). 1.0 as default

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
    rain_conf_w_spec: float = 0.4 # 0.3 as default
    rain_conf_w_rise: float = 0.6 # 0.7 as default

    # EMA smoothing of confidence to stabilize the adaptive gain
    rain_conf_smooth_alpha: float = 0.8

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
    noise_psd_mode: str = "fft"  # or "fft"

    # New / refined PSD tracking params
    pre_smooth_frames: int = 0      # try 3–5, 0 disables
    ema_up: float = 0.6             # fast when noise increases
    ema_down: float = 0.95          # slow when noise decreases

    # ---------------- mode-frequency evidence (new) ----------------
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

    # ---------------- 3-state classification (new) ----------------
    rain_hi: float = 0.80
    noise_hi: float = 0.85
    noise_update_thresh: float = 0.85

    rain_hold_frames: int = 1
    noise_confirm_frames: int = 2

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Tuple

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

        # optional: normalize a few common tuple-ish fields
        if k in ("rain_band", "adj_band") and isinstance(v, (list, tuple)) and len(v) == 2:
            v = (float(v[0]), float(v[1]))

        if k == "mode_bands":
            # accept list[list[2]] / list[tuple[2]] / tuple[tuple[2]]
            v = tuple((float(a), float(b)) for (a, b) in v)

        if k == "mode_weights":
            v = tuple(float(x) for x in v)

        setattr(cfg, k, v)

    return cfg


__all__ = [
    "NoiseProcessorConfig",
    "build_noise_config",
]


# DEFAULT_NOISE_PROCESSOR_PARAMS: Dict[str, Any] = {
#     # -------------------------------------------------------
#     # STFT parameters
#     # -------------------------------------------------------
#     "n_fft": 256,
#     "hop": 128,

#     # -------------------------------------------------------
#     # Operating band for estimation + suppression
#     # -------------------------------------------------------
#     "fmin": 400.0,
#     "fmax": 3500.0,

#     # -------------------------------------------------------
#     # Rain detection bands
#     # -------------------------------------------------------
#     "rain_band": (400.0, 600.0),
#     "adj_band": (600.0, 800.0),

#     # Rain-frame gating thresholds
#     "N_db": 6.0,    # spectral peak difference
#     "M_db": 6.0,    # inter-frame rise difference

#     # -------------------------------------------------------
#     # Noise PSD estimation
#     # -------------------------------------------------------
#     "q": 0.4,        # quantile used for min-stats
#     "win_sec": 1.0, # rolling window length (seconds)

#     # Temporal smoothing
#     "median_frames": 0,     # disabled = 0
#     "ema_lambda": 0.80,     # exponential smoothing for raw noise PSD

#     # Numerical stability
#     "eps": 1e-9,

#     # -------------------------------------------------------
#     # High-pass filter
#     # -------------------------------------------------------
#     "hp_cutoff_hz": 350.0,
#     "hp_order": 4,

#     # PSD smoothing (optional, light)
#     "psd_smooth_alpha": 0.1,

#     # -------------------------------------------------------
#     # NEW: Soft confidence + adaptive suppression parameters
#     # -------------------------------------------------------

#     # weights for rain confidence components
#     "rain_conf_w_spec": 0.3,     # weight for dAB
#     "rain_conf_w_rise": 0.7,     # weight for drise

#     # EMA smoothing factor for confidence
#     "rain_conf_smooth_alpha": 0.8,

#     # adaptive oversubtraction limits
#     "oversub_base": 1.0,         # minimum oversubtraction (rain-like frames)
#     "oversub_max": 3.0,          # strong oversubtraction (noise-like frames)

#     # bounds for gain
#     "gain_floor": 0.0,           # optionally raise to 0.05 to avoid tonal noise
#     "gain_ceil": 1.0,
#     "gain_mode": "sqrt_sub",       # or "wiener"
#     "gain_smooth_alpha": 0.7,  # add this field; default in dataclass
#     "n_mels": 24,  # number of mel bands
#     "rain_like_thresh": 0.7,
#     "noise_psd_mode": "mel",  # or "fft"

#     # New / refined PSD tracking params
#     "pre_smooth_frames": 0.0,    # 3–5, 0 disables
#     "ema_up": 0.6,             # fast when noise increases
#     "ema_down": 0.95,          # slow when noise decreases
    
# }
# def build_noise_config(
#     sample_rate: int,
#     params: Dict[str, Any],
# ) -> NoiseProcessorConfig:
#     """
#     Merge framework params + defaults into a NoiseProcessorConfig instance.

#     Priority:
#         1. params[...]  (overrides)
#         2. DEFAULT_NOISE_PROCESSOR_PARAMS
#         3. explicit fs from audio sample rate

#     Returns:
#         NoiseProcessorConfig
#     """

#     # 1) merge caller overrides on top of defaults
#     merged = {**DEFAULT_NOISE_PROCESSOR_PARAMS, **params}

#     cfg = NoiseProcessorConfig(
#         # Core
#         fs=sample_rate,
#         n_fft=merged["n_fft"],
#         hop=merged["hop"],

#         # High-pass filtering
#         hp_cutoff_hz=merged["hp_cutoff_hz"],
#         hp_order=merged["hp_order"],

#         # Operating band
#         fmin=merged["fmin"],
#         fmax=merged["fmax"],

#         # Rain detector bands
#         rain_band=merged["rain_band"],
#         adj_band=merged["adj_band"],

#         # Rain detection thresholds
#         N_db=merged["N_db"],
#         M_db=merged["M_db"],

#         # Noise tracking
#         q=merged["q"],
#         win_sec=merged["win_sec"],
#         ema_lambda=merged["ema_lambda"],
#         eps=merged["eps"],

#         # Confidence thresholds (with safe fallbacks)
#         dab_conf_low=merged.get("dab_conf_low", 4.0),
#         dab_conf_high=merged.get("dab_conf_high", merged.get("N_db", 8.0)),
#         drise_conf_low=merged.get("drise_conf_low", 4.0),
#         drise_conf_high=merged.get("drise_conf_high", merged.get("M_db", 8.0)),

#         # Confidence weights / smoothing
#         rain_conf_w_spec=merged["rain_conf_w_spec"],
#         rain_conf_w_rise=merged["rain_conf_w_rise"],
#         rain_conf_smooth_alpha=merged["rain_conf_smooth_alpha"],

#         # Adaptive oversubtraction & gain limits
#         oversub_base=merged["oversub_base"],
#         oversub_max=merged["oversub_max"],
#         gain_floor=merged["gain_floor"],
#         gain_ceil=merged["gain_ceil"],

#         # Gain smoothing + mode
#         gain_smooth_alpha=merged["gain_smooth_alpha"],
#         gain_mode=merged["gain_mode"],
        
#         # number of mel bands
#         n_mels=merged["n_mels"],
#         # rain-like threshold
#         rain_like_thresh=merged["rain_like_thresh"],
#         # select between fft and mel
#         noise_psd_mode=merged["noise_psd_mode"],
        
#         # New / refined PSD tracking params
#         pre_smooth_frames=merged["pre_smooth_frames"],
#         ema_up=merged["ema_up"], 
#         ema_down=merged["ema_down"],   
#     )

#     return cfg

