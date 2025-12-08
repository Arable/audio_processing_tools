"""
noise_defaults.py

Defines:
    - DEFAULT_NOISE_PROCESSOR_PARAMS: default settings for the spectral noise processor
    - build_noise_config(): utility to merge defaults with parameter overrides
"""

from typing import Dict, Any
from audio_processing_tools.edge.noise_processor import NoiseProcessorConfig


# -----------------------------------------------------------
# Default Noise Processor Parameters
# -----------------------------------------------------------

DEFAULT_NOISE_PROCESSOR_PARAMS: Dict[str, Any] = {
    # -------------------------------------------------------
    # STFT parameters
    # -------------------------------------------------------
    "n_fft": 256,
    "hop": 128,

    # -------------------------------------------------------
    # Operating band for estimation + suppression
    # -------------------------------------------------------
    "fmin": 400.0,
    "fmax": 3500.0,

    # -------------------------------------------------------
    # Rain detection bands
    # -------------------------------------------------------
    "rain_band": (400.0, 600.0),
    "adj_band": (600.0, 800.0),

    # Rain-frame gating thresholds
    "N_db": 6.0,    # spectral peak difference
    "M_db": 6.0,    # inter-frame rise difference

    # -------------------------------------------------------
    # Noise PSD estimation
    # -------------------------------------------------------
    "q": 0.4,        # quantile used for min-stats
    "win_sec": 0.50, # rolling window length (seconds)

    # Temporal smoothing
    "median_frames": 0,     # disabled = 0
    "ema_lambda": 0.80,     # exponential smoothing for raw noise PSD

    # Numerical stability
    "eps": 1e-9,

    # -------------------------------------------------------
    # High-pass filter
    # -------------------------------------------------------
    "hp_cutoff_hz": 350.0,
    "hp_order": 4,

    # PSD smoothing (optional, light)
    "psd_smooth_alpha": 0.1,

    # -------------------------------------------------------
    # NEW: Soft confidence + adaptive suppression parameters
    # -------------------------------------------------------

    # weights for rain confidence components
    "rain_conf_w_spec": 0.5,     # weight for dAB
    "rain_conf_w_rise": 0.5,     # weight for drise

    # EMA smoothing factor for confidence
    "rain_conf_smooth_alpha": 0.8,

    # adaptive oversubtraction limits
    "oversub_base": 1.0,         # minimum oversubtraction (rain-like frames)
    "oversub_max": 3.0,          # strong oversubtraction (noise-like frames)

    # bounds for gain
    "gain_floor": 0.0,           # optionally raise to 0.05 to avoid tonal noise
    "gain_ceil": 1.0,
    "gain_mode": "sqrt_sub",       # or "wiener"
    "gain_smooth_alpha": 0.7,  # add this field; default in dataclass

}

def build_noise_config(
    sample_rate: int,
    params: Dict[str, Any],
) -> NoiseProcessorConfig:
    """
    Merge framework params + defaults into a NoiseProcessorConfig instance.

    Priority:
        1. params[...]  (overrides)
        2. DEFAULT_NOISE_PROCESSOR_PARAMS
        3. explicit fs from audio sample rate

    Returns:
        NoiseProcessorConfig
    """

    # 1) merge caller overrides on top of defaults
    merged = {**DEFAULT_NOISE_PROCESSOR_PARAMS, **params}

    cfg = NoiseProcessorConfig(
        # Core
        fs=sample_rate,
        n_fft=merged["n_fft"],
        hop=merged["hop"],

        # High-pass filtering
        hp_cutoff_hz=merged["hp_cutoff_hz"],
        hp_order=merged["hp_order"],

        # Operating band
        fmin=merged["fmin"],
        fmax=merged["fmax"],

        # Rain detector bands
        rain_band=merged["rain_band"],
        adj_band=merged["adj_band"],

        # Rain detection thresholds
        N_db=merged["N_db"],
        M_db=merged["M_db"],

        # Noise tracking
        q=merged["q"],
        win_sec=merged["win_sec"],
        ema_lambda=merged["ema_lambda"],
        eps=merged["eps"],

        # Confidence thresholds (with safe fallbacks)
        dab_conf_low=merged.get("dab_conf_low", 3.0),
        dab_conf_high=merged.get("dab_conf_high", merged.get("N_db", 6.0)),
        drise_conf_low=merged.get("drise_conf_low", 3.0),
        drise_conf_high=merged.get("drise_conf_high", merged.get("M_db", 6.0)),

        # Confidence weights / smoothing
        rain_conf_w_spec=merged["rain_conf_w_spec"],
        rain_conf_w_rise=merged["rain_conf_w_rise"],
        rain_conf_smooth_alpha=merged["rain_conf_smooth_alpha"],

        # Adaptive oversubtraction & gain limits
        oversub_base=merged["oversub_base"],
        oversub_max=merged["oversub_max"],
        gain_floor=merged["gain_floor"],
        gain_ceil=merged["gain_ceil"],

        # Gain smoothing + mode
        gain_smooth_alpha=merged["gain_smooth_alpha"],
        gain_mode=merged["gain_mode"],
    )

    return cfg

__all__ = [
    "DEFAULT_NOISE_PROCESSOR_PARAMS",
    "build_noise_config",
]