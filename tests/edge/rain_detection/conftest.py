"""Shared configuration and deterministic signals for rain-detector tests."""

from typing import Any

import numpy as np
import pytest

SAMPLE_RATE = 11_162


@pytest.fixture
def detector_params() -> dict[str, Any]:
    """Deployed CM7 detector configuration used by the regression tests."""
    return {
        "mode_bands": [(450.0, 650.0), (800.0, 1050.0), (1500.0, 1800.0),
                       (2350.0, 2550.0), (3150.0, 3350.0)],
        "mode_weights": [1.0, 0.8, 0.7, 0.6, 0.5],
        "run_frame_level_comparison": True,
        "run_streaming_comparison": True,
        "run_nowinsor_replay": True,
        "detector_use_noise_norm": True,
        "flux_modes_winsor_enable": False,
        "mode_flux_norm_enable": True,
        "mode_flux_norm_win_sec": 0.5,
        "mode_flux_norm_q": 20.0,
        "mode_flux_norm_min": 1.0,
        "mode_flux_noise_max": 2.5,
        "new_rain_primary_flux_min": 2.19,
        "new_rain_mode1_flux_min": 2.63,
        "new_rain_mode2_flux_min": 2.57,
        "new_rain_mode3_flux_min": 2.45,
        "new_rain_min_support_count": 3,
        "noise_hi": 0.85,
        "td_gate_threshold": 3.4,
        "td_kurtosis_upper_threshold": 12.0,
        "td_soft_enable": False,
        "td_soft_subframe_len": 16,
        "td_soft_subframe_hop": 16,
        "td_input_mode": "default",
        "td_envelope_features_enable": False,
        "td_block_energy_len": 8,
        "td_block_energy_hop": None,
        "td_block_energy_post_pre_blocks": 6,
        "td_block_energy_smooth_enable": True,
        "td_apply_input_prefilter": True,
        "td_prefilter_mode": "highpass",
        "raw_spectral_shape_enable": True,
        "raw_spectral_rain_band": (400.0, 700.0),
        "raw_spectral_low_band": (60.0, 200.0),
        "raw_spectral_rolloff_fraction": 0.85,
        "feature_dump_dense_enable": True,
        "feature_dump_sparse_enable": False,
        "feature_dump_include_frame_class": True,
        "process_dtype": "float32",
    }


@pytest.fixture
def processor_params(detector_params: dict[str, Any]) -> dict[str, Any]:
    """Build the self-contained classifier-only processor configuration."""
    return {
        "sample_rate": SAMPLE_RATE,
        "operating_band": (400.0, 3500.0),
        "n_fft": 256,
        "hop": 128,
        "eps": 1e-9,
        "q": 0.30,
        "clip_rain_min_frames": 4,
        "detector": detector_params,
        "classifier_only_mode": True,
        "dump_features": True,
        "feature_decim": 1,
    }


@pytest.fixture
def deterministic_audio() -> np.ndarray:
    """Broadband decaying impulses over stationary seeded noise."""
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 0.002, 2 * SAMPLE_RATE)
    pulse_t = np.arange(96, dtype=np.float64) / SAMPLE_RATE
    pulse = 0.08 * np.exp(-pulse_t * 85.0) * sum(
        np.sin(2.0 * np.pi * frequency * pulse_t)
        for frequency in (550.0, 900.0, 1650.0, 2450.0, 3250.0)
    )
    for start in (2_000, 5_000, 9_000, 13_000, 17_000):
        audio[start : start + pulse.size] += pulse
    return audio.astype(np.float32)
