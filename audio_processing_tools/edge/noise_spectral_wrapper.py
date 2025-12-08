from typing import Dict, Any, Tuple
import numpy as np

from .noise_processor import NoiseProcessorConfig, SpectralNoiseProcessor
from .noise_defaults import build_noise_config

def spectral_noise_fn(audio_data, **params):
    sample_rate = params.get("sample_rate", 11162)

    # Build config (defaults + overrides)
    cfg = build_noise_config(sample_rate, params)

    proc = SpectralNoiseProcessor(cfg)
    # print(f"{cfg.psd_smooth_alpha = }")
    out = proc.process(audio_data, sr=sample_rate)
    print("Noise cfg:", cfg)
    # Extract outputs as before…
    y = out["y"]
    noise_psd = out["noise_psd"]
    freqs = out["freqs"]
    is_rain = out["is_rain"]
    x_hp = out["x_hp"]

    # Metrics
    band_mask = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
    noise_band = noise_psd[band_mask]
    noise_db = 10 * np.log10(noise_band + cfg.eps)

    metrics = {
        "mean_noise_floor_db": float(np.mean(noise_db)),
        "median_noise_floor_db": float(np.median(noise_db)),
        "rain_frame_fraction": float(np.mean(is_rain)),
    }

    # State
    state = {
        "input_audio": x_hp,
        "denoised_audio": y,
        "noise_psd": noise_psd,
        "is_rain": is_rain,
        "freqs": freqs,
        "times": out["times"],
        "S": out["S"],
        "S_hat": out["S_hat"],
        "debug": out["debug"],
        "config": cfg,
    }

    return metrics, state