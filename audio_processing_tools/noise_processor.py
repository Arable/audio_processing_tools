# noise_processor.py
# ----------------------------------------------------------------------
# NoiseProcessor: generic noise estimation wrapper
# ----------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .edge.spectral_noise_processor import SpectralNoiseProcessor, build_noise_config
from .processors import BaseProcessor


@dataclass
class NoiseProcessor(BaseProcessor):
    """
    Framework-level noise processor that wraps the spectral noise engine.

    This class is the entry point used by the audio processing framework.
    Internally it:
      - builds a noise configuration from params
      - instantiates a SpectralNoiseProcessor
      - runs the spectral noise algorithm
      - returns:
          * scalar KPIs in `results`
          * rich debug state in `state`

    The processor is identified by its `name`, which is also used to namespace
    metric columns in the orchestrator (e.g. "noise_spectral_minstats__snr_db").
    """

    # No `fn` field anymore; all the logic lives in `run()` / `_run_spectral_noise()`

    def run(
        self,
        audio_data: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main entry point called by the orchestrator.

        Validates the input buffer, times the call to the internal noise algorithm,
        and decorates outputs with latency and processor name.
        """
        # Sanity checks
        self._validate_audio(audio_data, params)

        # Time the core spectral noise algorithm
        (metrics, state), latency = self._with_timing(
            self._run_spectral_noise,
            audio_data,
            params,
        )

        # Normalize and decorate outputs
        if not isinstance(metrics, dict):
            metrics = {"value": metrics}
        if not isinstance(state, dict):
            state = {"state": state}

        metrics_out: Dict[str, Any] = dict(metrics)
        metrics_out["latency_s"] = latency

        state_out: Dict[str, Any] = dict(state)
        state_out["processor"] = self.name
        state_out["latency_s"] = latency

        return metrics_out, state_out

    # ------------------------------------------------------------------
    # Internal spectral noise implementation (moved from spectral_noise_fn)
    # ------------------------------------------------------------------

    def _run_spectral_noise(
        self,
        audio_data: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Core spectral min-stats noise logic that previously lived in spectral_noise_fn.

        Builds a configuration, runs SpectralNoiseProcessor, and converts the output
        into (metrics, state) dictionaries.
        """
        sample_rate = int(params.get("sample_rate", 11162))

        # Build algorithm configuration from the global params
        cfg = build_noise_config(sample_rate, params)

        # Instantiate the underlying spectral noise engine
        proc = SpectralNoiseProcessor(cfg)

        # Run the spectral processor
        out = proc.process(audio_data, sr=sample_rate)

        # Unpack outputs
        y = out["y"]
        noise_psd = out["noise_psd"]
        freqs = out["freqs"]
        is_rain = out["is_rain"]
        x_hp = out["x_hp"]

        # Compute simple band-limited noise statistics using operating_band
        f_lo, f_hi = cfg.operating_band
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        noise_band = noise_psd[band_mask]
        noise_db = 10.0 * np.log10(noise_band + cfg.eps)

        metrics: Dict[str, Any] = {
            "mean_noise_floor_db": float(np.mean(noise_db)),
            "median_noise_floor_db": float(np.median(noise_db)),
            "rain_frame_fraction": float(np.mean(is_rain)),
        }

        state: Dict[str, Any] = {
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