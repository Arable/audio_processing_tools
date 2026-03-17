# edge/rain_detector.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spectral_noise_processor import (
    NoiseProcessorConfig,
    SpectralNoiseProcessor,
    build_noise_config,
)
from .time_domain_detector import (
    TimeDomainDetectorConfig,
    TimeDomainRainDetector,
    build_time_domain_config,
)


class RainDetector:
    """
    Audio-processing-framework-compatible two-stage rain detector.

    Stage-1:
      SpectralNoiseProcessor
        - spectral rain/noise classification
        - PSD estimation
        - optional suppression

    Stage-2:
      TimeDomainRainDetector
        - run only on stage-1 rain frames
        - time-domain droplet confirmation

    Final output:
      - preserves stage-1 outputs
      - adds stage-2 outputs
      - sets top-level `is_rain` to the final confirmed result

    Notes:
      - Full two-stage initialization is supported via `RainDetector()` followed by
        `setup(params)`.
      - Constructing `RainDetector(config=...)` initializes stage-1 only; stage-2
        still requires framework params and is not reconstructed from the stage-1
        config object.
    """

    def __init__(self, config: Optional[NoiseProcessorConfig] = None):
        self.cfg = config
        self._is_setup = config is not None

        self.stage2_cfg: Optional[TimeDomainDetectorConfig] = None
        self.stage2: Optional[TimeDomainRainDetector] = None

        if self._is_setup:
            # A standalone stage-1 NoiseProcessorConfig is not sufficient to
            # reconstruct the full stage-2 config here because stage-2 is
            # intentionally built from the orchestrator/framework params.
            self.stage2 = None

        self.stage1 = (
            SpectralNoiseProcessor(config=config)
            if config is not None
            else SpectralNoiseProcessor()
        )

    def setup(self, params: Dict[str, Any]) -> None:
        """
        Framework-style setup method.
        Safe to call once before repeated process(...) calls.
        """
        if self._is_setup:
            return

        sr = int(params.get("sample_rate", params.get("fs", 11162)))
        self.cfg = build_noise_config(sample_rate=sr, params=params)
        self.stage1 = SpectralNoiseProcessor(config=self.cfg)

        self.stage2_cfg = build_time_domain_config(params)
        self.stage2 = TimeDomainRainDetector(self.stage2_cfg)

        self._is_setup = True

    def run_stage1(self, x: np.ndarray, sr: int) -> Dict[str, Any]:
        return self.stage1.process(x, sr=sr)

    def run_stage2(
        self,
        x: np.ndarray,
        sr: int,
        stage1_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.stage2 is None:
            raise RuntimeError("Time-domain detector is not initialized.")

        stage1_is_rain = np.asarray(stage1_out["is_rain"], dtype=bool)
        x_filt = np.asarray(stage1_out.get("x_filt", x), dtype=np.float64)

        return self.stage2.process(
            x_filt,
            stage1_is_rain=stage1_is_rain,
            sr=sr,
        )

    def merge_outputs(
        self,
        stage1_out: Dict[str, Any],
        stage2_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        stage1_is_rain = np.asarray(stage1_out["is_rain"], dtype=bool)
        stage2_confirmed = np.asarray(stage2_out["confirmed_mask"], dtype=bool)
        confirmed_raindrops = np.asarray(
            stage2_out["confirmed_counts"],
            dtype=np.int32,
        )

        debug = dict(stage1_out.get("debug", {}) or {})
        debug.update(
            {
                "is_rain_stage1": stage1_is_rain,
                "time_domain_params": (
                    vars(self.stage2_cfg).copy()
                    if self.stage2_cfg is not None
                    else {}
                ),
                "stage2_confirmed": stage2_confirmed,
                "confirmed_raindrops": confirmed_raindrops,
                "stage2_crest_factor": stage2_out.get("crest_factor"),
                "stage2_kurtosis": stage2_out.get("kurtosis"),
                "stage2_candidate_peaks": stage2_out.get("candidate_peaks"),
                "stage2_details": stage2_out.get("details"),
                "stage2_mode_signal": stage2_out.get("x_mode"),
            }
        )

        out = dict(stage1_out)
        out["debug"] = debug
        out["is_rain_stage1"] = stage1_is_rain
        out["stage2_confirmed"] = stage2_confirmed
        out["confirmed_raindrops"] = confirmed_raindrops

        # Final detector output used by the framework.
        out["is_rain"] = stage2_confirmed
        return out

    def process(
        self,
        x: np.ndarray,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Framework-compatible process method.
        This is the method the audio processing framework should call.
        """
        if self.cfg is None:
            raise RuntimeError(
                "RainDetector is not initialized. Call setup(params) before "
                "process(...)."
            )

        cfg = self.cfg
        if sr is None:
            sr = cfg.fs

        stage1_out = self.run_stage1(x, sr=sr)
        if self.stage2 is None:
            raise RuntimeError(
                "Time-domain detector is not initialized. Construct RainDetector "
                "with no config and call setup(params), or initialize stage-2 "
                "explicitly from framework params."
            )
        stage2_out = self.run_stage2(x=x, sr=sr, stage1_out=stage1_out)
        return self.merge_outputs(stage1_out=stage1_out, stage2_out=stage2_out)