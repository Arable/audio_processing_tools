#!/usr/bin/env python
from __future__ import annotations

"""
run_rain_noise_analysis.py

Driver script for running the audio processing framework with:
  - RainProcessor (rain_detection_algo)
  - NoiseProcessor (example or real)
  - Post-processing using audio_processing_tools.postprocess.rain/noise
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from audio_processing_tools.audio_processing_framework import process_audio_batches_v2
from audio_processing_tools.processors import RainProcessor, NoiseProcessor
from audio_processing_tools.edge.rain_detection_algo import rain_detection_algo

from audio_processing_tools.postprocess.rain import postprocess_rain
from audio_processing_tools.postprocess.noise import postprocess_noise


# ---------------------------------------------------------------------
# 1) Configuration
# ---------------------------------------------------------------------

INPUT_TYPE = "LocalPath"   # or "RemotePath"

TEST_VECTOR_PATH = "/absolute/path/to/audio/files"  # <-- EDIT for LocalPath

QUERY: Optional[str] = None     # for RemotePath
ADSE_ENGINE = None              # SQLAlchemy engine if using RemotePath

LOCAL_AUDIO_CACHE: Optional[str] = "raw_audio_cache"

FS = 11162
CHECK_DURATION_S = 10.0

PARAMS: Dict[str, Any] = {
    "sample_rate": FS,
    "check_duration": CHECK_DURATION_S,
    "freq_resolution": 45,
    "time_resolution_ms": 10,
    "op_freq_range": [400, 3500],
    "n_freq_range": [400, 700],
    "fn": 400,
    "num_harmonics": 6,
    "harmonic_threshold": [4.5, 4.0, 3.5, 3.5, 3.5, 3.5],
    "max_peaks": 3,
    "log_factor": 0,
    "ns_duration_ms": 470,
    "nf": 0,
    "min_drop_count": 0.3,
    "rain_drop_min_thr": 3,
    "rain_drop_max_thr": 50,
    "rain_peaks_min_thr": 9,
    "rain_peaks_max_thr": 30,
    "kurtosis_thr": 2.5,
    "crest_thr": 3.75,
    "diff_energy_thr": 6.5,
    "t_band": [400, 3500],
    "handle_fp": True,
    "handle_fn": True,
    "enable_nov_wind_dection": False,
    "enable_energy_peak_detection": False,
}

DEBUG_PARAMS: Dict[str, Any] = {
    "print_mismatched": True,
    "debug_all": False,
    "rain_drop_min_thr": PARAMS["rain_drop_min_thr"],
}


# ---------------------------------------------------------------------
# 2) Example noise algorithm (replace with your real one)
# ---------------------------------------------------------------------

def example_noise_fn(audio_data: np.ndarray, **params):
    """
    Example noise function. Replace with your real noise algorithm.

    Must return:
        metrics_dict, state_dict
    """
    energy = float(np.mean(audio_data**2) + 1e-12)
    snr_db = 10.0 * np.log10(energy + 1e-9)

    metrics = {"snr_db": snr_db}
    state = {"energy": energy}
    return metrics, state


# ---------------------------------------------------------------------
# 3) Main driver
# ---------------------------------------------------------------------

def main() -> None:
    if INPUT_TYPE == "LocalPath":
        if not TEST_VECTOR_PATH or not Path(TEST_VECTOR_PATH).exists():
            raise SystemExit(
                f"TEST_VECTOR_PATH does not exist: {TEST_VECTOR_PATH!r}. "
                "Edit run_rain_noise_analysis.py and set a valid folder."
            )

    # 1. Build processors
    rain_proc = RainProcessor(name="rain", fn=rain_detection_algo)
    noise_proc = NoiseProcessor(name="noise", fn=example_noise_fn)

    processors = [rain_proc, noise_proc]

    # 2. Run framework
    print("Running process_audio_batches_v2...")
    results_df, states_by_proc = process_audio_batches_v2(
        processors=processors,
        params_global=PARAMS,
        params_by_processor=None,
        debug_params=DEBUG_PARAMS,
        InputType=INPUT_TYPE,
        test_vector_path=TEST_VECTOR_PATH if INPUT_TYPE == "LocalPath" else None,
        query=QUERY if INPUT_TYPE == "RemotePath" else None,
        adse_engine=ADSE_ENGINE,
        batch_size=1000,
        local_cache=LOCAL_AUDIO_CACHE,
        localStatus=False,
        get_keys_fn=None,        # use defaults from audio_io
        get_input_data_fn=None,  # use defaults from audio_io
    )

    print("\n=== results_df (head) ===")
    print(results_df.head())

    # 3. Post-process rain using your dedicated module
    rain_states_df = states_by_proc.get("rain", pd.DataFrame())
    rain_results_df, rain_feature_df = postprocess_rain(
        results_df, rain_states_df, PARAMS
    )

    # 4. Post-process noise using your dedicated module
    noise_states_df = states_by_proc.get("noise", pd.DataFrame())
    noise_summary_df = postprocess_noise(
        results_df, noise_states_df, PARAMS
    )

    # 5. Save outputs
    rain_results_df.to_csv("rain_test_results.csv", index=False)
    rain_feature_df.to_csv("rain_feature_data.csv", index=False)
    noise_summary_df.to_csv("noise_summary.csv", index=False)

    print("\nWrote:")
    print("  rain_test_results.csv")
    print("  rain_feature_data.csv")
    print("  noise_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
