from __future__ import annotations

"""
Smoke test for the audio_processing_framework + processors.

Run this from the repo root after installing audio_processing_tools
(editable install recommended):

    uv pip install -e .
    python smoke_test_framework.py

This test:
    - Uses LocalPath input with a folder of .wav/.bin files
    - Runs a RainProcessor (real rain_detection_algo if available, else dummy)
    - Uses the default key + audio loading from audio_io
    - Writes small CSVs with results and state
"""

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from audio_processing_tools.audio_processing_framework import (
    process_audio_batches_v2,
)
from audio_processing_tools.processors import RainProcessor

# Try to use your real rain_detection_algo if it exists
try:
    from audio_processing_tools.edge.rain_detection_algo import rain_detection_algo as real_rain_algo
except ImportError:
    real_rain_algo = None


# ----------------------------------------------------------------------
# Dummy rain algorithm (fallback)
# ----------------------------------------------------------------------


def dummy_rain_detection_algo(audio_data: np.ndarray, **params) -> tuple[int, float, Dict[str, Any]]:
    """
    Extremely simple stand-in for rain_detection_algo:
        - counts "drops" = number of samples above a threshold / 100
        - frain_mean = mean of positive samples
        - returns a small state dict
    """
    thr = params.get("dummy_abs_thr", 0.1)
    mask = np.abs(audio_data) > thr
    drops = int(mask.sum() // 100)

    if np.any(audio_data > 0):
        frain_mean = float(audio_data[audio_data > 0].mean())
    else:
        frain_mean = 0.0

    state = {
        "rain_drop_count": drops,
        "rain_peaks_count": drops * 2,
        "rain_drop_count_mod": max(0, drops - 1),
        "kurtosis": 3.0,
        "crest_factor": 4.0,
        "diff_energy": 7.0,
        "nov": 0.2,
    }
    return drops, frain_mean, state


# ----------------------------------------------------------------------
# Main smoke test
# ----------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Configure input folder and basic params
    # ------------------------------------------------------------------
    TEST_VECTOR_PATH = os.environ.get(
        "SMOKE_TEST_VECTOR_PATH", "/Users/vikrantoak/Downloads/tv_sets/label_data"
    )
    InputType = os.environ.get("SMOKE_TEST_INPUT_TYPE", "LocalPath")

    if not Path(TEST_VECTOR_PATH).exists():
        raise SystemExit(
            f"TEST_VECTOR_PATH does not exist: {TEST_VECTOR_PATH!r}\n"
            "Please edit smoke_test_framework.py and set a valid folder path."
        )

    # Sample rate and duration should match (or be compatible with) your data.
    # WAV files will be resampled to this Fs if needed.
    Fs = 11162
    duration_s = 10.0

    params_global: Dict[str, Any] = {
        "sample_rate": Fs,
        "check_duration": duration_s,
        # If you later use the real rain algo, it will expect more params here.
        "rain_drop_min_thr": 3,
        "dummy_abs_thr": 0.1,  # only used by the dummy algorithm
    }

    debug_params: Dict[str, Any] = {
        "print_mismatched": True,
        "debug_all": False,
        "rain_drop_min_thr": params_global["rain_drop_min_thr"],
    }

    # ------------------------------------------------------------------
    # 2) Choose which rain algorithm to wrap
    # ------------------------------------------------------------------
    if real_rain_algo is not None:
        print("Using REAL rain_detection_algo from audio_processing_tools.edge")
        rain_fn = real_rain_algo
    else:
        print("Using DUMMY rain_detection_algo (real one not found)")
        rain_fn = dummy_rain_detection_algo

    rain_proc = RainProcessor(name="rain", fn=rain_fn)
    processors = [rain_proc]

    # ------------------------------------------------------------------
    # 3) Run the framework
    # ------------------------------------------------------------------
    print("\nRunning process_audio_batches_v2...")
    results_df, states_by_proc = process_audio_batches_v2(
        processors=processors,
        params_global=params_global,
        params_by_processor={},      # no per-processor overrides yet
        debug_params=debug_params,
        InputType=InputType,
        test_vector_path=TEST_VECTOR_PATH,
        query=None,
        adse_engine=None,
        batch_size=10,
        local_cache=None,
        localStatus=False,
        # get_keys_fn / get_input_data_fn default to audio_io.get_keys/get_input_data
        get_keys_fn=None,
        get_input_data_fn=None,
    )

    # ------------------------------------------------------------------
    # 4) Inspect and save outputs
    # ------------------------------------------------------------------
    print("\n=== RESULTS_DF (head) ===")
    if not results_df.empty:
        print(results_df.head())
    else:
        print("results_df is empty (no valid audio files processed).")

    print("\nColumns in results_df:")
    print(results_df.columns.tolist())

    rain_state_df = states_by_proc.get("rain", pd.DataFrame())
    print("\n=== RAIN STATE DF (head) ===")
    if not rain_state_df.empty:
        print(rain_state_df.head())
    else:
        print("No rain state rows collected.")

    # Save snapshots to disk
    results_df.to_csv("framework_results.csv", index=False)
    rain_state_df.to_csv("framework_states_rain.csv", index=False)

    print("\nSmoke test complete. Wrote:")
    print("  framework_results.csv")
    print("  framework_states_rain.csv")


if __name__ == "__main__":
    main()
