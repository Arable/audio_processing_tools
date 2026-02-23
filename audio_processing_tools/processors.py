from __future__ import annotations

"""
processors.py

Concrete processor implementations for the audio processing framework.

Exposes:
    - BaseProcessor : convenience base class (validation, timing)
    - RainProcessor : adapter around a rain_detection_algo-style function
    - NoiseProcessor: generic wrapper for noise / SNR estimation functions

All processors are structurally compatible with the AudioProcessor protocol
defined in audio_processing_framework.py (name + run(...) signature).
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Callable

import time
import numpy as np


# ----------------------------------------------------------------------
# Base processor with shared helpers
# ----------------------------------------------------------------------


@dataclass
class BaseProcessor:
    """
    Base class for audio processors.

    Provides:
        - name          : short identifier for namespacing (e.g. "rain", "noise")
        - _validate_audio: basic sanity checks on the input buffer
        - _with_timing  : measure runtime of the wrapped function

    Concrete processors inherit from this and implement .run().
    """

    name: str

    def _validate_audio(self, audio_data: np.ndarray, params: Dict[str, Any]) -> None:
        """
        Perform basic validation on the input audio buffer.

        Checks:
            - audio_data is a NumPy array
            - audio_data is 1-D (mono)
            - if sample_rate and check_duration are present in params,
              length is at least sample_rate * check_duration
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError(f"audio_data must be a NumPy array, got {type(audio_data)}")

        if audio_data.ndim != 1:
            raise ValueError(f"audio_data must be 1-D, got shape {audio_data.shape}")

        sr = params.get("sample_rate")
        dur = params.get("check_duration")
        if sr is not None and dur is not None:
            min_len = int(sr * dur)
            if audio_data.size < min_len:
                raise ValueError(
                    f"audio_data too short: {audio_data.size} < required {min_len} samples"
                )

    def _with_timing(self, func: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
        """
        Execute fn(*args, **kwargs) and return (result, elapsed_time_seconds).
        """
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        return result, dt


# ----------------------------------------------------------------------
# RainProcessor: adapter over rain_detection_algo
# ----------------------------------------------------------------------


@dataclass
class RainProcessor(BaseProcessor):
    """
    Adapter for a rain detection algorithm.

    Expected function signature:
        fn(audio_data: np.ndarray, **params) -> (rain_drops, frain_mean, state_dict)

    state_dict may contain keys such as:
        - "rain_drop_count"
        - "rain_peaks_count"
        - "rain_drop_count_mod"
        - "kurtosis"
        - "crest_factor"
        - "diff_energy"
        - "nov"
        - etc.

    The processor returns:
        - results: compact KPIs (namespaced later as rain__*)
        - state  : internal state (one dict per file)
    """

    fn: Callable[..., Tuple[int, float, Dict[str, Any]]]

    def run(
        self,
        audio_data: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Validate input buffer
        self._validate_audio(audio_data, params)

        # Call underlying algorithm with timing
        (rain_drops, frain_mean, state), latency = self._with_timing(
            self.fn,
            audio_data,
            **params,
        )

        # Compact metrics for main results table
        results: Dict[str, Any] = {
            "rain_drops": rain_drops,
            "frain_mean": frain_mean,
            "latency_s": latency,
        }

        # Promote common counters into top-level metrics if present
        if isinstance(state, dict):
            for k in ("rain_drop_count", "rain_peaks_count", "rain_drop_count_mod"):
                if k in state:
                    results[k] = state[k]

        # Normalize state to a dict and tag with processor info
        state_out: Dict[str, Any] = dict(state) if isinstance(state, dict) else {"state": state}
        state_out["processor"] = self.name
        state_out["latency_s"] = latency

        return results, state_out

def has_processor(processors, name: str) -> bool:
    """
    Check whether a processor with a given name exists in a list of processors.

    Parameters
    ----------
    processors : Iterable
        A collection of processor objects. Each processor is expected
        to expose a `.name` attribute.

    name : str
        The processor name to look for (e.g. "rain",
        "noise_spectral", "spectral_features").

    Returns
    -------
    bool
        True if any processor in `processors` has `p.name == name`,
        False otherwise.

    """
    return any(p.name == name for p in processors)
# ----------------------------------------------------------------------
# Minimal self-test / example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    def dummy_rain_algo(x: np.ndarray, **params):
        drops = int((np.abs(x) > 0.1).sum() // 100)
        frain_mean = float(max(0.0, x.mean()))
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

    def dummy_noise_fn(x: np.ndarray, **params):
        energy = float(np.mean(x**2) + 1e-12)
        snr_db = 10.0 * np.log10(energy + 1e-9)
        metrics = {"snr_db": snr_db}
        state = {"energy": energy}
        return metrics, state

    Fs = 11162
    duration = 10.0
    t = np.arange(int(Fs * duration)) / Fs
    audio = 0.05 * np.random.randn(t.size).astype(np.float32)

    params = {
        "sample_rate": Fs,
        "check_duration": duration,
        "rain_drop_min_thr": 3,
    }

    rain_proc = RainProcessor(name="rain", fn=dummy_rain_algo)
    noise_proc = NoiseProcessor(name="noise", fn=dummy_noise_fn)

    for proc in (rain_proc, noise_proc):
        print(f"\nProcessor: {proc.name}")
        results, state = proc.run(audio, params)
        print("  results:", results)
        print("  state keys:", list(state.keys()))