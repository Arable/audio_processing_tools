from __future__ import annotations

"""
audio_processing_framework.py

Batch orchestration for audio processors in the audio_processing_tools package.

This module defines:
    - AudioProcessor             : protocol describing the processor interface
    - process_audio_batches_v2() : main orchestration entry point
    - process_audio_batches      : backward-compatible alias

By default, it uses the key-discovery and audio-loading utilities from
audio_io (get_keys, get_input_data), but custom implementations can be
injected via the get_keys_fn and get_input_data_fn parameters.
"""

import gc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from .audio_io import get_keys as default_get_keys, get_input_data as default_get_input_data


# ----------------------------------------------------------------------
# Processor contract
# ----------------------------------------------------------------------


@runtime_checkable
class AudioProcessor(Protocol):
    """
    Interface for audio processors used by the framework.

    A processor receives a standardized audio buffer and a parameter dict,
    and returns:
        - results: scalar metrics (for the main results DataFrame)
        - state  : internal state (for debugging, plotting, or analysis)
    """

    @property
    def name(self) -> str:
        """
        Short identifier used as a namespace prefix in result columns.

        Examples
        --------
        "rain", "noise", "spectral", etc.
        """
        ...

    def run(
        self,
        audio_data: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute the processor on a single audio buffer.

        Parameters
        ----------
        audio_data :
            1-D float32 NumPy array at params['sample_rate'] Hz. Length is
            typically sample_rate * check_duration after pre-processing.
        params :
            Combined parameter dictionary, including global settings and any
            per-processor overrides.

        Returns
        -------
        results :
            Dict of scalar metrics (e.g., counts, averages, scores). These are
            flattened into the main results DataFrame with a '<name>__' prefix.
        state :
            Dict of internal state (arrays, intermediates, diagnostics) for
            this processor. One state dict is collected per file.
        """
        ...


# ----------------------------------------------------------------------
# Helper for namespaced result columns
# ----------------------------------------------------------------------


def _flatten_with_namespace(ns: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefix keys in a metrics dict with a processor namespace.

    Parameters
    ----------
    ns :
        Processor name used as a namespace, e.g. "rain", "noise".
    d :
        Metrics dictionary returned by a processor.

    Returns
    -------
    dict
        New dictionary where each key is prefixed as '<ns>__<key>'.

    Examples
    --------
    >>> _flatten_with_namespace("rain", {"frain_mean": 0.8, "rain_drops": 10})
    {'rain__frain_mean': 0.8, 'rain__rain_drops': 10}
    """
    return {f"{ns}__{k}": v for k, v in d.items()}


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------


def process_audio_batches_v2(
    *,
    processors: List[AudioProcessor],
    params_global: Dict[str, Any],
    params_by_processor: Optional[Dict[str, Dict[str, Any]]] = None,
    debug_params: Optional[Dict[str, Any]] = None,
    InputType: Optional[str] = None,
    test_vector_path: Optional[str] = None,
    query: Optional[str] = None,
    adse_engine=None,
    batch_size: int = 1000,
    local_cache: Optional[str] = None,
    localStatus: bool = True,
    get_keys_fn: Optional[
        Callable[..., List[Dict[str, Any]]]
    ] = None,
    get_input_data_fn: Optional[
        Callable[..., Dict[str, Dict[str, Any]]]
    ] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run one or more AudioProcessors over an audio corpus in batches.

    Parameters
    ----------
    processors :
        List of AudioProcessor implementations. Each processor must define
        a unique `.name` and a `.run(audio_data, params)` method.
    params_global :
        Global parameter dictionary shared across processors, e.g.
        {
            "sample_rate": ...,
            "check_duration": ...,
            ...
        }
        Must contain "sample_rate" and "check_duration".
    params_by_processor :
        Optional mapping from processor name to a dict of overrides that
        will be merged on top of `params_global`.
    debug_params :
        Optional debug configuration, e.g.
        {
            "print_mismatched": True,
            "debug_all": False,
            "rain_drop_min_thr": 3,
        }
    InputType :
        Data source type, passed through to `get_keys_fn` and `get_input_data_fn`.
        Typical values: "LocalPath", "RemotePath".
    test_vector_path :
        Local directory for input files when InputType="LocalPath".
    query :
        SQL query string when InputType="RemotePath".
    adse_engine :
        SQLAlchemy engine for DB-backed sources.
    batch_size :
        Number of items to process per batch.
    local_cache :
        Local cache root for remote audio (forwarded to get_input_data_fn).
    localStatus :
        Default raining flag for local data when label cannot be inferred.
    get_keys_fn :
        Optional key-discovery function. If None, uses audio_io.get_keys.
    get_input_data_fn :
        Optional audio-loading function. If None, uses audio_io.get_input_data.

    Returns
    -------
    results_df :
        DataFrame with one row per file. Includes:
            - "file_key"
            - "rain_actual"
            - namespaced metrics "<processor>__<metric>" for each processor.
            - optional "rain__predicted" and "rain__mismatch" if a "rain"
              processor is present and debug options are enabled.
    states_df_by_proc :
        Mapping from processor name to a DataFrame of internal state for that
        processor. Each row corresponds to one input file and includes "file_key"
        plus whatever keys the processor returned in its state dict.
    """
    if params_by_processor is None:
        params_by_processor = {}
    if debug_params is None:
        debug_params = {}

    # Require basic global parameters
    if "sample_rate" not in params_global or "check_duration" not in params_global:
        raise KeyError("params_global must contain 'sample_rate' and 'check_duration'.")

    Fs = params_global["sample_rate"]
    check_duration = params_global["check_duration"]
    required_samples = int(Fs * check_duration)

    # Use default implementations if none provided
    if get_keys_fn is None:
        get_keys_fn = default_get_keys
    if get_input_data_fn is None:
        get_input_data_fn = default_get_input_data

    # ------------------------------------------------------------------
    # 1) Discover all keys to process
    # ------------------------------------------------------------------
    keys: List[Dict[str, Any]] = get_keys_fn(
        InputType,
        test_vector_path=test_vector_path,
        query=query,
        adse_engine=adse_engine,
        batch_size=batch_size,
        localStatus=localStatus,
    )
    print(f"received {len(keys)} test vectors")

    results_rows: List[Dict[str, Any]] = []
    states_by_processor: Dict[str, List[Dict[str, Any]]] = {p.name: [] for p in processors}

    print_mismatched = bool(debug_params.get("print_mismatched", False))
    debug_all = bool(debug_params.get("debug_all", False))
    rain_min_thr = debug_params.get("rain_drop_min_thr", params_global.get("rain_drop_min_thr"))

    total_batches = (len(keys) + batch_size - 1) // batch_size if batch_size > 0 else 1

    # ------------------------------------------------------------------
    # 2) Process keys in batches
    # ------------------------------------------------------------------
    for batch_idx, batch_start in enumerate(range(0, len(keys), batch_size), start=1):
        batch_keys = keys[batch_start : batch_start + batch_size]
        print(f"Processing batch {batch_idx} of ~{total_batches}")

        dir_content = get_input_data_fn(
            batch_keys,
            InputType,
            Fs,
            check_duration,
            localStatus,
            local_cache,
            read_size=None,
            bytes_per_sample=2,
        )

        # dir_content: {file_key: {"file_contents": np.ndarray, "raining": bool, ...}}
        for file_key, meta in dir_content.items():
            audio = meta.get("file_contents")
            rain_actual = meta.get("raining", None)

            if audio is None:
                continue

            audio = np.asarray(audio)
            if audio.ndim != 1:
                raise ValueError(f"audio for {file_key} must be 1-D, got shape {audio.shape}")
            if audio.size < required_samples:
                # Skip files shorter than required duration
                continue

            row: Dict[str, Any] = {
                "file_key": file_key,
                "rain_actual": rain_actual,
            }

            # ----------------------------------------------------------
            # 2.1 Run all processors on this audio buffer
            # ----------------------------------------------------------
            for proc in processors:
                proc_params = {**params_global, **params_by_processor.get(proc.name, {})}
                proc_results, proc_state = proc.run(audio, proc_params)

                proc_state = dict(proc_state)
                proc_state["file_key"] = file_key

                row.update(_flatten_with_namespace(proc.name, proc_results))
                states_by_processor[proc.name].append(proc_state)

            # ----------------------------------------------------------
            # 2.2 Optional rain/no-rain mismatch diagnostics
            # ----------------------------------------------------------
            if (
                "rain__rain_drops" in row
                and rain_actual is not None
                and rain_min_thr is not None
            ):
                rain_predicted = bool(row["rain__rain_drops"] > rain_min_thr)
                row["rain__predicted"] = rain_predicted
                row["rain__mismatch"] = (rain_predicted != bool(rain_actual))

                if (print_mismatched and row["rain__mismatch"]) or debug_all:
                    rd = row.get("rain__rain_drop_count", row["rain__rain_drops"])
                    print(
                        f"[mismatch] {file_key}  "
                        f"actual={rain_actual}  predicted={rain_predicted}  "
                        f"rain_drops={rd}"
                    )

            results_rows.append(row)

        # Free per-batch data to keep memory bounded
        del dir_content
        gc.collect()

    # ------------------------------------------------------------------
    # 3) Collate results into DataFrames
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_rows)
    if not results_df.empty:
        results_df = results_df.sort_values("file_key").reset_index(drop=True)

    states_df_by_proc: Dict[str, pd.DataFrame] = {}
    for name, rows in states_by_processor.items():
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values("file_key").reset_index(drop=True)
        else:
            df = pd.DataFrame()
        states_df_by_proc[name] = df

    return results_df, states_df_by_proc


# Backwards-compatible alias
process_audio_batches = process_audio_batches_v2