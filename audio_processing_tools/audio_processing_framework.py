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
import os
import time
try:
    import psutil
except ImportError:
    psutil = None
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
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

def _extract_param_updates(obj: Any) -> Dict[str, Any]:
    """
    Extract parameter updates from processor output/state.

    Convention:
      - state["_param_updates"] : Dict[str, Any]
      - results["_param_updates"]: Dict[str, Any]  (optional)

    Returns empty dict if none / invalid.
    """
    if not isinstance(obj, dict):
        return {}
    upd = obj.get("_param_updates")
    return upd if isinstance(upd, dict) else {}

# ----------------------------------------------------------------------
# Helper for namespaced result columns
# ----------------------------------------------------------------------

def _discover_keys(
    *,
    InputType: Optional[str],
    test_vector_path: Optional[str],
    query: Optional[str],
    adse_engine,
    batch_size: int,
    localStatus: bool,
    max_files: Optional[int],
    get_keys_fn: Callable[..., List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    keys: List[Dict[str, Any]] = get_keys_fn(
        InputType,
        test_vector_path=test_vector_path,
        query=query,
        adse_engine=adse_engine,
        batch_size=batch_size,
        localStatus=localStatus,
    )

    if max_files is not None:
        if max_files < 0:
            raise ValueError("max_files must be >= 0 or None")
        keys = keys[:max_files]

    return keys


def _process_single_file_task(
    *,
    file_key: str,
    meta: Dict[str, Any],
    processors: List[AudioProcessor],
    params_global: Dict[str, Any],
    params_by_processor: Dict[str, Dict[str, Any]],
    required_samples: int,
    rain_min_thr,
) -> Optional[Dict[str, Any]]:
    audio = meta.get("file_contents")
    rain_actual = meta.get("raining", None)

    if audio is None:
        return None

    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"audio for {file_key} must be 1-D, got shape {audio.shape}")
    if audio.size < required_samples:
        return None

    row: Dict[str, Any] = {
        "file_key": file_key,
        "rain_actual": rain_actual,
    }
    states_for_file: Dict[str, Dict[str, Any]] = {}

    ctx_params: Dict[str, Any] = dict(params_global)

    for proc in processors:
        proc_params = dict(ctx_params)
        proc_params.update(params_by_processor.get(proc.name, {}))

        if hasattr(proc, "setup"):
            proc.setup(proc_params)

        proc_results, proc_state = proc.run(audio, proc_params)

        proc_results = dict(proc_results) if isinstance(proc_results, dict) else {"value": proc_results}
        proc_state = dict(proc_state) if isinstance(proc_state, dict) else {"state": proc_state}

        proc_state["file_key"] = file_key
        states_for_file[proc.name] = proc_state

        row.update(_flatten_with_namespace(proc.name, proc_results))

        updates = {}
        updates.update(_extract_param_updates(proc_results))
        updates.update(_extract_param_updates(proc_state))
        if updates:
            ctx_params.update(updates)

    if (
        "rain__rain_drops" in row
        and rain_actual is not None
        and rain_min_thr is not None
    ):
        rain_predicted = bool(row["rain__rain_drops"] > rain_min_thr)
        row["rain__predicted"] = rain_predicted
        row["rain__mismatch"] = (rain_predicted != bool(rain_actual))

    return {
        "row": row,
        "states": states_for_file,
    }


def _run_batch_serial(
    *,
    dir_content: Dict[str, Dict[str, Any]],
    processors: List[AudioProcessor],
    params_global: Dict[str, Any],
    params_by_processor: Dict[str, Dict[str, Any]],
    required_samples: int,
    rain_min_thr,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for file_key, meta in dir_content.items():
        item = _process_single_file_task(
            file_key=file_key,
            meta=meta,
            processors=processors,
            params_global=params_global,
            params_by_processor=params_by_processor,
            required_samples=required_samples,
            rain_min_thr=rain_min_thr,
        )
        if item is not None:
            outputs.append(item)
    return outputs


def _run_batch_parallel(
    *,
    dir_content: Dict[str, Dict[str, Any]],
    processors: List[AudioProcessor],
    params_global: Dict[str, Any],
    params_by_processor: Dict[str, Dict[str, Any]],
    required_samples: int,
    rain_min_thr,
    num_workers: Optional[int] = None,
    executor: Optional[ProcessPoolExecutor] = None,
) -> List[Dict[str, Any]]:
    max_workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 1) - 1)
    outputs: List[Dict[str, Any]] = []

    owns_executor = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=max_workers)

    try:
        futures = [
            executor.submit(
                _process_single_file_task,
                file_key=file_key,
                meta=meta,
                processors=processors,
                params_global=params_global,
                params_by_processor=params_by_processor,
                required_samples=required_samples,
                rain_min_thr=rain_min_thr,
            )
            for file_key, meta in dir_content.items()
        ]

        for fut in as_completed(futures):
            item = fut.result()
            if item is not None:
                outputs.append(item)
    finally:
        if owns_executor and executor is not None:
            executor.shutdown(wait=True)

    return outputs


def _collect_batch_outputs(
    *,
    batch_outputs: List[Dict[str, Any]],
    results_rows: List[Dict[str, Any]],
    states_by_processor: Dict[str, List[Dict[str, Any]]],
    print_mismatched: bool,
    debug_all: bool,
) -> None:
    for item in batch_outputs:
        row = item["row"]
        states_for_file = item["states"]

        if (
            "rain__mismatch" in row
            and ((print_mismatched and row["rain__mismatch"]) or debug_all)
        ):
            rd = row.get("rain__rain_drop_count", row.get("rain__rain_drops"))
            print(
                f"[mismatch] {row['file_key']}  "
                f"actual={row.get('rain_actual')}  predicted={row.get('rain__predicted')}  "
                f"rain_drops={rd}"
            )

        results_rows.append(row)
        for proc_name, proc_state in states_for_file.items():
            states_by_processor[proc_name].append(proc_state)

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
# Memory logging helper
# ----------------------------------------------------------------------

def _log_memory_usage(prefix: str = "") -> None:
    """
    Log memory usage of main process and child processes (if psutil available).
    """
    if psutil is None:
        print(f"{prefix} psutil not available for memory logging")
        return

    try:
        proc = psutil.Process(os.getpid())
        main_mb = proc.memory_info().rss / 1024**2

        child_mb = 0.0
        for c in proc.children(recursive=True):
            try:
                child_mb += c.memory_info().rss / 1024**2
            except Exception:
                pass

        total_mb = main_mb + child_mb
        print(f"{prefix} memory: main={main_mb:.1f} MB  children={child_mb:.1f} MB  total={total_mb:.1f} MB")
    except Exception as e:
        print(f"{prefix} memory logging failed: {e}")



# ----------------------------------------------------------------------
# Parquet batch helpers
# ----------------------------------------------------------------------

def _write_parquet_chunk(rows: List[Dict[str, Any]], path: Path, sort_by_file_key: bool = True) -> None:
    """
    Write a list of dict rows to a parquet file if non-empty.
    """
    if not rows:
        return

    df = pd.DataFrame(rows)
    if sort_by_file_key and not df.empty and "file_key" in df.columns:
        df = df.sort_values("file_key").reset_index(drop=True)
    df.to_parquet(path, index=False)


# ----------------------------------------------------------------------
# Parquet compatibility helpers for nested numpy payloads in state dicts
# ----------------------------------------------------------------------

def _to_parquet_compatible_value(value: Any) -> Any:
    """
    Convert nested numpy-heavy values into parquet-friendly Python objects.

    Rules
    -----
    - np.ndarray -> list (recursively via tolist)
    - np scalar  -> Python scalar via .item()
    - dict       -> dict with converted values
    - list/tuple -> list with converted values
    - other      -> unchanged
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _to_parquet_compatible_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_parquet_compatible_value(v) for v in value]
    return value



def _make_state_rows_parquet_safe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert state rows into parquet-safe dict rows.

    Special handling:
    - If a row has a `features` dict containing `normalized_mode_flux_by_mode`
      with shape (n_modes, n_frames), expand it into separate top-level columns:
      `normalized_mode_flux_by_mode_0`, `..._1`, etc.
    - Keep the remaining `features` payload, but convert nested numpy objects to
      plain Python lists/scalars.

    The in-memory state rows are not modified; a transformed copy is returned
    for parquet writing only.
    """
    safe_rows: List[Dict[str, Any]] = []

    for row in rows:
        safe_row = dict(row)

        # Convert existing top-level numpy/list-like values first.
        for key, value in list(safe_row.items()):
            if key == "features":
                continue
            safe_row[key] = _to_parquet_compatible_value(value)

        features = safe_row.get("features")
        if isinstance(features, Mapping):
            features_copy = dict(features)
            nmfbm = features_copy.pop("normalized_mode_flux_by_mode", None)

            if nmfbm is not None:
                nmfbm_arr = np.asarray(nmfbm)
                if nmfbm_arr.ndim != 2:
                    raise ValueError(
                        "features['normalized_mode_flux_by_mode'] must be a 2-D array "
                        f"when present; got shape {nmfbm_arr.shape}"
                    )
                for mode_idx in range(nmfbm_arr.shape[0]):
                    safe_row[f"normalized_mode_flux_by_mode_{mode_idx}"] = nmfbm_arr[mode_idx].tolist()

            safe_row["features"] = _to_parquet_compatible_value(features_copy)
        else:
            safe_row["features"] = _to_parquet_compatible_value(features)

        safe_rows.append(safe_row)

    return safe_rows


def _flush_saved_batches(
    *,
    results_rows: List[Dict[str, Any]],
    states_by_processor: Dict[str, List[Dict[str, Any]]],
    save_dir: Path,
    save_prefix: str,
    flush_idx: int,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Flush accumulated results/state rows to parquet and return saved paths.

    State rows are converted to parquet-safe objects on write so that nested
    NumPy payloads (including 2-D `normalized_mode_flux_by_mode` inside the
    `features` dict) do not break parquet serialization.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_result_paths: List[str] = []
    saved_state_paths: Dict[str, List[str]] = {name: [] for name in states_by_processor}

    if results_rows:
        results_path = save_dir / f"{save_prefix}__results_part_{flush_idx:05d}.parquet"
        _write_parquet_chunk(results_rows, results_path)
        saved_result_paths.append(str(results_path))

    for name, rows in states_by_processor.items():
        if not rows:
            continue
        state_path = save_dir / f"{save_prefix}__state__{name}_part_{flush_idx:05d}.parquet"
        parquet_safe_rows = _make_state_rows_parquet_safe(rows)
        _write_parquet_chunk(parquet_safe_rows, state_path)
        saved_state_paths[name].append(str(state_path))


    return saved_result_paths, saved_state_paths


# ----------------------------------------------------------------------
# Helper for restoring state DataFrame from parquet
# ----------------------------------------------------------------------

def restore_state_df_from_parquet(path: str | Path) -> pd.DataFrame:
    """
    Restore a single saved state parquet file into the in-memory state schema.

    This reverses the parquet-write compatibility transform applied by
    `_make_state_rows_parquet_safe()` for the special
    `features['normalized_mode_flux_by_mode']` payload. If the parquet file
    contains top-level columns named `normalized_mode_flux_by_mode_<i>`, those
    columns are reassembled into a 2-D NumPy array and inserted back into the
    per-row `features` dict under the key `normalized_mode_flux_by_mode`.

    Parameters
    ----------
    path :
        Path to a single parquet file previously written by
        `_flush_saved_batches()` for processor state rows.

    Returns
    -------
    pd.DataFrame
        Restored DataFrame matching the in-memory state schema as closely as
        possible for a single parquet chunk.
    """
    df = pd.read_parquet(path).copy()

    nmf_cols = sorted(
        [
            col
            for col in df.columns
            if col.startswith("normalized_mode_flux_by_mode_")
        ],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )

    if not nmf_cols:
        return df

    restored_features: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        feat = dict(row["features"]) if isinstance(row.get("features"), dict) else {}

        parts = []
        valid = True
        for col in nmf_cols:
            v = row[col]
            if v is None:
                valid = False
                break
            parts.append(np.asarray(v))

        if valid:
            feat["normalized_mode_flux_by_mode"] = np.stack(parts, axis=0)

        restored_features.append(feat)

    df["features"] = restored_features
    df = df.drop(columns=nmf_cols)

    return df


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
    max_files: Optional[int] = None,
    max_batch_save: int = 10_000,
    batch_save_dir: Optional[str] = "./save_dir",
    batch_save_prefix: str = "audio_processing_dump",
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
    max_files :
        Optional limit on the number of input files (keys) to process. If
        provided, only the first `max_files` discovered keys will be processed.
    local_cache :
        Local cache root for remote audio (forwarded to get_input_data_fn).
    localStatus :
        Default raining flag for local data when label cannot be inferred.
    get_keys_fn :
        Optional key-discovery function. If None, uses audio_io.get_keys.
    get_input_data_fn :
        Optional audio-loading function. If None, uses audio_io.get_input_data.

    max_batch_save :
        Maximum number of accumulated result rows to keep in memory before
        flushing both results and per-processor states to parquet files.
        Set to 0 or a negative value to disable periodic flushing.
    batch_save_dir :
        Directory where parquet chunks will be written when in-memory rows
        exceed `max_batch_save`. Defaults to "./save_dir". Set to None to
        disable periodic flushing.
    batch_save_prefix :
        File prefix used for saved parquet chunks.

    Returns
    -------
    results_df :
        DataFrame with one row per file. Includes:
            - "file_key"
            - "rain_actual"
            - namespaced metrics "<processor>__<metric>" for each processor.
            - optional "rain__predicted" and "rain__mismatch" if a "rain"
              processor is present and debug options are enabled.
        When periodic flushing is enabled, intermediate chunks are written to
        parquet during processing. The returned DataFrame contains only the
        final in-memory remainder (which is also flushed at the end), while
        all saved parquet chunk paths are attached via DataFrame `.attrs`.
    states_df_by_proc :
        Mapping from processor name to a DataFrame of internal state for that
        processor. Each row corresponds to one input file and includes "file_key"
        plus whatever keys the processor returned in its state dict.
        When periodic flushing is enabled, intermediate chunks are written to
        parquet during processing. Each returned state DataFrame contains only
        the final in-memory remainder for that processor (which is also flushed
        at the end), while all saved parquet chunk paths are attached via
        DataFrame `.attrs`.
    """
    _wall_t0 = time.perf_counter()
    if params_by_processor is None:
        params_by_processor = {}
    if debug_params is None:
        debug_params = {}

    if max_batch_save is None:
        max_batch_save = 10_000
    if batch_save_dir is not None and max_batch_save <= 0:
        raise ValueError("max_batch_save must be > 0 when batch_save_dir is provided")

    save_dir_path = Path(batch_save_dir) if batch_save_dir is not None else None

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
    keys = _discover_keys(
        InputType=InputType,
        test_vector_path=test_vector_path,
        query=query,
        adse_engine=adse_engine,
        batch_size=batch_size,
        localStatus=localStatus,
        max_files=max_files,
        get_keys_fn=get_keys_fn,
    )

    if max_files is None:
        print(f"received {len(keys)} test vectors")
    else:
        print(f"received {len(keys)} test vectors (limited by max_files={max_files})")

    results_rows: List[Dict[str, Any]] = []
    states_by_processor: Dict[str, List[Dict[str, Any]]] = {p.name: [] for p in processors}
    saved_result_paths: List[str] = []
    saved_state_paths: Dict[str, List[str]] = {p.name: [] for p in processors}
    flush_idx = 0

    print_mismatched = bool(debug_params.get("print_mismatched", False))
    debug_all = bool(debug_params.get("debug_all", False))
    rain_min_thr = debug_params.get("rain_drop_min_thr", params_global.get("rain_drop_min_thr"))
    log_memory = bool(debug_params.get("log_memory", False))

    parallel = bool(debug_params.get("parallel", False))
    num_workers = debug_params.get("num_workers")

    total_batches = (len(keys) + batch_size - 1) // batch_size if batch_size > 0 else 1
    executor: Optional[ProcessPoolExecutor] = None
    if parallel:
        max_workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 1) - 1)
        executor = ProcessPoolExecutor(max_workers=max_workers)

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
        if parallel:
            batch_outputs = _run_batch_parallel(
                dir_content=dir_content,
                processors=processors,
                params_global=params_global,
                params_by_processor=params_by_processor,
                required_samples=required_samples,
                rain_min_thr=rain_min_thr,
                num_workers=num_workers,
                executor=executor,
            )
        else:
            batch_outputs = _run_batch_serial(
                dir_content=dir_content,
                processors=processors,
                params_global=params_global,
                params_by_processor=params_by_processor,
                required_samples=required_samples,
                rain_min_thr=rain_min_thr,
            )

        _collect_batch_outputs(
            batch_outputs=batch_outputs,
            results_rows=results_rows,
            states_by_processor=states_by_processor,
            print_mismatched=print_mismatched,
            debug_all=debug_all,
        )

        if log_memory:
            _log_memory_usage(prefix=f"[batch {batch_idx}]")

        # Flush accumulated rows once they exceed the configured in-memory threshold.
        if save_dir_path is not None and max_batch_save > 0 and len(results_rows) >= max_batch_save:
            flush_idx += 1
            chunk_result_paths, chunk_state_paths = _flush_saved_batches(
                results_rows=results_rows,
                states_by_processor=states_by_processor,
                save_dir=save_dir_path,
                save_prefix=batch_save_prefix,
                flush_idx=flush_idx,
            )
            saved_result_paths.extend(chunk_result_paths)
            for name, paths in chunk_state_paths.items():
                saved_state_paths[name].extend(paths)

            results_rows.clear()
            for rows in states_by_processor.values():
                rows.clear()
            gc.collect()

        # Free per-batch data to keep memory bounded
        del dir_content
        gc.collect()

    if executor is not None:
        executor.shutdown(wait=True)

    # Final flush: persist the last in-memory remainder as parquet as well,
    # while still returning it to the caller for backward compatibility.
    has_pending_state = any(rows for rows in states_by_processor.values())
    if save_dir_path is not None and (results_rows or has_pending_state):
        flush_idx += 1
        chunk_result_paths, chunk_state_paths = _flush_saved_batches(
            results_rows=results_rows,
            states_by_processor=states_by_processor,
            save_dir=save_dir_path,
            save_prefix=batch_save_prefix,
            flush_idx=flush_idx,
        )
        saved_result_paths.extend(chunk_result_paths)
        for name, paths in chunk_state_paths.items():
            saved_state_paths[name].extend(paths)

    # ------------------------------------------------------------------
    # 3) Collate results into DataFrames
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_rows)
    if not results_df.empty:
        results_df = results_df.sort_values("file_key").reset_index(drop=True)
    results_df.attrs["saved_parquet_files"] = saved_result_paths

    states_df_by_proc: Dict[str, pd.DataFrame] = {}
    for name, rows in states_by_processor.items():
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values("file_key").reset_index(drop=True)
        else:
            df = pd.DataFrame()
        df.attrs["saved_parquet_files"] = saved_state_paths.get(name, [])
        states_df_by_proc[name] = df

    _wall_t1 = time.perf_counter()
    total_wall_time_sec = _wall_t1 - _wall_t0
    total_files_processed = len(keys)
    files_per_sec_total = (
        (total_files_processed / total_wall_time_sec)
        if total_wall_time_sec > 0 else None
    )

    results_df.attrs["wall_time_sec"] = total_wall_time_sec
    results_df.attrs["num_files_processed_total"] = total_files_processed
    results_df.attrs["files_per_sec_total"] = files_per_sec_total

    for df in states_df_by_proc.values():
        df.attrs["wall_time_sec"] = total_wall_time_sec
        df.attrs["num_files_processed_total"] = total_files_processed
        df.attrs["files_per_sec_total"] = files_per_sec_total

    print(f"Total wall time: {total_wall_time_sec:.3f} s")
    print(f"Total files processed: {total_files_processed}")
    if files_per_sec_total is not None:
        print(f"Throughput: {files_per_sec_total:.3f} files/s")

    return results_df, states_df_by_proc


# Backwards-compatible alias
process_audio_batches = process_audio_batches_v2

