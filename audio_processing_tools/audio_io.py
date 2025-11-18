from __future__ import annotations

"""
audio_io.py

Key discovery and audio loading utilities for the audio_processing_tools package.

Responsibilities:
    - Converting raw PCM / arrays to normalized float32
    - Ensuring mono, fixed-duration, fixed-sr signals
    - Discovering audio "keys" from local or DB-backed sources
    - Loading audio batches from local files or remote storage
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import librosa
import numpy as np
import pandas as pd

from .db_tools import get_db_data
from .fetch import get_device_raw_audio_data
from .parse import parse_mark_audio_file


# ----------------------------------------------------------------------
# Converters
# ----------------------------------------------------------------------


def safe_to_float(
    data: np.ndarray | bytes | bytearray | memoryview,
    bytes_per_sample: int = 2,
    signed: bool = True,
) -> np.ndarray:
    """
    Convert raw PCM samples or numeric arrays to float32 audio in the range [-1, 1].

    Parameters
    ----------
    data :
        Raw PCM buffer (bytes / bytearray / memoryview) or a NumPy array
        of dtype int16 or float.
    bytes_per_sample :
        Bytes per PCM sample. Only 2-byte (16-bit) signed PCM is supported.
    signed :
        Whether PCM data is signed. Only signed=True is supported.

    Returns
    -------
    np.ndarray
        1-D float32 array with values in [-1, 1].
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        if bytes_per_sample != 2 or not signed:
            raise ValueError("Only 16-bit signed PCM input is supported.")
        arr = np.frombuffer(data, dtype="<i2")  # little-endian int16
    else:
        arr = np.asarray(data)

    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float32, copy=False)
        return np.clip(out, -1.0, 1.0)

    if arr.dtype != np.int16:
        raise ValueError(f"Unsupported dtype {arr.dtype}; expected int16 or float.")

    scale = np.float32(32767.0)
    return arr.astype(np.float32) / scale


def ensure_mono_len_sr(
    y: np.ndarray,
    sr_in: int,
    sr_out: int,
    duration_s: float,
) -> Optional[np.ndarray]:
    """
    Force audio to mono, resample if needed, and trim to a fixed duration.

    Parameters
    ----------
    y :
        Input audio array, mono or multi-channel.
    sr_in :
        Sampling rate of the input signal.
    sr_out :
        Target sampling rate.
    duration_s :
        Required output duration in seconds.

    Returns
    -------
    np.ndarray or None
        1-D float32 mono array of length sr_out * duration_s in [-1, 1],
        or None if the input is too short after processing.
    """
    y = np.asarray(y)

    # Reduce multi-channel input to mono
    if y.ndim == 2:
        # Support both (channels, samples) and (samples, channels)
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)

    if sr_in != sr_out:
        y = librosa.resample(
            y.astype(np.float32, copy=False),
            orig_sr=sr_in,
            target_sr=sr_out,
        )

    required_len = int(sr_out * duration_s)
    if y.size < required_len:
        return None

    y = y[:required_len].astype(np.float32, copy=False)
    return np.clip(y, -1.0, 1.0)


# ----------------------------------------------------------------------
# Key discovery
# ----------------------------------------------------------------------


def get_db_file_list(
    query: str,
    adse_engine,
    file_path: str = "db_keys.csv",
) -> List[Dict[str, Any]]:
    """
    Fetch ['source_file', 'raining'] from the database, optionally using a local cache.

    If the cache file exists and contains the required columns, it is used.
    Otherwise, the query is executed against the provided engine.

    Parameters
    ----------
    query :
        SQL query that returns at least 'source_file' and 'raining' columns.
    adse_engine :
        SQLAlchemy engine to use for the query.
    file_path :
        Optional CSV cache file that stores the query result.

    Returns
    -------
    list of dict
        Each dict contains 'source_file' and 'raining'.
    """
    df: Optional[pd.DataFrame] = None

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not {"source_file", "raining"}.issubset(df.columns):
            print(f"Warning: {file_path} missing required columns; ignoring cache.")
            df = None

    if df is None:
        print("Querying database for audio keys...")
        df = get_db_data(query, adse_engine)
        required = {"source_file", "raining"}
        if not required.issubset(df.columns):
            raise ValueError("DB result must contain columns: 'source_file', 'raining'")
        # Optional cache:
        # df.to_csv(file_path, index=False)

    return df[["source_file", "raining"]].to_dict(orient="records")


def get_local_file_list(
    test_vector_path: str | Path,
    file_path: str = "local_keys.csv",
    localStatus: bool = True,
) -> List[Dict[str, Any]]:
    """
    Discover local audio files and infer raining labels.

    If the cache file exists and contains the required columns, it is used.
    Otherwise, the directory is scanned recursively for .bin and .wav files.

    Parameters
    ----------
    test_vector_path :
        Root directory to search for audio files.
    file_path :
        Optional CSV cache file that stores discovered keys.
    localStatus :
        Default raining flag when it cannot be inferred from the filename.

    Returns
    -------
    list of dict
        Each dict contains 'source_file' and 'raining'.
    """
    df: Optional[pd.DataFrame] = None

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not {"source_file", "raining"}.issubset(df.columns):
            print(f"Warning: {file_path} missing required columns; ignoring cache.")
            df = None

    if df is not None:
        return df[["source_file", "raining"]].to_dict(orient="records")

    if not test_vector_path:
        raise ValueError("test_vector_path must be provided for LocalPath input.")

    keys: List[Dict[str, Any]] = []
    for fname in Path(test_vector_path).rglob("*"):
        if not fname.is_file():
            continue
        suffix = fname.suffix.lower()
        if suffix in [".bin", ".wav"]:
            fstr = str(fname).lower()
            if "true" in fstr:
                raining = True
            elif "false" in fstr:
                raining = False
            else:
                raining = localStatus
            keys.append({"source_file": str(fname), "raining": raining})

    # Optional cache:
    # pd.DataFrame(keys).to_csv(file_path, index=False)

    return keys


def batched_query_to_dict_records(
    source_files: List[str],
    adse_engine,
    batch_size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Hydrate raining labels for a list of source_files in batches.

    Parameters
    ----------
    source_files :
        List of source_file identifiers to query.
    adse_engine :
        SQLAlchemy engine used to execute the queries.
    batch_size :
        Number of source_files per batch.

    Returns
    -------
    list of dict
        Each dict contains 'source_file' and 'raining'.

    Notes
    -----
    Assumes a table public.device_audio_rain_classification with columns:
        - source_file
        - raining
    """
    all_records: List[Dict[str, Any]] = []

    for i in range(0, len(source_files), batch_size):
        batch = source_files[i : i + batch_size]
        placeholders = ", ".join(f"'{s}'" for s in batch)
        q = f"""
            SELECT source_file, raining
            FROM public.device_audio_rain_classification
            WHERE source_file IN ({placeholders});
        """
        records = get_db_file_list(q, adse_engine)
        all_records.extend(records)

    return all_records


def get_keys(
    InputType: str,
    test_vector_path: Optional[str] = None,
    query: Optional[str] = None,
    adse_engine=None,
    batch_size: int = 1000,
    localStatus: bool = True,
    csv_inp_file: Optional[str] = None,
    key_list: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return a list of key records with 'source_file' and 'raining' fields.

    Supported InputType values:
        - "LocalPath"  : recursively scan a local directory
        - "RemotePath" : execute SQL query against a DB
        - "CsvInput"   : read source_file values from a CSV and hydrate via DB
        - "KeyList"    : hydrate via DB for a provided list of source_files

    The caller must supply a valid SQLAlchemy engine for any DB-backed option
    (RemotePath, CsvInput, KeyList).
    """
    if InputType == "LocalPath":
        if not test_vector_path:
            raise ValueError("LocalPath requires 'test_vector_path'.")
        print(f"LocalPath: scanning directory {test_vector_path}")
        return get_local_file_list(test_vector_path, localStatus=localStatus)

    if InputType == "RemotePath":
        if not query:
            raise ValueError("RemotePath requires 'query'.")
        if adse_engine is None:
            raise ValueError("RemotePath requires a valid 'adse_engine'.")
        print("RemotePath: executing SQL query")
        return get_db_file_list(query, adse_engine)

    if InputType == "CsvInput":
        if not csv_inp_file:
            raise ValueError("CsvInput requires 'csv_inp_file'.")
        if adse_engine is None:
            raise ValueError("CsvInput requires a valid 'adse_engine'.")
        print(f"CsvInput: loading file list from {csv_inp_file}")

        df = pd.read_csv(csv_inp_file)
        if "source_file" not in df.columns:
            raise ValueError("CsvInput CSV must contain a 'source_file' column.")

        source_files = (
            df["source_file"]
            .dropna()
            .astype(str)
            .tolist()
        )
        return batched_query_to_dict_records(source_files, adse_engine, batch_size=batch_size)

    if InputType == "KeyList":
        if not key_list:
            raise ValueError("KeyList requires 'key_list'.")
        if adse_engine is None:
            raise ValueError("KeyList requires a valid 'adse_engine'.")
        print("KeyList: hydrating provided keys via DB")
        return batched_query_to_dict_records(key_list, adse_engine, batch_size=batch_size)

    raise ValueError(
        f"Unknown InputType '{InputType}'. Expected one of: "
        "'LocalPath', 'RemotePath', 'CsvInput', 'KeyList'."
    )


# ----------------------------------------------------------------------
# Audio loading
# ----------------------------------------------------------------------

def get_input_data(
    batch_keys: List[Dict[str, Any]],
    InputType: str,
    Fs: int,
    check_duration: float,
    localStatus: bool,
    local_cache: Optional[str],
    read_size: Optional[int],
    bytes_per_sample: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Load audio for a batch of keys and return a normalized float32 buffer per file.

    Parameters
    ----------
    batch_keys :
        List of records containing at least 'source_file' and optionally 'raining'.
    InputType :
        "LocalPath" for local filesystem, any other value is treated as remote/S3.
    Fs :
        Target sampling rate in Hz.
    check_duration :
        Duration in seconds to extract per file.
    localStatus :
        Default raining flag when not provided in batch_keys (LocalPath only).
    local_cache :
        Local cache directory for remote/S3 audio (used by get_device_raw_audio_data).
    read_size :
        Unused in this implementation; kept for API compatibility.
    bytes_per_sample :
        Bytes per PCM sample for raw Mark-3 data (typically 2).

    Returns
    -------
    dict
        Mapping:
            {
              source_file: {
                  "file_contents": np.ndarray(float32 mono at Fs),
                  "raining": bool,
              },
              ...
            }

        Files shorter than the required duration are skipped.
    """
    dir_content: Dict[str, Dict[str, Any]] = {}
    required_samples = int(Fs * check_duration)

    if InputType == "LocalPath":
        # Local filesystem
        for key in batch_keys:
            audio_path = key["source_file"]
            raining = key.get("raining", localStatus)

            # WAV files
            if audio_path.lower().endswith(".wav"):
                try:
                    y, sr = librosa.load(audio_path, sr=None, mono=False)
                except Exception as e:
                    print(f"Error loading WAV file {audio_path}: {e}")
                    continue

                y = ensure_mono_len_sr(y, sr_in=sr, sr_out=Fs, duration_s=check_duration)
                if y is None:
                    continue

                dir_content[audio_path] = {"file_contents": y, "raining": raining}
                continue

            # Non-WAV (e.g., Mark-3 .bin)
            try:
                with open(audio_path, "rb") as f:
                    raw = f.read()
                audio_i16, _meta = parse_mark_audio_file(raw)
                y = safe_to_float(audio_i16, bytes_per_sample=bytes_per_sample)


                y = ensure_mono_len_sr(y, sr_in=Fs, sr_out=Fs, duration_s=check_duration)
                if y is None:
                    continue

                dir_content[audio_path] = {"file_contents": y, "raining": raining}

            except Exception as e:
                print(f"Error reading local file {audio_path}: {e}")
                continue

    else:
        # Remote/S3 path (keys identify remote objects)
        source_files = [k["source_file"] for k in batch_keys]
        raw_audio_map = get_device_raw_audio_data(
            keys=source_files,
            local_cache_location=local_cache,
            header_only=False,
        )

        for key in batch_keys:
            s = key["source_file"]
            raining = key.get("raining", False)
            raw = raw_audio_map.get(s)
            if raw is None:
                continue

            # Ensure even length for int16 PCM interpretation
            if len(raw) % 2:
                raw = raw[:-1]

            if len(raw) < 2 * required_samples:
                # Not enough samples for the requested duration
                continue

            try:
                # Parse Mark-3 container and normalize
                audio_i16, _meta = parse_mark_audio_file(raw)
                y = safe_to_float(audio_i16, bytes_per_sample=bytes_per_sample)

                y = ensure_mono_len_sr(y, sr_in=Fs, sr_out=Fs, duration_s=check_duration)
                if y is None:
                    continue

                dir_content[s] = {"file_contents": y, "raining": raining}

            except Exception as e:
                print(f"Error parsing remote audio for {s}: {e}")
                continue

    return dir_content

