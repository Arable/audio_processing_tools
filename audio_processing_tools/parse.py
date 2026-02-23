from __future__ import annotations




from pathlib import Path

import numpy as np
import pandas as pd
import datetime as dt
import os
import wave
import tempfile
import logging
import shutil
import subprocess
from typing import Optional,Tuple, Dict, Any

import kaitaistruct
from kaitaistruct import ValidationNotEqualError
from kaitaistruct import KaitaiStruct


from audio_processing_tools.db_tools import upsert_df
from audio_processing_tools.fetch import get_device_raw_audio_data
from audio_processing_tools.alac_utils import rearrange


class AudioBinary(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.magic_bytes = self._io.read_bytes(4)
        if not self.magic_bytes == b"\xAD\xFB\xCA\xDE":
            raise kaitaistruct.ValidationNotEqualError(
                b"\xAD\xFB\xCA\xDE", self.magic_bytes, self._io, "/seq/0"
            )
        self.timestamp = self._io.read_u4le()
        self.sample_rate = self._io.read_u4le()
        self.num_channels = self._io.read_u1()
        self.adc_bitdepth = self._io.read_u1()
        self.endianness = self._io.read_u1()
        self.audio_file_version = self._io.read_u1()
        self.latitude = self._io.read_f4le()
        self.longitude = self._io.read_f4le()
        self.altitude = self._io.read_f4le()
        self.device_id = self._io.read_bytes(10).decode("UTF-8").rstrip("\x00")
        # Skip extra bytes that are added by FW because of some quirk
        self.skipped_bytes = self._io.read_bytes(2)
        self.audio = self._io.read_bytes_full()


def create_dict_by_kaitai(compressed_bin):
    audio_binary_obj = AudioBinary.from_bytes(compressed_bin)
    # print(audio_binary_obj.timestamp)

    # finally assemble the complete dictionary
    serialized = {
        "device": audio_binary_obj.device_id,
        "ts": audio_binary_obj.timestamp,
        "sample_rate": audio_binary_obj.sample_rate,
        "channels": audio_binary_obj.num_channels,
        "bit_depth": audio_binary_obj.adc_bitdepth,
        "endianness": audio_binary_obj.endianness,
        "gps": [
            audio_binary_obj.latitude,
            audio_binary_obj.longitude,
            audio_binary_obj.altitude,
        ],
        "audio_file_version": audio_binary_obj.audio_file_version,
        "audio": audio_binary_obj.audio,
    }
    return serialized


# def parse_mark_audio_file(file_contents: bytes, force_file_type=None):
#     """
#     Parse a MARK audio file, handling both legacy PCM and ALAC-encoded formats.
    
#     Parameters
#     ----------
#     file_contents : bytes
#         Raw binary contents of the audio file
#     force_file_type : str, optional
#         Override automatic format detection. Options: 'pcm', 'alac'
    
#     Returns
#     -------
#     sig : np.ndarray
#         Audio signal data
#     metadata : dict
#         Metadata about the audio file
#     """
#     try:
#         parsed_binary = create_dict_by_kaitai(file_contents)
#         sample_rate = parsed_binary["sample_rate"]
#         channels = parsed_binary["channels"]
#         bit_depth = parsed_binary["bit_depth"]
#         endianness = parsed_binary["endianness"]
#         gps = parsed_binary["gps"]
#         audio_data = parsed_binary["audio"]
#         device_id = parsed_binary["device"]
#         time = parsed_binary["ts"]
#         file_version = parsed_binary['audio_file_version']
#     except ValidationNotEqualError:
#         print("COULD NOT FIND HEADER ON AUDIO, USING DEFAULT IMPORT PARAMS")
#         sample_rate = 11162
#         channels = 1
#         bit_depth = 16
#         endianness = 0
#         file_version = 0

#     # Determine file type (PCM vs ALAC)
#     is_alac = force_file_type == 'alac' if force_file_type else file_version >= 1

#     if is_alac:
#         # Create temporary files for conversion
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Write original data to temp file
#             input_path = os.path.join(temp_dir, "input.bin")
#             with open(input_path, "wb") as f:
#                 f.write(audio_data)
            
#             # Create paths for intermediate CAF file and final WAV
#             caf_path = os.path.join(temp_dir, "audio.caf")
#             wav_path = os.path.join(temp_dir, "audio.wav")
            
#             # Convert to CAF and WAV
#             rearrange(input_path, caf_path)
#             os.system(f'ffmpeg -i "{caf_path}" "{wav_path}" -y -v error')
            
#             # Read the WAV file
#             with wave.open(wav_path, 'rb') as wav_file:
#                 audio_data = wav_file.readframes(wav_file.getnframes())
#                 sig = np.frombuffer(audio_data, dtype=np.int16)
#     else:
#         # Original PCM processing
#         sig = np.frombuffer(audio_data, dtype=np.int16)

#     duration = round(len(sig) / sample_rate, 2)

#     metadata = {
#         "sample_rate": sample_rate,
#         "channels": channels,
#         "bit_depth": bit_depth,
#         "endianness": endianness,
#         "device_id": device_id,
#         "time": time,
#         "lat": gps[0],
#         "long": gps[1],
#         "duration": duration,
#         "audio_file_version": file_version,
#         "format": "alac" if is_alac else "pcm"
#     }

#     return sig, metadata



def parse_mark_audio_file(
    file_contents: bytes,
    force_file_type: str | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a MARK audio file, handling both legacy PCM and ALAC-encoded formats.

    Ensures audio payload length is aligned to bit depth before decoding.

    Parameters
    ----------
    file_contents : bytes
        Raw binary contents of the audio file (full file, including header)
    force_file_type : {'pcm', 'alac'}, optional
        Override automatic format detection.

    Returns
    -------
    sig : np.ndarray
        Audio signal data (int16 PCM)
    metadata : dict
        Metadata about the audio file.
    """

    # ------------------------------------------------------------------
    # 1) Try to parse header with Kaitai
    # ------------------------------------------------------------------
    try:
        parsed_binary = create_dict_by_kaitai(file_contents)
        sample_rate = parsed_binary["sample_rate"]
        channels = parsed_binary["channels"]
        bit_depth = parsed_binary["bit_depth"]
        endianness = parsed_binary["endianness"]
        gps = parsed_binary["gps"]
        audio_data = parsed_binary["audio"]
        device_id = parsed_binary["device"]
        time = parsed_binary["ts"]
        file_version = parsed_binary["audio_file_version"]
    except ValidationNotEqualError:
        print("WARNING: Could not parse header, assuming raw PCM defaults")

        sample_rate = 11162
        channels = 1
        bit_depth = 16
        endianness = 0
        file_version = 0

        gps = (None, None)
        device_id = None
        time = None

        audio_data = file_contents

    # ------------------------------------------------------------------
    # 2) Normalize and validate bit depth
    # ------------------------------------------------------------------
    if bit_depth == 0:
        bit_depth = 16

    if bit_depth % 8 != 0:
        raise ValueError(f"Invalid bit depth {bit_depth}: must be multiple of 8")

    if bit_depth != 16:
        print(f"WARNING: Unsupported bit depth {bit_depth}; assuming 16-bit PCM compatibility")

    bytes_per_sample = bit_depth // 8

    # ------------------------------------------------------------------
    # 3) Align audio payload length BEFORE decoding
    # ------------------------------------------------------------------
    remainder = len(audio_data) % bytes_per_sample
    if remainder != 0:
        audio_data = audio_data[: len(audio_data) - remainder]

    # ------------------------------------------------------------------
    # 4) Decide file type (PCM vs ALAC)
    # ------------------------------------------------------------------
    if force_file_type == "alac":
        is_alac = True
    elif force_file_type == "pcm":
        is_alac = False
    else:
        # Default heuristic
        is_alac = file_version >= 1

    # ------------------------------------------------------------------
    # 5) Decode audio
    # ------------------------------------------------------------------
    if is_alac:
        sig = _decode_alac_to_pcm(audio_data)
    else:
        sig = _decode_pcm_payload(
            audio_data,
            bit_depth=bit_depth,
            channels=channels,
            endianness=endianness,
        )

    # ------------------------------------------------------------------
    # 6) Compute duration
    # ------------------------------------------------------------------
    if channels > 0:
        n_samples_per_channel = len(sig) / channels
    else:
        n_samples_per_channel = len(sig)

    duration = round(n_samples_per_channel / sample_rate, 2)

    # ------------------------------------------------------------------
    # 7) Metadata
    # ------------------------------------------------------------------
    metadata = {
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": bit_depth,
        "endianness": endianness,
        "device_id": device_id,
        "time": time,
        "lat": gps[0],
        "long": gps[1],
        "duration": duration,
        "audio_file_version": file_version,
        "format": "alac" if is_alac else "pcm",
    }

    return sig, metadata


logger = logging.getLogger(__name__)


def _resolve_ffmpeg_path(
    ffmpeg_path: Optional[str] = None,
) -> str:
    """
    Resolve the ffmpeg executable path in a cron-safe way.

    Resolution order:
      1) explicit argument `ffmpeg_path` (absolute or resolvable)
      2) env var FFMPEG_PATH
      3) shutil.which("ffmpeg") on current PATH
      4) common locations (macOS Homebrew, Linux)

    Raises
    ------
    FileNotFoundError
        If ffmpeg cannot be resolved.
    """
    candidates: list[str] = []

    if ffmpeg_path:
        candidates.append(ffmpeg_path)

    env_override = os.environ.get("FFMPEG_PATH")
    if env_override:
        candidates.append(env_override)

    # If user gave something like "ffmpeg" or "ffmpeg-7", try resolving via PATH
    for c in list(candidates):
        resolved = shutil.which(c)
        if resolved:
            return resolved
        p = Path(c)
        if p.exists() and os.access(str(p), os.X_OK):
            return str(p)

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    # Common fallback locations
    fallback_paths = [
        "/opt/homebrew/bin/ffmpeg",  # macOS Apple Silicon Homebrew
        "/usr/local/bin/ffmpeg",     # macOS Intel Homebrew
        "/usr/bin/ffmpeg",           # sometimes present on Linux
        "/bin/ffmpeg",
    ]
    for fp in fallback_paths:
        if os.path.exists(fp) and os.access(fp, os.X_OK):
            return fp

    raise FileNotFoundError(
        "ffmpeg not found. Install it (e.g., brew install ffmpeg / apt-get install ffmpeg) "
        "or set FFMPEG_PATH to the absolute path of ffmpeg (useful for cron)."
    )


def _subprocess_env_with_common_paths() -> dict:
    """
    Build a subprocess environment with extra PATH entries.
    This helps under cron where PATH can be minimal.
    """
    env = os.environ.copy()
    current = env.get("PATH", "")

    extras = [
        "/opt/homebrew/bin",  # macOS Apple Silicon
        "/usr/local/bin",     # macOS Intel / some Linux
        "/usr/bin",
        "/bin",
    ]

    parts = [p for p in current.split(os.pathsep) if p] if current else []
    # Prepend extras not already present
    new_parts = [p for p in extras if p not in parts] + parts
    env["PATH"] = os.pathsep.join(new_parts)
    return env


def _decode_alac_to_pcm(
    audio_data: bytes,
    *,
    ffmpeg_path: Optional[str] = None,
) -> np.ndarray:
    """
    Decode MARK ALAC payload to int16 PCM by:
      (1) writing payload to temp
      (2) rearranging into a CAF container
      (3) decoding CAF -> WAV via ffmpeg
      (4) loading WAV frames into numpy int16

    Parameters
    ----------
    audio_data:
        ALAC-encoded payload extracted from the MARK container.
    ffmpeg_path:
        Optional explicit ffmpeg path. If None, resolves via FFMPEG_PATH, PATH, and common fallbacks.

    Returns
    -------
    sig:
        Decoded PCM samples (int16). Interleaved if multichannel.

    Raises
    ------
    FileNotFoundError:
        If ffmpeg cannot be found.
    RuntimeError:
        If ffmpeg fails decoding.
    ValueError:
        If decoded WAV is not 16-bit PCM.
    """
    ffmpeg = _resolve_ffmpeg_path(ffmpeg_path)
    env = _subprocess_env_with_common_paths()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.bin")
        caf_path = os.path.join(temp_dir, "audio.caf")
        wav_path = os.path.join(temp_dir, "audio.wav")

        # Write raw ALAC payload
        with open(input_path, "wb") as f:
            f.write(audio_data)

        # Rearrange into CAF container (your existing function)
        rearrange(input_path, caf_path)

        # Decode via ffmpeg
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",  # only errors
            "-y",                  # overwrite
            "-i", caf_path,
            wav_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            # include some context for debugging
            raise RuntimeError(
                "ffmpeg failed while decoding ALAC.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {result.returncode}\n"
                f"stderr:\n{result.stderr.strip()}"
            )

        # Demux warnings sometimes appear even on success
        if result.stderr and "Error during demuxing" in result.stderr:
            logger.warning(
                "ffmpeg reported 'Error during demuxing' but decode succeeded. stderr=%s",
                result.stderr.strip(),
            )

        # Read back WAV as int16 PCM
        with wave.open(wav_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            if sampwidth != 2:
                raise ValueError(
                    f"Expected 16-bit WAV (sampwidth=2), got sampwidth={sampwidth}. "
                    f"n_channels={n_channels}"
                )

            raw = wav_file.readframes(wav_file.getnframes())
            sig = np.frombuffer(raw, dtype=np.int16)

        # If you want to return shape (N, C) instead of interleaved:
        # if n_channels > 1:
        #     sig = sig.reshape(-1, n_channels)

        return sig
# def _decode_alac_to_pcm(audio_data: bytes) -> np.ndarray:
#     """
#     Helper: decode ALAC payload to int16 PCM using rearrange() + ffmpeg.

#     Parameters
#     ----------
#     audio_data : bytes
#         ALAC-encoded payload extracted from the MARK container.

#     Returns
#     -------
#     sig : np.ndarray[int16]
#         Decoded PCM samples.
#     """
#     with tempfile.TemporaryDirectory() as temp_dir:
#         input_path = os.path.join(temp_dir, "input.bin")
#         caf_path = os.path.join(temp_dir, "audio.caf")
#         wav_path = os.path.join(temp_dir, "audio.wav")

#         # Write raw ALAC payload
#         with open(input_path, "wb") as f:
#             f.write(audio_data)

#         # Rearrange into CAF container
#         rearrange(input_path, caf_path)

#         # Run ffmpeg with proper error handling
#         result = subprocess.run(
#             [
#                 "ffmpeg",
#                 "-v", "error",      # only log errors
#                 "-y",               # overwrite
#                 "-i", caf_path,
#                 wav_path,
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )

#         if result.returncode != 0:
#             raise RuntimeError(
#                 f"ffmpeg failed while decoding ALAC (code {result.returncode}):\n{result.stderr}"
#             )

#         # Optional: warn if demuxing issues occurred but decode succeeded
#         if "Error during demuxing" in result.stderr:
#             print("Warning: ffmpeg reported 'Error during demuxing' but decode succeeded.")

#         # Read back WAV as int16 PCM
#         with wave.open(wav_path, "rb") as wav_file:
#             n_channels = wav_file.getnchannels()
#             sampwidth = wav_file.getsampwidth()
#             # You can re-check these against expected `bit_depth` if you want.
#             if sampwidth != 2:
#                 raise ValueError(f"Expected 16-bit WAV, got sampwidth={sampwidth}")

#             raw = wav_file.readframes(wav_file.getnframes())
#             sig = np.frombuffer(raw, dtype=np.int16)

#         # sig is interleaved if n_channels > 1; you can reshape if needed:
#         # sig = sig.reshape(-1, n_channels)

#         return sig


def _decode_pcm_payload(audio_data: bytes, bit_depth: int, channels: int, endianness: int) -> np.ndarray:
    """
    Helper: decode raw PCM payload from MARK file into int16 PCM.
    Currently assumes 16-bit PCM, mono or interleaved multi-channel.

    Parameters
    ----------
    audio_data : bytes
        Raw PCM payload.
    bit_depth : int
        Bits per sample (expected 16).
    channels : int
        Number of channels.
    endianness : int
        0 = little-endian, 1 = big-endian (based on your Kaitai spec).

    Returns
    -------
    sig : np.ndarray[int16]
        PCM samples.
    """
    if bit_depth != 16:
        # You can extend this if your format ever changes.
        raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")

    # Decide dtype string based on endianness
    # 0 => little-endian, 1 => big-endian (adapt if your spec uses different codes)
    if endianness == 0:
        dtype = "<i2"  # little-endian 16-bit signed
    else:
        dtype = ">i2"  # big-endian 16-bit signed

    sig = np.frombuffer(audio_data, dtype=dtype)

    # If you want native-endian int16:
    sig = sig.astype(np.int16, copy=False)

    # You can reshape to (n_frames, channels) if needed:
    # if channels > 1:
    #     sig = sig.reshape(-1, channels)

    return sig


def parse_s3_audio_key(key: str) -> dict:
    """
    Parse information stored in S3 Key for audio file

    Parameters
    ----------
    key - S3 Key

    Returns
    -------
    info - dictionary with information from key including device_id, location and time

    """
    components = key.split("/")
    # use parent folder to determine tpye of audio file. See comments in top of fetch.py file for detail
    parent_folder = components[0]
    if parent_folder == "audio":
        old_audio_files = True
    elif parent_folder == "raw_audio":
        old_audio_files = False
    else:
        raise Exception(
            f"Expected parent folder to be 'audio' or 'raw_audio' to determine file type for parsing but found: '{parent_folder}'"
        )

    if old_audio_files:
        info = dict(
            device_id=components[1],
            location=components[2],
            time=dt.datetime.fromtimestamp(int(components[3])),
        )
    else:
        date_format = "%Y%m%d_%H_%M_%S_000000"
        info = dict(
            device_id=components[1],
            time=dt.datetime.strptime(components[5].split("_rain_")[0], date_format),
        )

    return info


class AudioSignal:
    """
    Class to wrap audio signal in (as numpy array) primarily so that pandas can store it in a cell
    and not try to unravel it. May add other functionality later.
    """

    def __init__(self, contents):
        self.contents = contents


def tabularize_audio_data(binary_raw_audio: dict, device_metadata=True, force_file_type=None) -> pd.DataFrame:
    """
    convert raw audio files to tabular format with metadata
    Parameters
    ----------
    binary_raw_audio : dict
        Dictionary of binary audio files in the format {file_key:bytes_contents, ...}
    device_metadata : bool
        Boolean to control if additional metadata is added, can turn off to support audio files that do not have S3 path
    force_file_type : str, optional
        Override automatic format detection. Options: 'pcm', 'alac'
    Returns
    -------
    tabular_data : pd.DataFrame
        Pandas DataFrame with audio metadata and contents
    """
    tabular_data = pd.DataFrame()
    for key, datum in binary_raw_audio.items():
        sig, metadata = parse_mark_audio_file(datum, force_file_type=force_file_type)
        if device_metadata:
            additional_metadata = parse_s3_audio_key(key)
            metadata = {**metadata, **additional_metadata}

        tabular_data.loc[key, "signal"] = AudioSignal(sig)
        
        # add the source file (S3 key) as a column
        tabular_data.loc[key, "source_file"] = key

        for k, v in metadata.items():
            tabular_data.loc[key, k] = v
    return tabular_data


bytes_per_sample = 2


def pcm_to_float(signal, scale_factor=1 << (bytes_per_sample * 8 - 1)):
    return signal / scale_factor


class AudioMetadataHandler:
    def __init__(
        self,
        keys: list,
        sqlalchemy_db_engine,
        local_audio_cache="./raw_audio_cache",
        table_name="audio_metadata",
        batch_size=100  # Specify batch size for upserts
    ):
        self.keys = keys
        self.sqlalchemy_db_engine = sqlalchemy_db_engine
        self.local_audio_cache = local_audio_cache
        self.table_name = table_name
        self.batch_size = batch_size
        self.buffer = []  # Buffer to accumulate rows before upserting

    def fetch_and_store_metadata(self):
        # Fetch audio data using get_device_raw_audio_data function
        audio_data = get_device_raw_audio_data(
            keys=self.keys,
            local_cache_location=self.local_audio_cache,
            redownload=False,
            use_caching=True,
            header_only=False,
            verbose=False,
        )

        for key in self.keys:
            if key not in audio_data:
                print(f"Audio data for key {key} could not be fetched.")
                continue

            try:
                audio_binary = audio_data[key]
                _, metadata = parse_mark_audio_file(audio_binary)

                # Convert Unix time to human-readable format
                metadata["time"] = dt.datetime.utcfromtimestamp(metadata["time"])

                # Add source key to metadata
                metadata["source_key"] = key

                # Define column order, with any additional columns appended at the end
                column_order = [
                    "source_key",
                    "device_id",
                    "time",
                    "sample_rate",
                    "lat",
                    "long",
                    "duration",
                ]
                remaining_columns = [
                    col for col in metadata.keys() if col not in column_order
                ]
                metadata = {
                    col: metadata[col]
                    for col in column_order + remaining_columns
                }

                # Convert metadata to a DataFrame row and add to buffer
                self.buffer.append(pd.DataFrame([metadata]).set_index("source_key"))

                # Check if buffer is full, and if so, trigger batch upsert
                if len(self.buffer) >= self.batch_size:
                    self.upsert_batch()

            except Exception as e:
                print(f"Error processing key {key}: {e}")

        # Upsert any remaining data in the buffer
        if self.buffer:
            self.upsert_batch()

    def upsert_batch(self):
        # Concatenate all DataFrames in the buffer into a single DataFrame
        batch_data = pd.concat(self.buffer)
        self.buffer = []  # Clear the buffer

        try:
            upsert_df(batch_data, self.table_name, self.sqlalchemy_db_engine)
            print(f"Successfully inserted {len(batch_data)} entries.")
        except Exception as e:
            print(f"Error during database upsert: {e}")

# Example usage:
# db_engine = create_engine('your_sqlalchemy_connection_string_here')
# keys = ['key1', 'key2', 'key3']
# audio_handler = AudioMetadataHandler(keys=keys, sqlalchemy_db_engine=db_engine)
# audio_handler.fetch_and_store_metadata()
