import datetime as dt
import os
import wave
import tempfile

import kaitaistruct
import numpy as np
import pandas as pd
from kaitaistruct import KaitaiStruct
from kaitaistruct import ValidationNotEqualError

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


def parse_mark_audio_file(file_contents: bytes, force_file_type=None):
    """
    Parse a MARK audio file, handling both legacy PCM and ALAC-encoded formats.
    
    Parameters
    ----------
    file_contents : bytes
        Raw binary contents of the audio file
    force_file_type : str, optional
        Override automatic format detection. Options: 'pcm', 'alac'
    
    Returns
    -------
    sig : np.ndarray
        Audio signal data
    metadata : dict
        Metadata about the audio file
    """
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
        file_version = parsed_binary['audio_file_version']
    except ValidationNotEqualError:
        print("COULD NOT FIND HEADER ON AUDIO, USING DEFAULT IMPORT PARAMS")
        sample_rate = 11162
        channels = 1
        bit_depth = 16
        endianness = 0
        file_version = 0

    # Determine file type (PCM vs ALAC)
    is_alac = force_file_type == 'alac' if force_file_type else file_version >= 1

    if is_alac:
        # Create temporary files for conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write original data to temp file
            input_path = os.path.join(temp_dir, "input.bin")
            with open(input_path, "wb") as f:
                f.write(audio_data)
            
            # Create paths for intermediate CAF file and final WAV
            caf_path = os.path.join(temp_dir, "audio.caf")
            wav_path = os.path.join(temp_dir, "audio.wav")
            
            # Convert to CAF and WAV
            rearrange(input_path, caf_path)
            os.system(f'ffmpeg -i "{caf_path}" "{wav_path}" -y -v error')
            
            # Read the WAV file
            with wave.open(wav_path, 'rb') as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
                sig = np.frombuffer(audio_data, dtype=np.int16)
    else:
        # Original PCM processing
        sig = np.frombuffer(audio_data, dtype=np.int16)

    duration = round(len(sig) / sample_rate, 2)

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
        "format": "alac" if is_alac else "pcm"
    }

    return sig, metadata


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
