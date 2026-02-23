import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

# include compatibility for python  < and >= 3.8
try:
    from importlib.metadata import version
except ModuleNotFoundError:
    import pkg_resources
import sqlalchemy

try:
    import cachesql  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cachesql = None  # Optional fallback; only required if you use cachesql-specific functions

from scipy import signal, fft
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from tqdm import tqdm

from audio_processing_tools.db_tools import get_db_data, upsert_df
from audio_processing_tools.fetch import get_raw_audio_data, get_device_raw_audio_data
from audio_processing_tools.parse import parse_mark_audio_file, parse_s3_audio_key, pcm_to_float
from audio_processing_tools.edge.device_dsd_processing_emulator import DsdProcessingEmualtor
# from audio_processing_tools.edge.dsp_rain_detection import analyse_raw_audio_wrapper


def butter_bandpass(lowcut, highcut, fs, order=5):
    return signal.butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def get_real_fft_df(sig, sample_rate):
    n_samples = len(sig)
    sample_spacing = 1 / sample_rate
    y_fft = fft.fft(sig)  # calculate fourier transform (complex numbers list)
    x_fft = fft.fftfreq(n_samples, sample_spacing)[: n_samples // 2]

    amplitude = 2.0 / n_samples * np.abs(y_fft[0 : n_samples // 2])

    df = pd.DataFrame({"frequency": x_fft, "amplitude": amplitude})
    return df


def emulator_output_to_df(
    output, device_id, audio_start_timestamp, output_interval_min=1
):
    dsd_cols = [f"dsd{i}" for i in range(32)]
    pft_cols = [f"pft{i}" for i in range(30)]
    fft_cols = [f"fft{i}" for i in range(38)]
    col_names = dsd_cols + pft_cols + fft_cols

    df = pd.DataFrame(output, columns=col_names)

    # dsd data from device has right edge time label while audio files have left edge. adding one minute to timestamp for consistency
    timestamps = pd.date_range(
        audio_start_timestamp + dt.timedelta(minutes=1),
        periods=len(df),
        freq=f"{output_interval_min}min",
    )
    df["time"] = timestamps
    df["device"] = device_id
    return df

def validate_db_engine(db_engine):
    """
    Validate that db_engine is an ADSE-connected engine.

    Supports:
      - SQLAlchemy Engine (preferred)
      - cachesql.sql.Database (optional, only if cachesql is installed)
    """
    is_sqlalchemy = isinstance(db_engine, sqlalchemy.engine.base.Engine)
    is_cachesql = cachesql is not None and isinstance(db_engine, cachesql.sql.Database)

    if not (is_sqlalchemy or is_cachesql):
        raise Exception(f"Did not recognize db engine type: {type(db_engine)}")

    if is_sqlalchemy:
        engine_name = str(db_engine.url)
    elif is_cachesql:
        engine_name = db_engine.name
    else:
        # Should not be reachable, but keep for safety
        raise Exception(f"Unsupported db engine type: {type(db_engine)}")

    if "adse" not in engine_name:
        raise Exception("Must provide db_engine that connects to ADSE database")

def fetch_audio_data(s3_file_key):
    try:
        # if data is not sent to ALP, may be in ALS or ALT ("arable-device-data-test")
        raw_audio_data = get_raw_audio_data(
            s3_file_key, "arable-device-data", use_caching=False
        )
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "NoSuchKey":
            try:
                raw_audio_data = get_raw_audio_data(
                    s3_file_key, "arable-device-data-test", use_caching=False
                )
            except Exception as e:
                raise (f"ERROR ACCESSING AUDIO FILE ({s3_file_key}):{e}")
        else:
            raise (f"ERROR ACCESSING AUDIO FILE ({s3_file_key}):{ex}")
    return raw_audio_data


def get_package_version():
    try:
        package_version = version("audio_processing_tools")
    except:
        package_version = pkg_resources.get_distribution("audio_processing_tools").version
    return package_version


RAIN_ENERGY_THRESHOLD = 0.6
RAIN_LOG_FACTOR = 0.6


def reverse_binning_func(drop_bin, threshold=RAIN_ENERGY_THRESHOLD):
    return (((np.e ** (drop_bin * np.log(1.13))) - 1) / RAIN_LOG_FACTOR) + threshold


dsd_weights = {f"dsd{i}": reverse_binning_func(i) for i in range(32)}


def add_weighted_dsd_data(
    df, weights=dsd_weights.values(), add_to_df=True, add_weighted_dsd_sum=False
):
    dsd_columns = [f"dsd{i}" for i in range(32)]
    dsd_data = df[dsd_columns]
    weighted_dsd_data = (dsd_data * weights).add_suffix("_weighted")
    if add_weighted_dsd_sum:
        weighted_dsd_data["weighted_dsd_sum"] = weighted_dsd_data.sum(axis=1)
    if add_to_df:
        return pd.concat([df, weighted_dsd_data], axis=1)
    else:
        return weighted_dsd_data


# def dsp_classification_from_audio_keys(
#     s3_file_keys: list,
#     db_engine,
#     reprocess=False,
#     force=True,
#     verbose=False,
#     local_raw_audio_cache=None,
# ):
#     """input s3 keys for raw audio files, get back dsp algorithm output (i.e. 'drop count' data)
#     s3_file_keys - list of Mark 3 raw audio s3 file keys intended for processing
#     db_engine - pandas compatible db engine/connection object to use to connect to db
#     reprocess - if True will reprocess audio data through dsp classifier and rewrite to db, if False db cache will be checked
#     force -  if False will cache db queries locally for expedited future runs
#     verbose - if True will have more verbose print statements
# 
#     """
# 
#     # guard against upserts to other dbs
#     validate_db_engine(db_engine)
# 
#     # First check db to see if processed data for this audio key already exists in db
#     if verbose:
#         print("Fetching dsp classification data from db")
#     query = f"""SELECT * FROM dsp_classification_from_raw_audio WHERE key in {tuple(s3_file_keys)} """
#     dsp_classification_output_df = get_db_data(
#         query, db_engine, force=force, cache=False
#     )
#     if verbose:
#         print("Checking if audio files require processing")
#     for key in tqdm(s3_file_keys):
#         if key in dsp_classification_output_df["key"].to_list() and reprocess is True:
#             dsp_classification_output_df = dsp_classification_output_df[
#                 dsp_classification_output_df["key"] != key
#             ]
#         if (
#             key not in dsp_classification_output_df["key"].to_list()
#             or reprocess is True
#         ):
#             raw_audio_data = fetch_audio_data(key)
#             sig, metadata = parse_mark_audio_file(raw_audio_data)
#             metadata = {**metadata, **parse_s3_audio_key(key)}
# 
#             # Only process complete 1-minute segments through dsd emulator
#             seconds_per_minute = 60
#             sample_rate = metadata["sample_rate"]
#             mins_to_process = int(
#                 round(len(sig) / sample_rate, 1) // seconds_per_minute
#             )
#             if mins_to_process < 1:
#                 raise Exception(
#                     "Cannot process audio file with duration less than 1 minute"
#                 )
#             minute_series = []
#             for i in range(mins_to_process):
#                 start_index = i * seconds_per_minute * sample_rate
#                 end_index = (i + 1) * seconds_per_minute * sample_rate
#                 sig_to_process = sig[start_index:end_index]
# 
#                 rain_drop_count, frain_mean = analyse_raw_audio_wrapper(
#                     sig_to_process, ref_file=None
#                 )
# 
#                 # dsd data from device has right edge time label while audio files have left edge.
#                 # adding one minute to timestamp for consistency
#                 minute_data = pd.Series(
#                     {
#                         "key": key,
#                         "time": metadata["time"] + dt.timedelta(minutes=1 + i),
#                         "rain_drop_count": rain_drop_count,
#                         "frain_mean": frain_mean,
#                         "sample_rate": sample_rate,
#                     }
#                 )
#                 minute_series.append(minute_data)
#             output_df = pd.DataFrame(minute_series)
# 
#             output_df["dsp_classifier_version"] = get_package_version()
#             output_df["device"] = metadata["device_id"]
#             current_time = dt.datetime.utcnow()
#             output_df["update_time"] = current_time
#             if reprocess is False:
#                 output_df["create_time"] = current_time
# 
#             # save results to db
#             if verbose:
#                 print("Saving processed audio file to db")
#             upsert_df(
#                 output_df.set_index(["key", "time"]),
#                 "dsp_classification_from_raw_audio",
#                 db_engine if isinstance(db_engine, sqlalchemy.engine.base.Engine)
#                 # sqlalchemy engine class access required for cachesql engine
#                 else (
#                     db_engine.engine
#                     if isinstance(db_engine, cachesql.sql.Database)
#                     else None
#                 ),
#             )
#             dsp_classification_output_df = pd.concat(
#                 [dsp_classification_output_df, output_df]
#             )
#     return dsp_classification_output_df


def process_audio_file_dsd(key, local_cache_location, verbose, reprocess):
    raw_audio_data = get_device_raw_audio_data(
        local_cache_location=local_cache_location,
        header_only=False,
        keys=[key],
        verbose=verbose,
        # limit threads to avoid overflow as this process is spawned by threads
        max_threads=1,
        show_progress=False
    )[key]
    sig, metadata = parse_mark_audio_file(raw_audio_data)
    metadata = {**metadata, **parse_s3_audio_key(key)}

    # if sig is longer than 1 minute, process only the first minute
    # just simplifies a bunch of data joinign and logic for current place of classification
    #  test vector database and energy content comparison. Can add second minute/improve later
    sig_duration_seconds = round(len(sig) / metadata["sample_rate"])
    if sig_duration_seconds > 60:
        sig_to_process = sig[:60*metadata["sample_rate"]]
    else:
        sig_to_process = sig

    # Process raw audio data through DSD emulator
    rmodel = DsdProcessingEmualtor(
        fs=metadata["sample_rate"],
        frame_length=512,
        hop_length=512,
        bwindow=False,
        ts=0,
        verbose=verbose,
    )
    dsd_output = rmodel.process_audio_data(audio_data=pcm_to_float(sig_to_process), ts=0)
    _dsd_output_df = emulator_output_to_df(
        dsd_output, metadata["device_id"], metadata["time"]
    )
    _dsd_output_df["key"] = key
    _dsd_output_df["update_time"] = dt.datetime.utcnow()

    # Duration calculation
    sig_duration_seconds = round(len(sig_to_process) / metadata["sample_rate"])
    _dsd_output_df["duration"] = sig_duration_seconds

    # Calculate weighted DSD sum
    _dsd_output_df["weighted_dsd_sum"] = add_weighted_dsd_data(
        _dsd_output_df, add_to_df=False, add_weighted_dsd_sum=True
    )["weighted_dsd_sum"]

    # Include sample rate
    _dsd_output_df["sample_rate"] = metadata["sample_rate"]

    # Get DSD Emulator Version

    try:
        package_version = version("audio_processing_tools")
    except:
        package_version = pkg_resources.get_distribution(
            "audio_processing_tools"
        ).version
    _dsd_output_df["dsd_emulator_version"] = package_version
    if reprocess is False:
        _dsd_output_df["create_time"] = _dsd_output_df["update_time"]

    return _dsd_output_df


def dsd_from_audio_keys(
    s3_file_keys,
    db_engine,
    reprocess=False,
    verbose=False,
    local_cache_location="raw_audio_cache",
    max_workers=None,
):
    """
    Fetch or compute DSD data for a list of S3 file keys.

    Supports:
      - SQLAlchemy Engine (preferred)
      - cachesql.sql.Database (optional, only if cachesql is installed)
    """
    validate_db_engine(db_engine)

    if verbose:
        print("Fetching existing DSD data from db")

    query = f"""SELECT * FROM dsd_from_raw_audio WHERE key IN {tuple(s3_file_keys)}"""
    existing_dsd_df = get_db_data(query, db_engine, force=True)

    existing_keys = (
        set(existing_dsd_df["key"].tolist()) if not existing_dsd_df.empty else set()
    )

    # Determine which keys need processing
    if reprocess:
        keys_to_process = s3_file_keys
    else:
        keys_to_process = [key for key in s3_file_keys if key not in existing_keys]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_audio_file_dsd,
                key,
                local_cache_location,
                verbose,
                reprocess,
            ): key
            for key in keys_to_process
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing audio files",
        ):
            result_df = future.result()
            results.append(result_df)
            if verbose:
                print(f"Processed and fetched results for key: {futures[future]}")

    processed_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # Decide which underlying engine to pass to upsert_df
    is_sqlalchemy = isinstance(db_engine, sqlalchemy.engine.base.Engine)
    is_cachesql = cachesql is not None and isinstance(db_engine, cachesql.sql.Database)

    if is_sqlalchemy:
        upsert_engine = db_engine
    elif is_cachesql:
        upsert_engine = db_engine.engine  # cachesql Database wraps a SQLAlchemy engine
    else:
        raise Exception(
            f"Unsupported db_engine type for upsert_df: {type(db_engine)}"
        )

    if not processed_df.empty:
        upsert_df(
            processed_df.set_index(["key", "time"]),
            "dsd_from_raw_audio",
            upsert_engine,
        )

    # Combine existing data with newly processed data
    if not reprocess:
        if not processed_df.empty:
            final_df = pd.concat([existing_dsd_df, processed_df], ignore_index=True)
        else:
            final_df = existing_dsd_df
    else:
        final_df = processed_df

    return final_df


# commented out as it was replaced with parallelized version. Didnt have too much time to thoroughly test but basic tests work
# have this version here (past working example) for reference in case anything is not working with new version
# def dsd_from_audio_keys(
#     s3_file_keys: list,
#     db_engine,
#     reprocess=False,
#     force=True,
#     verbose=False,
#     local_cache_location="raw_audio_cache",
# ):
#     """input s3 keys for raw audio files, get back processed DSD data
#     s3_file_keys - list of Mark 3 raw audio s3 file keys intended for processing
#     db_engine - pandas compatible db engine/connection object to use to connect to db
#     reprocess - if True will reprocess audio data through dsd emulator and rewrite to db, if False db cache will be checked
#     force -  if False will cache db queries locally for expedited future runs
#     verbose - if True will print extensive detail for DSD conversion process
#
#     """
#
#     # guard against upserts to other dbs
#     validate_db_engine(db_engine)
#
#     # First check db to see if processed data for this audio key already exists in db
#     if verbose:
#         print("Fetching dsd data from db")
#     query = f"""SELECT * FROM dsd_from_raw_audio WHERE key in {tuple(s3_file_keys)} """
#     dsd_output_df = get_db_data(query, db_engine, force=force, cache=False)
#     if verbose:
#         print("Checking if audio files require processing")
#     for key in tqdm(s3_file_keys):
#         if key in dsd_output_df["key"].to_list() and reprocess is True:
#             dsd_output_df = dsd_output_df[dsd_output_df["key"] != key]
#         if key not in dsd_output_df["key"].to_list() or reprocess is True:
#             raw_audio_data = get_device_raw_audio_data(
#                 local_cache_location=local_cache_location,
#                 header_only=False,
#                 keys=[key],
#                 verbose=False,
#             )[key]
#             sig, metadata = parse_mark_audio_file(raw_audio_data)
#             metadata = {**metadata, **parse_s3_audio_key(key)}
#
#             # Only process complete 1-minute segments through dsd emulator
#             seconds_per_minute = 60
#             sample_rate = metadata["sample_rate"]
#             sig_duration_seconds = round(len(sig) / sample_rate)
#             # mins_to_process = int(
#             #     sig_length_seconds // seconds_per_minute
#             # )
#             # if mins_to_process < 1:
#             #     raise Exception(
#             #         "Cannot process audio file with duration less than 1 minute"
#             #     )
#             # start_index = 0
#             # end_index = mins_to_process * seconds_per_minute * sample_rate
#             # sig_to_process = sig[start_index:end_index]
#             sig_to_process = sig
#
#             # process raw audio data through dsd emulator
#             bytes_per_sample, frame_size, duration = (
#                 2,
#                 512,
#                 sig_duration_seconds,
#             )
#             rmodel = DsdProcessingEmualtor(
#                 fs=sample_rate,
#                 frame_length=frame_size,
#                 hop_length=512,
#                 bwindow=False,
#                 ts=0,
#                 verbose=verbose,
#             )
#             dsd_output = rmodel.process_audio_data(audio_data=sig_to_process, ts=0)
#             _dsd_output_df = emulator_output_to_df(
#                 dsd_output, metadata["device_id"], metadata["time"]
#             )
#             _dsd_output_df["weighted_dsd_sum"] = add_weighted_dsd_data(
#                 _dsd_output_df, add_to_df=False, add_weighted_dsd_sum=True
#             )["weighted_dsd_sum"]
#
#             _dsd_output_df["duration"] = sig_duration_seconds
#             _dsd_output_df["key"] = key
#             _dsd_output_df["sample_rate"] = sample_rate
#             try:
#                 package_version = version("audio_processing_tools")
#             except:
#                 package_version = pkg_resources.get_distribution(
#                     "audio_processing_tools"
#                 ).version
#             _dsd_output_df["dsd_emulator_version"] = package_version
#
#             current_time = dt.datetime.utcnow()
#             _dsd_output_df["update_time"] = current_time
#             if reprocess is False:
#                 _dsd_output_df["create_time"] = current_time
#             # save results to db
#             if verbose:
#                 print("Saving processed audio file to db")
#             upsert_df(
#                 _dsd_output_df.set_index(["key", "time"]),
#                 "dsd_from_raw_audio",
#                 db_engine if isinstance(db_engine, sqlalchemy.engine.base.Engine)
#                 # sqlalchemy engine class access required for cachesql engine
#                 else (
#                     db_engine.engine
#                     if isinstance(db_engine, cachesql.sql.Database)
#                     else None
#                 ),
#             )
#             dsd_output_df = pd.concat([dsd_output_df, _dsd_output_df])
#     return dsd_output_df

