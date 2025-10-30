import os
import datetime as dt
from tqdm import tqdm
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import NoCredentialsError, ProfileNotFound

PROD_AWS_PROFILE = "arable_prod"

# Notably, raw audio data from devices can end up in multiple locations:
# If the device is pointed to ALP, audio data will go to the arable-device-data bucket
# If the device is pointed to ALS or ALT, audio data will go to the arable-device-data-test bucket
# If the device is using the old format of sending data (as json chunks) then it will end up in the {bucket}/audio/{device}/ directory
# If the device is using the new format of sending data (as binary files) then it will end up in the {bucket}/raw_audio/{device}/ directory


def get_prod_boto_session(
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: Optional[str] = "us-east-1",
) -> boto3.session.Session:
    """
    Returns a boto3 session for accessing AWS.

    - By default, it tries to use the "arable_prod" profile.
    - If a different profile name is provided, it tries to use that.
    - If AWS access keys are provided, it creates a session using those.
    - If nothing is provided and no valid credentials exist, it falls back to the default session mechanism.
    """

    try:
        if aws_access_key_id and aws_secret_access_key:
            return boto3.session.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region,
            )
        elif profile_name:
            return boto3.session.Session(profile_name=profile_name)
        else:
            return boto3.session.Session(profile_name=PROD_AWS_PROFILE)
    except (ProfileNotFound, NoCredentialsError):
        print(
            f"WARNING: Could not find AWS credentials. Attempting to use default session."
        )
        return boto3.session.Session()


def fetch_raw_audio_from_s3(
    key_to_fetch: str,
    bucket: str,
    boto_session: Optional[boto3.session.Session] = None,
    header_only=False,
) -> bytes:
    """
    Simple function that returns contents of a file from S3

    Parameters
    ----------
    key_to_fetch - key for file to fetch
    bucket - bucket where file is stored
    boto_session - boto3 sessions with credentials for appropriate AWS account

    Returns
    -------
    object_content - bytes object containing raw audio file contents

    """

    if boto_session is None:
        boto_session = get_prod_boto_session()

    s3_client = boto_session.client("s3")
    start_byte, stop_byte = 0, 40
    s3_response_object = (
        s3_client.get_object(Bucket=bucket, Key=key_to_fetch)
        if header_only is False
        else s3_client.get_object(
            Bucket=bucket,
            Key=key_to_fetch,
            Range="bytes={}-{}".format(start_byte, stop_byte - 1),
        )
    )
    object_content = s3_response_object["Body"].read()
    return object_content


def get_raw_audio_data(
    file_key: str,
    bucket: str,
    boto_session: Optional[boto3.session.Session] = None,
    local_cache_location: str = "raw_audio_cache",
    redownload: bool = False,
    use_caching: bool = True,
    header_only: bool = False,
) -> bytes:
    """
    Returns contents of file from S3, with built-in caching functionality

    Parameters
    ----------
    file_key - key for file to fetch
    bucket - bucket where file is stored
    boto_session - boto3 sessions with credentials for appropriate AWS account
    local_cache_location - local directory to use for caching files
    redownload - if True, will redownload file from S3 even if it is present in cache and overwrite cached version
    use_caching - if False, will not attempt caching at all
    header_only - if True, will fetch only file header with metadata, not auido contents

    Returns
    -------
    bytes object containing raw audio file contents

    """
    if boto_session is None:
        boto_session = get_prod_boto_session()

    if use_caching:
        # make local cache if it doesn't exist
        if not os.path.isdir(local_cache_location):
            os.mkdir(local_cache_location)
        # check local cache for file
        local_path = os.path.join(local_cache_location, file_key)
        if os.path.isfile(local_path) and not redownload:
            with open(local_path, mode="rb") as file:
                raw_audio_content = file.read()
                return raw_audio_content
        # if file isn't present locally, pull from remote and store locally
        else:
            raw_audio_content = fetch_raw_audio_from_s3(
                file_key, bucket, boto_session, header_only
            )
            # save it to exact path but in local cache
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as file:
                file.write(raw_audio_content)
            return raw_audio_content
    else:
        return fetch_raw_audio_from_s3(file_key, bucket, boto_session, header_only)


def list_audio_keys(
    prefix: str,
    bucket: str,
    boto_session: Optional[boto3.session.Session] = None,
) -> list:
    """
    Return list of files with specific S3 prefix

    Parameters
    ----------
    prefix - Key prefix to use for keys of interest on S3
    bucket - bucket from which to pull keys
    boto_session - boto3 sessions with credentials for appropriate AWS account

    Returns
    -------
    key_list - list of S3 keys

    """

    if boto_session is None:
        boto_session = get_prod_boto_session()
    bucket_resource = boto_session.resource("s3").Bucket(bucket)
    key_list = [file.key for file in bucket_resource.objects.filter(Prefix=prefix)]
    return key_list


def get_device_audio_keys(
    device: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    bucket: str,
    parent_folder: str,
    boto_session: Optional[boto3.session.Session] = None,
) -> list:
    """
    Returns keys for audio files based on device id and time range

    Parameters
    ----------
    device - Mark device id
    start_date - start date for file query (must be datetime object)
    end_date - end date for file query (must be datetime object)
    bucket - bucket from which to pull audio files
    parent_folder - parent folder for audio files. 'audio' if firmware uses old format of sending data (as json chunks) and 'raw_audio' if firmware uses the new format of sending data (as binary files)
    boto_session - boto3 sessions with credentials for appropriate AWS account


    Returns
    -------
    keys_of_interest - list of keys that meet search criteria

    """
    if boto_session is None:
        boto_session = get_prod_boto_session()

    all_keys = list_audio_keys(
        f"{parent_folder}/{device}/",
        boto_session=boto_session,
        bucket=bucket,
    )
    if parent_folder == "audio":
        keys_by_start_date = {
            dt.datetime.fromtimestamp(int(p.split("/")[-1])): p for p in all_keys
        }
    elif parent_folder == "raw_audio":
        date_format = "%Y%m%d_%H_%M_%S_000000"
        keys_by_start_date = {
            dt.datetime.strptime(p.split("/")[-1].split("_rain_")[0], date_format): p
            for p in all_keys
        }
    else:
        raise Exception(
            f"Did not recognize parent folder: '{parent_folder}'. Expected 'audio' or 'raw_audio'. This info is necessary to determine how to parse file names. "
        )

    keys_of_interest = [
        key
        for date, key in keys_by_start_date.items()
        if end_date >= date >= start_date
    ]
    return keys_of_interest


def get_device_raw_audio_data(
    device: str = None,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    boto_session: Optional[boto3.session.Session] = None,
    local_cache_location: str = "raw_audio_cache",
    redownload: bool = False,
    use_caching: bool = True,
    header_only: bool = False,
    keys: Optional[List[str]] = None,
    verbose=False,
    max_threads: int = 10,
    show_progress: bool = True,
) -> dict:
    """
     Fetches raw audio data from AWS S3 buckets either by device and date range or by a list of specified keys,
     with options for caching and header-only retrieval, executed with concurrent threading to optimize performance.

     Parameters
     ----------
     device : str, optional
         Mark device ID for querying files by date range.
     start_date : datetime, optional
         Start date for the file query (required if device and end_date are provided).
     end_date : datetime, optional
         End date for the file query (required if device and start_date are provided).
     boto_session : boto3.session.Session
         boto3 session with credentials for the AWS account.
     local_cache_location : str
         Directory path for caching files locally.
     redownload : bool
         If True, redownloads file from S3 even if it is present in the cache and overwrites the cached version.
     use_caching : bool
         If False, caching will not be used.
     header_only : bool
         If True, fetches only the file header with metadata; if False, fetches full audio contents.
     keys : List[str], optional
         Specific list of file keys. If provided, device and date range are ignored.
     verbose : bool
         If True, prints additional information during execution.
     max_threads : int
         Maximum number of threads to use for concurrent file fetching. Helps manage performance and resource usage.
    show_progress : bool
        If True, shows progress bars during file fetching. If False, progress bars are hidden.
     Returns
     -------
     dict
         Dictionary mapping file keys to their content (bytes) or headers as specified.

     Raises
     ------
     Exception
         If neither device and date range nor keys are provided.
    """
    if boto_session is None:
        boto_session = get_prod_boto_session()

    if keys is None and (start_date is None or end_date is None or device is None):
        raise Exception(
            "Must provide start_date + end_date + device_id OR list of device keys"
        )

    buckets = ["arable-device-data-test", "arable-device-data"]
    all_files_contents_by_key = {}
    def fetch_file_content(key):
        for bucket in buckets:
            try:
                result = get_raw_audio_data(
                    key,
                    boto_session=boto_session,
                    bucket=bucket,
                    local_cache_location=local_cache_location,
                    redownload=redownload,
                    use_caching=use_caching,
                    header_only=header_only,
                )
                if result:
                    return key, result
            except Exception as e: # should make this less broad and only capture file not found...
                if verbose:
                    print(f"Error retrieving key {key} from bucket {bucket}: {e}")
        return key, None

    def process_keys(keys):
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_key = {executor.submit(fetch_file_content, key): key for key in keys}
            if show_progress:
                progress = tqdm(
                    as_completed(future_to_key),
                    total=len(keys),
                    desc="Fetching files",
                    unit="file",
                )
            else:
                progress = as_completed(future_to_key)
            for future in progress:
                key, result = future.result()
                if result:
                    all_files_contents_by_key[key] = result

    if keys is not None:
        if show_progress:
            tqdm.write("Downloading provided keys...")
        process_keys(keys)
    else:
        for bucket in buckets:
            parent_folders = (
                ["audio", "raw_audio"] if not header_only else ["raw_audio"]
            )
            for parent_folder in parent_folders:
                keys = get_device_audio_keys(
                    device,
                    start_date,
                    end_date,
                    bucket,
                    parent_folder,
                    boto_session=boto_session
                )
                if keys:
                    if show_progress:
                        tqdm.write(
                            f"Downloading {len(keys)} keys from {parent_folder} in bucket {bucket}"
                        )
                    process_keys(keys)

    return all_files_contents_by_key