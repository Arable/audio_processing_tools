from typing import Any, Tuple, List

import pandas as pd
import numpy as np
from audio_processing_tools.parse import parse_mark_audio_file
from audio_processing_tools.fetch import get_device_raw_audio_data
from audio_processing_tools.edge.dsp_rain_detection import (
    rain_detection_algo as python_rain_detection_algo,
)
from audio_processing_tools.edge.parameter_tuning.call_c_fun import (
    rain_detection_algo as c_rain_detection_algo,
)
# from audio_processing_tools.edge.dsp_rain_detection import (
#     analyse_raw_audio_wrapper as old_python_rain_detection_algo,
# )


def python_classifier_boolean_wrapper(
    audio_signal: np.ndarray, **kwargs
):
    """
    Evaluates whether an audio signal contains rain.

    Parameters:
    - audio_signal (np.ndarray): The audio signal to evaluate.
    - **kwargs (Any): Additional keyword arguments passed to the rain detection algorithm.

    Returns:
    - Union[bool, float]: Returns True if rain is detected above the threshold, False if below or equal to the threshold,
      and np.nan if the rain drop count is negative.
    """

    rain_drop_count, frain_mean = python_rain_detection_algo(audio_signal, **kwargs)
    if rain_drop_count > 0:
        return True
    elif rain_drop_count == 0:
        return False
    else:
        return np.nan


def c_classifier_boolean_wrapper(
    audio_signal: np.ndarray, **kwargs
):
    """
    Evaluates whether an audio signal contains rain.

    Parameters:
    - audio_signal (np.ndarray): The audio signal to evaluate.
    - **kwargs (Any): Additional keyword arguments passed to the rain detection algorithm.

    Returns:
    - Union[bool, float]: Returns True if rain is detected above the threshold, False if below or equal to the threshold,
      and np.nan if the rain drop count is negative.
    """

    rain_drop_count, frain_mean = c_rain_detection_algo(audio_signal, **kwargs)
    if rain_drop_count > 0:
        return True
    elif rain_drop_count == 0:
        return False
    else:
        return np.nan

def grid_search_classification_wrapper(
    audio_df: pd.DataFrame,
    local_audio_file_cache,
    boolean_algo,
    **params: Any
) -> Tuple[float, List[int], List[int], List[int], List[int]]:
    """
    Performs a single iteration of grid search  on audio data to determine the presence of rain by applying a specified algorithm.

    Parameters:
    - audio_df (pd.DataFrame): A DataFrame containing audio data with metadata information.
    - local_audio_file_cache (str): The directory path where audio files are cached for quick access.
    - boolean_algo (Callable): The classification algorithm to be used for evaluating the presence of rain. must accept audio signal as ndarray and return boolean reflecting rain classification
    - **params (Any): Additional keyword arguments containing parameters for the classifier algorithm.

    Returns:
    - Tuple[float, List[int], List[int], List[int], List[int]]: Returns a tuple containing the overall accuracy of the classification,
      and lists of unique identifiers (UIDs) for true positive, true negative, false positive, and false negative classifications, respectively.

    The function processes each segment within the audio data to extract relevant features and apply the classification algorithm. The results
    are then summarized to provide insights into the model's performance, specifically focusing on its ability to accurately detect rain sounds.
    """
    # reduce input data to cols of interest to speed slightly
    cols_of_interest = [
        "source_file",
        "raining",
        "segment_start_seconds",
        "segment_end_seconds",
    ]
    data_to_process = audio_df[cols_of_interest]

    # Initialize a dictionary to store classification results
    classification_results = {}

    for uid, row in data_to_process.iterrows():
        key = row["source_file"]
        audio_data = get_device_raw_audio_data(
            keys=[key],
            local_cache_location=local_audio_file_cache,
            header_only=False,
            verbose=False,
            show_progress=False,
        )
        audio_binary = audio_data[key]
        sig, metadata = parse_mark_audio_file(audio_binary)

        sample_rate = metadata["sample_rate"]
        start_time_s = row["segment_start_seconds"]
        end_time_s = row["segment_end_seconds"]
        start_frame = start_time_s * sample_rate
        end_frame = end_time_s * sample_rate

        sig_of_interest = sig[int(start_frame) : int(end_frame)]

        rain_status = boolean_algo(sig_of_interest, **params)

        # Store classification result using uid as key
        classification_results[uid] = rain_status

    # After loop, add all classification results to DataFrame at once
    # Convert classification_results to a Series and then assign it to avoid fragmentation
    data_to_process["classification_output"] = pd.Series(classification_results)

    true_positive_uids = data_to_process[
        (data_to_process["classification_output"] == True)
        & (data_to_process["raining"] == True)
    ].index.to_list()
    true_negative_uids = data_to_process[
        (data_to_process["classification_output"] == False)
        & (data_to_process["raining"] == False)
    ].index.to_list()
    false_positive_uids = data_to_process[
        (data_to_process["classification_output"] == True)
        & (data_to_process["raining"] == False)
    ].index.to_list()
    false_negative_uids = data_to_process[
        (data_to_process["classification_output"] == False)
        & (data_to_process["raining"] == True)
    ].index.to_list()

    overall_accuracy = 1 - (
        (len(false_negative_uids) + len(false_positive_uids)) / len(data_to_process)
    )

    return (
        overall_accuracy,
        true_positive_uids,
        true_negative_uids,
        false_positive_uids,
        false_negative_uids,
    )
