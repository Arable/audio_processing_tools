import glob, json, os

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from audio_processing_tools.edge.device_dsd_processing_emulator import DsdProcessingEmualtor
from audio_processing_tools.parse import parse_mark_audio_file
from audio_processing_tools.fetch import get_device_raw_audio_data
from audio_processing_tools.transform import emulator_output_to_df


def load_results(pattern: str) -> pd.DataFrame:
    """
    Load results from JSON files matching a given pattern.

    Parameters:
    - pattern (str): The glob pattern used to find files.

    Returns:
    - pd.DataFrame: A DataFrame containing the aggregated results from the JSON files.
    """
    all_results = []
    for filename in glob.glob(pattern):
        with open(filename, "r") as file:
            result = json.load(file)
            results_of_interest = {
                "experiment": result["experiment"],
                "overall_accuracy": result["overall_accuracy"],
                "param_hash": filename.split("/")[-1].split("_")[-3],
                **result["parameters"],
                "n_tp": len(result["tp_classifications"]),
                "n_tn": len(result["tn_classifications"]),
                "n_fp": len(result["fp_classifications"]),
                "n_fn": len(result["fn_classifications"]),
            }
            all_results.append(results_of_interest)
    return pd.DataFrame(all_results)


def add_derived_metrics(result_df: pd.DataFrame) -> None:
    """
    Add derived metrics to the results DataFrame.

    Parameters:
    - result_df (pd.DataFrame): The DataFrame to which the derived metrics will be added.

    The function adds the following columns to the DataFrame:
    - 'truncated_hash': A truncated version of the 'param_hash' for display purposes.
    - 'true_positive_rate': The true positive rate calculated from 'n_tp' and 'n_fn'.
    - 'true_negative_rate': The true negative rate calculated from 'n_tn' and 'n_fp'.
    """
    result_df["truncated_hash"] = result_df["param_hash"].apply(
        lambda x: f"{x[:5]}...{x[-5:]}"
    )
    result_df["true_positive_rate"] = result_df["n_tp"] / (
        result_df["n_tp"] + result_df["n_fn"]
    )
    result_df["true_negative_rate"] = result_df["n_tn"] / (
        result_df["n_tn"] + result_df["n_fp"]
    )


def visualize_performance(result_df: pd.DataFrame, extra_params = None, extra_param_names =None) -> None:
    """
    Visualize the performance of different algorithm parameters using a Plotly graph.

    Parameters:
    - result_df (pd.DataFrame): The DataFrame containing the performance metrics.

    This function creates a Plotly figure visualizing the overall accuracy, true positive rate,
    and true negative rate of different algorithm parameters, distinguished by the test name.
    """

    # Sort the DataFrame for consistent plotting
    sorted_df = result_df.sort_values("overall_accuracy")

    markers = ["circle", "square", "diamond", "cross", "hexagon", "star"]
    marker_by_test_name = {
        name: shape for name, shape in zip(result_df["test_name"].unique(), markers)
    }
    fig = go.Figure()
    # Loop through each unique test name and add a trace for overall_accuracy
    for test_name, marker_symbol in marker_by_test_name.items():
        filtered_df = sorted_df[sorted_df["test_name"] == test_name]
        fig.add_trace(
            go.Scatter(
                x=filtered_df["truncated_hash"],
                y=filtered_df["overall_accuracy"],
                mode="markers",
                marker=dict(symbol=marker_symbol, size=9),  # Adjust size as needed
                name=f"{test_name}<br> Overall Accuracy",
                customdata=filtered_df["param_hash"],
                hovertemplate="<br>".join(
                    ["Accuracy: %{y}", "Hash: %{customdata}", "Test: " + test_name]
                ),
            )
        )

        # Add additional params
        if extra_params:
            if len(extra_params) > 0 and len(extra_param_names)>0 and len(extra_params)!=len(extra_param_names):
                raise Exception("if extra_param_names is provided, it must be of equal length to extra_params ")
            for param, name in zip(extra_params, extra_param_names):
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df["truncated_hash"],
                        y=filtered_df[param],
                        mode="markers",
                        marker=dict(
                            symbol=marker_symbol, size=7
                        ),  # Adjust size & color as needed
                        name=f"{test_name}<br>{name}",
                        customdata=filtered_df["param_hash"],
                        hovertemplate="<br>".join(
                            [name + " : %{y}", "Hash: %{customdata}", "Test: " + test_name]
                        ),
                    )
                )

    # Update the layout if needed, and then show the figure
    fig.update_layout(
        height=500,
        title="Performance of Different Algo Parameters",
        xaxis_title="Parameter Hash",
        yaxis_title="Metric Value",
        legend_title="Test and Metric",
    )

    fig.show()


def plot_energy_histogram_with_classification_results(
    df, title_suffix, raining_condition, log=True
):
    """Plot histogram of acoustic energy distribution with secondary x-axis for rain rate."""
    fig = go.Figure()
    for condition, name, opacity in [
        (
            True,
            "Raining<br>Correct Classification"
            if raining_condition
            else "Not Raining<br>False Positive",
            0.75,
        ),
        (
            False,
            "Raining<br>False Negative"
            if raining_condition
            else "Not Raining<br>Correct Classification",
            0.75,
        ),
    ]:
        filtered_df = df[
            (df["raining"] == raining_condition)
            & (df["classification_result"] == condition)
        ]
        fig.add_trace(
            go.Histogram(
                x=filtered_df["normalized_wdsd_sum_5min"],
                nbinsx=100,
                name=name,
                opacity=opacity,
            )
        )

    # Primary x-axis layout
    fig.update_layout(
        title=f"Acoustic Energy Distribution - {title_suffix}",
        yaxis={"title": "Count", "type": "log" if log else "linear"},
        xaxis={"title": "Weighted DSD Sum<br>(normalized)"},
        barmode="overlay",
    )

    # Calculating secondary x-axis values
    dsd_sum_ticks = np.linspace(
        start=df["normalized_wdsd_sum_5min"].min(),
        stop=df["normalized_wdsd_sum_5min"].max(),
        num=15,
    )

    def rain_regression_model(weighted_dsd_sum):
        """Regression model to calculate rain rate from weighted dsd sum."""
        a, b, c = 1.41079931e-03, 2.52101903e01, -7.44399129e-06
        return (a * weighted_dsd_sum) / (b + c * weighted_dsd_sum)

    ticktext = [f"{rain_regression_model(v) * (60 / 5):.2f}" for v in dsd_sum_ticks]

    # Add dummy trace for secondary x-axis to ensure it's displayed
    fig.add_trace(
        go.Scatter(
            x=[dsd_sum_ticks[0]],
            y=[0],
            xaxis="x2",
            showlegend=False,
            marker={"size": 1},
        )
    )

    # Update layout for secondary x-axis
    fig.update_layout(
        xaxis2={
            "title": "Inferred Rain Rate (mm/hr)",
            "overlaying": "x",
            "side": "top",
            "tickvals": dsd_sum_ticks,
            "ticktext": ticktext,
            "range": [min(dsd_sum_ticks), max(dsd_sum_ticks)],
            "position": 0.95,
        },
        margin=dict(t=150),  # Adjust top margin to make room for secondary x-axis title
    )

    fig.show()


def find_single_matching_file(directory, pattern):
    """Find a single file matching a pattern, raise exceptions for 0 or multiple files."""
    matching_files = glob.glob(os.path.join(directory, pattern))
    if not matching_files:
        raise FileNotFoundError("Could not find results matching the pattern")
    if len(matching_files) > 1:
        raise FileExistsError("Found more than one file matching the pattern")
    return matching_files[0]


def process_audio_data_through_dsd_emulator(df, local_audio_cache):
    """Fetch and process audio data through dsd emulator."""
    for uid, label_metadata in tqdm(df.iterrows(), total=len(df)):
        source_file = label_metadata["source_file"]
        audio_data = get_device_raw_audio_data(
            keys=[source_file], local_cache_location=local_audio_cache, verbose=False, header_only=False
        )
        sig, audio_metadata = parse_mark_audio_file(audio_data[source_file])

        # Process through dsd emulator
        sample_rate = audio_metadata["sample_rate"]
        sig_to_process = sig[
            int(label_metadata["segment_start_seconds"] * sample_rate) : int(
                label_metadata["segment_end_seconds"] * sample_rate
            )
        ]

        rmodel = DsdProcessingEmualtor(
            fs=sample_rate,
            frame_length=512,
            hop_length=512,
            bwindow=False,
            verbose=False,
            ts=0,
        )
        dsd_output = rmodel.process_audio_data(audio_data=sig_to_process, ts=0)

        # Save results
        _dsd_output_df = emulator_output_to_df(
            dsd_output, label_metadata["device"], label_metadata["start_time"]
        )

        RAIN_ENERGY_THRESHOLD = 0.6
        RAIN_LOG_FACTOR = 0.6

        def reverse_binning_func(drop_bin, threshold=RAIN_ENERGY_THRESHOLD):
            return (
                ((np.e ** (drop_bin * np.log(1.13))) - 1) / RAIN_LOG_FACTOR
            ) + threshold

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

        dsd_sum = add_weighted_dsd_data(_dsd_output_df, add_weighted_dsd_sum=True)[
            "weighted_dsd_sum"
        ].iloc[0]
        df.loc[uid, "weighted_dsd_sum"] = dsd_sum
    audio_duration_s = df["segment_end_seconds"] - df["segment_start_seconds"]
    df["normalized_wdsd_sum_5min"] = (
        df["weighted_dsd_sum"] / audio_duration_s * (60 * 5)
    )

    return df
