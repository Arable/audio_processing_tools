# rain_postprocess.py

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def postprocess_rain(
    results_df: pd.DataFrame,
    rain_states_df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the old-style (test_results_df, feature_df) from framework outputs
    for the 'rain' processor.
    """
    if results_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "test_count",
                    "file_key",
                    "rain_actual",
                    "rain_predicted",
                    "rain_drop_count",
                    "rain_peaks_count",
                    "rain_drop_count_mod",
                    "frain_mean",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "test_count",
                    "file_key",
                    "rain_actual",
                    "frain_mean",
                    "kurtosis",
                    "crest_factor",
                    "diff_energy",
                    "nov",
                ]
            ),
        )

    # --- test_results_df ---
    rd_col = (
        "rain__rain_drop_count"
        if "rain__rain_drop_count" in results_df.columns
        else "rain__rain_drops"
    )
    rain_drops = results_df[rd_col]
    frain_mean = results_df["rain__frain_mean"]
    thr = params.get("rain_drop_min_thr", 3)

    if "rain__predicted" in results_df.columns:
        rain_predicted = results_df["rain__predicted"].astype(bool)
    else:
        rain_predicted = (rain_drops > thr)

    test_results_df = pd.DataFrame(
        {
            "test_count": np.arange(len(results_df), dtype=int),
            "file_key": results_df["file_key"],
            "rain_actual": results_df.get(
                "rain_actual", pd.Series([None] * len(results_df))
            ),
            "rain_predicted": rain_predicted.astype(bool),
            "rain_drop_count": rain_drops,
            "rain_peaks_count": results_df.get("rain__rain_peaks_count", np.nan),
            "rain_drop_count_mod": results_df.get("rain__rain_drop_count_mod", np.nan),
            "frain_mean": frain_mean,
        }
    )

    # --- feature_df ---
    base = pd.DataFrame(
        {
            "test_count": np.arange(len(results_df), dtype=int),
            "file_key": results_df["file_key"],
            "rain_actual": results_df.get(
                "rain_actual", pd.Series([None] * len(results_df))
            ),
            "frain_mean": frain_mean,
        }
    )

    cols_needed = ["file_key", "nov"]
    if params.get("handle_fp") or params.get("handle_fn"):
        cols_needed += ["kurtosis", "crest_factor", "diff_energy"]

    cols_available = [c for c in cols_needed if c in rain_states_df.columns]

    if cols_available:
        states_slice = rain_states_df[cols_available].copy()
        feature_df = base.merge(states_slice, on="file_key", how="left")
    else:
        feature_df = base.copy()
        feature_df["nov"] = np.nan
        if params.get("handle_fp") or params.get("handle_fn"):
            feature_df["kurtosis"] = np.nan
            feature_df["crest_factor"] = np.nan
            feature_df["diff_energy"] = np.nan

    return test_results_df, feature_df
