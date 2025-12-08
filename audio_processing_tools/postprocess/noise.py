# noise_postprocess.py
import numpy as np
import pandas as pd
from typing import Dict, Any


def postprocess_noise(
    results_df: pd.DataFrame,
    noise_states_df: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Example: return a per-file noise summary.
    """
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "file_key",
                "rain_actual",
                "noise_snr_db",
                "noise_floor_db",
            ]
        )

    out = pd.DataFrame(
        {
            "file_key": results_df["file_key"],
            "rain_actual": results_df.get("rain_actual", None),
            "noise_snr_db": results_df.get("noise__snr_db", np.nan),
            "noise_floor_db": results_df.get("noise__floor_db", np.nan),
        }
    )
    return out
