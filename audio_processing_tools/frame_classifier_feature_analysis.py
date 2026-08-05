import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


# Helper: build lagged boolean target for framewise classifier analysis
def build_lagged_boolean_target(
    y: np.ndarray,
    lag_frames: int = 0,
    tolerance_frames: int = 0,
) -> np.ndarray:
    """
    Build a lag-aware boolean target.

    For frame t, the output is True if any input positive label exists in the
    window [t - lag_frames - tolerance_frames, t - lag_frames + tolerance_frames].
    This is useful when the detector is known to lag the reference labels.
    """
    y = np.asarray(y, dtype=bool).reshape(-1)
    n = len(y)
    out = np.zeros(n, dtype=bool)

    for t in range(n):
        center = t - int(lag_frames)
        i0 = max(0, center - int(tolerance_frames))
        i1 = min(n, center + int(tolerance_frames) + 1)
        if i1 > i0:
            out[t] = bool(np.any(y[i0:i1]))

    return out


def plot_feature_rain_vs_no_rain(
    frame_df: pd.DataFrame,
    feature_cols,
    label_col: str = "td_soft_label",
    positive_label=True,
    bins: int = 60,
    max_features_per_fig: int = 6,
    dropna: bool = True,
    log_x_features=None,
    show_boxplot: bool = False,
):
    """
    Plot per-feature distributions for rain vs no-rain frames.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Frame-level dataframe.
    feature_cols : list[str] | str
        Feature column(s) to plot.
    label_col : str
        Column used to split frames into rain / no-rain.
        Examples: 'td_soft_label', 'is_rain_raw', 'raining'
    positive_label : Any
        Value in label_col treated as "rain".
    bins : int
        Histogram bins.
    max_features_per_fig : int
        Number of features to show before creating a new figure.
    dropna : bool
        Whether to drop NaN/inf values before plotting.
    log_x_features : set[str] | list[str] | None
        Feature names to plot on log-x scale.
    show_boxplot : bool
        If True, also show a compact boxplot below each histogram.
    """
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    feature_cols = [c for c in feature_cols if c in frame_df.columns]

    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found in frame_df.")

    if label_col not in frame_df.columns:
        raise ValueError(f"label_col={label_col!r} not found in frame_df.")

    log_x_features = set(log_x_features or [])

    rain_mask = frame_df[label_col] == positive_label
    no_rain_mask = ~rain_mask

    n_per_feature = 2 if show_boxplot else 1
    chunk_size = max_features_per_fig

    for start in range(0, len(feature_cols), chunk_size):
        cols_chunk = feature_cols[start:start + chunk_size]
        n_feats = len(cols_chunk)

        fig, axes = plt.subplots(
            nrows=n_feats * n_per_feature,
            ncols=1,
            figsize=(10, 4 * n_feats * n_per_feature),
            squeeze=False,
        )
        axes = axes.flatten()

        for i, col in enumerate(cols_chunk):
            ax_hist = axes[i * n_per_feature]

            rain_vals = frame_df.loc[rain_mask, col]
            no_rain_vals = frame_df.loc[no_rain_mask, col]

            rain_vals = pd.to_numeric(rain_vals, errors="coerce")
            no_rain_vals = pd.to_numeric(no_rain_vals, errors="coerce")

            if dropna:
                rain_vals = rain_vals.replace([np.inf, -np.inf], np.nan).dropna()
                no_rain_vals = no_rain_vals.replace([np.inf, -np.inf], np.nan).dropna()

            if len(rain_vals) == 0 and len(no_rain_vals) == 0:
                ax_hist.set_title(f"{col} (no valid values)")
                ax_hist.axis("off")
                continue

            # Use common plotting range from combined finite values
            combined = pd.concat([rain_vals, no_rain_vals], axis=0)
            combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

            if len(combined) == 0:
                ax_hist.set_title(f"{col} (no finite values)")
                ax_hist.axis("off")
                continue

            if col in log_x_features:
                positive_combined = combined[combined > 0]
                if len(positive_combined) > 0:
                    xmin, xmax = positive_combined.min(), positive_combined.max()
                    if xmin < xmax:
                        bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)
                        ax_hist.hist(
                            no_rain_vals[no_rain_vals > 0],
                            bins=bin_edges,
                            alpha=0.5,
                            density=True,
                            label="No rain",
                        )
                        ax_hist.hist(
                            rain_vals[rain_vals > 0],
                            bins=bin_edges,
                            alpha=0.5,
                            density=True,
                            label="Rain",
                        )
                        ax_hist.set_xscale("log")
                    else:
                        ax_hist.hist(no_rain_vals, bins=bins, alpha=0.5, density=True, label="No rain")
                        ax_hist.hist(rain_vals, bins=bins, alpha=0.5, density=True, label="Rain")
                else:
                    ax_hist.hist(no_rain_vals, bins=bins, alpha=0.5, density=True, label="No rain")
                    ax_hist.hist(rain_vals, bins=bins, alpha=0.5, density=True, label="Rain")
            else:
                ax_hist.hist(no_rain_vals, bins=bins, alpha=0.5, density=True, label="No rain")
                ax_hist.hist(rain_vals, bins=bins, alpha=0.5, density=True, label="Rain")

            rain_med = np.nanmedian(rain_vals) if len(rain_vals) else np.nan
            no_rain_med = np.nanmedian(no_rain_vals) if len(no_rain_vals) else np.nan

            ax_hist.set_title(
                f"{col} | rain n={len(rain_vals)}, no-rain n={len(no_rain_vals)}"
            )
            ax_hist.set_xlabel(col)
            ax_hist.set_ylabel("Density")
            ax_hist.legend()
            ax_hist.grid(alpha=0.3)

            txt = (
                f"rain median={rain_med:.3g}\n"
                f"no-rain median={no_rain_med:.3g}"
            )
            ax_hist.text(
                0.98, 0.95, txt,
                transform=ax_hist.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", alpha=0.15),
            )

            if show_boxplot:
                ax_box = axes[i * n_per_feature + 1]
                vals = []
                labels = []
                if len(no_rain_vals):
                    vals.append(no_rain_vals.values)
                    labels.append("No rain")
                if len(rain_vals):
                    vals.append(rain_vals.values)
                    labels.append("Rain")

                if len(vals):
                    ax_box.boxplot(vals, tick_labels=labels, vert=False, showfliers=False)
                    ax_box.set_title(f"{col} boxplot")
                    ax_box.grid(alpha=0.3)
                else:
                    ax_box.axis("off")

        plt.tight_layout()
        plt.show()


def plot_roc_curve(
    frame_df: pd.DataFrame,
    feature: str,
    label_col: str = "td_soft_label",
    positive_label=True,
    lag_frames: int = 0,
    tolerance_frames: int = 0,
):
    """
    Plot ROC curve for one feature against a binary label column.
    """
    if feature not in frame_df.columns:
        raise ValueError(f"feature={feature!r} not found in frame_df")
    if label_col not in frame_df.columns:
        raise ValueError(f"label_col={label_col!r} not found in frame_df")

    y_true = (frame_df[label_col] == positive_label).astype(bool).to_numpy()
    if lag_frames != 0 or tolerance_frames != 0:
        y_true = build_lagged_boolean_target(
            y_true,
            lag_frames=lag_frames,
            tolerance_frames=tolerance_frames,
        )
    y_true = y_true.astype(int)
    y_score = pd.to_numeric(frame_df[feature], errors="coerce")

    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC requires both positive and negative samples.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if lag_frames != 0 or tolerance_frames != 0:
        plt.title(
            f"ROC: {feature} | lag={lag_frames}, tol={tolerance_frames}"
        )
    else:
        plt.title(f"ROC: {feature}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }), roc_auc



def plot_pr_curve(
    frame_df: pd.DataFrame,
    feature: str,
    label_col: str = "td_soft_label",
    positive_label=True,
):
    """
    Plot precision-recall curve for one feature against a binary label column.
    """
    if feature not in frame_df.columns:
        raise ValueError(f"feature={feature!r} not found in frame_df")
    if label_col not in frame_df.columns:
        raise ValueError(f"label_col={label_col!r} not found in frame_df")

    y_true = (frame_df[label_col] == positive_label).astype(int)
    y_score = pd.to_numeric(frame_df[feature], errors="coerce")

    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        raise ValueError("PR curve requires both positive and negative samples.")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve: {feature}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # thresholds is length n-1, align for convenience with NaN padding
    thresholds_out = np.concatenate([thresholds, [np.nan]])
    return pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": thresholds_out,
    })



def ks_test_feature(
    frame_df: pd.DataFrame,
    feature: str,
    label_col: str = "td_soft_label",
    positive_label=True,
):
    """
    Run a two-sample KS test comparing the feature distributions for the
    positive and negative classes.
    """
    if feature not in frame_df.columns:
        raise ValueError(f"feature={feature!r} not found in frame_df")
    if label_col not in frame_df.columns:
        raise ValueError(f"label_col={label_col!r} not found in frame_df")

    vals = pd.to_numeric(frame_df[feature], errors="coerce")
    labels = frame_df[label_col] == positive_label

    pos = vals[labels].replace([np.inf, -np.inf], np.nan).dropna()
    neg = vals[~labels].replace([np.inf, -np.inf], np.nan).dropna()

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("KS test requires non-empty positive and negative samples.")

    stat, pval = ks_2samp(pos, neg)
    return {
        "feature": feature,
        "ks_stat": float(stat),
        "p_value": float(pval),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "positive_median": float(np.nanmedian(pos)),
        "negative_median": float(np.nanmedian(neg)),
    }



def threshold_sweep(
    frame_df: pd.DataFrame,
    feature: str,
    label_col: str = "td_soft_label",
    positive_label=True,
    thresholds=None,
    n_thresholds: int = 50,
):
    """
    Sweep thresholds for one feature and compute binary classification metrics.
    """
    if feature not in frame_df.columns:
        raise ValueError(f"feature={feature!r} not found in frame_df")
    if label_col not in frame_df.columns:
        raise ValueError(f"label_col={label_col!r} not found in frame_df")

    y_true = (frame_df[label_col] == positive_label).astype(int)
    y_score = pd.to_numeric(frame_df[feature], errors="coerce")

    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_score) == 0:
        raise ValueError("No valid values available for threshold sweep.")

    if thresholds is None:
        vmin = float(np.nanmin(y_score))
        vmax = float(np.nanmax(y_score))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Could not determine finite threshold range.")
        if vmin == vmax:
            thresholds = np.array([vmin], dtype=np.float64)
        else:
            thresholds = np.linspace(vmin, vmax, n_thresholds)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)

    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append({
            "threshold": float(thr),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "accuracy": float(accuracy),
            "f1": float(f1),
        })

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd


def build_frame_tuning_df(
    results_df,
    states_by_proc,
    proc_name="rain_detector",
    return_clip_df: bool = False,
    return_feature_cols: bool = False,
):
    rows = []
    clip_rows = []

    proc_states_df = states_by_proc[proc_name]
    if not isinstance(proc_states_df, pd.DataFrame):
        raise TypeError(
            f"states_by_proc[{proc_name!r}] is {type(proc_states_df)}, expected pandas DataFrame"
        )

    if not isinstance(results_df, pd.DataFrame):
        raise TypeError(
            f"results_df is {type(results_df)}, expected pandas DataFrame"
        )

    result_pos_by_file_key = None
    if "file_key" in results_df.columns:
        if results_df["file_key"].duplicated().any():
            raise ValueError(
                "results_df contains duplicate file_key values; build_frame_tuning_df "
                "requires unique file_key entries for robust file_key-based alignment."
            )
        result_pos_by_file_key = {
            fk: pos for pos, fk in enumerate(results_df["file_key"].tolist())
        }

    def _result_pos_for_state(state_row, idx):
        state_file_key = state_row.get("file_key", None)
        if (
            result_pos_by_file_key is not None
            and state_file_key is not None
            and state_file_key in result_pos_by_file_key
        ):
            return result_pos_by_file_key[state_file_key]

        if idx in results_df.index:
            try:
                return results_df.index.get_loc(idx)
            except KeyError:
                return None

        return None

    def _result_val(result_pos, col, default=np.nan):
        if result_pos is None or col not in results_df.columns:
            return default
        try:
            return results_df.iloc[result_pos][col]
        except Exception:
            return default

    def _first_valid(*vals, default=np.nan):
        for v in vals:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            return v
        return default

    for i, state_row in proc_states_df.iterrows():
        feat = state_row.get("features", None)
        result_pos = _result_pos_for_state(state_row, i)
        clip_idx = _first_valid(
            state_row.get("clip_idx", None),
            _result_val(result_pos, "clip_idx", default=np.nan),
            i,
            default=i,
        )

        clip_rec = {
            "clip_idx": clip_idx,
            "file_key": _first_valid(
                _result_val(result_pos, "file_key", default=None),
                state_row.get("file_key", None),
                default=None,
            ),
            "rain_actual": _result_val(result_pos, "rain_actual", default=np.nan),
            "rain_frame_count": _first_valid(
                _result_val(result_pos, f"{proc_name}__rain_frame_count", default=np.nan),
                state_row.get("rain_frame_count", np.nan),
                default=np.nan,
            ),
            "clip_rain_fraction": _first_valid(
                _result_val(result_pos, f"{proc_name}__clip_rain_fraction", default=np.nan),
                state_row.get("clip_rain_fraction", np.nan),
                default=np.nan,
            ),
            "clip_is_rain": _first_valid(
                _result_val(result_pos, f"{proc_name}__clip_is_rain", default=np.nan),
                state_row.get("clip_is_rain", np.nan),
                default=np.nan,
            ),
            "clip_rain_conf": _first_valid(
                _result_val(result_pos, f"{proc_name}__clip_rain_conf", default=np.nan),
                state_row.get("clip_rain_conf", np.nan),
                default=np.nan,
            ),
            "median_rain_conf": _first_valid(
                _result_val(result_pos, f"{proc_name}__median_rain_conf", default=np.nan),
                state_row.get("median_rain_conf", np.nan),
                default=np.nan,
            ),
        }
        clip_rows.append(clip_rec)

        # ------------------------------------------------------------------
        # New path: compact exported feature payload
        # ------------------------------------------------------------------
        if isinstance(feat, dict):
            frame_times = feat.get("frame_times", None)
            if frame_times is None:
                continue

            frame_times = np.asarray(frame_times)
            n_frames = len(frame_times)
            if n_frames == 0:
                continue

            # Frame-level metadata only: keep file_key + clip_idx as join keys.
            row_meta = {
                "clip_idx": clip_idx,
                "file_key": clip_rec["file_key"],
            }

            frame_class = np.asarray(feat.get("frame_class", np.full(n_frames, -1)))
            is_rain = np.asarray(feat.get("is_rain", np.full(n_frames, False)))
            rain_conf = np.asarray(feat.get("rain_conf", np.full(n_frames, np.nan)))
            noise_conf = np.asarray(feat.get("noise_conf", np.full(n_frames, np.nan)))
            mode_flux_score = np.asarray(feat.get("mode_flux_score", np.full(n_frames, np.nan)))
            peak_ratio = np.asarray(feat.get("peak_ratio", np.full(n_frames, np.nan)))
            norm_by_mode = feat.get("normalized_mode_flux_by_mode", None)
            td_soft_label = np.asarray(feat.get("td_soft_label", np.full(n_frames, False)))
            td_time_flux_score = np.asarray(feat.get("td_time_flux_score", np.full(n_frames, np.nan)))
            td_crest_factor = np.asarray(feat.get("td_crest_factor", np.full(n_frames, np.nan)))
            td_kurtosis = np.asarray(feat.get("td_kurtosis", np.full(n_frames, np.nan)))
            td_rise_time_sec = np.asarray(feat.get("td_rise_time_sec", np.full(n_frames, np.nan)))
            td_fall_time_sec = np.asarray(feat.get("td_fall_time_sec", np.full(n_frames, np.nan)))
            td_rise_slope = np.asarray(feat.get("td_rise_slope", np.full(n_frames, np.nan)))
            td_fall_slope = np.asarray(feat.get("td_fall_slope", np.full(n_frames, np.nan)))

            def _safe_get(arr, t, default=np.nan):
                arr = np.asarray(arr)
                return arr[t] if t < len(arr) else default

            for t in range(n_frames):
                rec = dict(row_meta)
                rec.update({
                    "frame_idx": t,
                    "frame_time": frame_times[t],
                    "frame_class": _safe_get(frame_class, t, -1),
                    "is_rain": bool(_safe_get(is_rain, t, False)),
                    "rain_conf": _safe_get(rain_conf, t, np.nan),
                    "noise_conf": _safe_get(noise_conf, t, np.nan),
                    "mode_flux_score": _safe_get(mode_flux_score, t, np.nan),
                    "peak_ratio": _safe_get(peak_ratio, t, np.nan),
                    "td_soft_label": bool(_safe_get(td_soft_label, t, False)),
                    "td_time_flux_score": _safe_get(td_time_flux_score, t, np.nan),
                    "td_crest_factor": _safe_get(td_crest_factor, t, np.nan),
                    "td_kurtosis": _safe_get(td_kurtosis, t, np.nan),
                    "td_rise_time_sec": _safe_get(td_rise_time_sec, t, np.nan),
                    "td_fall_time_sec": _safe_get(td_fall_time_sec, t, np.nan),
                    "td_rise_slope": _safe_get(td_rise_slope, t, np.nan),
                    "td_fall_slope": _safe_get(td_fall_slope, t, np.nan),
                })

                if isinstance(norm_by_mode, np.ndarray) and norm_by_mode.ndim == 2:
                    for m in range(norm_by_mode.shape[0]):
                        if t < norm_by_mode.shape[1]:
                            rec[f"norm_mode_flux_{m}"] = norm_by_mode[m, t]

                rows.append(rec)

            continue

        # ------------------------------------------------------------------
        # Fallback path: old debug-based layout
        # ------------------------------------------------------------------
        dbg = state_row.get("debug", None)
        if not isinstance(dbg, dict):
            continue

        det = dbg.get("detector", {})
        if not isinstance(det, dict):
            det = {}

        td = det.get("td_soft", {})
        if not isinstance(td, dict):
            td = {}

        frame_times = td.get("frame_times", None)
        if frame_times is None:
            frame_times = dbg.get("times_s", None)
        if frame_times is None:
            frame_times = state_row.get("times", None)
        if frame_times is None:
            continue

        frame_times = np.asarray(frame_times)
        n_frames = len(frame_times)
        if n_frames == 0:
            continue

        row_meta = {
            "clip_idx": clip_idx,
            "file_key": clip_rec["file_key"],
        }

        frame_class = np.asarray(det.get("frame_class", np.full(n_frames, -1)))
        rain_conf = np.asarray(dbg.get("rain_conf", np.full(n_frames, np.nan)))
        noise_conf = np.asarray(dbg.get("noise_conf", np.full(n_frames, np.nan)))
        is_rain_raw = np.asarray(det.get("is_rain_raw", np.full(n_frames, False)))
        mode_flux_score = np.asarray(det.get("mode_flux_score", np.full(n_frames, np.nan)))
        peak_ratio = np.asarray(det.get("peak_ratio", np.full(n_frames, np.nan)))
        norm_by_mode = det.get("normalized_mode_flux_by_mode", None)
        td_time_flux_score = np.asarray(td.get("time_flux_score", np.full(n_frames, np.nan)))
        td_crest_factor = np.asarray(td.get("crest_factor", np.full(n_frames, np.nan)))
        td_kurtosis = np.asarray(td.get("kurtosis", np.full(n_frames, np.nan)))
        td_soft_label = np.asarray(td.get("soft_label", np.full(n_frames, False)))
        td_rise_time_sec = np.asarray(td.get("rise_time_sec", np.full(n_frames, np.nan)))
        td_fall_time_sec = np.asarray(td.get("fall_time_sec", np.full(n_frames, np.nan)))
        td_rise_slope = np.asarray(td.get("rise_slope", np.full(n_frames, np.nan)))
        td_fall_slope = np.asarray(td.get("fall_slope", np.full(n_frames, np.nan)))

        for t in range(n_frames):
            rec = dict(row_meta)
            rec.update({
                "frame_idx": t,
                "frame_time": frame_times[t],
                "frame_class": frame_class[t],
                "rain_conf": rain_conf[t],
                "noise_conf": noise_conf[t],
                "is_rain_raw": bool(is_rain_raw[t]),
                "mode_flux_score": mode_flux_score[t],
                "peak_ratio": peak_ratio[t],
                "td_time_flux_score": td_time_flux_score[t],
                "td_crest_factor": td_crest_factor[t],
                "td_kurtosis": td_kurtosis[t],
                "td_soft_label": bool(td_soft_label[t]),
                "td_rise_time_sec": td_rise_time_sec[t],
                "td_fall_time_sec": td_fall_time_sec[t],
                "td_rise_slope": td_rise_slope[t],
                "td_fall_slope": td_fall_slope[t],
            })

            if isinstance(norm_by_mode, np.ndarray) and norm_by_mode.ndim == 2:
                for m in range(norm_by_mode.shape[0]):
                    rec[f"norm_mode_flux_{m}"] = norm_by_mode[m, t]

            rows.append(rec)

    frame_df = pd.DataFrame(rows)
    clip_df = pd.DataFrame(clip_rows)

    if not clip_df.empty:
        dedupe_subset = [c for c in ["clip_idx", "file_key"] if c in clip_df.columns]
        if dedupe_subset:
            clip_df = clip_df.drop_duplicates(subset=dedupe_subset).reset_index(drop=True)
        else:
            clip_df = clip_df.reset_index(drop=True)

    frame_feature_cols = [
        c for c in frame_df.columns
        if c not in {"file_key", "clip_idx", "frame_idx", "frame_time"}
    ]

    if return_clip_df and return_feature_cols:
        return frame_df, clip_df, frame_feature_cols
    if return_clip_df:
        return frame_df, clip_df
    if return_feature_cols:
        return frame_df, frame_feature_cols
    return frame_df


def review_rain_detector_from_debug(one_state: dict, max_time_sec=None):
    """
    Visual review helper.

    Supports either:
      1) new compact `features` payload in one_state["features"]
      2) older debug-based layout in one_state["debug"]["detector"]
    """
    feat = one_state.get("features", None)
    if isinstance(feat, dict):
        t = np.asarray(feat.get("frame_times", []))
        if t.size == 0:
            raise KeyError("features['frame_times'] missing or empty")

        frame_class = np.asarray(feat.get("frame_class", np.full(len(t), -1)))
        is_rain = np.asarray(feat.get("is_rain", np.full(len(t), False)))
        rain_conf = np.asarray(feat.get("rain_conf", np.full(len(t), np.nan)))
        noise_conf = np.asarray(feat.get("noise_conf", np.full(len(t), np.nan)))

        flux_primary = np.asarray(feat.get("flux_primary", np.full(len(t), np.nan)))
        mode_flux_score = np.asarray(feat.get("mode_flux_score", np.full(len(t), np.nan)))
        peak_ratio = np.asarray(feat.get("peak_ratio", np.full(len(t), np.nan)))
        peak_gate_score = np.asarray(feat.get("peak_gate_score", np.full(len(t), np.nan)))
        norm_mode_flux = feat.get("normalized_mode_flux_by_mode", None)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax = axes[0]
        ax.plot(t, frame_class, linewidth=1.2, label="frame_class")
        ax.plot(t, is_rain.astype(float), linewidth=1.0, label="is_rain")
        ax.plot(t, rain_conf, linewidth=1.0, label="rain_conf")
        ax.plot(t, noise_conf, linewidth=1.0, label="noise_conf")
        ax.set_ylabel("class / conf")
        ax.set_title("Detector outputs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        ax = axes[1]
        ax.plot(t, mode_flux_score, linewidth=1.2, label="mode_flux_score")
        ax.plot(t, flux_primary, linewidth=1.2, label="flux_primary")
        ax.plot(t, peak_ratio, linewidth=1.2, label="peak_ratio")
        ax.plot(t, peak_gate_score, linewidth=1.2, label="peak_gate_score")
        ax.set_ylabel("score")
        ax.set_title("Main detector features")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        ax = axes[2]
        if isinstance(norm_mode_flux, np.ndarray) and norm_mode_flux.ndim == 2:
            for i in range(norm_mode_flux.shape[0]):
                ax.plot(t, norm_mode_flux[i], linewidth=1.0, label=f"mode_{i}")
            ax.legend(loc="upper right", ncol=2)
        ax.set_ylabel("norm flux")
        ax.set_title("normalized_mode_flux_by_mode")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)

        if max_time_sec is not None:
            axes[-1].set_xlim(0, max_time_sec)

        plt.tight_layout()
        return fig, axes

    dbg = one_state.get("debug", None)
    if not isinstance(dbg, dict):
        raise KeyError("state['features'] and state['debug'] are both missing or invalid")

    det = dbg.get("detector", None)
    if not isinstance(det, dict):
        raise KeyError("state['debug']['detector'] is missing or not a dict")

    t = np.asarray(dbg.get("times_s", one_state.get("times", None)))
    if t is None or len(t) == 0:
        raise KeyError("Could not find time axis in debug['times_s'] or state['times']")

    frame_class = np.asarray(det.get("frame_class", np.full(len(t), -1)))
    is_rain = np.asarray(dbg.get("is_rain", det.get("is_rain", np.full(len(t), False))))
    rain_conf = np.asarray(dbg.get("rain_conf", det.get("rain_conf", np.full(len(t), np.nan))))
    noise_conf = np.asarray(dbg.get("noise_conf", det.get("noise_conf", np.full(len(t), np.nan))))

    flux_primary = np.asarray(det.get("flux_primary", np.full(len(t), np.nan)))
    mode_flux_score = np.asarray(det.get("mode_flux_score", np.full(len(t), np.nan)))
    peak_ratio = np.asarray(det.get("peak_ratio", np.full(len(t), np.nan)))
    peak_gate_score = np.asarray(det.get("peak_gate_score", np.full(len(t), np.nan)))
    norm_mode_flux = det.get("normalized_mode_flux_by_mode", None)
    peak_count_by_mode = det.get("peak_count_by_mode", None)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, frame_class, linewidth=1.2, label="frame_class")
    ax.plot(t, is_rain.astype(float), linewidth=1.0, label="is_rain")
    ax.plot(t, rain_conf, linewidth=1.0, label="rain_conf")
    ax.plot(t, noise_conf, linewidth=1.0, label="noise_conf")
    ax.set_ylabel("class / conf")
    ax.set_title("Detector outputs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(t, mode_flux_score, linewidth=1.2, label="mode_flux_score")
    ax.plot(t, flux_primary, linewidth=1.2, label="flux_primary")
    ax.plot(t, peak_ratio, linewidth=1.2, label="peak_ratio")
    ax.plot(t, peak_gate_score, linewidth=1.2, label="peak_gate_score")
    ax.set_ylabel("score")
    ax.set_title("Main detector features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[2]
    if isinstance(norm_mode_flux, np.ndarray) and norm_mode_flux.ndim == 2:
        for i in range(norm_mode_flux.shape[0]):
            ax.plot(t, norm_mode_flux[i], linewidth=1.0, label=f"mode_{i}")
        ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel("norm flux")
    ax.set_title("normalized_mode_flux_by_mode")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if isinstance(peak_count_by_mode, np.ndarray) and peak_count_by_mode.ndim == 2:
        for i in range(peak_count_by_mode.shape[0]):
            ax.plot(t, peak_count_by_mode[i], linewidth=1.0, label=f"mode_{i}")
        ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel("peak count")
    ax.set_title("peak_count_by_mode")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)

    if max_time_sec is not None:
        axes[-1].set_xlim(0, max_time_sec)

    plt.tight_layout()
    return fig, axes

def add_combined_mode_flux_old_style(
    frame_df: pd.DataFrame,
    mode_cols=None,
    mode_thresholds=None,
    combine_threshold=None,
    hi_factor: float = 1.6,
    clip_factor: float = 1.5,
    mode_weights=None,
    out_score_col: str = "combined_mode_flux_old_style",
    out_rain_col: str = "rain_status_old_style",
    out_vote_col: str = "mode_vote_count_old_style",
):
    """
    Add old-style per-mode thresholding + clipping + summed score to frame_df.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Frame-level dataframe from build_frame_tuning_df().
    mode_cols : list[str] | None
        Columns to use as per-mode inputs.
        Default: norm_mode_flux_0..4 if present.
    mode_thresholds : list[float]
        Per-mode thresholds.
    combine_threshold : float | None
        Threshold on summed processed score for final rain decision.
        If None, defaults to sum of first 3 thresholds (old-style heuristic).
    hi_factor : float
        Saturation trigger multiplier, old logic used ~1.6.
    clip_factor : float
        Clipped output value multiplier, old logic used ~1.5.
    mode_weights : list[float] | None
        Optional per-mode weights applied after thresholding/clipping.
    """
    df = frame_df.copy()

    if mode_cols is None:
        mode_cols = [c for c in df.columns if c.startswith("norm_mode_flux_")]
        mode_cols = sorted(mode_cols, key=lambda x: int(x.split("_")[-1]))

    if not mode_cols:
        raise ValueError("No norm_mode_flux_* columns found")

    n_modes = len(mode_cols)

    if mode_thresholds is None:
        raise ValueError("mode_thresholds must be provided")

    if len(mode_thresholds) != n_modes:
        raise ValueError(
            f"mode_thresholds length ({len(mode_thresholds)}) "
            f"does not match number of mode columns ({n_modes})"
        )

    if mode_weights is None:
        mode_weights = np.ones(n_modes, dtype=np.float64)
    else:
        mode_weights = np.asarray(mode_weights, dtype=np.float64)
        if len(mode_weights) != n_modes:
            raise ValueError(
                f"mode_weights length ({len(mode_weights)}) "
                f"does not match number of mode columns ({n_modes})"
            )

    if combine_threshold is None:
        combine_threshold = float(np.sum(mode_thresholds[: min(3, n_modes)]))

    vals = df[mode_cols].to_numpy(dtype=np.float64)   # shape: (N, n_modes)
    processed = np.zeros_like(vals, dtype=np.float64)

    for i in range(n_modes):
        thr = float(mode_thresholds[i])
        v = vals[:, i]

        processed[:, i] = np.where(
            v > hi_factor * thr,
            clip_factor * thr,
            np.where(v > thr, v, 0.0),
        )

    weighted_processed = processed * mode_weights[None, :]
    combined_score = np.sum(weighted_processed, axis=1)
    vote_count = np.sum(processed > 0, axis=1)

    df[out_score_col] = combined_score
    df[out_vote_col] = vote_count
    df[out_rain_col] = combined_score > float(combine_threshold)

    # Optional: keep processed per-mode values for inspection
    for i in range(n_modes):
        df[f"old_style_mode_{i}"] = processed[:, i]

    return df

__all__ = [
    "add_combined_mode_flux_old_style",
    "build_lagged_boolean_target",
    "plot_feature_rain_vs_no_rain",
    "plot_roc_curve",
    "plot_pr_curve",
    "ks_test_feature",
    "threshold_sweep",
    "build_frame_tuning_df",
    "review_rain_detector_from_debug",
]
