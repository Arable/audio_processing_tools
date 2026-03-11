from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_noise_processing_results(
    x,
    y,
    sr,
    S,
    S_hat,
    freqs,
    times,
    noise_psd=None,
    fmax=4000.0,
    title_prefix="",
    debug=None,
):
    if title_prefix:
        title_prefix = title_prefix.strip() + " - "

    # --------- Playback ---------
    print(f"{title_prefix}Original audio:")
    ipd.display(ipd.Audio(x, rate=sr))

    print(f"{title_prefix}Denoised audio:")
    ipd.display(ipd.Audio(y, rate=sr))
    
    eps = 1e-9

    mag_orig_db = 20 * np.log10(np.abs(S) + eps)
    mag_deno_db = 20 * np.log10(np.abs(S_hat) + eps)

    noise_db = None
    if noise_psd is not None:
        noise_db = 10 * np.log10(noise_psd + eps)

    # Frequency mask for plotting
    fmask = freqs <= fmax

    # Decide layout: base spectrogram count
    n_specs = 3 if noise_db is not None else 2  # orig, denoised, (optional noise)

    # --- Debug signals from process() ---
    G = P_band = N_band = None
    if debug is not None:
        G = debug.get("G", None)
        # Your keys: "P_band_all", "N_band_all"
        P_band = debug.get("P_band_all", None)
        N_band = debug.get("N_band_all", None)

    extra_rows = 0
    if G is not None:
        extra_rows += 1
    if P_band is not None:
        extra_rows += 1
    if N_band is not None:
        extra_rows += 1

    # Total rows: 1 (waveforms) + spectrograms + debug rows
    total_rows = 1 + n_specs + extra_rows

    fig, axes = plt.subplots(
        total_rows, 1, figsize=(10, 3 * total_rows), sharex=False
    )

    # If only 1 axis, wrap in list for consistent indexing
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    row = 0

    # 1) Waveforms
    t_x = np.arange(len(x)) / sr
    t_y = np.arange(len(y)) / sr
    ax = axes[row]
    row += 1
    ax.plot(t_x, x, label="Original", alpha=0.7)
    ax.plot(t_y, y, label="Denoised", alpha=0.7)
    ax.set_title(f"{title_prefix}Waveforms")
    ax.set_xlabel("Time [s]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Original spectrogram
    ax = axes[row]
    row += 1
    im1 = ax.pcolormesh(
        times,
        freqs[fmask],
        mag_orig_db[fmask, :],
        shading="auto",
    )
    ax.set_ylabel("Freq [Hz]")
    ax.set_title("Original spectrogram [dB]")
    fig.colorbar(im1, ax=ax, label="dB")

    # 3) Denoised spectrogram
    ax = axes[row]
    row += 1
    im2 = ax.pcolormesh(
        times,
        freqs[fmask],
        mag_deno_db[fmask, :],
        shading="auto",
    )
    ax.set_ylabel("Freq [Hz]")
    ax.set_title("Denoised spectrogram [dB]")
    fig.colorbar(im2, ax=ax, label="dB")

    # 4) Noise PSD (if provided)
    if noise_db is not None:
        ax = axes[row]
        row += 1
        im3 = ax.pcolormesh(
            times,
            freqs[fmask],
            noise_db[fmask, :],
            shading="auto",
        )
        ax.set_ylabel("Freq [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_title("Estimated noise PSD [dB]")
        fig.colorbar(im3, ax=ax, label="dB")
    else:
        # If we didn't plot noise PSD, last spectrogram gets the xlabel
        axes[row - 1].set_xlabel("Time [s]")

    # --------- Debug Plots ---------

    def _plot_1d_or_2d(ax, data, times, title, ylabel):
        """Helper: if data is 1D -> line plot, if 2D -> pcolormesh."""
        data = np.asarray(data)
        if data.ndim == 1:
            ax.plot(times, data)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        elif data.ndim == 2:
            # assume shape = (bands, frames)
            im = ax.pcolormesh(
                times,
                np.arange(data.shape[0]),
                data,
                shading="auto",
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Band index")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "Unsupported shape", ha="center", va="center")
            ax.set_title(title)

    # Gain G
    if G is not None:
        ax = axes[row]
        row += 1
        _plot_1d_or_2d(ax, G, times, "Gain G", "Gain")

    # P_band_all
    if P_band is not None:
        ax = axes[row]
        row += 1
        _plot_1d_or_2d(ax, P_band, times, "Signal band power P_band_all", "Power")

    # N_band_all
    if N_band is not None:
        ax = axes[row]
        row += 1
        _plot_1d_or_2d(ax, N_band, times, "Noise band power N_band_all", "Power")

    fig.tight_layout()
    plt.show()



from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _frames_to_df(det_debug: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert det_debug["frames"] (List[FrameFeatures]) into a tidy DataFrame.
    Falls back to det_debug arrays if frames are missing.
    """
    frames = det_debug.get("frames", None)

    if frames is not None and len(frames) > 0:
        rows = []
        for ff in frames:
            if is_dataclass(ff):
                d = asdict(ff)
            else:
                # tolerate plain dicts
                d = dict(ff)
            rows.append(d)
        df = pd.DataFrame(rows)

        # Ensure some expected columns exist
        if "t" not in df.columns:
            df["t"] = np.arange(len(df))
        if "time_s" not in df.columns:
            df["time_s"] = df["t"].astype(float)

        return df

    # ---- fallback: build from arrays in det_debug ----
    # This path is useful if you decide to not store frames.
    T = len(det_debug.get("rain_score_raw", []))
    t = np.arange(T)
    df = pd.DataFrame(
        {
            "t": t,
            "time_s": t.astype(float),
            "Ltot": det_debug.get("Ltot", np.full(T, np.nan)),
            "dLtot": det_debug.get("dLtot", np.full(T, np.nan)),
            "rain_score_raw": det_debug.get("rain_score_raw", np.full(T, np.nan)),
            "label": det_debug.get("label", np.array([""] * T)),
            "use_for_noise_psd": det_debug.get("use_for_noise_psd", np.zeros(T, dtype=bool)),
        }
    )
    return df


def plot_frame_classifier_debug(
    det_debug: Dict[str, Any],
    *,
    times_s: Optional[np.ndarray] = None,
    operating_band: Optional[tuple[float, float]] = None,
    primary_mode_idx: int = 0,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    title: str = "Rain/Noise Frame Classifier Debug",
    show: bool = True,
) -> go.Figure:
    """
    Interactive (zoomable) plot for RainFrameClassifierMixin outputs.

    Parameters
    ----------
    det_debug : dict
        The debug dict returned by _detect_rain_over_time (your det_debug).
        Must contain either:
          - "frames": List[FrameFeatures]
        OR
          - arrays: "rain_score_raw", "label", "use_for_noise_psd", etc.
    primary_mode_idx : int
        Which mode index is "primary" for extracting per-mode series from FrameFeatures lists.
    audio, sr : optional
        If provided, adds a waveform panel aligned to time_s.
    """
    df = _frames_to_df(det_debug).copy()

    # ----- extract list-based mode fields to scalar series -----
    def extract_mode_col(list_col: str, idx: int) -> np.ndarray:
        if list_col not in df.columns:
            return np.full(len(df), np.nan)
        out = np.full(len(df), np.nan)
        for i, v in enumerate(df[list_col].tolist()):
            if isinstance(v, (list, tuple)) and len(v) > idx:
                out[i] = float(v[idx])
        return out

    df["mode_L_primary_db"] = extract_mode_col("mode_L", primary_mode_idx)
    df["mode_rise_primary_db"] = extract_mode_col("mode_rise_db", primary_mode_idx)
    df["mode_peak_primary_db"] = extract_mode_col("mode_peak_db", primary_mode_idx)
    df["mode_score_primary"] = extract_mode_col("mode_score", primary_mode_idx)

    # Ensure common fields exist
    if "rain_score_raw" not in df.columns:
        df["rain_score_raw"] = det_debug.get("rain_score_raw", np.full(len(df), np.nan))
    if "label" not in df.columns:
        df["label"] = det_debug.get("label", [""] * len(df))
    if "use_for_noise_psd" not in df.columns:
        df["use_for_noise_psd"] = det_debug.get("use_for_noise_psd", np.zeros(len(df), dtype=bool))

    # Boolean gates (may be missing)
    for c in ["gate_primary_rise_ok", "gate_primary_peak_ok", "gate_top_peaks_mode_ok"]:
        if c not in df.columns:
            df[c] = False

    # ----- make numeric label tracks for visualization -----
    # map label -> y-levels
    label_map = {"noise": 0.0, "uncertain": 0.5, "rain": 1.0}
    df["label_y"] = df["label"].map(label_map).fillna(np.nan)

    # X axis
    if times_s is not None:
        x = np.asarray(times_s, dtype=float)
        # If shapes mismatch, fall back to df time axis
        if len(x) != len(df):
            x = df["time_s"].to_numpy(dtype=float)
    else:
        x = df["time_s"].to_numpy(dtype=float)

    # ----- subplot layout -----
    nrows = 4 + (1 if audio is not None and sr is not None else 0)
    row = 1

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.18] * nrows,
    )

    # (optional) waveform
    if audio is not None and sr is not None:
        t_audio = np.arange(len(audio)) / float(sr)
        fig.add_trace(
            go.Scatter(x=t_audio, y=audio, mode="lines", name="audio", line=dict(width=1)),
            row=row, col=1
        )
        fig.update_yaxes(title_text="audio", row=row, col=1)
        row += 1

    # score + label
    fig.add_trace(
        go.Scatter(x=x, y=df["rain_score_raw"], mode="lines+markers", name="rain_score_raw", marker=dict(size=4)),
        row=row, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["label_y"], mode="lines+markers", name="label (noise/unc/rain)", marker=dict(size=4)),
        row=row, col=1
    )
    # mark PSD-update frames
    psd_mask = df["use_for_noise_psd"].astype(bool).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=x[psd_mask],
            y=df["rain_score_raw"].to_numpy()[psd_mask],
            mode="markers",
            name="use_for_noise_psd",
            marker=dict(symbol="x", size=9),
        ),
        row=row, col=1
    )
    fig.update_yaxes(title_text="score/label", range=[-0.1, 1.1], row=row, col=1)
    row += 1

    # loudness + rise
    fig.add_trace(go.Scatter(x=x, y=df["Ltot"], mode="lines", name="Ltot (dB)"), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["dLtot"], mode="lines", name="dLtot (dB)"), row=row, col=1)
    fig.update_yaxes(title_text="Ltot/dLtot", row=row, col=1)
    row += 1

    # primary mode evidence
    fig.add_trace(go.Scatter(x=x, y=df["mode_L_primary_db"], mode="lines", name="mode_L_primary (dB)"), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["mode_rise_primary_db"], mode="lines", name="mode_rise_primary (dB)"), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["mode_peak_primary_db"], mode="lines", name="mode_peak_primary (dB)"), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["mode_score_primary"], mode="lines", name="mode_score_primary"), row=row, col=1)
    fig.update_yaxes(title_text="primary mode", row=row, col=1)
    row += 1

    # gates
    # Plot as 0/1 tracks so it’s easy to see where they flip.
    for c in ["gate_primary_rise_ok", "gate_primary_peak_ok", "gate_top_peaks_mode_ok"]:
        fig.add_trace(
            go.Scatter(x=x, y=df[c].astype(float), mode="lines+markers", name=c, marker=dict(size=4)),
            row=row, col=1
        )
    fig.update_yaxes(title_text="gates", range=[-0.1, 1.1], row=row, col=1)
    row += 1

    fig.update_layout(
        title=title if operating_band is None else f"{title}  (operating_band={operating_band[0]:.0f}-{operating_band[1]:.0f} Hz)",
        height=250 * nrows,
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="time (s)", row=nrows, col=1)

    if show:
        fig.show()

    return fig

def plot_frame_classifier_tuning(
    dbg: Dict[str, Any],
    *,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    title: str = "Frame Classifier Tuning",
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    show: bool = True,
) -> go.Figure:
    """Interactive tuning dashboard for the rain/noise frame classifier.

    This function is designed to be called with the *container* debug dict returned
    by `SpectralNoiseProcessor.process()` (typically `row["debug"]`). It will
    gracefully fall back if some fields are missing.

    Plots:
      - (optional) waveform
      - rain_conf / noise_conf / label
      - use_for_noise_psd markers
      - (optional) must-have gates from FrameFeatures list

    Parameters
    ----------
    dbg:
        Debug container, usually `row["debug"]`.
    audio, sr:
        Optional waveform to show on top.
    title:
        Plot title.
    t0, t1:
        Optional time window (seconds).
    """

    # Detector debug may be nested
    det_debug: Dict[str, Any] = dbg.get("detector", dbg) if isinstance(dbg, dict) else {}

    # ---- time axis ----
    if isinstance(dbg, dict) and "times_s" in dbg:
        times = np.asarray(dbg["times_s"], dtype=float)
    else:
        # fall back to frame-features DF or indices
        df_tmp = _frames_to_df(det_debug)
        if "time_s" in df_tmp.columns:
            times = df_tmp["time_s"].to_numpy(dtype=float)
        else:
            times = np.arange(len(df_tmp), dtype=float)

    T = len(times)

    # ---- series ----
    def _get_series(key: str, default_val: float = np.nan) -> np.ndarray:
        if isinstance(dbg, dict) and key in dbg:
            return np.asarray(dbg[key], dtype=float)
        if isinstance(det_debug, dict) and key in det_debug:
            v = det_debug[key]
            if isinstance(v, (list, tuple, np.ndarray)):
                return np.asarray(v, dtype=float)
        return np.full(T, default_val, dtype=float)

    rain_conf = _get_series("rain_conf")
    noise_conf = _get_series("noise_conf")

    use_for_noise_psd = None
    if isinstance(dbg, dict) and "use_for_noise_psd" in dbg:
        use_for_noise_psd = np.asarray(dbg["use_for_noise_psd"], dtype=bool)
    elif isinstance(det_debug, dict) and "use_for_noise_psd" in det_debug:
        use_for_noise_psd = np.asarray(det_debug["use_for_noise_psd"], dtype=bool)
    else:
        use_for_noise_psd = np.zeros(T, dtype=bool)

    label = None
    if isinstance(det_debug, dict) and "label" in det_debug:
        label = np.asarray(det_debug["label"], dtype=object)

    # thresholds (prefer container, then detector)
    def _get_scalar(key: str) -> Optional[float]:
        for src in (dbg, det_debug):
            if isinstance(src, dict) and key in src:
                try:
                    return float(src[key])
                except Exception:
                    return None
        return None

    rain_hi = _get_scalar("rain_hi")
    noise_hi = _get_scalar("noise_hi")
    noise_update_thresh = _get_scalar("noise_update_thresh")

    # ---- optional FrameFeatures -> gates ----
    df_frames = None
    if isinstance(det_debug, dict) and det_debug.get("frames") is not None:
        df_frames = _frames_to_df(det_debug)
        # align with times if possible
        if len(df_frames) == T:
            pass
        else:
            df_frames = None

    # ---- time window mask ----
    if t0 is None:
        t0 = float(times[0]) if T > 0 else 0.0
    if t1 is None:
        t1 = float(times[-1]) if T > 0 else 0.0

    m = (times >= float(t0)) & (times <= float(t1))

    times_w = times[m]
    rain_w = rain_conf[m]
    noise_w = noise_conf[m]
    use_w = use_for_noise_psd[m]

    # label -> y levels
    label_y = None
    if label is not None and len(label) == T:
        label_w = label[m]
        label_map = {"noise": 0.0, "uncertain": 0.5, "rain": 1.0}
        label_y = np.array([label_map.get(str(v), np.nan) for v in label_w], dtype=float)

    # ---- layout ----
    nrows = 2
    if df_frames is not None:
        nrows += 1
    if audio is not None and sr is not None:
        nrows += 1

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    row = 1

    # waveform
    if audio is not None and sr is not None:
        t_audio = np.arange(len(audio), dtype=float) / float(sr)
        ma = (t_audio >= float(t0)) & (t_audio <= float(t1))
        fig.add_trace(
            go.Scatter(x=t_audio[ma], y=np.asarray(audio)[ma], mode="lines", name="audio", line=dict(width=1)),
            row=row, col=1,
        )
        fig.update_yaxes(title_text="audio", row=row, col=1)
        row += 1

    # score / label / psd update
    fig.add_trace(
        go.Scatter(x=times_w, y=rain_w, mode="lines", name="rain_conf"),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times_w, y=noise_w, mode="lines", name="noise_conf"),
        row=row, col=1,
    )

    if label_y is not None:
        fig.add_trace(
            go.Scatter(x=times_w, y=label_y, mode="lines+markers", name="label (noise/unc/rain)", marker=dict(size=4)),
            row=row, col=1,
        )

    # PSD update markers at rain_conf values
    if np.any(use_w):
        fig.add_trace(
            go.Scatter(
                x=times_w[use_w],
                y=rain_w[use_w],
                mode="markers",
                name="use_for_noise_psd",
                marker=dict(symbol="x", size=10),
            ),
            row=row, col=1,
        )

    # threshold overlays
    shapes = []
    def _add_hline(y: float, name: str, dash: str = "dash"):
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref=f"y{row}" if row > 1 else "y",
                x0=float(t0), x1=float(t1),
                y0=float(y), y1=float(y),
                line=dict(dash=dash, width=1),
            )
        )
        fig.add_trace(
            go.Scatter(x=[np.nan], y=[np.nan], mode="lines", name=name, line=dict(dash=dash)),
            row=row, col=1,
        )

    if rain_hi is not None:
        _add_hline(rain_hi, "rain_hi", dash="dash")
    if noise_hi is not None:
        _add_hline(1.0 - noise_hi, "1-noise_hi", dash="dot")
    if noise_update_thresh is not None:
        _add_hline(1.0 - noise_update_thresh, "1-noise_update_thresh", dash="dashdot")

    fig.update_yaxes(title_text="conf/label", range=[-0.1, 1.1], row=row, col=1)
    row += 1

    # loudness/rise if present
    Ltot = _get_series("Ltot")
    dLtot = _get_series("dLtot")
    if np.any(np.isfinite(Ltot)) or np.any(np.isfinite(dLtot)):
        fig.add_trace(go.Scatter(x=times_w, y=Ltot[m], mode="lines", name="Ltot (dB)"), row=row, col=1)
        fig.add_trace(go.Scatter(x=times_w, y=dLtot[m], mode="lines", name="dLtot (dB)"), row=row, col=1)
        fig.update_yaxes(title_text="loudness", row=row, col=1)
    else:
        fig.add_trace(go.Scatter(x=times_w, y=rain_w, mode="lines", name="rain_conf (dup)"), row=row, col=1)
        fig.update_yaxes(title_text="(no Ltot)", row=row, col=1)
    row += 1

    # gates panel (if FrameFeatures available)
    if df_frames is not None:
        dfw = df_frames.loc[m].copy()
        for c in ["gate_primary_rise_ok", "gate_primary_peak_ok", "gate_top_peaks_mode_ok"]:
            if c in dfw.columns:
                fig.add_trace(
                    go.Scatter(
                        x=times_w,
                        y=dfw[c].astype(float).to_numpy(),
                        mode="lines+markers",
                        name=c,
                        marker=dict(size=4),
                    ),
                    row=row, col=1,
                )
        fig.update_yaxes(title_text="gates", range=[-0.1, 1.1], row=row, col=1)
        row += 1

    fig.update_layout(
        title=title,
        height=260 * nrows,
        hovermode="x unified",
        legend=dict(orientation="h"),
        shapes=shapes,
    )
    fig.update_xaxes(title_text="time (s)", row=nrows, col=1)

    if show:
        fig.show()

    return fig
def plot_noise_suppressor_debug(
    dbg: Dict[str, Any],
    *,
    title: str = "Noise suppressor debug",
    operating_band: Optional[tuple[float, float]] = None,
    show: bool = True,
) -> go.Figure:
    """Zoomable view of suppressor internals.

    Expects the *container* debug dict from SpectralNoiseProcessor (row["debug"]).

    Uses (if present):
      - dbg["times_s"]
      - dbg["freqs"]
      - dbg["G"]           : gain (F,T) or (T,)
      - dbg["P_band_all"]  : (bands,T) or (T,)
      - dbg["N_band_all"]  : (bands,T) or (T,)
    """
    times_s = np.asarray(dbg.get("times_s", []), dtype=float)
    freqs = np.asarray(dbg.get("freqs", []), dtype=float)

    G = dbg.get("G", None)
    P_band = dbg.get("P_band_all", None)
    N_band = dbg.get("N_band_all", None)

    nrows = 0
    if G is not None:
        nrows += 1
    if P_band is not None:
        nrows += 1
    if N_band is not None:
        nrows += 1
    if nrows == 0:
        # Empty figure with message
        fig = go.Figure()
        fig.add_annotation(text="No suppressor debug arrays found (G/P_band_all/N_band_all).", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title)
        if show:
            fig.show()
        return fig

    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    row = 1

    def _add_heat_or_line(name: str, data: Any, y_label: str):
        nonlocal row
        arr = np.asarray(data)
        if arr.ndim == 1:
            fig.add_trace(go.Scatter(x=times_s, y=arr, mode="lines", name=name), row=row, col=1)
            fig.update_yaxes(title_text=y_label, row=row, col=1)
        elif arr.ndim == 2:
            # (F,T) or (bands,T)
            y = np.arange(arr.shape[0])
            # If freqs matches first dim, use it
            if freqs.size == arr.shape[0]:
                y = freqs
            fig.add_trace(go.Heatmap(x=times_s, y=y, z=arr, name=name, colorbar=dict(title=name)), row=row, col=1)
            fig.update_yaxes(title_text=y_label, row=row, col=1)
        else:
            fig.add_annotation(text=f"{name}: unsupported shape {arr.shape}", x=0.5, y=0.5, showarrow=False, row=row, col=1)
        row += 1

    if G is not None:
        _add_heat_or_line("G", G, "freq/bin")
    if P_band is not None:
        _add_heat_or_line("P_band_all", P_band, "band")
    if N_band is not None:
        _add_heat_or_line("N_band_all", N_band, "band")

    if operating_band is not None:
        title2 = f"{title} (operating_band={operating_band[0]:.0f}-{operating_band[1]:.0f} Hz)"
    else:
        title2 = title

    fig.update_layout(title=title2, height=260 * nrows, hovermode="x unified", legend=dict(orientation="h"))
    fig.update_xaxes(title_text="time (s)", row=nrows, col=1)

    if show:
        fig.show()
    return fig


__all__ = [
    "show_noise_processing_results",
    "plot_frame_classifier_debug",
    "plot_frame_classifier_tuning",
    "plot_noise_suppressor_debug",
]