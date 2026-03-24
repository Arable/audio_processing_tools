import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import spectrogram


def plot_audio_signal(
    signal: np.array,
    sample_rate: int,
    title: str = "Audio Data",
    add_range_slider: bool = True,
    renderer=None,
):
    """
    Convenience function for plotting audio data

    Parameters
    ----------
    signal - audio signal data
    sample_rate - sample rate of audio data
    title - title for plot
    add_range_slider - if True, interactive range slider will be included in plot
    renderer - reneder engine for plotly plots


    """
    plot_df = pd.DataFrame()
    plot_df["signal"] = signal
    plot_df["time"] = [i / sample_rate for i in range(len(signal))]
    fig = go.Figure(
        [
            go.Scatter(
                x=plot_df["time"],
                y=plot_df["signal"],
                name="signal",
            )
        ],
        go.Layout(
            title=title,
            xaxis={"title": "seconds"},
            yaxis={"title": "signal"},
        ),
    )

    if add_range_slider:
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(), rangeslider=dict(visible=True), type="linear"
            )
        )
    fig.show(renderer=renderer)


def plot_fft(fft_df: pd.DataFrame, title: str = "FFT ", renderer=None):
    """
    Convenience function for plotting FFT

    Parameters
    ----------
    fft_df - fft df with 'frequency' and 'amplitude' column, can generate with transform.get_real_fft_df()
    title - title for plot
    renderer - reneder engine for plotly plots
    """

    fig = go.Figure(go.Scatter(x=fft_df["frequency"], y=fft_df["amplitude"]))
    fig.show(renderer)

def plot_audio_spectrogram(
    signal,
    fs,
    title="Spectrogram of the Audio Signal",
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    freq_range=None,
    colorscale="Viridis",
    db_floor=60.0,
):
    """
    Plot an audio spectrogram using Plotly with cleaner defaults for analysis.

    Args:
        signal: Audio signal array.
        fs: Sampling frequency.
        title: Plot title.
        window: Window passed to scipy.signal.spectrogram.
        nperseg: Segment length.
        noverlap: Overlap between segments. Defaults to nperseg // 2.
        nfft: FFT length. Defaults to nperseg.
        freq_range: Optional tuple (fmin, fmax) to limit displayed frequencies.
        colorscale: Plotly colorscale. Default is 'Viridis'.
        db_floor: Dynamic range below max dB to display.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    f, t, Sxx = spectrogram(
        signal,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
        mode="psd",
    )

    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    if freq_range is not None:
        fmin, fmax = freq_range
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        Sxx_dB = Sxx_dB[mask, :]

    zmax = np.max(Sxx_dB)
    zmin = zmax - db_floor

    fig = go.Figure(
        data=go.Heatmap(
            x=t,
            y=f,
            z=Sxx_dB,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Intensity [dB]"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="Frequency [Hz]"),
        template="plotly_white",
    )

    fig.show()

