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
    signal, fs, title="Spectrogram of the Audio Signal", window="hann", nperseg=256
):
    """
    Plots a spectrogram for the given audio signal using Plotly.

    Args:
        signal (numpy.array): The audio signal array.
        fs (int): The sampling frequency of the audio signal.
        title (str, optional): Title of the plot. Defaults to 'Spectrogram of the Audio Signal'.
        window (str or tuple or array_like, optional): Desired window to use. If window is a string or tuple, it is passed to
            `get_window` to generate the window values, which are DFT-even by default. See scipy.signal.get_window for a list
            of windows and required parameters. If window is array_like it will be used directly as the window and its length
            must be nperseg. Defaults to 'hann'.
        nperseg (int, optional): Length of each segment. Defaults to 256.

    """
    # Compute the spectrogram
    f, t, Sxx = spectrogram(signal, fs, window=window, nperseg=nperseg)

    # Convert power spectrogram (square of the magnitude) to decibels (dB)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Adding a small value to avoid log(0)

    # Creating the spectrogram plot
    fig = go.Figure(
        data=go.Heatmap(
            x=t, y=f, z=Sxx_dB, colorscale="Jet", colorbar=dict(title="Intensity [dB]")
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="Frequency [Hz]"),
        template="plotly_dark",
    )

    # Show the plot
    fig.show()
