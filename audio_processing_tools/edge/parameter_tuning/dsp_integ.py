#4th March
import os, sys
import sys
import numpy as np
import scipy
from scipy import signal
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
import csv
import pandas as pd
from collections import Counter
import statistics as stats
import math
import copy

###################################################
#### settings #####################################
###################################################
sys.path.append('..')
#import matplotlib as mpl
energy_thr_db = -70
# -60 = 20*log10(amplitude/max)
# amplitude threshold = 10**(-3.5)
# energy threshold = 10**(-7)
amp_thr = pow(10, energy_thr_db / 20)
energy_thr = pow(10, energy_thr_db / 10)

default_params = {
    'sample_rate': 11162,
    'freq_resolution': 45,
    'time_resolution_ms': 10,
    'check_duration': 2,
    'op_freq_range': [400, 3000],
    'n_freq_range': [400, 600],
    'fn': 400,
    'num_harmonics': 6,
    'harmonic_threshold': [5, 4, 4, 4, 4, 4],
    'max_peaks': 3,
    'log_factor': 10,
    'ns_duration_ms': 470,
    'nf': 0,
    'min_drop_count': 1
}

#default_params['fn'], default_params[op_freq_range][1]
# # frame and hop size
frame_length = 256
hop_length = 128

bytes_per_sample = 2
# # input vector duration in seconds
rain_check_duration = default_params['check_duration']
# offset for confirmation
offset_in_seconds = rain_check_duration

# # input bandpass filter to improve SNR
operating_freq_range = default_params['op_freq_range']
hn_max = default_params['num_harmonics']
max_harmonics = hn_max

# # log compression factor to compensate weak signals
log_compress_factor = default_params['log_factor']

min_average_len = 20

# # thresholds
rain_thr = default_params['harmonic_threshold']
rain_thr_hn = rain_thr[0] + rain_thr[1] + rain_thr[2]
# corr_thr = 0.25

min_drop_count_per_second = default_params['min_drop_count']
# # number of peaks to look for the primary resonance frequency
max_num_peaks = default_params['max_peaks']

# # the natural frequency of top dome is assumed to be in this range
natural_freq_range = default_params['n_freq_range']

noise_factor = default_params['nf']
noise_threshold = frame_length*.0025

HEADER_SIZE = 38
old_vecs = 1

enable_fmean_method_1 = 0
enable_correlation_mask = 0
enable_filter_design = True

if enable_filter_design != True:
    sos = [
        [0.07756312, 0.15512624, 0.07756312, 1.0, -0.06001812, 0.06505373],
        [1.0, 2.0, 1.0, 1.0, 0.14807404, 0.51446797],
        [1.0, -2.0, 1.0, 1.0, -1.59012430, 0.64155344],
        [1.0, -2.0, 1.0, 1.0, -1.82797234, 0.8711928],
    ]


F_natural = default_params['fn']
F_max = default_params['op_freq_range'][1]
search_freq_range = [operating_freq_range,
    [(F_natural * 2 - 200), (F_natural * 2 + 300)],
    [(F_natural * 3 - 200), (F_natural * 3 + 300)],
    [(F_natural * 4 - 200), (F_natural * 4 + 300)],
    [(F_natural * 5 - 200), (F_natural * 5 + 300)],
                    [(F_natural*6 - 200), operating_freq_range[1]]]

enable_plot_data = False
max_rows = 2

if enable_plot_data == True:
    old_vecs = 1
    enable_plot_spectrogram = 1
    enable_plot_noise_supressed_signal = 0
    enable_plot_freq_flux = 1
    enable_plot_frequency_harmonics = 1
    enable_plot_individual_harmonics = 1
    enable_plot_freq_flux_a_d = 0
    enable_plot_correlation_mask = 0
else:
    enable_plot_spectrogram = 0
    enable_plot_noise_supressed_signal = 0
    enable_plot_freq_flux = 0
    enable_plot_frequency_harmonics = 0
    enable_plot_individual_harmonics = 0
    enable_plot_freq_flux_a_d = 0
    enable_plot_correlation_mask = 0

if enable_plot_spectrogram == 1:
    max_rows += 1

if enable_plot_noise_supressed_signal == 1:
    max_rows += 1

if enable_plot_freq_flux == 1:
    max_rows += 1

if enable_plot_frequency_harmonics == 1:
    max_rows += 1

if enable_plot_individual_harmonics == 1:
    max_rows += hn_max - 1

if enable_plot_freq_flux_a_d == 1:
    max_rows += 1

if enable_plot_correlation_mask == 1:
    max_rows += 1

###################################################
# cant have magic statements outside notebooks
# %matplotlib inline

def check_energy_threshold(magnitude_spectrum, freqs, Fs, N, threshold):

    f_res = Fs / N
    idx1 = int(freqs[0] // f_res + 1)
    idx2 = int(freqs[1] // f_res)
    
    # Extract the relevant portion of the spectrum
    relevant_spectrum = magnitude_spectrum[idx1:idx2 + 1]

    # Calculate the energy in the specified frequency range
    energy_in_range = np.sum(np.square(relevant_spectrum))
    
    # Check if the energy is above the threshold
    if energy_in_range > threshold:
        return True
    else:
        return False


def configure_parameters (sample_rate=11162, 
                          freq_resolution=45, 
                          time_resolution_ms=10,
                          check_duration=2,
                          op_freq_range=[400, 3000],
                          n_freq_range=[400, 600],
                          fn=400, 
                          num_harmonics=6,
                          harmonic_threshold=[5, 4, 4, 4, 4, 4],
                          max_peaks=3,
                          log_factor=10,
                          ns_duration_ms=470,
                          nf=0,
                          min_drop_count=1):
    global Fs, frame_length, hop_length, rain_check_duration, operating_freq_range, hn_max, \
    log_compress_factor, min_average_len, rain_thr, rain_thr_hn, corr_thr, \
    min_drop_count_per_second, max_num_peaks,  natural_freq_range, noise_factor, noise_threshold,\
    search_freq_range, noise_duration_ms, F_natural, sos
    Fs = sample_rate
    val = Fs / freq_resolution 
    frame_length = 2 ** math.ceil(math.log2(val)) 
    
    # time_resolution_ms = (hop_length * 1000)/Fs
    hop_length = 2 ** math.ceil (math.log2((time_resolution_ms * Fs) / 1000))
    
    rain_check_duration = check_duration
    # offset for confirmation
    offset_in_seconds = rain_check_duration

    #base value of the natural frequency
    F_natural = fn  
    
    operating_freq_range = op_freq_range
    #the natural frequency of top dome is assumed to be in this range
    # natural_freq_range = [fn - 25, fn + 300]
    natural_freq_range = n_freq_range
    hn_max = num_harmonics
    max_harmonics = hn_max
    
    #log compression factor to compensate weak signals
    log_compress_factor = log_factor

    #noise_duration = ((2 * min_average_len + 1) * hop_length) / Fs
    noise_duration_ms = ns_duration_ms
    min_average_len = math.ceil(((noise_duration_ms * Fs / 1000) / hop_length - 1) / 2)
    #min_average_len = 20
    
    #thresholds
    rain_thr  = harmonic_threshold
    rain_thr_hn = rain_thr[0] + rain_thr[1]  + rain_thr[2]
    
    corr_thr = 0.25
    #rain_thr_hn = 10
    min_drop_count_per_second = min_drop_count
    # number of peaks to look for the primary resonance frequency
    max_num_peaks = max_peaks

    
    noise_factor = nf
    noise_threshold = frame_length*.0025

    if (enable_filter_design != True) :
        sos = [[ 0.07756312,  0.15512624, 0.07756312, 1.        , -0.06001812, 0.06505373],
               [ 1.        , 2.         , 1.        , 1.        ,  0.14807404, 0.51446797],
               [ 1.        , -2.        , 1.        , 1.        , -1.59012430, 0.64155344],
               [ 1.        , -2.        , 1.        , 1.        , -1.82797234, 0.8711928]]

    search_freq_range = [op_freq_range,
                        [(F_natural * 2 - 200), (F_natural * 2 + 300)],
                        [(F_natural * 3 - 200), (F_natural * 3 + 300) ],
                        [(F_natural * 4 - 200), (F_natural * 4 + 300)],
                        [(F_natural * 5 - 200), (F_natural * 5 + 300)],
                        [(F_natural * 6 - 200), operating_freq_range[1]]]
    
def update_search_freq_range (f_natural_est):
    global search_freq_range, max_harmonics
    #print (f"estimated natural frequency = {f_natural_est}")
    for i in range(1, 6):
        search_freq_range [i][0] = f_natural_est * (i+1) - 200
        if (search_freq_range[i][0] < operating_freq_range[0]):
            search_freq_range[i][0] = operating_freq_range[0]
        search_freq_range [i][1] = f_natural_est * (i+1) + 300        
        #if the search range is less than 400 Hz don't use that harmonic for calculation
        if search_freq_range[i][1] > operating_freq_range[1] + 100:
            max_harmonics = i
        if (search_freq_range[i][1] > operating_freq_range[1]):
            search_freq_range[i][1] = operating_freq_range[1]
     
def print_configuration ():
    print (f"freq_resolution = {Fs/frame_length} frame_length = {frame_length}")
    print (f"time_resolution_ms = {hop_length*1000/Fs} hop_length = {hop_length}")
    print (f"min_average_len = {min_average_len}, noise_duration_ms = {noise_duration_ms}")
    print (f"rain_check_duration = {rain_check_duration}") 
    print (f"top dome natural frequency = {F_natural}")
    print (f"operating frequency range = {operating_freq_range}")
    print (f"natural frequency range = {natural_freq_range}")
    print (f"number of harmonics = {hn_max}")
    print (f"log compression factor = {log_compress_factor}");
    print (f"harmonic threshold = {rain_thr}")
    print (f"overall threshold = {rain_thr_hn}")
    print (f"min_drop_count_per_sec = {min_drop_count_per_second}")
    print (f"max number of frequency peaks to check = {max_num_peaks}")
    print (f"noise_factor = {noise_factor} noise_threshold = {noise_threshold}")
    #print (f"search frequency ranges = {search_freq_range}")
    #print (f"sos filter coefficients = {sos}")


def change_plotting_status(status=False):
    global enable_plot_data, max_rows
    global enable_plot_spectrogram, enable_plot_noise_supressed_signal
    global enable_plot_freq_flux, enable_plot_frequency_harmonics, enable_plot_individual_harmonics
    global enable_plot_freq_flux_a_d, enable_plot_correlation_mask

    enable_plot_data = status
    if enable_plot_data == True:
        enable_plot_spectrogram = 1
        enable_plot_noise_supressed_signal = 1
        enable_plot_freq_flux = 1 
        enable_plot_frequency_harmonics = 1
        enable_plot_individual_harmonics = 1
        enable_plot_freq_flux_a_d = 0
        enable_plot_correlation_mask = 0
    else:
        enable_plot_spectrogram = 0
        enable_plot_noise_supressed_signal = 0
        enable_plot_freq_flux = 0
        enable_plot_frequency_harmonics = 0
        enable_plot_individual_harmonics = 0
        enable_plot_freq_flux_a_d = 0
        enable_plot_correlation_mask = 0
    max_rows = 2
    if enable_plot_spectrogram == 1:
        max_rows += 1

    if enable_plot_noise_supressed_signal == 1 and noise_factor != 0:
        max_rows += 1

    if enable_plot_freq_flux == 1:
        max_rows += 1

    if enable_plot_frequency_harmonics == 1:
        max_rows += 1

    if enable_plot_individual_harmonics == 1:
        max_rows += hn_max - 1

    if enable_plot_freq_flux_a_d == 1:
        max_rows += 1

    if enable_plot_correlation_mask == 1:
        max_rows += 1
    if enable_plot_data == True:
        print(f"number of subplots = {max_rows}")


def estimate_and_supress_noise(stft, threshold, noise_lpf_coef):
    """
    Estimate the noise spectrum from an STFT using a threshold and the specified equation.

    Parameters:
    - stft: Short-Time Fourier Transform of the audio signal (complex valued).
    - threshold: Energy threshold below which to estimate the noise spectrum.
    - noise_lpf_coef: Coefficient for the noise estimation equation.

    Returns:
    - estimate_and_reduce_noise: Estimated noise spectrum (complex valued).
    """
    # print (f"shape of stft is {stft.shape}");
    # Calculate the magnitude of the STFT
    stft_magnitude = np.abs(stft)
    # Initialize an array for the estimated noise spectrum
    noise_spectrum = np.zeros_like(stft)

    # Iterate through the STFT bins
    for i in range(stft.shape[1]):
        # Calculate the energy of the current STFT bin
        bin_energy = np.sum(stft_magnitude[:, i])

        # Check if the energy is below the threshold
        if bin_energy < threshold:
            # If below threshold, estimate noise using the given equation
            noise_spectrum[:, i] = (
                noise_lpf_coef * noise_spectrum[:, i]
                + (1 - noise_lpf_coef) * stft[:, i]
            )
    # noise = noise_spectrum[:,-1]
    for i in range(stft.shape[1]):
        stft[:, i] = stft[:, i] - noise_spectrum[:, i]
        stft[stft < 0] = 0

    return stft


def find_signal_occurrences(input_signal, reference_signal, threshold):
    occurrences = np.zeros(len(input_signal))
    occurrences_mask = np.zeros(len(input_signal))
    reference_norm = np.linalg.norm(
        reference_signal
    )  # Precompute reference signal's norm

    for i in range(len(input_signal) - len(reference_signal) + 1):
        window = input_signal[i : i + len(reference_signal)]
        window_norm = np.linalg.norm(window)  # Precompute current window's norm
        correlation_coefficient = np.dot(window, reference_signal) / (
            window_norm * reference_norm
        )

        if correlation_coefficient >= threshold:
            occurrences_mask[i] = correlation_coefficient * 10
        occurrences[i] = correlation_coefficient

    ref_len = len(reference_signal)
    audio_len = len(input_signal)
    occurrences = np.insert(occurrences, 0, np.zeros(ref_len // 2))
    occurrences = occurrences[:audio_len]
    occurrences_mask = np.insert(occurrences_mask, 0, np.zeros(ref_len // 2))
    occurrences_mask = occurrences_mask[:audio_len]
    return occurrences, occurrences_mask


def calculate_correlation_mask(audio_data, reference_audio, peaks, corr_thr):
    occurrences, occurrences_mask = find_signal_occurrences(
        audio_data, reference_audio, corr_thr
    )

    length = len(occurrences_mask)
    temp = np.zeros(length)
    win_len = 1
    for k in range(length):
        k_start = 0
        k_end = length
        if k > win_len:
            k_start = k - win_len
        if k < length - win_len:
            k_end = k + win_len
        temp[k] = sum(occurrences_mask[k_start:k_end])
        if temp[k] >= 1:
            temp[k] = 1
        else:
            temp[k] = 0
    # print ( temp);
    # print ( occurrences_mask)
    r_mask = np.add.reduceat(temp, np.arange(0, len(temp), hop_length))
    # r_mask = [r_mask, 0]
    # print ("len of r_mask is", len(r_mask), len(nov_3))
    val = np.diff(peaks)
    val[val < 0] = 0
    peaks_count = np.sum(val)

    return r_mask, peaks_count


def calculate_cepstrum(audio, sr, n_fft=512, H=256):
    # Compute the STFT of the audio signal
    stft = librosa.stft(audio, n_fft=n_fft, H=hop_length, window="hann")

    # Convert the magnitude spectrogram to dB scale
    mag_spec = np.abs(stft)
    mag_spec_db = librosa.amplitude_to_db(mag_spec)

    # Compute the inverse STFT (using the magnitude spectrogram)
    inv_stft = librosa.istft(mag_spec_db, H=hop_length)

    # Compute the cepstrum by taking the real part of the inverse STFT
    cepstrum = np.real(inv_stft)
    return cepstrum


def harmonic_enhancer(
    stft, fft_length=512, H=256, base_frequency=500, Fs=11162, Hn=6
):
    """
    Enhances the harmonics of an audio signal's STFT.

    Args:
        stft (np.ndarray): STFT of the audio signal
        fft_length (int): Length of FFT window
        H (int): Hop size
        base_frequency (float): Base frequency (in Hz)
        Fs (int): Sampling frequency (in Hz)
        Hn (int): Number of harmonics to add

    Returns:
        enhanced_stft (np.ndarray): Harmonic-enhanced STFT
    """
    num_bins, n_frames = stft.shape

    enhanced_stft = np.zeros_like(stft)

    # Determine the base frequency bin and the maximum bin for the base frequency band
    base_freq_bin = int(base_frequency * fft_length / Fs)

    # Add the amplitudes of frequencies within the base frequency band with the first N harmonics
    for frame_idx in range(n_frames):
        for freq_bin in range(base_freq_bin):
            for harmonic in range(1, Hn + 1):
                harmonic_bin = freq_bin * harmonic
                # if harmonic_bin < num_bins:
                val = stft[harmonic_bin, frame_idx]
                stft[harmonic_bin, frame_idx] = 0
                enhanced_stft[freq_bin, frame_idx] += val
    return enhanced_stft


def moving_average_smoothing(input_signal, k):
    """
    Smoothens the input signal using a moving average filter of length k samples.

    Parameters:
    - input_signal (numpy.ndarray): The input signal to be smoothed.
    - k (int): The length of the moving average filter.

    Returns:
    - smoothed_signal (numpy.ndarray): The smoothed signal.
    """
    if k <= 0:
        raise ValueError(
            "The length of the moving average filter (k) must be a positive integer."
        )

    # Pad the input signal at both ends to handle edge effects
    pad_width = k // 2
    padded_signal = np.pad(input_signal, (pad_width, pad_width), mode="edge")

    # Apply the moving average filter
    smoothed_signal = np.convolve(padded_signal, np.ones(k) / k, mode="valid")

    return smoothed_signal


def find_peaks_in_frequency_range(
    magnitude, freq_range, fpeak_range, fn, num_peaks=max_num_peaks
):
    # Convert the frequencies f1 and f2 to bins using the sampling frequency fs
    bin_f1 = int((freq_range[0] * magnitude.shape[0]) / fn)
    bin_f2 = int((freq_range[1] * magnitude.shape[0]) / fn)

    # Extract the magnitude within the specified frequency range
    magnitude_range = magnitude[bin_f1:bin_f2, :]

    # Find peaks within the magnitude_range
    peaks = []
    peaks_found = []
    fpeak_array = []
    for t in range(magnitude_range.shape[1]):
        # peak_indices = np.argsort(magnitude_range[:, t])[-num_peaks:][::-1]
        peak_indices, _ = find_peaks(magnitude_range[:, t])
        peak_indices = peak_indices + bin_f1
        peak_count = len(peak_indices)
        if peak_count > num_peaks:
            peak_count = num_peaks

        peak_frequencies = (peak_indices * fn) / magnitude.shape[0]
        # print(f"peak_frequencies = {peak_frequencies}")
        peak_amplitudes = magnitude[peak_indices, t]

        fpeak, Apeak, found = 0, 0, 0
        for k in range(peak_count):
            if (
                fpeak_range[0] < peak_frequencies[k]
                and peak_frequencies[k] < fpeak_range[1]
            ):
                found = found + 1
                fpeak = peak_frequencies[k]
                Apeak = peak_amplitudes[k]
            if found > 0:
                break

        peaks_found.append(found)
        fpeak_array.append(fpeak)
        peaks.append(list(zip(peak_frequencies, peak_amplitudes)))
    # print (f"fpeak_array =", fpeak_array)
    # print(f"shape of fpeak_array {len(fpeak_array)} {stft_output.shape}")
    return peaks_found, fpeak_array


def find_top_n_peaks_sorted_by_prominence(data, N):
    """
    Find the top N peaks from the columns of a two-dimensional array, sorted by prominence.

    Args:
        data (np.ndarray): Two-dimensional input array
        N (int): Number of peaks to find

    Returns:
        top_peaks_indices (np.ndarray): Array of indices of the top N peaks for each column, sorted by prominence
    """
    num_rows, num_cols = data.shape
    top_peaks_indices = np.zeros((N, num_cols), dtype=int)

    for col_idx in range(num_cols):
        column = data[:, col_idx]
        peaks, _ = find_peaks(column, prominence=True)

        if len(peaks) > N:
            # Sort peaks based on their prominence
            sorted_peaks = sorted(
                peaks, key=lambda x: _["prominences"][x], reverse=True
            )
            top_peaks_indices[:, col_idx] = sorted_peaks[:N]
        else:
            top_peaks_indices[: len(peaks), col_idx] = peaks

    return top_peaks_indices


def find_top_n_peaks_sorted_by_amplitude(data, N):
    """
    Find the top N peaks from the columns of a two-dimensional array, sorted by amplitude.

    Args:
        data (np.ndarray): Two-dimensional input array
        N (int): Number of peaks to find

    Returns:
        top_peaks_indices (np.ndarray): Array of indices of the top N peaks for each column, sorted by amplitude
    """
    num_rows, num_cols = data.shape
    # print (num_rows, num_cols);
    top_peaks_indices = np.zeros((N, num_cols), dtype=int)
    # data1 = [row[16:32] for row in data]
    data1 = data[16:32, :]
    # print ("shape of data1", data1.shape);
    for col_idx in range(num_cols):
        column = data1[:, col_idx]
        # column = data1[:, col_idx]
        # print("column = ", column.shape);
        peaks, _ = find_peaks(column)
        peaks_found = []
        if len(peaks) > N:
            # Sort peaks based on their amplitude
            sorted_peaks = sorted(peaks, key=lambda x: column[x], reverse=True)
            top_peaks_indices[:, col_idx] = sorted_peaks[:N]
        else:
            top_peaks_indices[: len(peaks), col_idx] = peaks

        for k in range(len(peaks) - 1):
            peak_idx = top_peaks_indices[k, col_idx]
            if 16 < peak_idx and peak_idx < 32:
                peaks_found.append(peak_idx)
    return peaks_found  # top_peaks_indices


def apply_bandpass_filter(input_signal, pass_band, sample_rate):
    """
    designs bandpass filter with pass band specified by low_freq and high_freq.
    The input_siganl is filtered with the bandpass filter and filter signal
    is returned

    Args:
        input_signal (np.ndarray): input audio siganl
        low_freq (int): lower cutoff frequency
        high_freq (int): higher cutoff frequency
        sample_rate (int): sampling frequency (in Hz)
    Returns:
        filtered_signal (np.ndarray): bandpass filtered signal
    """
    # Design the bandpass filter
    global sos
    global enable_filter_design
    if enable_filter_design == True:
        nyquist_freq = 0.5 * sample_rate
        low = pass_band[0] / nyquist_freq
        high = pass_band[1] / nyquist_freq
        sos = signal.butter(4, [low, high], btype="band", output="sos")
        # enable_filter_design = False
        # print(f"sos filter coefficient = {sos}")
        enable_filter_design = False

    # Apply the filter to the input signal
    filtered_signal = signal.sosfilt(sos, input_signal)

    return filtered_signal


def filter_frequencies(arr, fc_low, fc_high, Fs, N):
    """
    zeros out fft bins between f1 to f2 frequencies
    Args:
        arr (np.ndarray): STFT of the audio signal
        fc_low (int): lower cutoff frequency
        fc_high (int): higher cutoff frequency
        sample_rate (int): sampling frequency (in Hz)
        N (int): frame length
    Returns:
        returns modified spectogram
    """

    f_res = Fs / N
    idx1 = int(fc_low // f_res + 1)
    idx2 = int(fc_high // f_res)
    arr[idx1 : idx2 + 1, :] = 0
    return arr


def bp_filter_frequencies(arr, freqs, Fs, N):
    """
    zeros out fft bins between f1 to f2 frequencies
    Args:
        arr (np.ndarray): STFT of the audio signal
        fc_low (int): lower cutoff frequency
        fc_high (int): higher cutoff frequency
        sample_rate (int): sampling frequency (in Hz)
        N (int): frame length
    Returns:
        returns modified spectogram
    """
    f_res = Fs / N
    idx1 = int(freqs[0] // f_res + 1)
    idx2 = int(freqs[1] // f_res)
    arr[0:idx1] = 0
    arr[idx2 + 1 :] = 0

    return arr


def compute_novelty_energy(
    x, Fs=1, N=512, H=256, gamma=10, norm=True
):
    """Compute energy-based novelty function

    Notebook: C6/C6S1_NoveltyEnergy.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = frame length)
        H (int): Hop size (Default value = hop length)
        gamma (float): Parameter for logarithmic compression (Default value = 10.0)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_energy (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    # x_power = x**2
    w = signal.hann(N)
    Fs_feature = Fs / H
    energy_local = np.convolve(x**2, w**2, "same")
    energy_local = energy_local[::H]
    max_energy = np.max(energy_local)
    if gamma is not None:
        energy_local = np.log(1 + gamma * energy_local)
    energy_local_diff = np.diff(energy_local)
    energy_local_diff = np.concatenate((energy_local_diff, np.array([0.0])))
    novelty_energy = np.copy(energy_local_diff)
    novelty_energy[energy_local_diff < 0] = 0
    if norm:
        max_value = max(novelty_energy)
        if max_value > 0:
            novelty_energy = novelty_energy / max_value

    novelty_energy = novelty_energy * max_energy
    # novelty_energy[energy_local_diff < 0.005] = 0

    return novelty_energy, Fs_feature


def compute_local_average_new (x, M):
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = int(max(m - M, 0))
        b = int(min(m + M + 1, L))
        
        if a == 0:
            b = min (2*M + 1, L)
        if b == L :
            a = max( L - 2*M - 1, 0)
            
        xd = sorted (x[a:b])
        win_len = len(xd)

        if win_len > M//6:
            win_len = M//6
        
        if (win_len < 3):
            win_len = 3
        win_len = 1
        local_average[m] = (1/win_len)*np.sum(xd[:win_len])
    return local_average

def compute_local_average(x, M):
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        # win_len = 1
        # if (b > a) :
        #     win_len = b - a
        xd = sorted(x[a:b])
        win_len = len(xd)

        if win_len > M // 6:
            win_len = M // 6
        if win_len < 3:
            win_len = 3
        local_average[m] = (1 / win_len) * np.sum(xd[:win_len])
    return local_average


def compute_min_average(x):
    """Compute min average of signal

    Args:
        x (np.ndarray): Signal
    Returns:
        min_average (np.ndarray):
    """
    L = len(x)
    local_average = np.ones(L)
    xd = sorted(x)
    # print(xd)
    avg = np.mean(xd[:min_average_len])
    if avg == 0:
        avg = amp_thr
    # print("avg = ",avg, "max", np.max(x), "min", np.min(x))

    return avg


def process_novelty_spectrum(Y_diff, M, norm=True):
    # print(f'process_novelty = {np.sum(Y_diff, axis=0)}')
    novelty_spectrum = np.sum(Y_diff, axis=0)

    # print (f'novelty_spectrum is {novelty_spectrum.shape} values {novelty_spectrum}')
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0])))
    # print("novelty spectrum shape", novelty_spectrum.shape)
    # print("shape of novelty spectrum", novelty_spectrum.shape)
    if M > 0:
        #print (f"{novelty_spectrum = }")
        local_average = compute_local_average(novelty_spectrum, M)
        # local_average = compute_min_average(novelty_spectrum)
        # local_average = np.sum(g_noise_spectrum)
        
        for i in range(len(local_average)) :

            if local_average[i] <= 0 :
                local_average[i] = np.max(novelty_spectrum)/5
        #print (f"{local_average = }")
        novelty_spectrum = np.divide(novelty_spectrum, local_average)

    return novelty_spectrum
def calculate_snr (novelty_spectrum, M):
    local_average = compute_local_average(novelty_spectrum, M)
    for i in range(len(local_average)) :        
        if local_average[i] <= 0 :
            local_average[i] = np.max(novelty_spectrum)/5
    novelty_spectrum [novelty_spectrum == 0] = 1
    local_average[local_average == 0 ] = 1
    novelty_spectrum = np.divide(novelty_spectrum, local_average)
    return novelty_spectrum
    
def compute_novelty_spectrum_new (Y, Fs=1, N=256, H=128, M=20, norm=True, threshold=5):
 
    Y_diff_a = np.diff(Y, n=1, axis=0)
    Y_diff_a[Y_diff_a <= 0] = 0
    
    novelty_spectrum = np.sum(Y_diff_a, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0])))
    novelty_spectrum_a = copy.deepcopy(novelty_spectrum)
    
    novelty_spectrum = calculate_snr (novelty_spectrum, M)
    
    # local_average = compute_local_average(novelty_spectrum, M)
    # for i in range(len(local_average)) :
    #     if local_average[i] <= 0 :
    #         local_average[i] = np.max(novelty_spectrum)/5
    # novelty_spectrum = np.divide(novelty_spectrum, local_average)

    peaks, _ = signal.find_peaks(novelty_spectrum, prominence=(None, None))
    mask = np.zeros(len(novelty_spectrum))
    mask[peaks] = 1

    for k in range(len(novelty_spectrum)):

        if novelty_spectrum[k] > threshold:
            novelty_spectrum[k] = novelty_spectrum [k]
            if (novelty_spectrum[k] > threshold*1.5) :
                novelty_spectrum[k] = threshold*1.5
        else:
            novelty_spectrum[k] = 0

    novelty_spectrum = novelty_spectrum * mask

    return novelty_spectrum, (novelty_spectrum_a * mask)  # + novelty_spectrum_d

 

def compute_novelty_spectrum(Y, Fs=1, N=256, H=128, M=20, norm=True, threshold=5):
    """Compute spectral-based novelty function

    Notebook: modified version of C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 256)
        H (int): Hop size (Default value = 128)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
    """
    Y_diff_d = np.diff(Y, n=1, axis=0)
    Y_diff_d[Y_diff_d > 0] = 0

    Y_diff_a = np.diff(Y, n=1, axis=0)
    Y_diff_a[Y_diff_a <= 0] = 0
    # print (f"shape of Y_diff_a = {Y_diff_a.shape}")
    novelty_spectrum_a = process_novelty_spectrum(Y_diff_a, M, norm)
    # novelty_spectrum_d = process_novelty_spectrum(-Y_diff_d, M, norm)
    #np.set_printoptions(precision=2)
    #print (f"{Y_diff_a }")
    
    #print([ "{:0.2f}".format(x) for x in Y_diff_a ])
    novelty_spectrum = np.copy(novelty_spectrum_a) # + novelty_spectrum_d;

    peaks, _ = signal.find_peaks(novelty_spectrum, prominence=(None, None))
    mask = np.zeros(len(novelty_spectrum))
    mask[peaks] = 1
    # print("novelty spectrum peaks", peaks);

    for k in range(len(novelty_spectrum)):

        if novelty_spectrum[k] > threshold:
            novelty_spectrum[k] = threshold  # novelty_spectrum [k]  #

        else:
            novelty_spectrum[k] = 0

    novelty_spectrum = novelty_spectrum * mask

    return novelty_spectrum, (novelty_spectrum_a * mask)  # + novelty_spectrum_d


def plot_data(sp_rows, sp_cols, sp_idx, ax1, val, duration, title):
    # ax = plt.subplot(sp_rows, sp_cols, sp_idx, sharex=ax1)
    ###
    # fig, ax1 = plt.subplots(sp_rows, sp_cols, figsize=(10, 10))
    if enable_plot_data == True:
        if sp_idx == 0:
            fig = plt.figure(figsize=(10, 20))
            fig.suptitle(title)
            gs = fig.add_gridspec(sp_rows)
            ax1 = gs.subplots(sharex=True)

            # fig, ax1 = plt.subplots(sp_rows, sp_cols, figsize=(10, 8), sharex=True)
            time = np.arange(len(val)) * duration / len(val)
            ax1[sp_idx].plot(time, val)

            plt.grid(which="both", axis="both")
            plt.tight_layout()

            return fig, ax1
        ####
        else:
            time = np.arange(len(val)) * duration / len(val)
            ax1[sp_idx].plot(time, val)
            ax1[sp_idx].set_title(title)  # , y=1.0, pad=-14
            plt.grid(which="both", axis="both")
            plt.tight_layout()

            return 0, ax1
    return 0, 0


def frequency_of_nonzero_values(arr):
    # Exclude zeros from the input array
    non_zero_arr = [x for x in arr if x != 0]

    # Use Counter to count the frequency of non-zero values
    frequency_counter = Counter(non_zero_arr)
    # frequency_counter = Counter(arr)
    if not non_zero_arr:
        return None  # Return None if the input array is all zeros
    # Sort the items by count in descending order
    sorted_items = sorted(frequency_counter.items(), key=lambda x: x[1], reverse=True)

    return sorted_items


def get_indices_for_freq_range(freq_range, freqs, Fs, N):
    f_res = Fs / N
    idx1 = int(freqs[0] // f_res + 1)
    idx2 = int(freqs[1] // f_res)
    arr[0:idx1] = 0
    arr[idx2 + 1, :] = 0

    return [idx1, idx2]

def find_nonzero_mean(arr):
    non_zero_arr = [x for x in arr if x != 0]
    if not non_zero_arr:
        return 0
    else:
        return np.mean(non_zero_arr)


def find_mean_frequency_method1(arr):
    frequency_result = frequency_of_nonzero_values(arr)
    fmean = 0
    if frequency_result is not None:
        fmean = frequency_result[0][0]
        frequency_result = frequency_result[:3]
        count = len(frequency_result)
        ftotal = 0
        total_count = 0
        # fmean = np.mean(frequency_result[:][1])
        for f in range(count):
            ftotal = ftotal + frequency_result[f][0] * frequency_result[f][1]
            total_count = total_count + frequency_result[f][1]
        fmean = ftotal / total_count
    return fmean


def find_mean_frequency(arr):
    frain_mean = 0
    if enable_fmean_method_1:
        frain_mean = find_mean_frequency_method1(arr)
    else:
        frain_mean = find_nonzero_mean(arr)
    return frain_mean


def read_audio_file(audio_file, read_size, read_offset):
    """

    Parameters
    ----------
    audio_file - path to file OR signal contents
    read_size
    read_offset

    Returns
    -------

    """
    if type(audio_file) == str:
        if audio_file.lower().endswith(".wav"):
            audio_data_in, sr_ref = librosa.load(audio_file, sr=None)

        else:
            with open(audio_file, "rb") as file:
                # skip the audio file header
                file.seek(HEADER_SIZE),
                # read the file in to an ndarray
                audio_data_in = file.read()
                # scale it to float values between -1.0 to 1.0
                scale_factor = (1 << (bytes_per_sample * 8 - 1)) - 1
                audio_data_in = (
                    np.frombuffer(audio_data_in, dtype=np.int16) / scale_factor
                )
    elif type(audio_file) == np.ndarray:
        audio_data_in = audio_file
    else:
        raise Exception(
            "Did not recognize 'audio_file' type. Must be path to WAV, audio binary, or numpy signal array"
        )

    audio_data_in = audio_data_in[read_offset : read_offset + read_size]

    return audio_data_in


def analyse_raw_audio(
    audio_file,
    ref_file,
    duration=rain_check_duration,
    offset=offset_in_seconds,
    audio=True,
    sample_rate = 11162
):
    Fs, frame_size, hop_size, bytes_per_sample = (
        sample_rate,
        frame_length,
        hop_length,
        2,
    )

    n_frames = duration * Fs / frame_size
    # read audio data
    read_size = int(frame_size * n_frames)
    read_offset = int(Fs * offset)

    audio_data_in = read_audio_file(audio_file, read_size, read_offset)

    audio_data = apply_bandpass_filter(audio_data_in, operating_freq_range, Fs)
    if ref_file is not None:
        reference_audio_in, sr_ref = librosa.load(ref_file, sr=None)
        reference_audio = apply_bandpass_filter(
            reference_audio_in, operating_freq_range, Fs
        )

    colmap = "magma"  # 'viridis' # None #

    # Create a time array for plotting
    duration = len(audio_data) / Fs

    N, H = frame_length, hop_length
    stft0 = librosa.stft(audio_data, n_fft=N, hop_length=H, win_length=N, window="hann")
    sp_idx = 0

    if enable_plot_data == True:
        title = audio_file + " waveform"
        fig, ax1 = plot_data(max_rows, 1, sp_idx, 0, audio_data, duration, title)
    stft0 = abs(stft0)
    stft = np.copy(stft0)

    if noise_factor != 0:
        # stft = estimate_and_supress_noise(abs(stft0), noise_threshold, noise_factor);
        noise_spectrum = estimate_noise_lpf (stft0, noise_threshold, noise_factor)
        # noise_spectrum = estimate_noise_ss (abs(stft0))
        # stft = stft - noise_spectrum
        # stft[stft < 0] = 0
        for i in range(stft.shape[1]):
            stft[:,i] -= noise_spectrum[:,-1]
        stft[stft < 0 ] = 0

    if log_compress_factor == 0:
        Y = abs(stft)
    else:
        Y = np.log(1 + log_compress_factor * abs(stft))

    if enable_plot_spectrogram == True:
        if log_compress_factor == 0:
            Yp = abs(stft0)
        else:
            Yp = np.log(1 + log_compress_factor*abs(stft0)) 
            
        Sp_db = librosa.amplitude_to_db(Yp, ref=np.max)
        sp_idx = sp_idx + 1
        img = librosa.display.specshow(
            Sp_db,
            sr=Fs,
            hop_length=H,
            cmap=colmap,
            x_axis="time",
            y_axis="linear",
            ax=ax1[sp_idx],
        )
        ax1[sp_idx].set(title=f"spectrogram: gamma ={log_compress_factor}")

    if noise_factor != 0:
        S_db = librosa.amplitude_to_db(Y, ref=np.max)
        if enable_plot_noise_supressed_signal == True:
            sp_idx = sp_idx + 1
            img = librosa.display.specshow(
                S_db,
                sr=Fs,
                hop_length=H,
                cmap=colmap,
                x_axis="time",
                y_axis="linear",
                ax=ax1[sp_idx],
            )
            ax1[sp_idx].set(
                title=f"spectrogram noise supressed: gamma ={log_compress_factor}"
            )

    nov = []
    nov1 = []
    # find the natural frequency of top dome.
    freq_range = [F_natural, F_natural+300] #natural_freq_range
    Y1 = copy.deepcopy(Y)
    Y1 = bp_filter_frequencies(Y1, freq_range, Fs, N)
    hn=1
    #print (f'{hn:*>20}Harmonic -{hn:*<20}')
    novk, novt = compute_novelty_spectrum_new(
        Y1,
        Fs=sample_rate,
        N=frame_length,
        H=hop_length,
        M=min_average_len,
        norm=True,
        threshold=rain_thr[0],
    )

    peaks, fpeak_array = find_peaks_in_frequency_range(
        abs(stft), search_freq_range[0], freq_range, Fs / 2, max_num_peaks
    )

    # print (f"fpeak_array = {fpeak_array}")
    frain_mean = 0
    # print("novk before = ", novk)
    # novk = np.diff(novd)
    #novt_hn = []
    for k in range(len(fpeak_array)):
        if novk[k] != 0:
            if peaks[k] == 0:
                novk[k] = 0
                novt[k] = 0
    nov.append(copy.deepcopy(novk))
    nov1.append(copy.deepcopy(novt))
    # print("novk after = ", novk)
    frain_mean = find_mean_frequency(fpeak_array)
    f_mean_harmonics = []
    f_mean_harmonics.append(frain_mean)
    # print(f"frain_mean 0 = {frain_mean}")
    # print(f"frain_mean {frain_mean}")
    # find frequency flux at harmonics of the fundamental frequency
    if enable_plot_freq_flux == 1:
        sp_idx = sp_idx + 1
        plot_data(max_rows, 1, sp_idx, ax1, nov[0], duration, "frequency flux")

    if enable_plot_freq_flux_a_d == 1:
        sp_idx = sp_idx + 1
        plot_data(max_rows, 1, sp_idx, ax1, novk, duration, "frequency flux a+d")
    update_search_freq_range (frain_mean)
    #print (frain_mean)
    #search_freq_out = [[round(value, 2) for value in nested] for nested in search_freq_range]
    #print (search_freq_out, max_harmonics)
    
    enable_combined_harmonic_analysis = False
    if (enable_combined_harmonic_analysis):
        print (f"combined harmonic analysis")
        #raining = harmonic_analysis ()
        
    if (frain_mean >= natural_freq_range[0] and frain_mean <= natural_freq_range[1]):
        for hn in range(1, max_harmonics):
            f1 = frain_mean * (hn + 1) - 100
            fpeak_range = [f1, f1+300]
            Y1 = copy.deepcopy(Y)
            freq_range = [f1,  f1+300]
            Y1 = bp_filter_frequencies(Y1, freq_range, Fs, N)
            #print (f'{hn:*>20}Harmonic -{hn:*<20}')
            novx, novt = compute_novelty_spectrum_new(
                Y1,
                Fs=sample_rate,
                N=frame_length,
                H=hop_length,
                M=min_average_len,
                norm=True,
                threshold=rain_thr[hn],
            )
            #print ("harmonic = ", hn, "search_range = ", search_freq_range[hn])
            peaks_hn, fpeak_array_hn = find_peaks_in_frequency_range(
                abs(stft), search_freq_range[hn], fpeak_range, Fs / 2, max_num_peaks
            )

            for k in range(len(fpeak_array_hn)):
                if novx[k] != 0:
                    if fpeak_array_hn[k] == 0:
                        novx[k] = 0

            frain_mean_hn = find_mean_frequency(fpeak_array_hn)
            f_mean_harmonics.append(frain_mean_hn)
            
            nov.append(copy.deepcopy(novx))
            nov1.append(copy.deepcopy(novt))
 
            if enable_plot_individual_harmonics:
                sp_idx = sp_idx + 1
                plot_data(
                    max_rows, 1, sp_idx, ax1, nov[hn], duration, f"frequency flux {hn}"
                )
    
    nov0 = nov[0]
    #print(f'{f_mean_harmonics = }')
    # if frequency flux is not above threshold for the base frequency then set the novelty for
    # all the harmonic frequencies to zero

    for k in range(len(nov0)):
        if nov0[k] == 0:
            for j in range(1, len(nov)):
                nov[j][k] = 0

    nov_hn = np.sum(nov, axis=0)
    # print(f" length of nov is {len(nov)}")

    if enable_plot_frequency_harmonics:
        sp_idx = sp_idx + 1
        plot_data(
            max_rows,
            1,
            sp_idx,
            ax1,
            np.transpose(nov1[:]),
            duration,
            "frequency flux hn",
        )  # nov_hn #np.transpose(novt_hn[1:])

    if 0:
        sp_idx = sp_idx + 1
        plot_data(max_rows, 1, sp_idx, ax1, peaks, duration, "freq peak found")
    if enable_correlation_mask:
        r_mask, peak_count = calculate_correlation_mask(
            audio_data, reference_audio, peaks, corr_thr
        )
        if enable_plot_correlation_mask == 1:
            sp_idx = sp_idx + 1
            plot_data(max_rows, 1, sp_idx, ax1, r_mask, duration, "correlation mask")

    nov_hn[nov_hn > rain_thr_hn] = rain_thr_hn
    nov_hn[nov_hn < rain_thr_hn] = 0
    raining = nov_hn
    # raining = r_mask*nov[0][:-1] #*peak
    # raining[raining > 0] = 1;
    if enable_plot_data == True:
        sp_idx = sp_idx + 1
        plot_data(max_rows, 1, sp_idx, ax1, raining, duration, "raining status")

    rain_drops = (raining >= 1).sum()
    # print (type(rain_drops))
    # if (rain_drops < 3):
    #     print(f"{rain_drops} less than 5 setting to zero")
    #     rain_drops = 0

    # plot_data (max_rows, 1, 4, ax1, peaks, duration)
    # nov_4, Fs_nov4 = compute_novelty_spectrum(X, Fs=11162, N=frame_length, H=hop_length, M=20, norm=True)
    # plot_data (max_rows, 1, 6, ax1, nov_4, duration)

    if enable_plot_data == True:
        plt.tight_layout()
        for i in range(len(ax1)):
            ax1[i].grid()
        plt.show()
        if audio: ipd.display(ipd.Audio(audio_data, rate=Fs))
    return rain_drops, frain_mean
    
    # sample_rate = 11162, 
    # freq_resolution = 45, 
    # time_resolution_ms = 10,
    # check_duration = 2,
    # op_freq_range = [400, 3000],
    # n_freq_range = [400, 600],
    # fn = 400,
    # num_harmonics = 6,
    # harmonic_threshold = [5, 4, 4, 4, 4, 4],
    # max_peaks = 3,
    # log_factor = 10,
    # ns_duration_ms = 470,
    # nf = 0,
    # min_drop_count = 1
def sample_classifier_to_evaluate (
    audio_data,
    threshold=2,
    **kwargs) -> bool:

    default_params = {
        'sample_rate': 11162,
        'freq_resolution': 45,
        'time_resolution_ms': 10,
        'check_duration': 2,
        'op_freq_range': [400, 3000],
        'n_freq_range': [400, 600],
        'fn': 400,
        'num_harmonics': 6,
        'harmonic_threshold': [5, 4, 4, 4, 4, 4],
        'max_peaks': 3,
        'log_factor': 10,
        'ns_duration_ms': 470,
        'nf': 0,
        'min_drop_count': 1
    }
    configure_parameters (**{**default_params, **kwargs})
    #print_configuration ()
    
    rain_drop_count, frain_mean = analyse_raw_audio_wrapper(audio_data, ref_file=None)
    #print (f"{rain_drop_count = } {frain_mean = }")
    if rain_drop_count > threshold:
        return True
    elif 0 <= rain_drop_count <= threshold:
        return False
    else:
        return np.nan
    
def rain_detection_algo (
    audio_data,
    **kwargs):
        
    configure_parameters (**kwargs)
    #print_configuration ()
    
    rain_drop_count, frain_mean = analyse_raw_audio_wrapper(audio_data, ref_file=None)
    
    return rain_drop_count, frain_mean

def analyse_raw_audio_wrapper(audio_file, ref_file):
    duration = rain_check_duration
    offset = 0
    rain_drop_count, frain_mean = analyse_raw_audio(
        audio_file, ref_file, duration, offset
    )

    rain_drops_threshold = math.ceil(min_drop_count_per_second * duration * 2)
    if rain_drop_count < rain_drops_threshold:
        offset = offset_in_seconds
        # print("check again for next 3 seconds")
        rain_drop_count1, frain_mean = analyse_raw_audio(
            audio_file, ref_file, duration, offset
        )
        if rain_drop_count + rain_drop_count1 > rain_drops_threshold:
            # print("aggrigate droplet count greater than 10")
            rain_drop_count = rain_drops_threshold
        else:
            # print("aggrigate droplet count less than 10 hence setting droplet count to 0")
            rain_drop_count = 0
    return rain_drop_count, frain_mean


## SAMPLE USE
if __name__ == "__main__":

    def write_results(csv_file_name, csv_columns, data):

        with open(csv_filename, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    ref_file = os.path.join("/", "Users", "vikrantoak", "Downloads", "rain_testvecs", "rain_drop_19.wav")
    # configure_parameters (sample_rate = 11162, 
    #                       freq_resolution = 45, 
    #                       time_resolution_ms = 10,
    #                       check_duration = 2,
    #                       op_freq_range = [400, 3000],
    #                       n_freq_range = [400, 600],
    #                       fn = 400,
    #                       num_harmonics = 6,
    #                       harmonic_threshold = [5, 4, 4, 4, 4, 4],
    #                       max_peaks = 3,
    #                       log_factor = 10,
    #                       ns_duration_ms = 470,
    #                       nf = 0,
    #                       min_drop_count = 1)
    #print_configuration ()

    if old_vecs == 1:
        #current_dir = "/home/vikrant/rain_testvecs/"
        current_dir = "/Users/vikrantoak/Downloads/rain_testvecs/"
        os.chdir(current_dir)
        list1 = [
            (1, "False_77361_142881--audio-B301115-63bd84623c8454000a205ed8-1680295920.wav"),
            (1, "False_54373_119893--audio-B301115-63bd84623c8454000a205ed8-1680324540.wav"), 
            (1, "63482897.RAW"),
            (1, "6348211C.RAW"),
            (1, "1678743960"),
            (0, "1678200300"),
            (1, "634D15FC.RAW"),
        ]
        # 634824EC
        data = []
        idx = 0
        change_plotting_status(True)
        
        for x in list1:
            category = x[0]
            filename = x[1]
            #rain_drop_count, frain_mean = analyse_raw_audio_wrapper(x[1], ref_file)
            rain_drop_count, frain_mean = rain_detection_algo(
                x[1],
                **default_params)
            
                # sample_rate = 11162,                                    
                # freq_resolution = 45, 
                # time_resolution_ms = 10,
                # check_duration = 2,
                # op_freq_range = [400, 3000],
                # n_freq_range = [400, 600],
                # fn = 400,
                # num_harmonics = 6,
                # harmonic_threshold = [5, 4, 4, 4, 4, 4],
                # max_peaks = 3,
                # log_factor = 10,
                # ns_duration_ms = 470,
                # nf = 0,
                # min_drop_count = 1)
            print(
                f"{idx:5} {rain_drop_count:5} {int(frain_mean):5} {category:2}:{filename}"
            )
            row = {
                "file_name": filename,
                "rain_drops": rain_drop_count,
                "category": category,
            }
            data.append(row)
        csv_filename = "test_results.csv"
        csv_columns = ["file_name", "rain_drops", "category"]
        write_results(csv_filename, csv_columns, data)
        print(data)

    else:
        # path = os.path.join('/','home', 'vikrant', 'Downloads', 'NN Training data','test1')
        # path = os.path.join('/','home', 'vikrant', 'Downloads', 'manually validated')
        path = os.path.join(
            "/", "Users", "vikrantoak", "Downloads", "balanced_validation_set_for_vikrant"
        )

        os.chdir(path)
        # print(os.listdir());
        data = []
        idx = 0
        for filename in os.listdir():

            if filename.endswith(".wav"):
                change_plotting_status(False)
                # rain_drop_count, frain_mean = analyse_raw_audio_wrapper(
                #     filename, ref_file
                # )
                rain_drop_count, frain_mean = rain_detection_algo(
                    filename,
                    **default_params)
                
                    # sample_rate = 11162,                                    
                    # freq_resolution = 45, 
                    # time_resolution_ms = 10,
                    # check_duration = 2,
                    # op_freq_range = [400, 3000],
                    # n_freq_range = [400, 600],
                    # fn = 400,
                    # num_harmonics = 6,
                    # harmonic_threshold = [5, 4, 4, 4, 4, 4],
                    # max_peaks = 3,
                    # log_factor = 10,
                    # ns_duration_ms = 470,
                    # nf = 0,
                    # min_drop_count = 1)

                # if (rain_drop_count < 5):
                #     rain_drop_count = 0
                if filename.startswith("True"):
                    category = 1
                else:
                    category = 0
                # print(f'{idx:5} {rain_drop_count:5} {int(frain_mean):5} {category:2}:{filename}')
                row = {
                    "file_name": filename,
                    "rain_drops": rain_drop_count,
                    "category": category,
                }
                data.append(row)

                if (idx % 25) == 0:
                    print(idx)
                idx = idx + 1
                # if (count > 10):
                #    break;
        csv_filename = "test_results.csv"
        csv_columns = ["file_name", "rain_drops", "category"]
        write_results(csv_filename, csv_columns, data)



