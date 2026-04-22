########################################################
# Date : Jun 24st 2025                                  #
# Commit ID : 106e0f72ffe5d087531886fd0669a7adf351dd7c #
# Repo : https://github.com/Arable/raw-audio-tools.git branch vo/roe_improvement #
# https://docs.google.com/document/d/1YJk9tjKD_xPxy54aiR_XPhg7rTtAJHDJ8QNbMPdnaD4/edit?tab=t.0#heading=h.1r0srr7c0mlq
########################################################
#!/usr/bin/env python
# coding: utf-8

# ##  Algorithmic changes ##
# 
# The existing RoE algorithm is running in the field for one season now and with the new enchancements to audio upload capability we have collected a lot of new data and the new csases of false positive.
# current algorithm looks at overlapping audio frames estimates the natural resoance frequency of top dome and harmonicity of the siganl using frequency spectrum novelty. The algorithm is looking for sudden change in signal.
# The new set of algorithm changes are trying to address some of these false positive issues with the wind induced noise. It does so by looking at the time domain statistics of the signal.

# ## Install latest version of tools ##

# In[ ]:


# !pip install --upgrade pip
# !pip install raw_audio_tools==1.0.10 --no-cache-dir
# !pip install ipympl
# !pip install tabulate
## import os
## print (os.getcwd())
## /Users/vikrantoak/my_jupyter_project/ds_local/wind_proj


# ## Import necessary packages ##

# In[1]:


# import packages
import os, sys
import numpy as np

import pandas as pd
import scipy
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt, sosfilt
from scipy.signal import hilbert, decimate
from scipy.stats import kurtosis
from scipy.fft import fft
from scipy.signal import find_peaks

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.ticker as ticker

import librosa
import IPython.display as ipd
from IPython.display import Audio, display
import csv

from collections import Counter
import statistics as stats
import math
from math import ceil, floor, log
import copy

import librosa
import librosa.display
import itertools
from itertools import islice
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabulate import tabulate
from numpy.lib.stride_tricks import as_strided
from tqdm.notebook import tqdm


# ## Path settings ##

# In[2]:


# path settings
InputType =    "LocalPath" # "LocalPath" # "RemotePath"  # "KeyList"  "CsvInput"

#Csv file path
csv_inp_file = "/Users/vikrantoak/Downloads/final_data_with_wind_rain.csv"

# localPath - path to local directory with test vectors

#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/rain_testvecs"
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/balanced_rain_test_vectors"
#localStatus = True
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/sample_noise"
#localStatus = False
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/fp_investigation"
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/firmware_share"
#localStatus = False

#TEST_VECTOR_PATH="/Users/vikrantoak/my_jupyter_project/ds_local/wind_proj/outliers"
# TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/balanced_validation_set_for_vikrant"
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/D003735"
# TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/intense_wind_resonance"
# localStatus = False
TEST_VECTOR_PATH = "/home/adithya/Code/mark3-firmware-trunk/Toolshed/audio_downloads_20250627_161122_"
localStatus = True
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/Demo"

#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/manually_validated"
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/Customer_False_Positives"
#localStatus = False
#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/DSP_bad_classification/False_fp"


# RemotePath - database query

LOCAL_AUDIO_CACHE =  "/Users/vikrantoak/my_jupyter_project/audio_header_cache"
#"/Users/vikrantoak/my_jupyter_project/audio_header_cache"  #"/Volumes/Passport_Re/user_data/audio_cache"
debug_all = False
print_mismatched = False

plot_status = False
audio_player = False



# In[3]:


# settings to be used in sagemaker studio
# InputType =  "LocalPath"         # "RemotePath"   # 
# LOCAL_AUDIO_CACHE = "user-default-efs/Users/vikrantoak/audio_cache"
# #TEST_VECTOR_PATH = "user-default-efs/Users/vikrantoak/Downloads/balanced_rain_test_vectors"
# #TEST_VECTOR_PATH = "user-default-efs/Users/vikrantoak/Downloads/interesting_testcases"
# TEST_VECTOR_PATH = "user-default-efs/Users/vikrantoak/Downloads/rain_testvecs"
# #TEST_VECTOR_PATH = "user-default-efs/Users/vikrantoak/Downloads/intense wind resonance"
# #TEST_VECTOR_PATH="user-default-efs/user-default-efs/Users/vikrantoak/Downloads/balanced_validation_set_for_vikrant"
# #TEST_VECTOR_PATH="user-default-efs/Users/vikrantoak/Downloads/wind_with_rain_t1"
# #TEST_VECTOR_PATH="user-default-efs/Users/vikrantoak/Downloads/Customer False Positives"
# #TEST_VECTOR_PATH="user-default-efs/Users/vikrantoak/Downloads/false_negative_rain"
# #TEST_VECTOR_PATH="user-default-efs/Users/vikrantoak/Downloads/manually validated"
# #TEST_VECTOR_PATH="user-default-efs/Users/vikrantoak/Downloads/DSP_bad_classifications - 20231025"


# ## FP Approach-1: Frequency Novelty Based ##

# In[4]:


# frequency novelty based metrics

def compare_novelties(nov_wind_raw, nov_rain_raw, nov_wind, nov_rain):
    """
    Compares statistical features of wind and rain novelty spectra.

    Returns:
    - Dictionary summarizing key metrics
    """
    comparison = {
        "wind_raw_max": np.max(nov_wind_raw),
        "rain_raw_max": np.max(nov_rain_raw),
        "wind_raw_mean": np.mean(nov_wind_raw),
        "rain_raw_mean": np.mean(nov_rain_raw),
        "wind_thresh_sum": np.sum(nov_wind),
        "rain_thresh_sum": np.sum(nov_rain),
        "wind_spike_count": np.sum(nov_wind > 0),
        "rain_spike_count": np.sum(nov_rain > 0),
        "overlap_spikes": np.sum((nov_wind > 0) & (nov_rain > 0)),
    }
    return comparison

def detect_gusts (mag, Fs, wind_band=(200, 300), rain_band = (400, 700),
                         n_fft=256, hop_length=128, win_length=128,
                         threshold=5, M=20):
    """
    Detect gust events using novelty spectrum in wind band.

    Returns:
    - times: timestamps (in seconds) of detected gusts
    - novelty_spectrum: full novelty spectrum (masked)
    """

    # Step 1: Get frequency vector
    freqs = librosa.fft_frequencies(sr=Fs, n_fft=n_fft)
    
    # calculate novelty spectrum for wind
    band_indices = np.where((freqs >= wind_band[0]) & (freqs <= wind_band[1]))[0]
    
    # Step 2: Extract wind band magnitude
    Y_wind = mag[band_indices, :]
    threshold_wind = 10
    # Step 3: Compute novelty spectrum
    novelty_wind, novelty_wind_raw = compute_novelty_spectrum_new (Y_wind, M=M, threshold=threshold_wind)


    # calculate novelty spectrum for wind
    band_indices = np.where((freqs >= rain_band[0]) & (freqs <= rain_band[1]))[0]
    
    # Step 2: Extract wind band magnitude
    Y_rain = mag[band_indices, :]

    # Step 3: Compute novelty spectrum
    novelty_rain, novelty_rain_raw = compute_novelty_spectrum_new (Y_rain, M=M, threshold=threshold)

    # Step 4: Get time vector for STFT frames
    times = librosa.frames_to_time(np.arange(len(novelty_wind)), sr=Fs, hop_length=hop_length)

    # Step 5: Find gust onset times
    gust_times = times[novelty_wind > 0]

    return gust_times, novelty_wind_raw, novelty_rain_raw, novelty_wind, novelty_rain
def plot_novelties(time_audio, nov_wind_raw, nov_rain_raw, nov_wind=None, nov_rain=None):
    """
    Plot raw and thresholded novelty curves for wind and rain.

    Parameters:
    - times: time vector (in seconds)
    - nov_wind_raw, nov_rain_raw: unthresholded novelty spectra
    - nov_wind, nov_rain: thresholded novelty spectra (optional)
    """
    print (f"{ len(time_audio) = }")
    print (f"{ len(nov_wind_raw) = }")
    print (f"{ len(nov_rain_raw) = }")
    print (f"{ len(nov_wind) = }")
    print (f"{ len(nov_rain) = }")
    plt.figure(figsize=(12, 6))

    # Raw novelties
    plt.plot(time_audio, nov_wind_raw, label="Wind Novelty (Raw)", color="blue", alpha=0.5)
    plt.plot(time_audio, nov_rain_raw, label="Rain Novelty (Raw)", color="green", alpha=0.5)

    # Thresholded novelties (optional)
    if nov_wind is not None:
        plt.plot(time_audio, nov_wind, label="Wind Novelty (Thresholded)", color="blue", linestyle="--")
    if nov_rain is not None:
        plt.plot(time_audio, nov_rain, label="Rain Novelty (Thresholded)", color="green", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Novelty")
    plt.title("Comparison of Wind and Rain Novelty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        

import numpy as np
import matplotlib.pyplot as plt


def compute_rain_mod(nov_rain, nov_gust, raining, rain_thr):
    """
    Compute rain_mod time series using novelty signals and rain status.

    Parameters:
    - nov_rain: np.ndarray, rain band novelty (per frame)
    - nov_gust: np.ndarray, gust band novelty (per frame)
    - raining: np.ndarray, binary or float-valued rain activity per frame

    Returns:
    - raining_mod: np.ndarray, modified rain indicator
    """
    nov_gust_safe = nov_gust.copy()
    nov_gust_safe[nov_gust_safe == 0] = np.nan  # Avoid division by zero
    ratio = nov_rain / nov_gust_safe

    raining_mod = np.where(nov_gust > 0,
                        ratio * raining,
                        nov_rain * raining)

    # Optional: replace any NaN values caused by 0/0 with 0
    raining_mod = np.nan_to_num(raining_mod)
    raining_mod = np.where(raining_mod >= rain_thr, rain_thr, 0)

    return raining_mod



# ## FP Approach-2: Instantaneous Frequency and EAC approach ##

# In[5]:


# instantaneous frequency and EAC based metrics
from scipy.signal import correlate, find_peaks

def compute_eac_for_frames(audio_frames, center_clip_threshold=0.3):
    """
    Compute Enhanced Autocorrelation (EAC) for each audio frame.

    Parameters:
    - audio_frames: np.ndarray, shape (num_frames, frame_length)
    - center_clip_threshold: float, threshold for center clipping (fraction of max)

    Returns:
    - eac_matrix: np.ndarray, shape (num_frames, frame_length), autocorrelation per frame
    """
    num_frames, frame_length = audio_frames.shape
    eac_matrix = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        frame = audio_frames[i]
        max_val = np.max(np.abs(frame))
        clip_thresh = center_clip_threshold * max_val

        # Center clipping
        clipped = frame #np.where(np.abs(frame) >= clip_thresh, frame, 0)

        # Normalized autocorrelation
        autocorr = correlate(clipped, clipped, mode='full')
        mid = len(autocorr) // 2
        ac = autocorr[mid:mid + frame_length]

        # Normalize to prevent energy scaling
        if np.max(np.abs(ac)) > 0:
            ac /= np.max(np.abs(ac))

        eac_matrix[i] = ac

    return eac_matrix

def estimate_pitch_from_eac(eac_matrix, fs, fmin=50, fmax=1000, harmonic_weights=[1.0, 0.5, 0.25]):
    """
    Estimate pitch (F0) per frame from Enhanced Autocorrelation (EAC) using harmonic summation.

    Parameters:
    - eac_matrix: np.ndarray, shape (num_frames, frame_length), output from compute_eac_for_frames
    - fs: int, sampling rate
    - fmin: float, minimum pitch frequency to detect (Hz)
    - fmax: float, maximum pitch frequency to detect (Hz)
    - harmonic_weights: list of floats, weights for harmonics (default: [1, 0.5, 0.25])

    Returns:
    - f0_estimates: np.ndarray, shape (num_frames,), estimated pitch in Hz
    """
    num_frames, frame_length = eac_matrix.shape
    lag_min = int(fs / fmax)
    lag_max = int(fs / fmin)

    f0_estimates = np.zeros(num_frames)

    for i in range(num_frames):
        eac = eac_matrix[i]

        best_score = -np.inf
        best_lag = 0

        for lag in range(lag_min, min(lag_max, frame_length)):
            score = 0
            for h, w in enumerate(harmonic_weights, start=1):
                h_lag = lag * h
                if h_lag < frame_length:
                    score += w * eac[int(h_lag)]
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag > 0:
            f0_estimates[i] = fs / best_lag
        else:
            f0_estimates[i] = 0

    return f0_estimates


# ## Approach 3: Time pulse characterisation ##

# In[6]:


# time domain pulse characterisation and related metrics 

def bandpass_filter_sos(signal, fs, lowcut, highcut, order=8):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass', output='sos')
    return sosfilt(sos, signal)

def calculate_block_energy(signal, block_length):
    num_blocks = len(signal) // block_length
    energy = np.array([
        np.sum(signal[i*block_length:(i+1)*block_length]**2)
        for i in range(num_blocks)
    ])
    return energy

def analyze_energy_peaks(audio_data, Fs=11162, freq_band=(60, 1500), 
                         block_length=48, tx_ms=400, peak_ratio_thr=4.0,
                         max_db_drop=20):
    """
    Analyze block energy envelope to extract sharp peaks and their timing features.

    Returns:
    - results: list of selected peaks with time information
    - energy: full block energy envelope
    - energy_fs: effective sampling rate of envelope (in Hz)
    """
    filtered = bandpass_filter_sos(audio_data, Fs, freq_band[0], freq_band[1])
    energy = calculate_block_energy(filtered, block_length)
    energy_fs = Fs / block_length
    time_per_block_ms = (block_length / Fs) * 1000
    half_tx_blocks = int((tx_ms / 2) / time_per_block_ms)
    total_blocks = len(energy)

    peaks, _ = find_peaks(energy)
    if len(peaks) == 0:
        return [], energy, energy_fs

    max_energy = np.max(energy[peaks])
    max_energy_db = 10 * np.log10(max_energy + 1e-12)
    valid_peaks = [p for p in peaks if 10 * np.log10(energy[p] + 1e-12) >= (max_energy_db - max_db_drop)]
    sorted_peaks = sorted(valid_peaks, key=lambda idx: energy[idx], reverse=True)

    used_mask = np.zeros_like(energy, dtype=bool)
    results = []

    for peak_idx in sorted_peaks:
        if used_mask[peak_idx]:
            continue

        a = max(peak_idx - half_tx_blocks, 0)
        b = min(peak_idx + half_tx_blocks + 1, total_blocks)
        min_energy = np.min(energy[a:b])

        if min_energy <= 0 or energy[peak_idx] / min_energy < peak_ratio_thr:
            continue

        end_idx = peak_idx
        for i in range(peak_idx + 1, b):
            if energy[i] <= 1.2 * min_energy:
                end_idx = i
                break

        start_idx = peak_idx
        for i in range(peak_idx - 1, a - 1, -1):
            if energy[i] <= 1.2 * min_energy:
                start_idx = i
                break

        rise_time_ms = (peak_idx - start_idx) * time_per_block_ms
        decay_time_ms = (end_idx - peak_idx) * time_per_block_ms


        pulse_time = rise_time_ms + decay_time_ms
        if pulse_time > 50 :
            used_mask[start_idx:end_idx + 1] = True
            continue
        offset = (block_length/(2*Fs))*1000  # adjust for the block alignment
        
        results.append({
            'peak_idx': peak_idx,
            'peak_time_ms': peak_idx * time_per_block_ms + offset,
            'peak_energy': energy[peak_idx],
            'start_time_ms': start_idx * time_per_block_ms + offset,
            'end_time_ms': end_idx * time_per_block_ms + offset,
            'rise_time_ms': rise_time_ms,
            'decay_time_ms': decay_time_ms,
            'pulse_time': pulse_time,
            'start_energy': energy[start_idx],
            'end_energy': energy[end_idx]
        })

        used_mask[start_idx:end_idx + 1] = True

    return results, energy, energy_fs
    

def print_peak_results_table(results):
    """
    Print a table of peak analysis results using time (ms) instead of indices.
    """
    if not results:
        print("No valid peaks detected.")
        return

    headers = [
        "Start Time (ms)", "Peak Time (ms)", "End Time (ms)",
        "Peak Energy", "Start Energy", "End Energy",
        "pulse Time",
        
    ]
    # "Rise Time (ms)", "Decay Time (ms)"
    table = [
        [
            round(r['start_time_ms'], 2),
            round(r['peak_time_ms'], 2),
            round(r['end_time_ms'], 2),
            round(r['peak_energy'], 6),
            round(r['start_energy'], 6),
            round(r['end_energy'], 6),
            round(r['pulse_time'], 6),
        ]
        for r in results
    ]
    # round(r['rise_time_ms'], 2),
    # round(r['decay_time_ms'], 2)

    print(tabulate(table, headers=headers, tablefmt="grid"))

def apply_time_offset_to_results(results, offset_msec):
    """
    Add a time offset (in milliseconds) to start, peak, and end times in each result entry.

    Parameters:
    - results: list of dictionaries (from analyze_energy_peaks)
    - offset_msec: float, offset to add (in ms)

    Returns:
    - new_results: list with updated time fields
    """
    new_results = []
    for r in results:
        new_entry = r.copy()
        new_entry['start_time_ms'] += offset_msec
        new_entry['peak_time_ms'] += offset_msec
        new_entry['end_time_ms'] += offset_msec
        new_results.append(new_entry)
    return new_results
def plot_energy_peaks_and_pulses(energy, Fs, block_length, results, title="Envelope with Detected Pulses"):
    """
    Plot the block energy envelope and overlay detected peaks and pulse regions.

    Parameters:
    - energy: np.ndarray, block energy values
    - Fs: sampling rate
    - block_length: samples per block
    - results: list of detected peaks (from analyze_energy_peaks)
    - title: plot title
    """
    time_per_block = block_length / Fs
    times = np.arange(len(energy)) * time_per_block * 1000  # convert to ms

    plt.figure(figsize=(12, 6))
    plt.plot(times, energy, label="Block Energy Envelope", color="black", linewidth=1)

    for pulse in results:
        # peak_idx = pulse['peak_index']
        # start_idx = pulse['start_index']
        # end_idx = pulse['end_index']
        
        start_time = pulse ['start_time_ms']
        peak_time = pulse ['peak_time_ms']
        end_time = pulse['end_time_ms']
        peak_idx = int (peak_time*Fs/(block_length*1000))
        # Mark the pulse region
        plt.axvspan(start_time, end_time, color='orange', alpha=0.3)

        # Mark the peak
        plt.plot(peak_time, energy[peak_idx], 'ro', label="Peak" if 'Peak' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel("Time (ms)")
    plt.ylabel("Energy")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ## Approach 4: Pulses Peak and Tail Characterisation ##

# In[7]:


# time domain approach-2 metrics using peakedness and tailedness of the pulses
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import get_window


def compute_instantaneous_frequency(frame, fs):
    """
    Compute instantaneous frequency of a real-valued audio frame using the Hilbert transform.

    Parameters:
    - frame: np.ndarray, real-valued audio frame (1D)
    - fs: float, sampling frequency (Hz)

    Returns:
    - f_inst: np.ndarray, instantaneous frequency (Hz) for each sample in the frame
    """
    # print (f"input to hilbert {len(frame)}")
    # Step 1: Compute analytic signal
    analytic_signal = hilbert(frame)

    # Step 2: Instantaneous phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Step 3: Derivative of phase
    dphi = np.diff(instantaneous_phase)

    # Step 4: Instantaneous frequency
    f_inst = (fs / (2.0 * np.pi)) * dphi

    # Match output length to original frame (pad last value)
    f_inst = np.append(f_inst, f_inst[-1])

    return f_inst
    
import numpy as np
from scipy.stats import kurtosis

def crest_factor(x):
    return np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12)

def peak_to_mean_ratio(x):
    return np.max(np.abs(x)) / (np.mean(np.abs(x)) + 1e-12)

def peak_to_median_ratio(x):
    return np.max(np.abs(x)) / (np.median(np.abs(x)) + 1e-12)

def get_min_neighbor_energy(energy_list, i, m):
    """
    Compute the minimum energy among current and surrounding frames (±m),
    including the current frame i. Excludes padded edge frames (0 and n-1).

    Parameters:
    - energy_list: np.ndarray of per-frame energies
    - i: int, current frame index
    - m: int, number of frames to consider on each side

    Returns:
    - float: minimum energy value among valid neighboring frames including i
    """
    n = len(energy_list)

    # Define valid index window (excluding padded 0 and n-1)
    left = max(1, i - m)
    right = min(n - 1, i + m + 1)  # end is exclusive

    if left >= right:
        return 0  # no valid neighbors

    window = energy_list[left:right]  # includes current i if in range
    return np.min(window)



def compute_overlapped_block_energy(audio_data, fft_length, hop_length):
    """
    Efficient computation of frame-wise energy using strided sliding window.
    Returns energy_list of shape (num_frames,)
    """
    n_samples = len(audio_data)
    num_frames = 1 + (n_samples - fft_length) // hop_length

    # Build strided frame view [num_frames x fft_length]
    shape = (num_frames, fft_length)
    strides = (hop_length * audio_data.strides[0], audio_data.strides[0])
    frames = as_strided(audio_data, shape=shape, strides=strides)

    # Compute energy = sum of squares across each row (axis=1)
    energy_list = np.sum(frames**2, axis=1)

    return energy_list

    
def calculate_pulse_characteristics(audio_data, num_frames, Fs, fft_length=256, hop_length=128, m=30):
    """
    Calculates time-domain pulse characteristics, energy difference, and pitch per frame.

    Parameters:
    - audio_data: np.ndarray, 1D audio signal
    - num_frames: int, number of audio frames
    - Fs: int, sampling frequency
    - fft_length: int, FFT size used per frame
    - hop_length: int, hop size between frames
    - m: int, number of frames to look around for energy normalization

    Returns:
    - dict with time vector and per-frame metrics including pitch
    """

    audio_data_padded = np.concatenate((np.zeros(hop_length), audio_data, np.zeros(hop_length)))

    # Apply bandpass filter (e.g., rain band)
    audio_data_filtered = apply_bandpass_filter(audio_data_padded, [400, 900], Fs)

    energy_list = np.zeros(num_frames)
    # Compute overlapped block energy
    energy_list = compute_overlapped_block_energy(audio_data_filtered, fft_length, hop_length)

    # # Frame the filtered signal for pitch estimation and EAC
    # total_samples = len(audio_data_filtered)
    # actual_num_frames = 1 + (total_samples - fft_length) // hop_length
    # audio_frames = np.lib.stride_tricks.sliding_window_view(audio_data_filtered, fft_length)[::hop_length][:num_frames]

    # # Compute EAC
    # eac_matrix = compute_eac_for_frames(audio_frames)

    # # Estimate pitch
    # pitch_list = estimate_pitch_from_eac(eac_matrix, Fs)

    # Initialize feature arrays
    k_list = np.zeros(num_frames)
    crest_list = np.zeros(num_frames)
    diff_energy = np.zeros(num_frames)
    min_energy = np.zeros(num_frames)

    # p2m_list = np.ones(num_frames)
    # p2med_list = np.ones(num_frames)
    # finst_list = np.zeros((fft_length, num_frames))

    for i in range(num_frames):
        start = i * hop_length
        end = start + fft_length
        if end > len(audio_data_padded) :
            break

        x_frame = audio_data_padded[start:end]
        min_energy[i] = get_min_neighbor_energy(energy_list, i, m)
        
        if i >= 2:
            
            last_frame_energy = energy_list[i-1]
            if (energy_list[i-2] < energy_list[i-1]):
                last_frame_energy = energy_list [i-2]
            if energy_list[i] > last_frame_energy:
                # diff_energy[i] = (energy_list[i]) / (1e-12 + energy_list[i - 2])
                # diff_energy[i] = (energy_list[i] - last_frame_energy) / (1e-12 + min_energy)
                diff_energy[i] = energy_list[i]/(last_frame_energy + 1e-12)
            else:
                diff_energy[i] = 0
        else:
            diff_energy[i] = 0

        if i > 0:
            k_list[i] = kurtosis(x_frame, fisher=True)
            crest_list[i] = crest_factor(x_frame)
            
            # p2m_list[i] = peak_to_mean_ratio(x_frame)
            # p2med_list[i] = peak_to_median_ratio(x_frame)
            # finst_list[:, i] = compute_instantaneous_frequency(x_frame, Fs)

    # Time axis
    time_vector = np.arange(num_frames) * hop_length / Fs

    # Pad results for consistency
    k_list = np.concatenate((k_list, [0]))
    crest_list = np.concatenate((crest_list, [0]))
    energy_list = np.concatenate((energy_list, [0]))
    min_energy = np.concatenate ((min_energy, [0]))
    diff_energy = np.concatenate((diff_energy, [0]))
    time_vector = np.concatenate(([0], time_vector))
    
    # p2m_list = np.concatenate((p2m_list, [1]))
    # p2med_list = np.concatenate((p2med_list, [1]))
    # pitch_list = np.concatenate((pitch_list, [0]))
    # fft_length = finst_list.shape[1]
    # zero_frame = np.zeros((1, fft_length))
    # finst_list_padded = np.vstack((finst_list, zero_frame))
    # finst_list = np.concatenate((finst_list, [0]))
    # energy_list = energy_list/(min_energy+1e-12)

    
    return {
        'times': time_vector,
        'kurtosis': k_list,
        'crest_factor': crest_list,
        'diff_energy': diff_energy,
        'energy_list': energy_list,
        'min_energy':min_energy,
        # 'energy': energy_list,
        # 'finst_list': finst_list,
        # 'peak_to_mean': p2m_list,
        # 'peak_to_median': p2med_list,
        # 'pitch': pitch_list,
    }


def time_domain_raining_status(algo_state, params):
    """
    Refines the raining status based on kurtosis and crest factor thresholds.

    Parameters:
    - raining: ndarray of original rain status (e.g., 0 or 1)
    - kurtosis: ndarray of kurtosis values per frame
    - crest_factor: ndarray of crest factor values per frame
    - kurtosis_thr: float, minimum required kurtosis
    - crest_thr: float, minimum required crest factor

    Returns:
    - raining_mod: ndarray of modified rain status
    """
    kurtosis_thr = params["kurtosis_thr"]
    crest_thr = params["crest_thr"]
    diff_energy_thr = params ["diff_energy_thr"]

    raining = algo_state["raining"]
    kurtosis = algo_state ["kurtosis"] 
    crest_factor = algo_state ["crest_factor"]
    diff_energy = algo_state ["diff_energy"]

    # print (f"{len(raining) = }) {len(kurtosis) = } {len(crest_factor) = }")
    raining = np.asarray(raining)
    kurtosis = np.asarray(kurtosis)
    crest_factor = np.asarray(crest_factor)

    peaks = (kurtosis > kurtosis_thr) & (crest_factor > crest_thr)  & (diff_energy > diff_energy_thr)
    #raining_mod = np.where(mask, raining, 0)
    
    return peaks



# ## Plotting Utilities ##

# In[8]:


# plotting code

def plot_frame_matrix_on_axis(matrix, Fs, hop_size, ax, title="Spectrogram", y_label="Frequency (Hz)"):
    """
    Generic function to plot a frame-wise matrix (e.g., spectrogram or instantaneous frequency)
    using librosa.display.specshow.

    Parameters:
    - matrix: 2D np.ndarray, shape (frequency_bins, time_frames), to be visualized
    - Fs: int, sampling rate
    - hop_size: int, hop length used in STFT
    - ax: matplotlib axis to plot on
    - title: str, title of the plot
    - y_label: str, label for the y-axis
    """
    librosa.display.specshow(
        matrix,
        sr=Fs,
        hop_length=hop_size,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True)
    
def plot_algo_state_summary(algo_state, params):
    """
    Plots synchronized subplots for audio waveform, spectrogram, novelty functions,
    and raining status using librosa's specshow for accurate spectrogram axis alignment.

    Parameters:
        algo_state (dict): Dictionary with keys like audio_data, spectrum_db, Nov0, etc.
        Fs (int): Sampling rate of the audio signal.
    """
    handle_fp = params["handle_fp"]
    handle_fn = params["handle_fn"]

    subplots = 3    
    kurtosis_thr = params["kurtosis_thr"]
    crest_thr = params["crest_thr"]
    diff_energy_thr = params["diff_energy_thr"]
    Fs= params["sample_rate"]

    audio_data = algo_state["audio_data"]
    file_key = algo_state ['file_key']
    Sp_db = algo_state["spectrum_db"]  # This is the one we'll use for spectrogram
    Nov0 = algo_state["Nov0"]
    novk = algo_state["novk"] # novk
    novt = algo_state["novt"]
    raining = algo_state["raining"]

    if enable_nov_wind_detect == True :
        nov_wind = algo_state["nov_wind"]
        nov_rain = algo_state["nov_rain"]
        nov_wind_raw = algo_state["nov_wind_raw"]
        nov_rain_raw = algo_state["nov_rain_raw"]
        

    if process_fn or process_fp:
        subplots += 4
        audio_data = algo_state["filtered"]
        kurtosis = algo_state ["kurtosis"] 
        crest_factor = algo_state ["crest_factor"]
        rain_peaks = algo_state["rain_peaks"]
        diff_energy = algo_state ["diff_energy"]
        energy_list = algo_state ["energy_list"]
        min_energy = algo_state["min_energy"]
        
        # peak_to_mean = algo_state ["peak_to_mean"]
        # peak_to_median = algo_state ["peak_to_median"]
        # pitch = algo_state ["pitch"]
        # finst_list = algo_state["finst_list"]
    
        # condition attributes for better visualization 
        crest_factor[crest_factor > 25] = 25
        kurtosis[kurtosis > 25] = 25
        diff_energy_max = 10
        diff_energy [diff_energy > diff_energy_max] = diff_energy_max
        energy_list_max = 100
        energy_list [energy_list > energy_list_max] = energy_list_max
        
        # spectral_flatness = algo_state ["spectral_flatness"]
        # peak_to_median[peak_to_median > 10] = 10
        # peak_to_mean[peak_to_mean > 10] = 10


    

    hop_size = 128
    n_frames = len(Nov0)
    frame_duration = hop_size / Fs
    duration = n_frames * frame_duration

    # Time axes
    time_audio = np.linspace(0, duration, len(audio_data))
    time_spec = np.linspace(0, duration, n_frames)

    fig, axs = plt.subplots(subplots, 1, figsize=(14, 14), sharex=True)

    # Audio waveform
    idx = 0
    axs[idx].plot(time_audio, audio_data, color='blue')
    axs[idx].set_title(f"{file_key}")
    axs[idx].set_ylabel("Amplitude")
    axs[idx].grid(True)

    # Spectrogram using librosa specshow
    idx += 1
    librosa.display.specshow(
        Sp_db,
        sr=Fs,
        hop_length=hop_size,
        x_axis="time",
        y_axis="linear",
        cmap= "magma",
        ax=axs[idx],
    )
    axs[idx].set_title("Spectrogram (Sp_db)")
    axs[idx].set_ylabel("Frequency (Hz)")
    axs[idx].grid(True)

    # # Nov0
    # idx += 1
    # axs[idx].plot(time_spec, Nov0, label="Nov0")
    # axs[idx].set_title("Nov0 (novelty function)")
    # axs[idx].set_ylabel("Amplitude")
    # axs[idx].grid(True)

    # Frequency flux (novk)
    # idx += 1
    # axs[3].plot(time_spec, novk, label="Freq Flux", color='tab:blue')
    # axs[3].set_title("Frequency Flux (Novk)")
    # axs[3].set_ylabel("Amplitude")
    # axs[3].grid(True)

    # Raining status
    idx += 1
    axs[idx].step(time_spec, raining, where='post', label="Raining", color='tab:green')
    axs[idx].set_title("Raining Status")
    axs[idx].set_ylabel("Status")
    axs[idx].set_xlabel("Time (sec)")
    axs[idx].grid(True)

    
    if process_fp or process_fn :
        # kurtosis
        idx += 1
        time_spec = np.linspace(0, duration, len(kurtosis))
        # Main step plot (all values)
        axs[idx].step(time_spec, kurtosis, where='post', label="kurtosis", color='tab:green')
        # Highlight where kurtosis > threshold
        above_mask = np.array(kurtosis) > kurtosis_thr
        highlighted = np.where(above_mask, kurtosis, np.nan)  # mask non-threshold values
        axs[idx].step(time_spec, highlighted, where='post', label=f'> {kurtosis_thr}', color='red')
        # Optional: horizontal threshold line
        axs[idx].axhline(kurtosis_thr, linestyle='--', color='gray', linewidth=1)
        # Labels and styling
        axs[idx].set_title("Kurtosis Status")
        axs[idx].set_ylabel("Kurtosis")
        axs[idx].set_xlabel("Time (sec)")
        axs[idx].grid(True)
        axs[idx].legend()
    
        # crest_factor
        idx += 1
        time_spec = np.linspace(0, duration, len(crest_factor))
        # Main step plot (all values)
        axs[idx].step(time_spec, crest_factor, where='post', label="crest_factor", color='tab:green')
        # Highlight where crest_factor > threshold
        above_mask = np.array(crest_factor) > crest_thr
        highlighted = np.where(above_mask, crest_factor, np.nan)  # mask non-threshold values
        axs[idx].step(time_spec, highlighted, where='post', label=f'> {crest_thr}', color='red')
        # Optional: horizontal threshold line
        axs[idx].axhline(crest_thr, linestyle='--', color='gray', linewidth=1)
        # Labels and styling
        axs[idx].set_title("crest_factor Status")
        axs[idx].set_ylabel("crest_factor")
        axs[idx].set_ylabel("Status")
        axs[idx].set_xlabel("Time (sec)")
        axs[idx].grid(True)
    
        # idx += 1
        # axs[idx].step(time_spec, diff_energy, where='post', label="diff_energy", color='tab:green')
        # above_mask = np.array(diff_energy) > diff_energy_thr
        # highlighted = np.where(above_mask, diff_energy, np.nan)  # mask non-threshold values
        # axs[idx].step(time_spec, highlighted, where='post', label=f'> {diff_energy_thr}', color='red')
        # axs[idx].set_title("diff energy Status")
        # axs[idx].set_ylabel("Status")
        # axs[idx].set_xlabel("Time (sec)")
        # axs[idx].grid(True)
    
        # idx += 1
        # axs[idx].step(time_spec, pitch, where='post', label="pitch", color='tab:green')
        # axs[idx].set_title("pitch Status")
        # axs[idx].set_ylabel("Status")
        # axs[idx].set_xlabel("Time (sec)")
        # axs[idx].grid(True)
        
        # idx += 1
        # plot_frame_matrix_on_axis(finst_list, Fs, hop_size, axs[idx], title="Instantaneous Frequency", y_label="Frequency (Hz)")
     
        # rain energy peaks    
        idx += 1
        axs[idx].step(time_spec, rain_peaks, where='post', label="rain_peaks", color='tab:green')
        axs[idx].set_title("rain peaks Status")
        axs[idx].set_ylabel("Status")
        axs[idx].set_xlabel("Time (sec)")
        axs[idx].grid(True)
    
        idx += 1
        axs[idx].step(time_spec, energy_list, where='post', label="energy_list", color='tab:green')
        axs[idx].step(time_spec, min_energy, where='post', label="min_energy", color='tab:red')
        # above_mask = np.array(energy_list) > energy_thr
        # highlighted = np.where(above_mask, energy_list, np.nan)  # mask non-threshold values
        # axs[idx].step(time_spec, highlighted, where='post', label=f'> {diff_energy_thr}', color='red')
        axs[idx].set_title("energy list")
        axs[idx].set_ylabel("Status")
        axs[idx].set_xlabel("Time (sec)")
        axs[idx].grid(True)

    plt.tight_layout()
    plt.show()

def plot_novelty_and_rain_status(nov_hn, nov_hn_new, rain_status, rain_status_existing, frame_duration_s=0.1):
    """
    Plot novelty values and rain status on separate subplots.

    Parameters:
    - nov_hn (np.ndarray): Original summed novelty per frame.
    - nov_hn_new (np.ndarray): Processed summed novelty per frame.
    - rain_status (np.ndarray): Boolean array indicating detected rain (new).
    - rain_status_existing (np.ndarray): Boolean array indicating existing rain label.
    - frame_duration_s (float): Duration of each frame in seconds (for x-axis scaling).
    """
    frames = np.arange(len(nov_hn))
    time_axis = frames * frame_duration_s

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot novelty values
    axs[0].plot(time_axis, nov_hn, label='Original nov_hn', linestyle='--')
    axs[0].plot(time_axis, nov_hn_new+2, label='Processed nov_hn_new', linewidth=2)
    axs[0].set_ylabel("Novelty Sum")
    axs[0].set_title("Novelty Over Time")
    axs[0].legend()
    axs[0].grid(True)

    # Plot rain status
    axs[1].step(time_axis, rain_status_existing.astype(int), label='Existing Rain Status', where='mid', linestyle='--')
    axs[1].step(time_axis, rain_status.astype(int), label='Detected Rain Status', where='mid', linewidth=2)
    axs[1].set_ylabel("Rain Status")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['No Rain', 'Rain'])
    axs[1].set_title("Rain Detection Comparison")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# ## RoE Algorithm Integrated With New Tools ##

# In[9]:


# RoE algorithm integration of the new tools

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
# handle_fp = True
# handle_fn = False
t_band = [400, 3500]

default_params = {
    "sample_rate": 11162,
    "freq_resolution": 45,
    "time_resolution_ms": 10,
    "check_duration": 10,
    "op_freq_range": [400, 3500],
    "n_freq_range": [400, 700],
    "fn": 400,
    "num_harmonics": 6,
    "harmonic_threshold": [4.5, 4.0, 3.5, 3.5, 3.5, 3.5],
    "max_peaks": 3,
    "log_factor": 0,
    "ns_duration_ms": 470,
    "nf": 0,
    "min_drop_count": 0.3,
    "rain_drop_min_thr": 3,
    "rain_drop_max_thr": 50,
    "rain_peaks_min_thr": 9,
    "rain_peaks_max_thr": 30,
    "kurtosis_thr": 2.5,
    "crest_thr": 3.75,
    "diff_energy_thr": 6.5,
    "t_band": [400, 3500],
    "handle_fp": True,
    "handle_fn": True,
    "enable_nov_wind_dection": False,
    "enable_energy_peak_detection": False,
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

HEADER_SIZE = 40

enable_fmean_method = 2
enable_correlation_mask = 0
enable_filter_design = True

enable_plot_data = False
max_rows = 2

if enable_plot_data == True:

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

def merge_algo_state(dict1, dict2, verbose=False):
    # print (f"{type(dict1) = }, { type(dict2) = }")
    if not isinstance(dict2, dict):
        return dict1
    

    for key, val2 in dict2.items():
        if key not in dict1:
            dict1[key] = val2
        else:
            val1 = dict1[key]
            if ( isinstance(val1, list)
                and isinstance(val2, list)
                and all(isinstance(x, np.ndarray) for x in val1)
                and all(isinstance(x, np.ndarray) for x in val2)
               ):
                if verbose:
                    # print(f"list of ndarray: {key=} {len(val1)=} {len(val2)=}")
                    print (f" {[arr.shape for arr in val1] = }")
                    print (f"{[arr.shape for arr in val2] = }")
                
                # Pad val2 if it's shorter than val1
                if len(val2) < len(val1):
                    for i in range(len(val2), len(val1)):
                        shape = val2[0].shape
                        zero_pad = np.zeros(shape, dtype=val2[0].dtype)
                        val2.append(zero_pad)
                
                # Or pad val1 if it's shorter (just in case)
                elif len(val1) < len(val2):
                    for i in range(len(val1), len(val2)):
                        shape = val1[0].shape
                        zero_pad = np.zeros(shape, dtype=val1[0].dtype)
                        val1.append(zero_pad)

                merged_list = [np.concatenate((x, y)) for x, y in zip(val1, val2)]
                dict1[key] = merged_list
                first_shape = merged_list[0].shape
                for arr in merged_list:
                    if arr.shape != first_shape:
                        print ([arr.shape for arr in merged_list])

            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if verbose:
                    print(f"ndarray {key = }: {val1.shape = }, {val2.shape = }")
                if val1.ndim > 1:
                    if val1.shape[0] != val2.shape[0]:
                        raise ValueError(f"Incompatible array shapes for key '{key}'")
                    dict1[key] = np.concatenate((val1, val2), axis=1)
                else:
                    dict1[key] = np.concatenate((val1, val2))

            elif isinstance(val1, list) and isinstance(val2, list):
                if verbose:
                    print(f"list {key = }: {len(val1) = }, {len(val2) = }")
                dict1[key] = val1 + val2

            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                dict1[key] = val1 + val2

            else:
                raise TypeError(f"Incompatible types for key '{key}': {type(val1)} and {type(val2)}")

    return dict1 


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
                          check_duration=10,
                          op_freq_range=[400, 3500],
                          n_freq_range=[400, 700],
                          fn=400, 
                          num_harmonics=6,
                          harmonic_threshold=[4.5, 4.0, 3.5, 3.5, 3.5, 3.5],
                          max_peaks=3,
                          log_factor=0,
                          ns_duration_ms=470,
                          nf=0,
                          min_drop_count=0.3,
                          kurtosis_thr = 2.5,
                          crest_thr = 3.75,
                          diff_energy_thr = 6.5,
                          rain_drop_min_thr = 3,
                          rain_drop_max_thr = 50,
                          rain_peaks_min_thr = 9,
                          rain_peaks_max_thr = 30,
                          t_band = [400, 3500],
                          handle_fp = True,
                          handle_fn = True,
                          enable_nov_wind_dection = False,
                          enable_energy_peak_detection = False,
                         ):
    global Fs, frame_length, hop_length, rain_check_duration, operating_freq_range, hn_max, \
    log_compress_factor, min_average_len, rain_thr, rain_thr_hn, corr_thr, \
    min_drop_count_per_second, max_num_peaks,  natural_freq_range, noise_factor, noise_threshold,\
    search_freq_range, noise_duration_ms, F_natural, sos, time_freq_band, process_fp, process_fn, \
    enable_nov_wind_detect, enable_energy_peak_detect

    enable_nov_wind_detect = enable_nov_wind_dection
    enable_energy_peak_detect = enable_energy_peak_detection
    
    process_fp = handle_fp
    process_fn = handle_fn
    # print (f" configuration {process_fp = } {process_fn}")
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
    time_freq_band = t_band
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
        enable_plot_individual_harmonics = 0
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
    """
    The function find_peaks_in_frequency_range analyzes stft magnitude spectrum to detect prominent frequency peaks
    within a specified frequency range, and reports whether a peak was found in a narrower target frequency band for
    each time frame.

    """
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
    # data1 = [row[16:32] for row in data]e
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

def design_bandstop_filter(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    sos = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='bandstop', output='sos')
    return sos

def apply_filter(audio_data, sos):
    return signal.sosfilt(sos, audio_data)
    
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
    # enable_filter_design = True
    if enable_filter_design == True:
        nyquist_freq = 0.5 * sample_rate
        low = pass_band[0] / nyquist_freq
        high = pass_band[1] / nyquist_freq
        sos = signal.butter(4, [low, high], btype="band", output="sos")
        # enable_filter_design = False
        # print(f"sos filter coefficient = {sos}")
        enable_filter_design = True

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



#     return novelty_spectrum
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
    # print(f"{Y.shape = }, {Y_diff_a.shape = }")
    novelty_spectrum = np.sum(Y_diff_a, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0])))
    novelty_spectrum_a = copy.deepcopy(novelty_spectrum)
    
    novelty_spectrum = calculate_snr (novelty_spectrum, M)

    peaks, _ = signal.find_peaks(novelty_spectrum, prominence=(None, None))
    mask = np.zeros(len(novelty_spectrum))
    mask[peaks] = 1
    # mask = np.ones(len(novelty_spectrum))
    
    # novelty spectrum without thresholding 
    novelty_spectrum_1 = novelty_spectrum * mask
    
    for k in range(len(novelty_spectrum)):

        if novelty_spectrum[k] > threshold:
            novelty_spectrum [k] = novelty_spectrum [k]
            if (novelty_spectrum[k] > threshold*1.5) :
                novelty_spectrum[k] = threshold*1.5
        else:
            novelty_spectrum [k] = 0

    novelty_spectrum = novelty_spectrum * mask

    return novelty_spectrum, novelty_spectrum_1 # (novelty_spectrum_a * mask)  # + novelty_spectrum_d

 


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
    if enable_fmean_method == 1:
        frain_mean = find_mean_frequency_method1(arr)
    else:  #enable_fmean_method == 2:
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
                if len(audio_data_in)%2 :
                    audio_data_in = audio_data_in[:-1]
                audio_data_in = (
                    np.frombuffer(audio_data_in, dtype=np.int16) / scale_factor
                )
    elif type(audio_file) == np.ndarray:
        audio_data_in = audio_file
    else:
        print (type(audio_file))
        raise Exception(
            "Did not recognize 'audio_file' type. Must be path to WAV, audio binary, or numpy signal array"
        )

    audio_data_in = audio_data_in[read_offset : read_offset + read_size]

    return audio_data_in
def novelty_based_gust_detection(
    Y,
    Fs,
    frame_length,
    hop_length,
    duration,
    wind_band=[150, 300],
    threshold=4.25,
    M=20,
    nov=None,
):
    """
    Performs novelty-based gust detection and returns an updated algo_state dictionary.

    Parameters:
        Y (ndarray): FFT amplitude spectrum (frequencies x frames)
        Fs (int): Sampling frequency
        frame_length (int): FFT window size
        hop_length (int): Hop size between frames
        duration (float): Duration of the signal in seconds
        wind_band (list): Frequency range for wind novelty [low, high]
        threshold (float): Novelty threshold
        M (int): Number of frames used to estimate noise floor
        nov (list or ndarray): Required to determine number of frames for time axis

    Returns:
        algo_state (dict): Dictionary containing computed novelty values
    """
    gust_time, nov_wind_raw, nov_rain_raw, nov_wind, nov_rain = detect_gusts(
        Y,
        Fs,
        wind_band=wind_band,
        n_fft=frame_length,
        hop_length=hop_length,
        threshold=threshold,
        M=M
    )

    # Optional novelty comparison logic
    comparison = compare_novelties(
        nov_wind_raw,
        nov_rain_raw,
        nov_wind_raw > 10,
        nov_rain_raw > 5
    )

    if nov is not None:
        n_frames = len(nov[0])
    else:
        n_frames = Y.shape[1]

    time_spec = np.linspace(0, duration, n_frames)

    algo_state = {
        "nov_wind": nov_wind,
        "nov_rain": nov_rain,
        "nov_wind_raw": nov_wind_raw,
        "nov_rain_raw": nov_rain_raw,
        "gust_time": gust_time,
        "time_spec": time_spec,
        "novelty_comparison": comparison
    }

    return algo_state
    
def update_algo_state_from_results(algo_state, t_results):
    """
    Updates the algo_state dictionary with selected keys from t_results.

    Parameters:
        algo_state (dict): The dictionary to be updated.
        t_results (dict): The dictionary containing the results.

    Returns:
        dict: The updated algo_state.
    """
    keys_to_update = [
        "times",
        "kurtosis",
        "crest_factor",
        "diff_energy",
        "energy_list",
        "min_energy",

        # "finst_list",
        # "peak_to_mean",
        # "peak_to_median",
        # "gini",
        # "spectral_flatness",
        # "pitch",
    ]

    for key in keys_to_update:
        if key in t_results:
            algo_state[key] = t_results[key]

    return algo_state

import numpy as np

def detect_rain_from_novelty(
    nov_hn_list,        # List of novelty arrays (one per harmonic)
    thresholds,         # List of thresholds (one per harmonic)
    rain_threshold      # Overall threshold for summed novelty
):
    """
    Detect rain presence from harmonic novelty arrays.

    Parameters:
    - nov_hn_list (list of np.ndarray): One novelty array per harmonic, each of shape (num_frames,)
    - thresholds (list of float): One threshold per harmonic.
    - rain_threshold (float): Overall novelty threshold to classify rain in each frame.

    Returns:
    - nov_hn (np.ndarray): Combined novelty score per frame (after clipping).
    - rain_status (np.ndarray): Boolean array indicating rain status per frame.
    """
    # print (len(nov_hn_list), len(thresholds))
    if (len(nov_hn_list) != len(thresholds)):
        thresholds = thresholds[:len(nov_hn_list)]
    assert len(nov_hn_list) == len(thresholds), "Mismatch between novelty arrays and thresholds"

    processed_nov = []
    for nov, thr in zip(nov_hn_list, thresholds):
        nov = np.asarray(nov)
        clipped_nov = np.where(
            nov > 1.6 * thr,
            1.5 * thr,
            np.where(nov > thr, nov, 0)
        )
        processed_nov.append(clipped_nov)

    # Sum across harmonics
    nov_hn = np.sum(processed_nov, axis=0)

    # Determine rain status per frame
    rain_status = nov_hn > rain_threshold

    return nov_hn, rain_status

def analyse_raw_audio(
    audio_file,
    ref_file = None,
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

    enable_nov_wind_detect = False
    enable_energy_peak_detect = False
    algo_state = {}

    n_frames = duration * Fs / frame_size
    # read audio data
    read_size = int(frame_size * n_frames)
    read_offset = int(Fs * offset)

    audio_data_in = read_audio_file(audio_file, read_size, read_offset)

    if (len(audio_data_in) < Fs):
        return 0, 0, 0

    
    # print (f"{read_offset = } {read_size = } {len(audio_data_in) = }")
    # audio_data = apply_bandpass_filter(audio_data_in, operating_freq_range, Fs)
    audio_data = bandpass_filter_sos(audio_data_in, Fs, operating_freq_range[0], operating_freq_range[1])
    
    enable_notch_filter = False
    notch_band = [350, 400]
    if enable_notch_filter:
        # Notch filter (350–400 Hz)
        notch_sos = design_bandstop_filter(notch_band[0], notch_band[1], Fs)
        audio_data = apply_filter(audio_data, notch_sos)
    
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
    
    algo_state = {}
    algo_state["audio_data"] = audio_data_in
    algo_state["duration"] = duration

    if enable_energy_peak_detect :
        time_window = 200  # msec
        block_length = 32
        peak_ratio_thr = 4
        max_db_drop = 15
        e_results, energy, energy_fs = analyze_energy_peaks(audio_data, Fs=Fs, freq_band= time_analysis_band,
                                                            block_length=block_length, tx_ms=time_window, 
                                                            peak_ratio_thr=peak_ratio_thr, max_db_drop=max_db_drop)
        # plot_energy_peaks_and_pulses(energy, Fs, block_length, e_results, title="Envelope with Detected Pulses")
        # print_peak_results_table (e_results)
        # tv_results = apply_time_offset_to_results(e_results, offset*1000)
        # algo_state ["pulse_data"] = tv_results
    
    if process_fp or process_fn : 
        # filtered = bandpass_filter_sos(audio_data_in, Fs, time_freq_band[0], time_freq_band[1])
        algo_state["filtered"] = audio_data
        t_results = calculate_pulse_characteristics(audio_data, stft0.shape[1], Fs, fft_length=N, hop_length=H, m = 30)
        algo_state = update_algo_state_from_results(algo_state, t_results)
        # print (f"{algo_state.keys()}")
        
    # if enable_plot_data == True:
    #     title = audio_file + " waveform"
    #     fig, ax1 = plot_data(max_rows, 1, sp_idx, 0, audio_data, duration, title)
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
    else:
        stft = stft0
        
    if log_compress_factor == 0:
        #with noise supression
        Y = abs(stft)
        # before noise supression
        Yp = abs(stft0)
    else:
        Y = np.log(1 + log_compress_factor * abs(stft))
        Yp = np.log(1 + log_compress_factor * abs(stft0))

    Sp_db0 = librosa.amplitude_to_db(Y, ref=np.max)  
    Sp_db = librosa.amplitude_to_db(Yp, ref=np.max)
    # print (f"{Sp_db0.shape =} , {Sp_db.shape =}")
    algo_state["spectrum_db0"] = Sp_db0
    algo_state["spectrum_db"] = Sp_db
    # print (f"{ type(Sp_db0) = }")
    # if enable_plot_spectrogram == True:

    #     sp_idx = sp_idx + 1
    #     img = librosa.display.specshow(
    #         Sp_db,
    #         sr=Fs,
    #         hop_length=H,
    #         cmap=colmap,
    #         x_axis="time",
    #         y_axis="linear",
    #         ax=ax1[sp_idx],
    #     )
        
    #     ax1[sp_idx].set(title=f"spectrogram: gamma ={log_compress_factor}")

    # if noise_factor != 0:
    #     if enable_plot_noise_supressed_signal == True:
    #         sp_idx = sp_idx + 1
    #         img = librosa.display.specshow(
    #             S_db0,
    #             sr=Fs,
    #             hop_length=H,
    #             cmap=colmap,
    #             x_axis="time",
    #             y_axis="linear",
    #             ax=ax1[sp_idx],
    #         )
    #         ax1[sp_idx].set(
    #             title=f"spectrogram noise supressed: gamma ={log_compress_factor}"
    #         )

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
    if enable_nov_wind_detect == True:
        algo_state = novelty_based_gust_detection(
            Y=Y,
            Fs=Fs,
            frame_length=256,
            hop_length=128,
            duration=10,
            nov=nov
        )

    # if enable_plot_freq_flux == 1:
    #     sp_idx = sp_idx + 1
    #     plot_data(max_rows, 1, sp_idx, ax1, nov[0], duration, "frequency flux")
    
    algo_state["Nov0"] = nov[0]
    algo_state["novt"] = novt
    algo_state["novk"] = novk
    # print (f"{ type(nov[0]) = }")

    # if enable_plot_freq_flux_a_d == 1:
    #     sp_idx = sp_idx + 1
    #     plot_data(max_rows, 1, sp_idx, ax1, novk, duration, "frequency flux a+d")
    
    # print (f"{ type(novk) = }")
        
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
 
            # if enable_plot_individual_harmonics:
            #     sp_idx = sp_idx + 1
            #     plot_data(
            #         max_rows, 1, sp_idx, ax1, nov[hn], duration, f"frequency flux {hn}"
            #     )
        algo_state ["nov"] = nov
        algo_state ["nov1"] = nov1
        # print (f"{ type(nov) = }")
    
    nov0 = nov[0]
    #print(f'{f_mean_harmonics = }')
    # if frequency flux is not above threshold for the base frequency then set the novelty for
    # all the harmonic frequencies to zero

    for k in range(len(nov0)):
        if nov0[k] == 0:
            for j in range(1, len(nov)):
                nov[j][k] = 0

    nov_hn = np.sum(nov, axis=0)
    nov_hn_copy = nov_hn.copy()
    # print(f" length of nov is {len(nov)}")

    # if enable_plot_frequency_harmonics:
    #     sp_idx = sp_idx + 1
    #     plot_data(
    #         max_rows,
    #         1,
    #         sp_idx,
    #         ax1,
    #         np.transpose(nov1[:]),
    #         duration,
    #         "frequency flux hn",
    #     )  # nov_hn #np.transpose(novt_hn[1:])

    # if 0:
    #     sp_idx = sp_idx + 1
    #     plot_data(max_rows, 1, sp_idx, ax1, peaks, duration, "freq peak found")
    if enable_correlation_mask:
        r_mask, peak_count = calculate_correlation_mask(
            audio_data, reference_audio, peaks, corr_thr
        )
        # if enable_plot_correlation_mask == 1:
        #     sp_idx = sp_idx + 1
        #     plot_data(max_rows, 1, sp_idx, ax1, r_mask, duration, "correlation mask")
        algo_state["corr_mask"] = r_mask
        # print (f"{ type(r_mask) = }")
    nov_hn[nov_hn > rain_thr_hn] = rain_thr_hn
    nov_hn[nov_hn < rain_thr_hn] = 0
    raining = nov_hn
    # raining = r_mask*nov[0][:-1] #*peak
    # raining[raining > 0] = 1;
    # if enable_plot_data == True:
    #     sp_idx = sp_idx + 1
    #     plot_data(max_rows, 1, sp_idx, ax1, raining, duration, "raining status")
    algo_state ["raining"] = raining
    # print (f"{ type(raining) = }")
    rain_drops = (raining >= 1).sum()
    # print (type(rain_drops))
    # if (rain_drops < 3):
    #     print(f"{rain_drops} less than 5 setting to zero")
    #     rain_drops = 0

    # plot_data (max_rows, 1, 4, ax1, peaks, duration)
    # nov_4, Fs_nov4 = compute_novelty_spectrum(X, Fs=11162, N=frame_length, H=hop_length, M=20, norm=True)
    # plot_data (max_rows, 1, 6, ax1, nov_4, duration)

    # if enable_plot_data == True:
    #     plt.tight_layout()
    #     for i in range(len(ax1)):
    #         ax1[i].grid()
    #     plt.show()
        # if audio: ipd.display(ipd.Audio(audio_data, rate=Fs))

    nov_hn_new, rain_status =  detect_rain_from_novelty(nov, rain_thr, rain_thr_hn)
    algo_state['rain_status_new'] = rain_status
    # rain_drops_new = (rain_status >= 1).sum()
    # plot_novelty_and_rain_status(nov_hn_copy, nov_hn_new, rain_status, raining, frame_duration_s=0.1)

    return rain_drops, frain_mean, algo_state
    


def rain_detection_algo (
    audio_data,
    **kwargs):
        
    configure_parameters (**kwargs)
    #print_configuration ()
    algo_state_all = {}
    rain_drop_count, frain_mean, offset, algo_state_all = analyse_raw_audio_wrapper(audio_data, kwargs, ref_file=None)
    
    return rain_drop_count, frain_mean, algo_state_all

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

    rain_drop_count, frain_mean,algo_state_all = rain_detection_algo (audio_signal, **kwargs)
    if rain_drop_count > 0:
        return True
    elif rain_drop_count == 0:
        return False
    else:
        return np.nan
        

MAX_DURATION_FW = 2

def analyse_raw_audio_in_parts (audio_file, ref_file, rain_drop_threshold, 
                                duration=2, offset=0, rain_drop_count=0):
    
    """
    This function makes it equivalent to C code. The raindrops are counted 
    for the entire check duration even when the threshold is exceed in the 
    first fw duration. 
    """
    raining = False
    results = []
    algo_state_parts = {}
    while duration > 0:
        rain_drop_count_part = 0.0
        frain_mean = 0.0
        part_duration = duration
        if part_duration > MAX_DURATION_FW:
            part_duration = MAX_DURATION_FW
        rain_drop_count_part, frain_mean, algo_state = analyse_raw_audio(
            audio_file, 
            ref_file, 
            part_duration, 
            offset)
        # print (f"{algo_state.keys() = }")
        algo_state_parts =  merge_algo_state(algo_state_parts, algo_state)
        duration -= part_duration
        offset += part_duration
        rain_drop_count += rain_drop_count_part
        if rain_drop_count > rain_drop_threshold:
            raining = True
    results.append ({"raining" : raining, 
                    "rain_drop_count" : rain_drop_count, 
                    "frain_mean" : frain_mean, 
                    "offset" :offset})
    return results, algo_state_parts
    
def combine_raining_status(params, rain_peaks_count, rain_drop_count, raining, rain_drop_threshold):
    """
    Adjusts raining status and modifies rain_drop_count_mod
    based on peak and drop count thresholds for FP and FN.

    Args:
        params (dict): Configuration parameters with thresholds and flags.
        rain_peaks_count (int): Count of detected rain peaks.
        rain_drop_count (int): Count of detected rain drops.
        raining (bool): Initial raining status.

    Returns:
        tuple: (raining (bool), rain_drop_count_mod (int))
    """
    process_fp = params["handle_fp"]
    process_fn = params["handle_fn"]
    rain_peaks_min_thr = params["rain_peaks_min_thr"]
    rain_peaks_max_thr = params["rain_peaks_max_thr"]
    rain_drop_max_thr = params["rain_drop_max_thr"]
    rain_drop_min_thr = params["rain_drop_min_thr"]

    rain_drop_count_mod = rain_drop_count

    if process_fn and not raining:
        if rain_drop_count > rain_drop_max_thr or rain_peaks_count > rain_peaks_max_thr:
            raining = True
            rain_drop_count_mod = max(rain_drop_count, rain_peaks_count)
    # print (f"{raining = }")
    if process_fp and raining:
        # print (f" process_fp {rain_peaks_count = }, {rain_peaks_min_thr = }, {rain_drop_count = }, {rain_drop_min_thr = }, {rain_drop_max_thr = }")
        if rain_peaks_count < rain_peaks_min_thr or rain_drop_count < rain_drop_threshold:
            if (True) or (rain_drop_count < rain_drop_max_thr):
                raining = False
                rain_drop_count_mod = 0
    # print (f"processed: {rain_drop_count = }, {rain_drop_count_mod = }")

    return raining, rain_drop_count_mod


def analyse_raw_audio_wrapper (audio_file, params, ref_file=None):
    offset = 0
    raining = False
    rain_drop_count = 0
    duration = rain_check_duration
    rain_drop_threshold = math.ceil(min_drop_count_per_second * duration)
    # print(f"{rain_drop_threshold = }")
    result_all = []
    algo_state_all = {}

    duration = rain_check_duration        
    results, algo_state_parts = analyse_raw_audio_in_parts (audio_file,
                                         ref_file, 
                                         rain_drop_threshold, 
                                         duration, 
                                         offset,
                                         rain_drop_count)
    
    algo_state_all =  merge_algo_state(algo_state_all, algo_state_parts)
    result_all += results
    offset = results[-1]["offset"]
    rain_drop_count = results[-1]["rain_drop_count"]
    if results [-1]["raining"] == True:
        raining = True
    
    results [-1]['rain_drop_count_mod'] = rain_drop_count
    # print (f"{len(algo_state_all['nov_hn_array']) = }")
    if (process_fp or process_fn):
        rain_peaks = time_domain_raining_status (algo_state_all, params)
        algo_state_all ['rain_peaks'] = rain_peaks
        rain_peaks_count = (rain_peaks > 0).sum()
        results [-1]['rain_peaks_count'] = rain_peaks_count 

        raining, rain_drop_count_mod = combine_raining_status(
            params, rain_peaks_count, rain_drop_count, raining, rain_drop_threshold
        )
        
        algo_state_all ['rain_drop_count'] = rain_drop_count
        algo_state_all ['rain_peaks_count'] = rain_peaks_count
        algo_state_all ['rain_drop_count_mod'] = rain_drop_count_mod
        results[-1]['rain_drop_count_mod'] = rain_drop_count_mod
    else :
        algo_state_all ['rain_drop_count_mod'] = rain_drop_count
        algo_state_all ['rain_drop_count'] = rain_drop_count
        algo_state_all ['rain_peaks_count'] = rain_drop_count
        
    if raining == False :
        rain_drop_count = 0
        results [-1]['rain_drop_count'] = rain_drop_count
        results [-1]['rain_drop_count_mod'] = rain_drop_count
    
    return results[-1]["rain_drop_count_mod"], results[-1]["frain_mean"], results[-1]["offset"], algo_state_all




# ## Utilities for Test Vector Batch Processing ##

# In[10]:


# Utilites to get test vectors and batch processing

    
    
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import gc
import types
def process_single_file(args):
    file_key, file_data, params, Fs, debug_params = args
    try:
        file_content = file_data["file_contents"]
        rain_actual = file_data["raining"]

        if len(file_content) % 2 != 0:
            file_content = file_content[:-1]

        required_length = params['sample_rate'] * params['check_duration']
        if len(file_content) < required_length:
            return None, None

        audio_signal = file_content
        algo_state_all = {}
        rain_drops, frain_mean, algo_state_all = rain_detection_algo(audio_signal, **params)
        algo_state_all['file_key'] = file_key

        rain_predicted = rain_drops > params['rain_drop_min_thr']
        rain_drop_count = algo_state_all['rain_drop_count']
        rain_peaks_count = algo_state_all['rain_peaks_count']
        rain_drop_count_mod = algo_state_all['rain_drop_count_mod']

        print(f"{file_key} actual = {rain_actual} predicted = {rain_predicted} rain_drops = {rain_drop_count} peaks = {rain_peaks_count} modified count = {rain_drop_count_mod}")
        mismatched = rain_predicted != rain_actual
        if (debug_params['print_mismatched'] and mismatched) or debug_params['debug_all']:
            state = "False Negative" if rain_actual else "False Positive"
            print(f"{file_key} {state} actual = {rain_actual} predicted = {rain_predicted} rain_drops = {rain_drop_count} peaks = {rain_peaks_count} modified count = {rain_drop_count_mod}")

        if (debug_params['audio_player'] and mismatched) or debug_params['debug_all']:
            ipd.display(ipd.Audio(algo_state_all['audio_data'], rate=Fs))

        if (debug_params['plot_status'] and mismatched) or debug_params['debug_all']:
            plot_algo_state_summary(algo_state_all, params)

        test_result = {
            "file_key": file_key,
            "rain_actual": rain_actual,
            "rain_predicted": rain_predicted,
            "rain_drop_count": rain_drop_count,
            "rain_peaks_count": rain_peaks_count,
            "rain_drop_count_mod": rain_drop_count_mod,
            "frain_mean": frain_mean,
        }

        if params['handle_fp'] or params['handle_fn']:
            feature = {
                "file_key": file_key,
                "rain_actual": rain_actual,
                "frain_mean": frain_mean,
                "kurtosis": algo_state_all["kurtosis"],
                "crest_factor": algo_state_all["crest_factor"],
                "diff_energy": algo_state_all["diff_energy"],
                "nov": algo_state_all["nov"],
            }
        else:
            feature = {
                "file_key": file_key,
                "rain_actual": rain_actual,
                "frain_mean": frain_mean,
                "nov": algo_state_all["nov"],
            }

        return test_result, feature

    except Exception as e:
        print(f"Error processing {file_key}: {e}")
        return None, None

def process_audio_batches(params, debug_params, InputType, test_vector_path=TEST_VECTOR_PATH, query=None, adse_engine=None, batch_size=1000, local_cache=LOCAL_AUDIO_CACHE, localStatus=True):
    keys = get_keys(InputType, test_vector_path, query, adse_engine, batch_size=batch_size, localStatus=localStatus)
    print(f"received {len(keys)} test vectors")

    Fs = params["sample_rate"]
    check_duration = params["check_duration"]
    header_size = 40
    bytes_per_sample = 2
    read_size = check_duration * Fs + header_size

    test_results_list = []
    feature_list = []

    for batch_count, batch_start in enumerate(range(0, len(keys), batch_size), start=1):
        batch_keys = keys[batch_start: batch_start + batch_size]
        print(f"Processing batch {batch_count}")

        dir_content = get_input_data(batch_keys, InputType, Fs, check_duration, localStatus, local_cache, read_size, bytes_per_sample, use_audio_tools=False)

        args_list = [
            (file_key, file_data, params, Fs, debug_params)
            for file_key, file_data in dir_content.items()
        ]

        with ProcessPoolExecutor() as executor:
            results = executor.map(process_single_file, args_list)

        for result in results:
            test_result, feature = result
            if test_result is not None and feature is not None:
                test_result["test_count"] = len(test_results_list)
                feature["test_count"] = len(feature_list)
                test_results_list.append(test_result)
                feature_list.append(feature)

        del dir_content
        gc.collect()

    test_results_df = pd.DataFrame(test_results_list)
    feature_df = pd.DataFrame(feature_list)
    return test_results_df, feature_df    

def process_audio_batches_old (
    params, debug_params, InputType, test_vector_path=TEST_VECTOR_PATH, query=None, adse_engine=None,
    batch_size=1000, local_cache = LOCAL_AUDIO_CACHE, localStatus = True):
    """
    Processes audio data in batches, performs analysis, and includes the "raining" status.

    Parameters:
    - InputType: If `LocalPath`, retrieves files from local directory; otherwise, queries the database.
    - test_vector_path (str or Path): Path to the local directory (used when `InputType = "LocalPath"`).
    - query (str): SQL query to retrieve data (used when `InputType = "RemotePath"`).
    - adse_engine (SQLAlchemy engine): Database engine (used when `InputType = "RemotePath").
    - batch_size (int): Number of audio files to process per batch (default = 100).

    Returns:
    - None (Processes and analyzes audio files in batches).
    """

    keys = get_keys(InputType, test_vector_path, query, adse_engine, batch_size=batch_size, localStatus = localStatus)
    # keys = keys[:25]
    print (f"received {len(keys)} test vectors")

    check_duration = params["check_duration"]
    Fs = params["sample_rate"]
 
    header_size =  40
    bytes_per_sample = 2

    read_size = check_duration * Fs + header_size
    # print_algo_parameters(params, file_params)
    
    test_count = 0
    results = []
    fp = {}
    fn = {}
    # test_results_all = {}
    test_results_list = []
    feature_list = []
    # batch_count = 0
    # Process keys in batches

    for batch_count, batch_start in enumerate(range(0, len(keys), batch_size), start=1):
        
        batch_keys = keys[batch_start: batch_start + batch_size]
        
        print(f"Processing batch {batch_count}")
        dir_content = get_input_data(batch_keys, InputType, Fs, check_duration, localStatus, local_cache, read_size, bytes_per_sample, use_audio_tools= False)

        for t_count, (file_key, file_data) in enumerate(dir_content.items()):
            audio_signal = []
            # try:
            file_content = file_data["file_contents"]
            rain_actual = file_data["raining"]

            # Parse the audio file and adjust file length if it is not even number of bytes
            if (len(file_content) % 2) != 0:
                file_content = file_content [:-1]
            required_length = params['sample_rate'] * params['check_duration']
            if (len (file_content) < required_length) :
                continue

            audio_signal = file_content
    
            algo_state_all = {}

            # for name, val in globals().items():
            #    if (
            #        not name.startswith("__") and
            #        not isinstance(val, type(__builtins__)) and
            #        not isinstance(val, types.FunctionType)
            #    ):
            #       print(f"{name}: {type(val)} -> {val}")
            # for obj in gc.get_objects():
            #   try:
            #       if isinstance(obj, dict) and "__name__" in obj:
            #           print(obj["__name__"])
            #   except ReferenceError:
            #       continue
            #   except Exception as e:
            #       print(f"Skipped object due to: {e}")
            # for obj in gc.get_objects():
            #    if isinstance(obj, dict) and "__name__" in obj:
            #        print(obj["__name__"])

            algo_state_all = {}

            rain_drops, frain_mean, algo_state_all = rain_detection_algo(
                audio_signal,
                **params)
            algo_state_all ['file_key'] = file_key

            rain_predicted = False
            
            if rain_drops > params['rain_drop_min_thr'] : rain_predicted = True

            rain_drop_count = algo_state_all ['rain_drop_count']
            rain_peaks_count = algo_state_all ['rain_peaks_count']
            rain_drop_count_mod = algo_state_all ['rain_drop_count_mod']
            nov = algo_state_all ['nov']
            file_name = Path(file_key).name 

            state = ""
            mismatched = False
            if (rain_predicted != rain_actual):
                mismatched = True
                state = "False Negative" if (rain_actual) else "False Positive"
            if ((print_mismatched and mismatched) or debug_all):
                print (f"{test_count} {file_key} {state} actual = {rain_actual} predicted = {rain_predicted} rain_drops = {rain_drop_count} peaks = {rain_peaks_count} modified count = {rain_drop_count_mod}")

            if (audio_player and mismatched) or (debug_all):
                ipd.display(ipd.Audio(algo_state_all['audio_data'], rate=Fs))
                
            if (plot_status and mismatched) or (debug_all):
                plot_algo_state_summary(algo_state_all, params)
                
            # Instead of updating a dictionary
            test_results_list.append({
                "test_count": test_count,
                "file_key": file_key,
                "rain_actual": rain_actual, 
                "rain_predicted": rain_predicted,
                "rain_drop_count": rain_drop_count,
                "rain_peaks_count": rain_peaks_count,
                "rain_drop_count_mod": rain_drop_count_mod,
                "frain_mean": frain_mean,
            })
            # print (algo_state_all.keys())
            if (params['handle_fp'] or params['handle_fn']):
                feature_list.append({
                    "test_count": test_count,
                    "file_key": file_key,
                    "rain_actual": rain_actual, 
                    "frain_mean": frain_mean,
                    "kurtosis": algo_state_all ["kurtosis"],
                    "crest_factor": algo_state_all["crest_factor"],
                    "diff_energy": algo_state_all ["diff_energy"],
                    "nov": algo_state_all ["nov"],
                })
            else:
                feature_list.append({
                    "test_count": test_count,
                    "file_key": file_key,
                    "rain_actual": rain_actual, 
                    "frain_mean": frain_mean,
                    "nov": algo_state_all ["nov"],
                })
            
            test_count += 1
            
            # except Exception as e:
            #     print(f"Error processing audio file {file_key}: {e}")
            
        del algo_state_all, dir_content
        gc.collect()

    # After processing all test vectors
    test_results_df = pd.DataFrame(test_results_list)
    feature_df = pd.DataFrame(feature_list)

    return test_results_df, feature_df     # test_results_all

#save results
import json

def save_results(fp, fn, output_path="results_fp_fn.json"):
    """
    Writes false positives and false negatives to a JSON file with structure:
    {
        "source_file": "file_path",
        "rain_actual": bool,
        "rain_predicted": bool 
    }

    Parameters:
        fp (dict): False positives, where predicted = True, actual = False
        fn (dict): False negatives, where predicted = False, actual = True
        output_path (str): Output file path for the JSON file
    """
    result = []

    for file_path, data in {**fp, **fn}.items():
        rain_actual = data["rain_actual"]
        rain_predicted = data["rain_predicted"]
        result.append({
            "source_file": file_path,
            "rain_actual": rain_actual,
            "rain_predicted": rain_predicted
        })

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results written to {output_path}")

import hashlib
import json

def hash_dict_sha256(input_dict):
    """
    Creates a SHA-256 hash of a dictionary by serializing it into a consistent JSON string.

    Parameters:
        input_dict (dict): The input dictionary to hash.

    Returns:
        str: SHA-256 hash as a hexadecimal string.
    """
    # Convert dictionary to a JSON string with sorted keys for consistency
    dict_str = json.dumps(input_dict, sort_keys=True)
    
    # Encode string and hash it
    hash_obj = hashlib.sha256(dict_str.encode('utf-8'))
    
    return hash_obj.hexdigest()

def write_dict_to_json_file(data_dict, file_path):
    """
    Writes a dictionary to a JSON file with the specified file name.

    Parameters:
        data_dict (dict): The dictionary to write.
        file_path (str): The path (including filename) to save the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"Dictionary written to {file_path}")

import csv

def write_dict_to_csv(data_dict, csv_file_path):
    """
    Writes a dictionary to a CSV file.
    
    Parameters:
    - data_dict: dict, where keys are column headers and values are lists of column data
    - csv_file_path: str or Path, the path to the output CSV file
    """
    if not data_dict:
        print("Empty dictionary provided.")
        return

    # Convert dictionary to list of rows
    keys = list(data_dict.keys())
    rows = list(zip(*[data_dict[key] for key in keys]))

    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)  # Write header
        writer.writerows(rows)  # Write data rows

    print(f"CSV written to {csv_file_path}")

def write_test_results_to_csv(data_dict, output_csv_path):
    """
    Writes test results stored in a dictionary to a CSV file.

    Parameters:
    - data_dict (dict): Dictionary of the form {
          file_key: {
              "test_count": ...,
              "rain_actual": ...,
              "rain_predicted": ...,
              ...
          }
      }
    - output_csv_path (str): Path to the output CSV file
    """
    if not data_dict:
        raise ValueError("Input dictionary is empty")

    # Get field names from one of the dictionary values
    first_entry = next(iter(data_dict.values()))
    fieldnames = ["file_key"] + list(first_entry.keys())

    # Write to CSV
    with open(output_csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file_key, metrics in data_dict.items():
            row = {"file_key": file_key}
            row.update(metrics)
            writer.writerow(row)

import pandas as pd
import pickle

def save_df (df, file_path):
    """
    Saves feature_df to a file using pickle to preserve ndarray and list structures.

    Parameters:
    - df (pd.DataFrame): The feature DataFrame to save.
    - file_path (str): The path where the file will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)

def load_df (file_path):
    """
    Loads the feature_df from a pickle file.

    Parameters:
    - file_path (str): Path to the pickle file.

    Returns:
    - pd.DataFrame: The loaded feature DataFrame.
    """
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    return df


# ## Test The Algorithm ##

# In[11]:


import pandas as pd
from pathlib import Path
import multiprocessing

params = {
    "sample_rate": 11162,
    "freq_resolution": 45,
    "time_resolution_ms": 10,
    "check_duration": 10,
    "op_freq_range": [400, 3500],
    "n_freq_range": [400, 700],
    "fn": 400,
    "num_harmonics": 6,
    "harmonic_threshold": [4.5, 4.0, 3.5, 3.5, 3.5, 3.5],
    "max_peaks": 3,
    "log_factor": 0,
    "ns_duration_ms": 470,
    "nf": 0,
    "min_drop_count": 0.3,
    "rain_drop_min_thr": 3,
    "rain_drop_max_thr": 50,
    "rain_peaks_min_thr": 9,
    "rain_peaks_max_thr": 30,
    "kurtosis_thr": 2.5,
    "crest_thr": 3.75,
    "diff_energy_thr": 6.5,
    "t_band": [400, 3500],
    "handle_fp": True,
    "handle_fn": True,
    "enable_nov_wind_dection": False,
    "enable_energy_peak_detection": False,
}

debug_params = {
    "enable_plot": True,
    "max_plots": 50,
    "enable_detailed_print": True,
    "audio": True,
    "plot_output": False,
    "print_mismatched": False,
    "debug_all": False,
    "audio_player": False,
    "plot_status": False,
}
query = f"""
SELECT *
FROM device_audio_rain_classification
"""
# N = 10000
# query = f"""
#     (
#         SELECT *
#         FROM device_audio_rain_classification
#         WHERE raining = True
#         ORDER BY start_time, device
#         LIMIT {N}
#     )
#     UNION ALL
#     (
#         SELECT *
#         FROM device_audio_rain_classification
#         WHERE raining = False
#         ORDER BY start_time, device
#         LIMIT {N}
#     );
# """

batch_size = 1000
localStatus = True
#InputType = "RemotePath"  # or "LocalPath" depending on your use case
print_mismatched = False

def main():
    print(f"audio cache path = {LOCAL_AUDIO_CACHE}")

    test_results_df, feature_df = process_audio_batches_old (
        params, debug_params, InputType, test_vector_path=TEST_VECTOR_PATH,
        query=query, adse_engine=None, batch_size=batch_size,
        local_cache=LOCAL_AUDIO_CACHE, localStatus=localStatus
    )

    results_fn = test_results_df[
        (test_results_df["rain_actual"] == True) &
        (test_results_df["rain_actual"] != test_results_df["rain_predicted"])
    ]

    results_fp = test_results_df[
        (test_results_df["rain_actual"] == False) &
        (test_results_df["rain_actual"] != test_results_df["rain_predicted"])
    ]

    print("...")
    print(f"\n{len(results_fn) = }, {len(results_fp) = }\n")
    print("processing complete")

    results_fn.to_csv("results_fn.csv", index=False)
    results_fp.to_csv("results_fp.csv", index=False)
    test_results_df.to_csv("test_results.csv", index=False)
    # feature_df.to_csv("feature_data.csv", index=False)

    save_df(feature_df, "feature_data.pkl")
    save_df(test_results_df, "test_results.pkl")

# Required for standalone execution with multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()


