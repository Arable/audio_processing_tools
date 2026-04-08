#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ver 3 code
# %%
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from audio_processing_tools.audio_processing_framework import process_audio_batches_v2
from audio_processing_tools.edge.band_noise_processor import BandNoiseEstimatorProcessor
from data_access_connectors.data_access_connectors.database import DatabaseInstances, get_engine

import os
print(os.getcwd())
# -------------------------
# Band-noise params (new estimator + separate detector)
# -------------------------
band_noise_params = {
    # --- framework / harness knobs ---
    "hop": 512,  # usually == frame_len
    "include_audio_in_state": True,
    "print_params_once": True,

    # --- BandNoiseEstimatorConfig overrides ---
    # NOTE: fs will be taken from params_global["sample_rate"] by _build_config()
    "frame_len": 512,

    "hp_cutoff_hz": 400.0,
    "hp_order": 4,

    "band_hz": (400.0, 700.0),
    "bpf_order": 4,

    "subframe_len": 128,
    "subhop": 128,

    "W": 25,
    "W_min": 20,
    "q": 0.3,
    "ema_alpha": 0.25,

    "beta": 1.0,
    "gain_floor": 0.10,
    "eps": 1e-12,

    "ne_attack_alpha_dry": 0.20,
    "ne_attack_alpha_wet": 0.02,
    "ne_release_alpha": 0.25,
    "smooth_N_E": True,

    # --- NoiseFrameDetectorConfig overrides (dotted keys) ---
    "det.n_fft": 512,  # will be overwritten to frame_len by your _build_config(), but harmless
    "det.M_db": 6.0,
    "det.N_db": 3.0,
    "det.primary_hz": (450.0, 650.0),
    "det.rain_bands_hz": (
        (450.0, 650.0),
        (800.0, 1050.0),
        (1500.0, 1800.0),
        (2350.0, 2550.0),
        (3150.0, 3350.0),
    ),
    "det.k_subframes": 2,
    "det.use_D_trigger": False,
    "det.D_db": 6.0,
    # NEW dB-onset detector knobs
    "det.band_rise_db": 8.0,
    "det.excess_rise_db": 4,
    "det.use_dE_over_Ehpf": False,
    "det.min_Ehpf": 1e-8,
    "det.min_Eband": 1e-10,
}

band_noise_params.pop("det.dE_over_Ehpf_thr", None)
# -------------------------
# One processor only
# -------------------------
band_proc = BandNoiseEstimatorProcessor(name="band_noise")  # mode ignored/compat

processors = [band_proc]

params_by_processor = {
    "band_noise": {**band_noise_params},
}

# -------------------------
# Global framework params
# -------------------------
params_global = {
    "sample_rate": 11162,
    "check_duration": 10.0,
}

DEBUG_PARAMS: Dict[str, Any] = {
    "print_mismatched": True,
    "debug_all": True,
    "enable_plot": True,
    "max_plots": 50,
    "enable_detailed_print": True,
    "audio": True,
    "plot_output": True,
}

# -------------------------
# Inputs
# -------------------------
INPUT_TYPE = "LocalPath" #"RemotePath" or "LocalPath"


#TEST_VECTOR_PATH = "/Users/santhoshp/mark3/rain/test_vector"
#TEST_VECTOR_PATH = "/Users/santhoshp/mark3/rain/test_vector"
TEST_VECTOR_PATH = "/Users/santhoshp/mark3/rain/audio_header_cache"
LOCAL_AUDIO_CACHE = "/Users/santhoshp/mark3/rain/audio_header_cache"
M3CLI_PATH = "/Users/santhoshp/repo/mark3-firmware-trunk/Utilities/M3cli/m3cli"
M3CLI_DIR  = str(Path(M3CLI_PATH).parent)

# =============================================================================
# >>>  TOGGLE THESE TO SELECT THE COMPARISON MODE  <<<
#
#   COMPARE_C_VS_DEVICE = True
#       Runs BOTH the PC C-library AND the Mark3 device (m3cli), then compares
#       the two against each other.
#       C-library output → NOS_BAND_C_<stem>.TXT   ("py" side in charts)
#       Device output    → NOS_BAND_<stem>.TXT      ("nb" side in charts)
#       (USE_LOCAL_LIB is ignored in this mode)
#
#   COMPARE_C_VS_DEVICE = False, USE_LOCAL_LIB = True
#       Compare Python reference vs PC C-library.
#
#   COMPARE_C_VS_DEVICE = False, USE_LOCAL_LIB = False
#       Compare Python reference vs Mark3 device (m3cli).
# =============================================================================
COMPARE_C_VS_DEVICE = True   # True → C-lib vs device;  False → Python vs one side
USE_LOCAL_LIB       = True    # (only used when COMPARE_C_VS_DEVICE = False)

ENABLE_COMBINED_PLOT = True   # combined scatter + confusion matrix (all files)
ENABLE_OTHER_PLOTS   = False  # per-file band-noise state plots

MAX_FILES = 10   # max number of files to process (None = all files)

N = 10
QUERY = f"""
    (
        SELECT *
        FROM device_audio_rain_classification
        WHERE raining = True
        ORDER BY start_time, device
        LIMIT {N}
    )
    UNION ALL
    (
        SELECT *
        FROM device_audio_rain_classification
        WHERE raining = False
        ORDER BY start_time, device
        LIMIT {N}
    );
"""

# Database Engine
adse_engine = get_engine(DatabaseInstances.ADSE)

if INPUT_TYPE == "LocalPath":
    if not TEST_VECTOR_PATH or not Path(TEST_VECTOR_PATH).exists():
        raise RuntimeError(f"TEST_VECTOR_PATH does not exist: {TEST_VECTOR_PATH!r}")



results_df, states_by_proc = process_audio_batches_v2(
    processors=processors,
    params_global=params_global,
    params_by_processor=params_by_processor,
    debug_params=DEBUG_PARAMS,
    InputType=INPUT_TYPE,
    test_vector_path=TEST_VECTOR_PATH if INPUT_TYPE == "LocalPath" else None,
    query=QUERY if INPUT_TYPE == "RemotePath" else None,
    adse_engine=adse_engine,
    batch_size=1000,
    local_cache=LOCAL_AUDIO_CACHE,
    localStatus=False,
)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig

def _design_hpf_sos(fs: float, cutoff_hz: float, order: int):
    if cutoff_hz <= 0:
        return None
    nyq = 0.5 * fs
    w = np.clip(cutoff_hz / nyq, 1e-6, 0.999)
    return spsig.butter(int(order), w, btype="highpass", output="sos")

def plot_band_noise_state(
    states_df,
    *,
    file_key: str,
    hpf_cutoff_hz: float = 400.0,
    hpf_order: int = 4,
    noise_amp_mode: str = "rms",   # "rms" or "std"
    figsize=(15, 10),
):
    """
    Plots:
      1) HPF(x_in) with +/- noise amplitude envelope
      2) E_band, N_E, N_E_raw (frame-level)
      3) Rain decision timeline (subframe OR frame-level)
    Assumptions:
      - states_df has one row per file_key with arrays stored in columns
      - cfg has fs, frame_len
    """
    hit = states_df[states_df["file_key"] == file_key]
    if hit.empty:
        sample = states_df["file_key"].head(5).tolist()
        raise KeyError("file_key not found. Example keys:\n" + "\n".join(sample))
    row = hit.iloc[0]

    x_in = np.asarray(row["x_in"], dtype=float)
    times_s = np.asarray(row["times_s"], dtype=float)
    E_band = np.asarray(row["E_band"], dtype=float)
    N_E = np.asarray(row["N_E"], dtype=float)
    N_E_raw = np.asarray(row["N_E_raw"], dtype=float)
    fft_rain_frame = np.asarray(row["fft_rain_frame"], dtype=bool)
    rain_submask = np.asarray(row["rain_submask"], dtype=bool)
    cfg = row["config"]

    fs = float(cfg.fs)
    N = int(cfg.frame_len)

    # truncate safely
    T = min(len(times_s), len(E_band), len(N_E), len(N_E_raw), len(fft_rain_frame))
    times_s = times_s[:T]
    E_band = E_band[:T]
    N_E = N_E[:T]
    N_E_raw = N_E_raw[:T]
    fft_rain_frame = fft_rain_frame[:T]

    # HPF audio
    sos = _design_hpf_sos(fs, hpf_cutoff_hz, hpf_order)
    x_hpf = spsig.sosfilt(sos, x_in) if sos is not None else x_in.copy()

    # time axis for audio
    t_audio = np.arange(x_hpf.size, dtype=float) / fs

    # ---- derive a per-frame noise amplitude from N_E (which is ENERGY in band)
    # If N_E is "band energy per frame", a rough amplitude scale is:
    #   A_rms ~= sqrt(N_E / N)
    # and then make a piecewise-constant envelope aligned to frames.
    n_frames_audio = x_hpf.size // N
    T_env = min(T, n_frames_audio)
    if T_env <= 0:
        raise ValueError("audio too short to form even one frame")

    N_E_env = N_E[:T_env]
    if noise_amp_mode.lower() == "std":
        amp = np.sqrt(np.maximum(N_E_env, 0.0) / max(N - 1, 1))
    else:  # "rms"
        amp = np.sqrt(np.maximum(N_E_env, 0.0) / max(N, 1))

    # Build sample-level envelope (step-hold per frame) and pad to full audio length
    amp_samples = np.repeat(amp, N)

    if amp_samples.size < x_hpf.size:
        # pad with last value so it matches x_hpf length
        pad = x_hpf.size - amp_samples.size
        amp_samples = np.pad(amp_samples, (0, pad), mode="edge")
    else:
        amp_samples = amp_samples[: x_hpf.size]

    # ---- rain decision signal
    # Prefer subframe mask if available; otherwise frame-level.
    rain_any_frame = None
    rain_t = None
    rain_sig = None

    if rain_submask.ndim == 2 and rain_submask.shape[0] > 0:
        Tm = min(rain_submask.shape[0], T)
        rain_any_frame = np.any(rain_submask[:Tm], axis=1)
        rain_t = times_s[:Tm]
        rain_sig = rain_any_frame.astype(float)
    else:
        rain_t = times_s
        rain_sig = fft_rain_frame.astype(float)

    # ---- plots
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=False, constrained_layout=True)

    # (1) HPF audio with +/- noise amplitude
    axs[0].plot(t_audio, x_hpf, linewidth=0.9, label="x_hpf")
    axs[0].plot(t_audio, +amp_samples, linewidth=1.0, label="+noise_amp")
    axs[0].plot(t_audio, -amp_samples, linewidth=1.0, label="-noise_amp")
    axs[0].set_title("HPF audio with ± noise amplitude envelope (derived from N_E)")
    axs[0].set_ylabel("amplitude")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    # (2) Energies
    axs[1].plot(times_s, E_band, label="E_band")
    axs[1].plot(times_s, N_E, label="N_E (smoothed/used)")
    axs[1].plot(times_s, N_E_raw, label="N_E_raw", alpha=0.8)
    axs[1].set_title("Frame energies: E_band, N_E, N_E_raw")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("energy (linear)")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    # (3) Rain decisions
    axs[2].plot(rain_t, rain_sig, label="rain (any subframe)" if (rain_any_frame is not None) else "fft_rain_frame")
    axs[2].set_title("Rain decision")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("decision (0/1)")
    axs[2].set_ylim(-0.1, 1.1)
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    plt.show()


# In[ ]:


states_df = states_by_proc["band_noise"]
cols = ["rain_actual", "band_noise__fft_rain_frac", "band_noise__n_frames"]

df_small = results_df.set_index("file_key")[cols]
for i in range(len(states_df)):
    fk = states_df["file_key"].iloc[i]
    row = df_small.loc[fk]
    print("=" * 80)
    print("file_key:", fk)
    print("rain_actual:", row["rain_actual"])
    print("fft_rain_frac:", row["band_noise__fft_rain_frac"])
    print("n_frames:", row["band_noise__n_frames"])
    if ENABLE_OTHER_PLOTS:
        #plot_band_noise_states_aligned(states_df, file_key=fk)
        plot_band_noise_state(
            states_df,
            file_key=fk,
            hpf_cutoff_hz=400.0,
            hpf_order=4,
            noise_amp_mode="rms",
            figsize=(15, 10),
        )


# In[ ]:


import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt

def _design_hpf_sos(fs: float, cutoff_hz: float, order: int):
    if cutoff_hz <= 0:
        return None
    nyq = 0.5 * fs
    w = np.clip(cutoff_hz / nyq, 1e-6, 0.999)
    return spsig.butter(int(order), w, btype="highpass", output="sos")

def _design_bpf_sos(fs: float, band_hz, order: int):
    lo, hi = band_hz
    nyq = 0.5 * fs
    w1 = np.clip(lo / nyq, 1e-6, 0.999)
    w2 = np.clip(hi / nyq, 1e-6, 0.999)
    if w2 <= w1:
        w2 = min(0.999, w1 + 1e-3)
    return spsig.butter(int(order), [w1, w2], btype="bandpass", output="sos")

def _frame_view(sig: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if sig.size < frame_len:
        return np.empty((0, frame_len), dtype=np.float64)
    Tn = 1 + (sig.size - frame_len) // hop
    stride = sig.strides[0]
    return np.lib.stride_tricks.as_strided(
        sig, shape=(Tn, frame_len), strides=(hop * stride, stride), writeable=False
    )

def plot_band_noise_states_aligned(
    states_df,
    *,
    file_key: str,
    hpf_cutoff_hz: float = 400.0,
    hpf_order: int = 4,
    figsize=(15, 11),
):
    hit = states_df[states_df["file_key"] == file_key]
    if hit.empty:
        sample = states_df["file_key"].head(5).tolist()
        raise KeyError("file_key not found. Example keys:\n" + "\n".join(sample))
    row = hit.iloc[0]

    x_in = np.asarray(row["x_in"], dtype=float)
    times_s = np.asarray(row["times_s"], dtype=float)
    E_band = np.asarray(row["E_band"], dtype=float)
    N_E = np.asarray(row["N_E"], dtype=float)
    fft_rain_frame = np.asarray(row["fft_rain_frame"], dtype=bool)
    rain_submask = np.asarray(row["rain_submask"], dtype=bool)
    cfg = row["config"]

    fs = float(cfg.fs)
    N = int(cfg.frame_len)
    sub_len = int(cfg.subframe_len)
    subhop = int(cfg.subhop)
    S = int(N // sub_len)

    # ---- DEFINE T EARLY (before any use)
    T = min(len(times_s), len(E_band), len(N_E), len(fft_rain_frame))
    times_s = times_s[:T]
    E_band = E_band[:T]
    N_E = N_E[:T]
    fft_rain_frame = fft_rain_frame[:T]

    # audio time axis
    t_audio = np.arange(x_in.size, dtype=float) / fs
    dur_s = float(t_audio[-1]) if t_audio.size else 0.0

    # HPF audio
    sos_hpf = _design_hpf_sos(fs, hpf_cutoff_hz, hpf_order)
    x_hpf = spsig.sosfilt(sos_hpf, x_in) if sos_hpf is not None else x_in.copy()

    # BPF audio (match estimator band)
    sos_bp = _design_bpf_sos(fs, cfg.band_hz, int(cfg.bpf_order))
    x_bp = spsig.sosfilt(sos_bp, x_hpf) if sos_bp is not None else x_hpf.copy()

    # energies per FFT frame from time-domain signals
    n_frames_audio = x_hpf.size // N
    T_energy = min(T, n_frames_audio)

    E_hpf = np.zeros(T_energy, dtype=float)
    E_bp = np.zeros(T_energy, dtype=float)
    for i in range(T_energy):
        fr_hpf = x_hpf[i * N : (i + 1) * N]
        fr_bp  = x_bp[i * N : (i + 1) * N]
        E_hpf[i] = float(np.sum(fr_hpf * fr_hpf))
        E_bp[i]  = float(np.sum(fr_bp * fr_bp))

    times_energy = times_s[:T_energy]

    # subframe timeline from stored rain_submask
    Tm, Sm = rain_submask.shape
    S = int(Sm)

    t0 = times_s[:Tm]                              # frame start times
    offsets = (np.arange(S) * (subhop / fs))       # offsets within frame
    t_sub = (t0[:, None] + offsets[None, :]).reshape(-1)

    rain_any_frame = np.any(rain_submask, axis=1)                 # (Tm,)
    rain_any_sub = np.repeat(rain_any_frame.astype(float), S)     # (Tm*S,)
    fft_up = np.repeat(fft_rain_frame[:Tm].astype(float), S)      # (Tm*S,)

    # ---- Plot (4 subplots)
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # 1) HPF audio
    axs[0].plot(t_audio, x_hpf)
    axs[0].set_title(f"HPF audio (cutoff={hpf_cutoff_hz} Hz, order={hpf_order})")
    axs[0].set_ylabel("amplitude")
    axs[0].grid(True)

    # 2) decisions
    axs[1].plot(t_sub, rain_any_sub, label="time mask (any lane)", linewidth=1.0)
    axs[1].plot(t_sub, fft_up, label="FFT rain frame (upsampled)", linewidth=1.0, alpha=0.8)
    axs[1].set_title("Rain decisions (subframe timeline)")
    axs[1].set_ylabel("decision (0/1)")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    # 3) E_band and N_E
    axs[2].plot(times_s, E_band, label="E_band")
    axs[2].plot(times_s, N_E, label="N_E")
    axs[2].set_title("E_band and N_E (frame-level)")
    axs[2].set_ylabel("energy (linear)")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    # 4) energy comparison
    axs[3].plot(times_energy, E_band[:T_energy], label="E_band (FFT-domain)")
    axs[3].plot(times_energy, E_bp, label="E_bp (time-domain BPF)")
    axs[3].plot(times_energy, E_hpf, label="E_hpf (time-domain HPF)", alpha=0.7)
    axs[3].set_title("Energy comparison: E_band vs E_bp vs E_hpf")
    axs[3].set_ylabel("energy (linear)")
    axs[3].set_xlabel("time (s)")
    axs[3].grid(True)
    axs[3].legend(loc="upper right")

    axs[3].set_xlim(0, dur_s)
    plt.show()


# In[ ]:


if ENABLE_OTHER_PLOTS:
    states_df = states_by_proc["band_noise"]
    for i in range(len(states_df)):
        fk = states_df["file_key"].iloc[i]
        print(fk)
        plot_band_noise_states_aligned(states_df, file_key=fk)


# In[ ]:


import pprint as pp
pd.set_option("display.max_colwidth", None)   # show full string
pd.set_option("display.width", None)          # no line wrapping/truncation


#display (results_df[['file_key', 'rain_actual', 'band_noise__fft_rain_frac', 'band_noise__n_frames']])
#print (results_df.columns)

pd.reset_option("display.max_colwidth")
pd.reset_option("display.width")


# In[ ]:




# =============================================================================
# CM4 E_band / M_band dump — Python equivalent of band_noise_process_nos_aud_file()
#
# The C code (disdrometer.c) does:
#   1. Skip 40-byte audio_header_t
#   2. Read NOS_AUD_FRAME_BYTES (1024) at a time as int16 PCM
#   3. Convert to float: sample / 32768.0f
#   4. Call band_noise_dsd_process_frame  (512 samples)
#   5. Write "frame_idx E_band M_band\n" to nos_band_out.txt
#
# This function reproduces step 2-5 in Python using BandNoiseEstimator with
# identical parameters to the CM4 defaults in band_noise_dsd.c.
# The output file can be diff'd directly against the device's nos_band_out.txt.
# =============================================================================

import sys
import os
import struct as _struct
import numpy as np
from pathlib import Path

# Band-noise estimator (same module used for CM4 comparison throughout this repo)
sys.path.insert(0, str(Path(__file__).parent.parent / "mark3-firmware-trunk"))
from audio_processing_tools.edge.band_noise_estimator import BandNoiseEstimator, BandNoiseEstimatorConfig

# Audio parser from this repo — handles PCM (version=0) and ALAC (version>=1)
from audio_processing_tools.parse import parse_mark_audio_file

# audio_header_t layout — must match CM4/Inc/audio.h
_HDR_FMT  = "<III4Bfff10s2x"   # 40 bytes total
_HDR_SIZE = _struct.calcsize(_HDR_FMT)
_MAGIC    = 0xDECAFBAD

# CM4 constants (disdrometer.c)
_FRAME_SAMPLES = 512   # NOS_AUD_FRAME_SAMPLES
_FRAME_BYTES   = _FRAME_SAMPLES * 2  # NOS_AUD_FRAME_BYTES  (int16)


def dump_eband_mband(audio_path: str,
                     output_path: str = "nos_band_out.txt",
                     *,
                     fs: int = 11162) -> int:
    """
    Python equivalent of band_noise_process_nos_aud_file() on CM4.

    Reads a Mark3 audio file (raw PCM or ALAC), runs BandNoiseEstimator
    frame-by-frame (512 samples, no overlap) and writes:

        <frame_idx> <E_band:.10e> <M_band:.10e>

    one line per frame — identical format to the device's nos_band_out.txt.

    Parameters
    ----------
    audio_path  : path to the Mark3 .bin audio file
    output_path : where to write the dump (default: nos_band_out.txt)
    fs          : sample rate override (default 11162; read from header if present)

    Returns
    -------
    int : number of frames processed
    """
    # ------------------------------------------------------------------
    # 1. Load file and decode (PCM or ALAC) via parse_mark_audio_file
    # ------------------------------------------------------------------
    with open(audio_path, "rb") as f:
        file_bytes = f.read()

    # Parse version / sample_rate from header if magic is present
    if (len(file_bytes) >= _HDR_SIZE and
            _struct.unpack_from("<I", file_bytes, 0)[0] == _MAGIC):
        hdr_fields = _struct.unpack_from(_HDR_FMT, file_bytes, 0)
        _, _, sample_rate, _, _, _, version = hdr_fields[:7]
        if sample_rate:
            fs = sample_rate
        fmt_str = "ALAC" if version >= 1 else "PCM"
    else:
        fmt_str = "PCM (no header)"

    pcm_int16, metadata = parse_mark_audio_file(file_bytes)
    if metadata.get("sample_rate"):
        fs = int(metadata["sample_rate"])

    print(f"dump_eband_mband: {os.path.basename(audio_path)}")
    print(f"  format={fmt_str}  sample_rate={fs}  samples={len(pcm_int16)}")

    # ------------------------------------------------------------------
    # 2. Normalise to float  (mirrors: (float)s16[i] / 32768.0f  in C)
    # ------------------------------------------------------------------
    x = pcm_int16.astype(np.float32) / np.float32(32768.0)

    # ------------------------------------------------------------------
    # 3. Build BandNoiseEstimator with CM4-matching defaults
    #    (band_noise_dsd.c: HPF 350 Hz, BPF 400-700 Hz, W=30, q=0.3)
    # ------------------------------------------------------------------
    cfg = BandNoiseEstimatorConfig()
    cfg.fs          = fs
    cfg.frame_len   = _FRAME_SAMPLES
    cfg.hp_cutoff_hz = 350.0
    cfg.hp_order    = 4
    cfg.band_hz     = (400.0, 700.0)
    cfg.bpf_order   = 4
    cfg.subframe_len = 128
    cfg.subhop      = 128
    cfg.W           = 30
    cfg.W_min       = 10
    cfg.q           = 0.3
    cfg.ema_alpha   = 1.0   # no extra EMA smoothing (matches C)
    cfg.smooth_N_E  = False
    cfg.validate()

    est = BandNoiseEstimator(cfg)
    est.reset()

    # ------------------------------------------------------------------
    # 4. Process frame-by-frame and write output
    #    Mirrors the while-loop in band_noise_process_nos_aud_file()
    # ------------------------------------------------------------------
    n_frames = len(x) // _FRAME_SAMPLES
    print(f"  frames={n_frames}  writing → {output_path}")

    with open(output_path, "w") as fout:
        for i in range(n_frames):
            frame = x[i * _FRAME_SAMPLES : (i + 1) * _FRAME_SAMPLES]
            out   = est.process_frame(frame)
            # Format matches C: snprintf(line, ..., "%u %.10e %.10e %.10e\n", ...)
            fout.write(f"{i} {out.E_band:.10e} {out.M_band:.10e} {out.M_clean:.10e}\n")

    print(f"  done — {n_frames} frames written.")
    return n_frames


# =============================================================================
# CODE PATH A – C shared library via ctypes  (USE_LOCAL_LIB = True)
# =============================================================================
#
# Build the library first:
#   cd /Users/santhoshp/repo/mark3-firmware-trunk/Utilities/band_noise_pylib
#   cmake -B build -DCMSISDSP_LIB=/path/to/libCMSISDSP.a
#   cmake --build build
#
# Then either:
#   • pass --lib /path/to/libband_noise_pylib.dylib on the command line, or
#   • set the BAND_NOISE_LIB environment variable.

import ctypes as _ctypes

# Default search locations for the shared library
_BAND_NOISE_LIB_CANDIDATES = [
    str(Path(__file__).parent.parent /
        "mark3-firmware-trunk/Utilities/band_noise_pylib/build/libband_noise_pylib.dylib"),
    str(Path(__file__).parent.parent /
        "mark3-firmware-trunk/Utilities/band_noise_pylib/build/libband_noise_pylib.so"),
]


def _load_c_lib(lib_path=None):
    """Load and configure libband_noise_pylib; return ctypes handle."""
    if lib_path is None:
        lib_path = os.environ.get("BAND_NOISE_LIB")
    if lib_path is None:
        for c in _BAND_NOISE_LIB_CANDIDATES:
            if os.path.exists(c):
                lib_path = c
                break
    if lib_path is None:
        raise FileNotFoundError(
            "Cannot find libband_noise_pylib.\n"
            "Pass --lib /path/to/libband_noise_pylib.dylib "
            "or set BAND_NOISE_LIB environment variable."
        )
    lib = _ctypes.CDLL(lib_path)
    # int band_noise_process_nos_aud_file(const char*, const char*)
    lib.band_noise_process_nos_aud_file.restype  = _ctypes.c_int
    lib.band_noise_process_nos_aud_file.argtypes = [_ctypes.c_char_p, _ctypes.c_char_p]
    # void bn_reset()
    lib.bn_reset.restype  = None
    lib.bn_reset.argtypes = []
    print(f"Loaded C library: {lib_path}")
    return lib


def run_c_lib_nos_aud(audio_path: str,
                      output_path: str,
                      lib_path: str = None) -> str:
    """
    Run band_noise_process_nos_aud_file() from the C shared library.

    Reads a Mark3 raw-PCM nos_aud.bin (40-byte header + int16 frames),
    runs the CM4 algorithm, and writes per-frame results to output_path:
        <frame_idx> <E_band> <M_band> <M_clean>

    Parameters
    ----------
    audio_path  : path to the Mark3 .bin audio file
    output_path : where to write the output text file
    lib_path    : path to libband_noise_pylib.dylib/.so (None = auto-detect)

    Returns
    -------
    str : output_path
    """
    lib = _load_c_lib(lib_path)
    print(f"\nrun_c_lib_nos_aud: {audio_path}")
    print(f"  output: {output_path}")
    rc = lib.band_noise_process_nos_aud_file(
        audio_path.encode(), output_path.encode()
    )
    if rc != 0:
        raise RuntimeError(
            f"band_noise_process_nos_aud_file() returned {rc} for {audio_path!r}"
        )
    return output_path


# =============================================================================
# CODE PATH B – Mark3 device via m3cli subprocess  (USE_LOCAL_LIB = False)
# =============================================================================

# m3cli binary and working directory — set via macros at the top of this file
_M3CLI_DEFAULT = M3CLI_PATH
_M3CLI_DIR     = M3CLI_DIR


def run_m3cli_nos_aud(audio_path: str,
                      m3cli_path: str = _M3CLI_DEFAULT,
                      nb_out_path: Optional[str] = None,
                      timeout: int = 300) -> str:
    """
    Run ``./m3cli 'nos_aud <audio_path>'``, wait for completion, and return
    the local path to the downloaded NOS_BAND.TXT.

    Parameters
    ----------
    audio_path  : absolute path to the Mark3 .bin audio file
    m3cli_path  : path to the m3cli binary (default: sibling repo location)
    nb_out_path : where to copy NOS_BAND.TXT after download; if None the file is
                  left in the m3cli working directory and its path is returned
    timeout     : seconds to wait for m3cli to finish (default 120 s)

    Returns
    -------
    str : path to the NOS_BAND.TXT file produced by the device
    """
    import subprocess

    m3cli_dir  = str(Path(m3cli_path).parent)
    nb_default = os.path.join(m3cli_dir, "NOS_BAND.TXT")

    cmd = [m3cli_path, f"nos_aud {audio_path}", "quit"]
    print(f"\nrun_m3cli_nos_aud: {' '.join(repr(c) for c in cmd)}")
    print(f"  working dir : {m3cli_dir}")

    # Remove any stale NOS_BAND.TXT so we can tell if this run produced a fresh one.
    if os.path.isfile(nb_default):
        os.remove(nb_default)

    result = subprocess.run(
        cmd,
        cwd=m3cli_dir,
        timeout=timeout,
        capture_output=True,    # capture so we can inspect output on failure
        text=True,
    )

    # Always echo captured output so the run is visible in the test log.
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"m3cli exited with code {result.returncode} "
            f"while processing {audio_path!r}"
        )

    if not os.path.isfile(nb_default):
        raise FileNotFoundError(
            f"NOS_BAND.TXT not found at {nb_default!r} after m3cli run"
        )

    if nb_out_path is not None and nb_out_path != nb_default:
        import shutil
        shutil.copy2(nb_default, nb_out_path)
        print(f"  NOS_BAND.TXT copied → {nb_out_path}")
        return nb_out_path

    print(f"  NOS_BAND.TXT  : {nb_default}")
    return nb_default


def _load_m_clean(path: str) -> np.ndarray:
    """
    Load the M_clean column (last, 4th column) from a 4-column output file.
    Format per line:  <frame_idx> <E_band> <M_band> <M_clean>
    """
    data = np.loadtxt(path, usecols=(0, 3))   # frame_idx, M_clean
    return data   # shape (N, 2)


def compare_m_clean(py_out_path: str,
                    nb_out_path: str,
                    *,
                    abs_tol: float = 1e-3) -> bool:
    """
    Compare the M_clean column between the Python output and the device
    (m3cli / NOS_BAND.TXT) output.

    A frame PASSES if |py - nb| < abs_tol (default 0.001).

    Prints statistics, a per-frame diff table for failing frames, a scatter
    plot of py vs nb M_clean, and a confusion matrix (PASS/FAIL per frame).

    Parameters
    ----------
    py_out_path : path to Python-generated output (4-column format)
    nb_out_path : path to device NOS_BAND.TXT (4-column format)
    abs_tol     : absolute tolerance — diff < abs_tol is a pass (default 1e-3)

    Returns
    -------
    bool : True if every frame is within tolerance
    """
    import matplotlib.pyplot as plt

    py_data = _load_m_clean(py_out_path)   # (N, 2): col0=frame, col1=M_clean
    nb_data = _load_m_clean(nb_out_path)

    py_frames = py_data[:, 0].astype(int)
    nb_frames = nb_data[:, 0].astype(int)

    # Deduplicate frame indices (keep first occurrence) so that isin-based
    # alignment always yields arrays of equal length.
    _, py_first = np.unique(py_frames, return_index=True)
    _, nb_first = np.unique(nb_frames, return_index=True)
    py_data = py_data[np.sort(py_first)]
    nb_data = nb_data[np.sort(nb_first)]
    py_frames = py_data[:, 0].astype(int)
    nb_frames = nb_data[:, 0].astype(int)

    # --- align on frame index (intersection) ---
    common = np.intersect1d(py_frames, nb_frames)
    if len(common) == 0:
        print("compare_m_clean: no common frame indices — cannot compare")
        return False, np.array([]), np.array([])

    py_mc = py_data[np.isin(py_frames, common), 1]
    nb_mc = nb_data[np.isin(nb_frames, common), 1]
    n = len(common)

    diff     = py_mc - nb_mc          # signed error (py − nb)
    abs_diff = np.abs(diff)
    within   = abs_diff < abs_tol
    n_pass   = int(np.sum(within))
    n_fail   = n - n_pass

    # --- statistics ---
    mae      = float(np.mean(abs_diff))
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    max_diff = float(np.max(abs_diff))
    avg_diff = float(np.mean(diff))                    # mean signed difference
    std_diff = float(np.std(diff))                     # std dev of signed error
    mean_err = avg_diff                                # bias
    max_dev  = float(np.max(np.abs(diff - avg_diff))) # max deviation from bias
    p90      = float(np.percentile(abs_diff, 90))
    p95      = float(np.percentile(abs_diff, 95))
    p99      = float(np.percentile(abs_diff, 99))

    print()
    print("=" * 72)
    print("  M_clean comparison:  Python  vs  Device (NOS_BAND.TXT)")
    print(f"  frames compared    : {n}")
    print(f"  MAE                : {mae:.6e}   (mean absolute error)")
    print(f"  RMSE               : {rmse:.6e}   (how close overall)")
    print(f"  Mean error (bias)  : {mean_err:.6e}   (signed: +ve = py > nb)")
    print(f"  Max |difference|   : {max_diff:.6e}   (worst case)")
    print(f"  Avg difference     : {avg_diff:.6e}")
    print(f"  Std deviation      : {std_diff:.6e}")
    print(f"  Max deviation      : {max_dev:.6e}   (max spread around bias)")
    print(f"  90th percentile    : {p90:.6e}")
    print(f"  95th percentile    : {p95:.6e}   (worst 5 % threshold)")
    print(f"  99th percentile    : {p99:.6e}   (worst 1 % threshold)")
    print(f"  tolerance          : |diff| < {abs_tol:.0e}")
    print(f"  frames PASS        : {n_pass} / {n}")
    print(f"  frames FAIL        : {n_fail}")
    print("=" * 72)

    if n_fail > 0:
        print(f"\n  {'frame':>6}  {'py_M_clean':>16}  {'nb_M_clean':>16}  {'abs_diff':>14}")
        print("  " + "-" * 58)
        for idx in np.where(~within)[0]:
            fi = int(common[idx])
            print(f"  {fi:>6}  {py_mc[idx]:>16.10e}  {nb_mc[idx]:>16.10e}"
                  f"  {abs_diff[idx]:>14.6e}")

    result_str = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  Overall result: {result_str}")
    print("=" * 72)

    return n_fail == 0, py_mc, nb_mc


def plot_combined_charts(all_py_mc: np.ndarray,
                         all_nb_mc: np.ndarray,
                         abs_tol: float = 1e-3,
                         title_prefix: str = "") -> None:
    """
    Plot scatter chart and confusion matrix from combined M_clean arrays
    accumulated across all processed files.

    Parameters
    ----------
    all_py_mc    : concatenated Python M_clean values (all files)
    all_nb_mc    : concatenated Device M_clean values (all files)
    abs_tol      : same tolerance used in compare_m_clean
    title_prefix : optional label prepended to figure title (e.g. file count)
    """
    import matplotlib.pyplot as plt

    if len(all_py_mc) != len(all_nb_mc):
        raise ValueError(
            f"plot_combined_charts: array size mismatch — "
            f"all_py_mc has {len(all_py_mc)} elements but "
            f"all_nb_mc has {len(all_nb_mc)} elements"
        )

    all_py_mc = np.asarray(all_py_mc)
    all_nb_mc = np.asarray(all_nb_mc)

    diff     = all_py_mc - all_nb_mc
    abs_diff = np.abs(diff)
    within   = np.asarray(abs_diff < abs_tol)
    n        = len(all_py_mc)
    n_pass   = int(np.sum(within))
    n_fail   = n - n_pass

    pass_idx = np.where(within)[0]
    fail_idx = np.where(~within)[0]

    mae      = float(np.mean(abs_diff))
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    max_diff = float(np.max(abs_diff))
    avg_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    mean_err = avg_diff
    max_dev  = float(np.max(np.abs(diff - avg_diff)))
    p90      = float(np.percentile(abs_diff, 90))
    p95      = float(np.percentile(abs_diff, 95))
    p99      = float(np.percentile(abs_diff, 99))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(
        f"{prefix}M_clean: Python vs Device  ({n} frames total)\n"
        f"MAE={mae:.3e}  RMSE={rmse:.3e}  Bias={mean_err:.3e}  "
        f"MaxDiff={max_diff:.3e}  Std={std_diff:.3e}  MaxDev={max_dev:.3e}  "
        f"P90={p90:.3e}  P95={p95:.3e}  P99={p99:.3e}",
        fontsize=8,
    )

    # ------------------------------------------------------------------ #
    #  Scatter plot                                                        #
    # ------------------------------------------------------------------ #
    ax = axes[0]
    ax.scatter(all_nb_mc[pass_idx], all_py_mc[pass_idx],
               s=8, alpha=0.5, color="steelblue", label=f"PASS ({n_pass})")
    if n_fail > 0:
        ax.scatter(all_nb_mc[fail_idx], all_py_mc[fail_idx],
                   s=15, alpha=0.85, color="crimson", label=f"FAIL ({n_fail})")
    lo = min(all_nb_mc.min(), all_py_mc.min())
    hi = max(all_nb_mc.max(), all_py_mc.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="ideal (y=x)")
    ax.set_xlabel("Device M_clean (nb)")
    ax.set_ylabel("Python M_clean (py)")
    ax.set_title("Scatter: py vs nb M_clean (all files combined)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # ------------------------------------------------------------------ #
    #  Confusion matrix (above/below combined median)                     #
    # ------------------------------------------------------------------ #
    threshold = np.median(np.concatenate([all_py_mc, all_nb_mc]))
    py_hi = (all_py_mc >= threshold).astype(int)
    nb_hi = (all_nb_mc >= threshold).astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for p, d in zip(py_hi, nb_hi):
        cm[d, p] += 1   # rows = device, cols = python

    ax2 = axes[1]
    im = ax2.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    labels = ["Below median", "Above median"]
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Python M_clean")
    ax2.set_ylabel("Device M_clean")
    ax2.set_title(f"Confusion Matrix (threshold={threshold:.3e}, all files combined)")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------------------------
# Run test for all files under TEST_VECTOR_PATH directory
# $ python test_cm4_band_noise_processor.py
#
# ---------------------------------------------------------------------------
# Run for single file
# $ python test_cm4_band_noise_processor.py /home/santhosh/Mark3/Doc/Rain/noise_cancel/test_vector/raw_audio/D009920/2025/04/05/20250405_07_40_00_000000_rain_09b.bin 
#
# ---------------------------------------------------------------------------
# CLI:  python test_cm4_band_noise_processor.py <audio.bin> [py_out.txt]
#       Add --skip-m3cli to only run the Python side (no device needed).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse as _argparse

    if COMPARE_C_VS_DEVICE:
        _mode_desc = "C-library vs device (m3cli)"
    elif USE_LOCAL_LIB:
        _mode_desc = "Python vs C-library"
    else:
        _mode_desc = "Python vs device (m3cli)"

    _ap = _argparse.ArgumentParser(
        description=(
            f"Compare M_clean outputs — active mode: {_mode_desc}\n"
            f"  COMPARE_C_VS_DEVICE={COMPARE_C_VS_DEVICE}  USE_LOCAL_LIB={USE_LOCAL_LIB}\n"
            "With no positional argument: processes every .bin file under TEST_VECTOR_PATH.\n"
            "With a positional argument: processes that single file only."
        )
    )
    _ap.add_argument("audio", nargs="?", default=None,
                     help="Mark3 audio .bin file to process (omit to run all files in TEST_VECTOR_PATH)")
    _ap.add_argument("--abs-tol", type=float, default=1e-3,
                     help="Absolute tolerance for M_clean comparison (default: 1e-3)")
    _ap.add_argument("--limit", type=int, default=MAX_FILES,
                     help=f"Max number of files to process (default: MAX_FILES={MAX_FILES})")

    # C-library args — needed for COMPARE_C_VS_DEVICE or USE_LOCAL_LIB
    if COMPARE_C_VS_DEVICE or USE_LOCAL_LIB:
        _ap.add_argument("--lib", default=None,
                         help="Path to libband_noise_pylib.dylib/.so (default: auto-detect)")

    # m3cli args — needed for COMPARE_C_VS_DEVICE or not USE_LOCAL_LIB
    if COMPARE_C_VS_DEVICE or not USE_LOCAL_LIB:
        _ap.add_argument("--m3cli", default=_M3CLI_DEFAULT,
                         help=f"Path to m3cli binary (default: {_M3CLI_DEFAULT})")
        _ap.add_argument("--skip-m3cli", action="store_true",
                         help="Skip m3cli call; use cached NOS_BAND.TXT instead")
        _ap.add_argument("--existing-nb-out", default=None,
                         help="Path to existing NOS_BAND.TXT (used with --skip-m3cli)")

    _args = _ap.parse_args()

    # Build list of files to process
    if _args.audio:
        _files = [_args.audio]
    else:
        _files = sorted(Path(TEST_VECTOR_PATH).rglob("*.bin"))
        if not _files:
            print(f"No .bin files found under TEST_VECTOR_PATH={TEST_VECTOR_PATH!r}")
            sys.exit(1)
        _total = len(_files)
        if _args.limit is not None:
            _files = _files[:_args.limit]
        print(f"Found {_total} .bin file(s) under {TEST_VECTOR_PATH}" +
              (f"; processing first {len(_files)}" if _args.limit is not None else ""))

    _results   = []   # list of (filename, passed)
    _all_py_mc = []   # list of (audio_path, py_mc array) per file
    _all_nb_mc = []   # list of (audio_path, nb_mc array) per file

    _results_dir = Path(__file__).parent / "results"
    _results_dir.mkdir(exist_ok=True)

    for _file_idx, _audio in enumerate(_files, 1):
        _audio = str(_audio)
        _stem  = Path(_audio).stem
        _py_out = str(_results_dir / f"py_out_{_stem}.txt")

        print(f"\n{'='*72}")
        print(f"  File [{_file_idx}/{len(_files)}]: {_audio}")
        print(f"  Mode: {_mode_desc}")
        print(f"{'='*72}")

        try:
            _nb_cache   = str(_results_dir / f"NOS_BAND_{_stem}.TXT")
            _nb_c_cache = str(_results_dir / f"NOS_BAND_C_{_stem}.TXT")

            if COMPARE_C_VS_DEVICE:
                # -----------------------------------------------------------
                # MODE: C-library (PC) vs Mark3 device (m3cli)
                # "py" side = C-library output  → NOS_BAND_C_<stem>.TXT
                # "nb" side = device output     → NOS_BAND_<stem>.TXT
                # -----------------------------------------------------------
                _c_out = run_c_lib_nos_aud(_audio, _nb_c_cache,
                                           lib_path=getattr(_args, "lib", None))

                if _args.skip_m3cli:
                    _nb = _args.existing_nb_out or os.path.join(_M3CLI_DIR, "NOS_BAND.TXT")
                    print(f"  Skipping m3cli — using existing NOS_BAND.TXT: {_nb}")
                elif os.path.isfile(_nb_cache):
                    _nb = _nb_cache
                    print(f"  [CACHE] Reusing {_nb_cache} — skipping m3cli")
                else:
                    _nb = run_m3cli_nos_aud(_audio, _args.m3cli, _nb_cache)
                    print(f"  [CACHE] Saved → {_nb_cache}")

                # Compare: C-library (as "py") vs device (as "nb")
                _passed, _py_mc, _nb_mc = compare_m_clean(_c_out, _nb,
                                                           abs_tol=_args.abs_tol)

            elif USE_LOCAL_LIB:
                # -----------------------------------------------------------
                # MODE: Python reference vs PC C-library
                # -----------------------------------------------------------
                dump_eband_mband(_audio, _py_out)
                _nb = run_c_lib_nos_aud(_audio, _nb_cache,
                                        lib_path=getattr(_args, "lib", None))
                _passed, _py_mc, _nb_mc = compare_m_clean(_py_out, _nb,
                                                           abs_tol=_args.abs_tol)

            else:
                # -----------------------------------------------------------
                # MODE: Python reference vs Mark3 device (m3cli)
                # -----------------------------------------------------------
                dump_eband_mband(_audio, _py_out)

                if _args.skip_m3cli:
                    _nb = _args.existing_nb_out or os.path.join(_M3CLI_DIR, "NOS_BAND.TXT")
                    print(f"  Skipping m3cli — using existing NOS_BAND.TXT: {_nb}")
                elif os.path.isfile(_nb_cache):
                    _nb = _nb_cache
                    print(f"  [CACHE] Reusing {_nb_cache} — skipping m3cli")
                else:
                    _nb = run_m3cli_nos_aud(_audio, _args.m3cli, _nb_cache)
                    print(f"  [CACHE] Saved → {_nb_cache}")

                _passed, _py_mc, _nb_mc = compare_m_clean(_py_out, _nb,
                                                           abs_tol=_args.abs_tol)
            _results.append((_audio, _passed))
            print(f"  x (nb_mc): size={len(_nb_mc)}  {_nb_mc}")
            print(f"  y (py_mc): size={len(_py_mc)}  {_py_mc}")
            if len(_nb_mc) != 2616 or len(_py_mc) != 2616:
                print(f"  [STOP] unexpected size (expected 2616) for {_audio} — stopping loop")
                break
            if len(_py_mc) != len(_nb_mc):
                print(f"  [SKIP combined plot] py/nb size mismatch for "
                      f"{_audio}: py={len(_py_mc)}, nb={len(_nb_mc)}")
            elif len(_py_mc) > 0:
                _all_py_mc.append((_audio, _py_mc))
                _all_nb_mc.append((_audio, _nb_mc))

        except (FileNotFoundError, RuntimeError) as _e:
            print(f"\n  [FAIL] {type(_e).__name__}: {_e}")
            print(f"  Treating as failed: {_audio}")
            _results.append((_audio, None))

    # --- Combined scatter + confusion matrix across all files ---
    if ENABLE_COMBINED_PLOT and _all_py_mc:
        # Find the most common frame count and skip files that differ
        from collections import Counter as _Counter
        _frame_counts = [len(py) for _, py in _all_py_mc]
        _expected_len = _Counter(_frame_counts).most_common(1)[0][0]

        _py_filtered, _nb_filtered = [], []
        for (_aud_py, _py_arr), (_aud_nb, _nb_arr) in zip(_all_py_mc, _all_nb_mc):
            if len(_py_arr) != _expected_len:
                print(f"  [SKIP combined plot] frame count {len(_py_arr)} "
                      f"!= expected {_expected_len}: {_aud_py}")
            else:
                _py_filtered.append(_py_arr)
                _nb_filtered.append(_nb_arr)

        _py_label = "C-lib" if COMPARE_C_VS_DEVICE else "Python"
        _nb_label = "device" if (COMPARE_C_VS_DEVICE or not USE_LOCAL_LIB) else "C-lib"
        print(f"\n  Combined plot ({_py_label} vs {_nb_label}): "
              f"{len(_py_filtered)} / {len(_all_py_mc)} files "
              f"with {_expected_len} frames each")
        print(f"  combined {_py_label} size: {sum(len(a) for a in _py_filtered)}, "
              f"combined {_nb_label} size: {sum(len(a) for a in _nb_filtered)}")

        _combined_py = np.concatenate(_py_filtered)
        _combined_nb = np.concatenate(_nb_filtered)
        plot_combined_charts(
            _combined_py, _combined_nb,
            abs_tol=_args.abs_tol,
            title_prefix=f"{len(_py_filtered)} file(s)",
        )

    # --- Summary across all files ---
    if len(_results) > 1:
        _n_pass = sum(1 for _, ok in _results if ok is True)
        _n_fail = sum(1 for _, ok in _results if ok is False)
        _n_skip = sum(1 for _, ok in _results if ok is None)
        print(f"\n{'='*72}")
        print(f"  SUMMARY: {_n_pass} PASSED  {_n_fail} FAILED  {_n_skip} SKIPPED  (of {len(_results)} files)")
        print(f"{'='*72}")
        for _f, _ok in _results:
            _tag = "PASS" if _ok is True else ("SKIP" if _ok is None else "FAIL")
            print(f"  [{_tag}]  {_f}")
        print(f"{'='*72}")

    sys.exit(0 if all(ok is True for _, ok in _results) else 1)


# In[ ]:
