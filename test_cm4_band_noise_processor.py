#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ver 3 code
# %%
from pathlib import Path
from typing import Dict, Any

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
INPUT_TYPE = "LocalPath"  # "RemotePath" or "LocalPath"

#TEST_VECTOR_PATH = "/Users/vikrantoak/Downloads/tv_sets/balanced_rain_test_vectors"
TEST_VECTOR_PATH = "/home/santhosh/Mark3/Doc/Rain/noise_cancel/test_vector/raw_audio/D009920"
LOCAL_AUDIO_CACHE = "/home/santhosh/Mark3/Doc/Rain/noise_cancel/audio_header_cache"

N = 100
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
    #plot_band_noise_states_aligned(states_df, file_key=fk)
    plot_band_noise_state(
        states_df,
        file_key=fk,
        hpf_cutoff_hz = 400.0,
        hpf_order  = 4,
        noise_amp_mode = "rms",   # "rms" or "std"
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
from band_noise_estimator import BandNoiseEstimator, BandNoiseEstimatorConfig

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
    x = pcm_int16.astype(np.float64) / 32768.0

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


# ---------------------------------------------------------------------------
# CLI:  python test_cm4_band_noise_processor.py <audio.bin> [output.txt]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_cm4_band_noise_processor.py <audio.bin> [output.txt]")
        sys.exit(1)
    _audio  = sys.argv[1]
    _output = sys.argv[2] if len(sys.argv) > 2 else "nos_band_out.txt"
    dump_eband_mband(_audio, _output)


# In[ ]:
