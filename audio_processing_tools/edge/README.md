# `edge/`: firmware-porting reference

This folder holds the Python reference implementations of the on-device rain algorithms. This README
covers the **low-level details that matter when porting to, or checking against, firmware**: frame
alignment, filter state, precision, Python↔C names, and the parity harnesses.

What the algorithms do, and their status, is in the top-level docs:

| Topic | Doc | Python | Firmware |
|---|---|---|---|
| NCL rain detection (CM7, pre-release) | [docs/rain_detection_ncl.md](../../docs/rain_detection_ncl.md) | `rain_signal_processor.py`, `rain_frame_classifier.py`, `noise_tracker.py`, `feature_extraction.py` | `edge` `NoiseCL/Src/dsp_noise_detection.c` |
| WRM rain detection (deployed) | [docs/rain_detection_wrm.md](../../docs/rain_detection_wrm.md) | `dsp_rain_detection.py` | `edge` `RainCL/Src/dsp_rain_detection.c` |
| Rain measurement (CM4): legacy DSD + band noise suppression | [docs/rain_measurement.md](../../docs/rain_measurement.md) | `rain_estimator.py`, `band_noise_estimator.py`, `band_noise_processor.py`, `../host_analysis/device_dsd_processing_emulator.py` | pluvira (production) `CM4/Src/disdrometer_legacy.c`, `disdrometer.c`, `band_noise_dsd.c`, `rain_models.c`, `fft_helper.c` |

---

## 1. Frame and filter conventions

**NCL (CM7)**
- 11,162 Hz; `n_fft = 256` (`NOISE_CL_FFT_SIZE`), `hop = 128` (`NOISE_CL_HOP_SIZE`); periodic Hann.
- STFT with `center=False`: frame `t` covers `x[t*hop : t*hop + n_fft]`, no symmetric padding.
  `center=True` would shift batch frames against the streaming/firmware frame boundaries. Frames per
  10 s clip: `(Fs*10 − n_fft) / hop + 1` (`NOISE_CL_MAX_FRAMES`).
- The STFT is taken on the **unfiltered** signal in both Python (`librosa.stft(x, …)`, not `x_proc`) and
  firmware (`finSignal`, "unfiltered — used for FFT/spectral path"). The 350 Hz high-pass
  (`NOISE_CL_HP_SOS_COEF_DEF`, 4th-order Butterworth, 2 biquads) feeds only the TD gate.
- TD filtering must be **causal** to match firmware: set `td_feature_timing_mode="causal_frame"` (Python
  default is `"offline"`, which uses non-causal `sosfiltfilt`). Firmware resets the HP state per clip and
  pre-filters the first `HOP_SIZE` samples so frame 0 sees a full filtered window; the PC build runs one
  causal pass over the clip (`sosfilt` equivalent).
- Streaming (`RainFrameClassifierState.process_audio_frame`): prime with `seed_audio(x[0:hop])`, then
  feed one hop per call so the rolling buffer holds exactly `x[t*hop : t*hop + n_fft]` before each FFT.

**CM4**
- Legacy DSD: 512-sample frames, no overlap, no window, unscaled FFT magnitude
  (`arm_rfft_fast_f32` + `arm_cmplx_mag_f32` == `np.abs(np.fft.fft(frame))`).
- Bin edges use firmware's **integer** spacing `DF = 11162 / 512 = 21`, not 21.80 Hz. See
  [rain_measurement.md §1.1](../../docs/rain_measurement.md); Python on `main` is one bin low until
  audio_processing_tools PR #7 merges.
- Band noise: 512-sample frames split into four 128-sample subframes; HPF (350 Hz) and BPF (400–700 Hz)
  are 4th-order Butterworth SOS whose state **carries across frames**, so the batch adapter requires
  `hop == frame_len`.

## 2. Precision

Firmware is float32 throughout. `SpectralNoiseProcessor` defaults to `process_dtype="float32"`;
`BandNoiseEstimatorConfig.dtype` defaults to float64. Compute and store in float32 when checking
parity; treat float64 as a declared exception. The remaining NCL C-vs-Python mismatch (2 of 2000 clip
decisions) is attributed to float32 (C) vs float64 (Python) per-mode band baselines.

## 3. State and memory

| Component | State carried between frames |
|---|---|
| `CausalNoiseTracker` | three arrays of K (operating-band bins): `_tracker`, `_tracker_scale`, `_prev_N` |
| `RainFrameClassifierState` | two previous spectrum columns (flux is `t` vs `t−2`), one low-quantile tracker per mode + one combined, an `n_fft`-sample rolling audio buffer (raw + filtered), and for `process_audio_frame` one `CausalNoiseTracker`: about 5.4 KB in total, independent of clip length (batch path: ~3.6 MB spectrogram) |
| Firmware NCL band state (`noise_cl_band_state_t`) | per mode band: `bin_lo/bin_hi`, `tracker`, `scale`, `prev_N`; plus `eta`, `band_eta` |
| `BandNoiseEstimator` / `band_noise_dsd.c` | HPF + BPF filter states, ring buffer of `W = 30` subframe energies with ages (TTL), previous-frame FFT band powers and dB levels, hold counter, effective q |

## 4. Python ↔ C names

**NCL** (`NoiseProcessorConfig` / detector params → `noise_cl_config_t`, `noisecldsp_config.h`):

| Python | C |
|---|---|
| `td_gate_threshold` | `td_gate_thr` |
| `td_kurtosis_upper_threshold` | `td_kurtosis_upper_thr` (0 disables) |
| `new_rain_primary_flux_min` | `primary_flux_min` |
| `new_rain_mode{1,2,3}_flux_min` | `mode{1,2,3}_flux_min` |
| `new_rain_min_support_count` | `NOISE_CL_MIN_SUPPORT_COUNT` |
| `clip_rain_min_frames` | `clip_rain_min_frames` |
| `q`, `win_sec` (noise PSD) | `NOISE_CL_QUANT_Q`, `NOISE_CL_QUANT_WIN_SEC` |
| `mode_flux_norm_q` (20) | `NOISE_CL_BAND_QUANT_Q` (0.20) |
| `mode_flux_norm_win_sec` (0.5) | `NOISE_CL_QUANT_WIN_SEC` (0.5; used with rounding for `band_eta`) |
| `ema_up`, `ema_down` | `NOISE_CL_EMA_UP`, `NOISE_CL_EMA_DOWN` |
| `mode_bands` | `NOISE_CL_MODE_BANDS_DEF` |
| `detector_use_noise_norm` | `use_noise_psd_norm` |
| `pre_filter_mode="highpass"` for TD features (causal with `td_feature_timing_mode="causal_frame"`) | `use_td_prefilter` |
| `mode_flux_norm_enable` | `use_flux_norm` |

Thresholds are read from `noise_cl_config()` in `dsp_noise_detection.c`, **not** from `config_list.h`
(`noise_cl_init()` ignores the latter). Release status: [rain_detection_ncl.md §3](../../docs/rain_detection_ncl.md).

**CM4 band noise:** full table in [rain_measurement.md §2.6](../../docs/rain_measurement.md)
(`BandNoiseEstimatorConfig` ↔ `BAND_NOISE_*` / `BAND_RAIN_*` in `band_noise_dsd.h`).

**Legacy DSD** (`disdrometer_legacy.h`): `RAIN_ENERGY_THRESHOLD` (0.6), `RAIN_LOG_FACTOR` (0.6),
`RAIN_LOG_BASE` (1.13), `FFT_LOW/HIGH_INDEX`, `PFT_LOW/HIGH_INDEX`, `LOWER/UPPER_WINDOW_*`.
In Python, `rain_estimator.py` defines `RAIN_ENERGY_THRESHOLD`, `RAIN_LOG_FACTOR`, `RAIN_LOG_BASE` and
`DSD_BINS` as module constants and computes the rain-band indices inside `estimate_rain_from_audio()`;
`DsdProcessingEmualtor` holds the indices as attributes (`rain_low_idx`/`rain_high_idx`,
`pft_low_idx`/`pft_high_idx`, `lwin_start_idx`/`lwin_end_idx`, `hwin_start_idx`/`hwin_end_idx`).

## 5. Numerical details that had to match (NCL bitmatch)

From `edge` `NoiseCL/TargetPC/Test/Util/bitmatch_summary.md` (stages 1–2, simplified configurations,
2000/2000) and `production_summary.md` (production configuration):

1. **Periodic** Hann (`2/N`), as librosa/scipy `fftbins=True`.
2. Band bin range `ceil(lo/res)`..`floor(hi/res)`, matching Python's `freqs >= lo & freqs <= hi`.
3. Crest factor `peak / sqrt(mean(x²) + eps)`, epsilon inside the square root.
4. Unbiased Pearson kurtosis, `scipy.stats.kurtosis(fisher=False, bias=False)`.
5. Baseline read **before** update (emit-before-ingest), as in `causal_stochastic_low_quantile_baseline`.
6. Scale EMA alpha `1 − eta`, not a fixed 0.95.
7. **Per-bin PSD tracker only:** window `W = floor(win_sec·Fs/hop)` = 43, `eta = 2/44`, not ceil
   (matches `CausalNoiseTracker`'s `int()`). Don't reuse this for the band baseline (item 11).
8. Inner clamp `N ≤ P` after each per-bin update, and outer clamp `N_lag ≤ P` before the log difference.

Production configuration (`use_noise_psd_norm=1`, `use_td_prefilter=1`, `use_flux_norm=1`):

9. Flux normalisation reads the **raw** band tracker (`tracker[i]`), not the EMA-smoothed `prev_N[i]`,
   and reads it before updating (emit-before-ingest).
10. TD pre-filter must be causal on both sides: Python `td_feature_timing_mode="causal_frame"`
    (`sosfilt`), not `"offline"` (`sosfiltfilt`).
11. **Band baseline** has its own step size, `band_eta`, from `W = round(win_sec·Fs/hop)` = 44
    (`eta = 2/45`), matching `causal_stochastic_low_quantile_baseline`'s `int(round(...))`.

## 6. Parity harnesses

| Algorithm | Harness | Notes |
|---|---|---|
| NCL | `edge` `NoiseCL/TargetPC/Test/Util/run_batch.py` → `call_c_fun.compare_algo()` | C library vs Python `SpectralNoiseProcessor`; PASS = rain frames within ±2. Release 1 (current NCL pre-release firmware): 1998/2000 clip decisions agree. Write-ups `bitmatch_summary.md`, `production_summary.md` |
| WRM | `edge` `RainCL/TargetPC/Test/Util/` (`call_c_fun.py`, `dsp_rain_detection.py`, `test_rain_classifier.py`) | 100/100 on set_100 (rain_anomaly_analysis `frame_level_processing_changes.md` §9) |
| CM4 band noise | pluvira `Utilities/band_noise_pylib/` | builds `band_noise_dsd.c` as a host library callable from Python |
| Python-only regression | `tests/edge/rain_detection/` here; golden regression in rain_anomaly_analysis `tests/rain_detection/golden_regression/` | deterministic synthetic tests; field-data baselines |

## 7. Known non-parity and open items

- **Python defaults vs firmware:** `td_feature_timing_mode` defaults to `"offline"` (firmware is causal);
  `tests/edge/rain_detection/conftest.py` uses `q = 0.30` (firmware 0.25); NCL thresholds are release 1,
  as in `edge` `master`/2.2.203.14. NCL build 2.2.204.14 (release 2) is reportedly on four devices; see
  [rain_detection_ncl.md §3](../../docs/rain_detection_ncl.md).
- **Streaming `process_audio_frame()`** differs from the batch path (which matches firmware) on 5.6 % of
  set_1000 clip decisions. Most of that comes from the 1-frame TD lag and pre-filter warm-up: ungated
  streaming already differs on 5.4 %, and its rain-gated noise learning (firmware doesn't gate) adds
  0.8 %. Use the batch path for firmware comparisons.
- **Streaming vs replay, by design:** filter warm-up after `reset()`, TD features one frame later, no
  per-clip flux winsorisation, and same-frame N(t) instead of N(t−1) for normalisation.
- **CM4:** Python legacy DSD and window bins differ from firmware until PR #7 (rain band 19..32 vs
  20..33, PFT high 67 vs 70, windows 13/45 vs 14/47); `smooth_N_E` exists only in
  Python; `BandNoiseEstimatorConfig` defaults to float64.
- **Spectral suppressor path** (`SpectralNoiseProcessor` gain/ISTFT) is Python-only, used to test and
  show the effect of spectral noise suppression; nothing to port.
