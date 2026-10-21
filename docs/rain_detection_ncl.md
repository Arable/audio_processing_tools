# NCL Rain Detection (CM7)

**Python:** `audio_processing_tools/edge/rain_signal_processor.py` (`SpectralNoiseProcessor`, `RainDetectorProcessor`),
`rain_frame_classifier.py` (`RainFrameClassifierMixin`, `RainFrameClassifierState`), `noise_tracker.py`
(`CausalNoiseTracker`), `feature_extraction.py`.
**Firmware:** `edge` repo, `NoiseCL/Src/dsp_noise_detection.c` (`noise_cl_config()`, `init_dsp_noise_model()`),
constants in `NoiseCL/Inc/noisecldsp_config.h`.

NCL is the per-minute "is it raining" gate that runs on the 10 s clip CM7 records each minute. It is
meant to replace the deployed WRM detector ([rain_detection_wrm.md](rain_detection_wrm.md)) and is
an NCL pre-release. The detector that currently gates rain reporting is WRM: CM4 firmware reads the
WRM (RainCL) rain status (`get_rain_status_from_raincl()`, pluvira `disdrometer.c`); NCL's output is
stored alongside, and how rain gating is applied downstream has not been verified here. The intended
design is that NCL's per-minute decision gates CM4's measurement ([rain_measurement.md](rain_measurement.md)).

Low-level conventions for porting (frame alignment, filter state, float32, Python↔C names) are in
[`audio_processing_tools/edge/README.md`](../audio_processing_tools/edge/README.md).

---

## 1. Overview

The detector looks for **energy above a continuously learned noise baseline** in the dome's resonance
modes, and confirms it with time-domain impulsiveness:

```
Input audio (11,162 Hz, 10 s)
   │
   ├── high-pass 350 Hz (4th-order Butterworth) ── TD features (crest factor, kurtosis)
   │
   └── STFT (n_fft 256, hop 128, Hann, center=False) on the UNFILTERED audio
          │
          ├── per-bin noise PSD  N(t)  (CausalNoiseTracker, q = 0.25)
          ├── noise-normalised spectrum  log P(t) − log N(t−1)
          ├── mode-band flux  →  per-mode low-quantile baseline (q = 0.20)  →  flux score
          └── frame decision: TD gate AND primary mode AND ≥ 3 of 3 support modes
                                     │
                                     ▼
                     clip_is_rain = rain_frame_count ≥ clip_rain_min_frames
```

Mode bands (Hz): primary 450–650; support 800–1050, 1500–1800, 2350–2550; a fifth band, 3150–3350, is
tracked but not used in the decision (`NOISE_CL_MODE_BANDS_DEF`; same list in the Python test config
`tests/edge/rain_detection/conftest.py`). The operating band is 400–3500 Hz.

The spectral suppressor path (section 6) is not part of the decision and is not in firmware.

## 2. Processing stages

### 2.1 Pre-filter and STFT

- `pre_filter_mode = "highpass"` (default; also `"bandpass"` to the operating band, or `"none"`),
  `hp_cutoff_hz = 350`, 4th order. The filtered signal feeds the **time-domain** features.
- STFT: `n_fft = 256`, `hop = 128`, Hann window, `center=False`, so frame `t` covers
  `x[t*hop : t*hop + n_fft]`. The STFT is taken on the **unfiltered** audio. Firmware does the same:
  `pcm_s16_to_float(...)` into `finSignal` is commented "unfiltered — used for FFT/spectral path",
  and the HP filter is used only for the TD gate.

### 2.2 Noise PSD (`CausalNoiseTracker`)

A causal stochastic low-quantile tracker per frequency bin over the operating band:

```
if P > tracker: increase slowly      (step ∝ q)
else:           decrease faster      (step ∝ 1 − q)
```

- `q = 0.25`, `win_sec = 0.5` (firmware `NOISE_CL_QUANT_Q`, `NOISE_CL_QUANT_WIN_SEC`).
- Asymmetric EMA on the output: `ema_up = 0.6` (fast when noise rises), `ema_down = 0.95` (slow when
  it falls).
- Safety clamp: `N ≤ noise_psd_max_ratio × P` with `noise_psd_max_ratio = 1.0`, per bin, per frame.
- For the **detector's** normalisation, the batch path (and firmware) runs the tracker **without rain
  gating**: every frame is a candidate noise frame. The streaming `process_audio_frame()` path does gate
  learning on `is_rain` after warm-up (section 5). The suppressor path keeps a second, gated estimate
  that only learns from frames classified as noise (section 6).

### 2.3 Time-lagged noise normalisation

The detector compares the current frame against the **previous frame's** noise estimate:

```
detector input = log P(t) − log N_lag(t),   N_lag(t) = N(t−1)
```

- The lag is unconditional on the detector side (`np.roll` on the noise-PSD array;
  `detector_use_noise_norm` defaults to True). Firmware uses the same lagged estimate (`psd_N_lag`).
- For the suppressor gain (section 6) the lag is opt-in: `use_lagged_noise_psd` defaults to False, so
  the gain uses the same-frame N(t) unless that flag is set.
- **Why the lag matters:** with the current frame's PSD, rain energy would leak into the noise
  estimate, normalisation would become self-referential and the detector would see a distorted
  spectrum. Using N(t−1) keeps the current frame out of its own baseline, so the input behaves like a
  causal, frame-level SNR: rain is detected as energy above the noise floor, not as raw amplitude.

### 2.4 Mode-band flux and per-mode baselines

1. **Flux:** half-wave-rectified difference between frame `t` and frame `t−2` (not `t−1`), summed per
   mode band.
2. **Per-mode baseline:** each mode's flux is normalised against its own causal low-quantile baseline
   (`mode_flux_norm_q = 20` percentile, `mode_flux_norm_win_sec = 0.5`; firmware
   `NOISE_CL_BAND_QUANT_Q = 0.20`, `NOISE_CL_QUANT_WIN_SEC = 0.5`). The score is an excess-over-baseline ratio,
   `max(flux − baseline, 0) / (baseline + floor)`, not a z-score.
3. Thresholds are applied in `log1p` space.

Spectral peak structure (position, prominence, bandwidth relative to the mode bands) is computed and
exported for feature dumps and offline tuning, but does not gate the decision.

### 2.5 Time-domain gate

On the high-pass-filtered frame:

```
td_gate = crest_factor > td_gate_threshold
          AND (kurtosis ≤ td_kurtosis_upper_threshold, if that threshold > 0)
```

Crest factor is `peak / sqrt(mean(x²) + eps)`; kurtosis is unbiased Pearson (non-excess), matching
`scipy.stats.kurtosis(fisher=False, bias=False)`. The gate multiplies the flux scores, so frames that
don't look impulsive can't be rain.

### 2.6 Frame decision

```
is_rain = primary_ok AND (support_hits ≥ new_rain_min_support_count)
```

`primary_ok` requires the primary mode's score to clear `new_rain_primary_flux_min`; `support_hits`
counts how many of the 3 support modes clear their own thresholds (`new_rain_mode{1,2,3}_flux_min`).
Requiring all 3 support modes on top of the primary guards against narrowband interference that
resembles a single mode. The per-mode adaptive baselines are a substantial change from WRM's fixed
SNR thresholds ([rain_detection_wrm.md](rain_detection_wrm.md)).
`rain_conf` is **binary** (0.0/1.0) so it stays aligned with the hard frame count used at clip level.
A small threshold miscalibration therefore flips a frame rather than shading a confidence, which is
worth keeping in mind when tuning against noisy labels.

### 2.7 Clip decision and outputs

```
clip_is_rain   = rain_frame_count ≥ clip_rain_min_frames
median_conf    = median(rain_conf on rain frames)
abundance_conf = clip(rain_frame_count / (2 × clip_rain_min_frames), 0, 1)
clip_rain_conf = max(median_conf, abundance_conf)
```

Requiring several rain frames stops short noise bursts and isolated misclassifications from
triggering a rain minute. With `estimate_clip_rain=True`, `RainDetectorProcessor` also attaches the
legacy DSD rain estimate (`rain_estimator.estimate_rain_from_audio`).

Outputs: metrics (`clip_is_rain`, `clip_rain_conf`, `rain_frame_count`, `clip_rain_fraction`,
`mean_noise_floor_db`) and per-frame state (`frame_class`, `rain_conf`, `noise_conf`, `features`,
`debug`).

**Configuration:** nested `{"suppressor": {...}, "detector": {...}}`; precedence is flat params > nested
params > defaults. **Modes:** full (detection + suppression); `classifier_only_mode` (features only);
`suppressor_bypass` (detector only: keeps detector normalisation, skips final PSD/gain/ISTFT).

## 3. Thresholds and release status

| Parameter (Python / firmware) | Release 1 (`trial_params3`) | Release 2 |
|---|---|---|
| `td_gate_threshold` / `td_gate_thr` | 3.4 | **3.7** |
| `new_rain_mode3_flux_min` / `mode3_flux_min` | 2.45 | **3.5** |
| `clip_rain_min_frames` | 4 | **2** |
| `new_rain_primary_flux_min` / `primary_flux_min` | 2.19 | 2.19 |
| `new_rain_mode1_flux_min` / `mode1_flux_min` | 2.63 | 2.63 |
| `new_rain_mode2_flux_min` / `mode2_flux_min` | 2.57 | 2.57 |
| `new_rain_min_support_count` / `NOISE_CL_MIN_SUPPORT_COUNT` | 3 | 3 |
| `td_kurtosis_upper_threshold` / `td_kurtosis_upper_thr` | 12.0 | 12.0 |
| noise PSD `q` / `NOISE_CL_QUANT_Q` | 0.25 | 0.25 |

- **Release status (as of 2026-10-21; fleet firmware versions not checked).** Release 2 is the
  recommendation in rain_anomaly_analysis
  [`learnings/ncl_fw_threshold_recommendation.md`](https://github.com/Arable/rain_anomaly_analysis/blob/main/learnings/ncl_fw_threshold_recommendation.md).
  The thresholds the model uses are hardcoded in `noise_cl_config()` (`dsp_noise_detection.c`);
  `config_list.h` is not read by the model.
  - `edge` `master` and `Release-2.2.203.14` (`22e0250`, PR #21) hardcode **release 1** (4 / 3.4 / 2.45).
    PR #21 changed only `config_list.h` (`ce97000` on `sp/noise_cancel_model`).
  - NCL build **2.2.204.14** (`Release-2.2.204.14` = `8e2e2ec`, the same commit as open `edge` PR #30,
    `sp/param_ch`) hardcodes **release 2**. It was reportedly rolled out to four devices as of 2026-10-12:
    D003974, D003733 and two HW3.5 EVT units, B050010 and B050008 (unpushed `edge` commit `7c159e3`).
  - So most NCL pre-release devices probably still run release 1. NCL is a pre-release on all of them;
    WRM is the deployed detector.
- **Python defaults are release 1** (the current NCL pre-release firmware on most devices): `rain_frame_classifier.py` (`td_gate_threshold` 3.4,
  `new_rain_mode3_flux_min` 2.45), `rain_signal_processor.py` (`clip_rain_min_frames` 4), and
  `tests/edge/rain_detection/conftest.py`. Move them together with the golden baselines in
  rain_anomaly_analysis, not on their own. Note that `conftest.py` uses `q = 0.30`, not firmware's 0.25.

## 4. Firmware alignment and parity

**Which Python path matches firmware:** the **batch** path (`SpectralNoiseProcessor` /
`RainDetectorProcessor`) with `td_feature_timing_mode="causal_frame"` and `q = 0.25`. The Python default
for `td_feature_timing_mode` is `"offline"` (TD features from non-causal `sosfiltfilt`); set
`"causal_frame"` to match firmware's causal filtering. The streaming `process_audio_frame()` path is
**not** the faithful one: it gates noise learning on rain (firmware doesn't), lags TD features by a
frame, and has pre-filter warm-up.

| Comparison | Result | Source |
|---|---|---|
| C library vs Python batch, NCL release 1 (current NCL pre-release firmware), 1000 rain + 1000 no-rain clips | clip decision agrees on 1998/2000; 1992/2000 within ±2 rain frames | `edge` `NoiseCL/TargetPC/Test/Util/` (`run_batch.py`, `production_summary.md`) |
| C vs Python, NCL and WRM, set_100 | 100/100 clip decisions each | rain_anomaly_analysis `tests/rain_detection/golden_regression/frame_level_processing_changes.md` §9 |
| Python `replay_clip()` vs batch `_detect_rain_over_time()` | 1.0000 frame-class agreement, flux MAE 0.000000 | [history/frame_level_rain_streaming.md](history/frame_level_rain_streaming.md) |
| Python `process_frame()` vs no-winsor replay | 0.9552 frame agreement; all disagreements from the 1-frame TD lag | same |
| Python batch vs streaming `process_audio_frame()`, set_1000 | rain-gated: 0.994 frame agreement, 56 clips (5.6 %) change decision; ungated: 54 clips (5.4 %). Gating alone flips 0.8 %; most of the gap is the 1-frame TD lag and pre-filter warm-up | `frame_level_processing_changes.md` §8 |

The last two rows are Python-vs-Python comparisons, not firmware ones. The remaining C-vs-Python
residual is attributed to float32 (C) vs float64 (Python) band baselines.

Validation and tuning harnesses live in rain_anomaly_analysis
(`tests/rain_detection/golden_regression/`). Its baseline scripts and DVC baselines are still on
release 1 with `q = 0.30`, so they need regenerating before they reflect firmware exactly.

## 5. Streaming implementation (`RainFrameClassifierState`)

A per-frame implementation for embedded use: O(n_fft) working state (~5.4 KB) instead of the batch
path's full spectrogram (~3.6 MB). Three modes over the same `_decide_frame()` core:

| Mode | Input | Use |
|---|---|---|
| `replay_clip(P, audio)` | Precomputed spectrogram + audio | Offline parity check against the batch mixin |
| `process_frame(spectrum, audio_hop)` | One noise-normalised spectrum column + one hop of audio | Streaming with caller-supplied normalisation |
| `process_audio_frame(chunk, is_rain)` | Raw audio only | Fully causal; own FFT and `CausalNoiseTracker` |

Known, by-design differences from the batch path: causal `sosfilt` (filter warm-up after `reset()`)
instead of `sosfiltfilt`; TD features one frame later; no per-clip flux winsorisation; and the streaming
path's normalisation uses the same-frame N(t) where the batch path uses N(t−1). Details and history:
[history/frame_level_rain_streaming.md](history/frame_level_rain_streaming.md).

## 6. Spectral suppressor path (Python validation only)

`SpectralNoiseProcessor` also produces a noise-suppressed signal. **This path is not in firmware.** It is
kept in Python to test and demonstrate how effective spectral noise suppression is (for example,
listening to or re-analysing the denoised output); it does not affect the rain decision.

- **Gain:** spectral subtraction with confidence-weighted oversubtraction (Boll, 1979). Square-root
  subtraction (default) `G = 1 − α·sqrt(N/P)`, or Wiener `G = max(P − αN, 0) / P`.
- **Adaptive oversubtraction:** suppression starts ramping only above a noise confidence of 0.7:
  `eff_noise = clip((noise_conf − 0.7) / 0.3, 0, 1)`, `α = base + eff_noise·(max − base)`, typically
  base 1.0, max 3.0. Rain-like frames get little suppression, noise-like frames strong suppression.
- **Stabilisation:** frequency smoothing against musical noise; temporal smoothing is turned off on rain
  frames to keep transients.
- **Optional SNR gating:** reduces suppression where SNR is high, to protect strong impulses.
- **Noise PSD:** a separate estimate from the detector's, updated only on frames classified as NOISE
  (RAIN and UNCERTAIN frames are excluded).
- **Output:** `S_hat = G·S`, optionally `y = ISTFT(S_hat)`.

The main risk is circular: noise has to be classified before it is suppressed, and classification is
least reliable in heavy noise. Warm-up (learning from all frames until the buffer is half full) handles
the cold start.

## 7. Behaviour in noise

**Intended strengths:** noise-normalised detection (wind bursts and slowly varying mechanical noise are
normalised out, rain impulses kept); a detector-side noise PSD that doesn't depend on detector
decisions, so detector errors don't feed back into it; asymmetric PSD tracking that follows rising noise quickly and decays
slowly; strong TD features (crest factor, kurtosis); a modular detector/suppressor split.

| Challenge | Typical system | This design |
|---|---|---|
| Wind bursts | Misclassified as rain | Normalised out |
| Mechanical vibration | Strong false positives | Filtered by TD gate + normalisation |
| Non-stationary noise | PSD drift or slow adaptation | Quantile tracking + lagged normalisation |
| Rain + noise overlap | Suppressed | Preserved via confidence + SNR (suppressor path) |

**Known failure modes:** rapidly changing noise (the PSD lags, giving temporary under- or
over-suppression); very light rain below the noise floor; strong impulsive mechanical noise that looks
like rain in the TD features (the anomalous-noise false positives studied in rain_anomaly_analysis);
frame misclassification propagating to the clip decision.

**Validation caveat:** the design has not been exhaustively validated across deployment conditions;
performance may differ where there is strong mechanical or highly structured vibration. Among the TD
features, time flux has been less discriminative in the datasets so far than crest factor and kurtosis.

**Limitations:** the FD features are highly correlated; decisions are threshold-based; PSD tracking can
lag fast noise changes; thresholds risk overfitting small tuning sets.

**Improvement ideas:** a learned model in place of thresholds; feature decorrelation; multi-timescale
PSD tracking; band-wise adaptive suppression; a secondary microphone as a noise reference; event-level
detection; distance-aware modelling.

One-line summary: *rain is detected as energy above a continuously learned noise baseline, which keeps
detection working in noisy environments.*

## 8. Related

- CM4/CM7 noise-estimate scale: CM4's `N_E` ≈ 0.114 × the sum of CM7's noise PSD over the 400–700 Hz
  bins (HW3.0, 250 no-rain clips), rain_anomaly_analysis PR #7
  ([`learnings/cm4_cm7_noise_scale_handoff.md`](https://github.com/Arable/rain_anomaly_analysis/blob/main/learnings/cm4_cm7_noise_scale_handoff.md),
  available once that PR merges).
- Development history: [history/noise_psd_by_mode_wiring.md](history/noise_psd_by_mode_wiring.md),
  [history/feature_dump_peak_and_envelope_wiring.md](history/feature_dump_peak_and_envelope_wiring.md).
