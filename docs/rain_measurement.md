# Rain Measurement (CM4)

**Python:** `audio_processing_tools/edge/rain_estimator.py` (`estimate_rain_from_audio`, `rain_regression_model`),
`audio_processing_tools/host_analysis/device_dsd_processing_emulator.py` (`DsdProcessingEmualtor`),
`audio_processing_tools/edge/band_noise_estimator.py` (`BandNoiseEstimator`, `NoiseFrameDetector`),
`audio_processing_tools/edge/band_noise_processor.py` (`BandNoiseEstimatorProcessor`).
**Firmware** (production CM4 source: pluvira, `CM4/`): `disdrometer_legacy.c/.h`, `disdrometer.c`,
`band_noise_dsd.c/.h`, `rain_models.c`, `fft_helper.c`. (mark3-firmware-trunk `master` has the legacy
files but not `band_noise_dsd.c`, which is only on its `band-noise-nc-fixes` branch.)

CM4 measures how much it rains. It runs continuously on 512-sample frames, unlike CM7, which records a
10 s clip once a minute. A per-minute detection decides whether each minute's CM4 measurement counts:
a missed detection drops a correct measurement, and a false detection lets through one that shouldn't be
reported. In current firmware CM4 reads the WRM (RainCL) rain status (`get_rain_status_from_raincl()`);
NCL is a pre-release whose output is stored alongside, and the downstream gating has not been verified
here. See [rain_detection_ncl.md](rain_detection_ncl.md).

Porting conventions (filter state, float32, Python↔C names): [`audio_processing_tools/edge/README.md`](../audio_processing_tools/edge/README.md).

---

## 1. Legacy measurement: band energy → DSD → mm

Per 512-sample frame (11,162 Hz, ~46 ms, no overlap, no window):

1. **FFT magnitude** (`fft_helper_do_fast_fft`: unscaled `arm_rfft_fast_f32` + `arm_cmplx_mag_f32`,
   the same as `np.abs(np.fft.fft(frame))`).
2. **Band energy** `drop_energy_level` = sum of magnitudes over the 400–700 Hz bins.
3. **Binning:** if `drop_energy_level > 0.6`,
   `bin = floor(log(1 + (E − 0.6)·0.6) / log(1.13))`, clipped to 0..31, and that DSD bin is incremented
   (`RAIN_ENERGY_THRESHOLD`, `RAIN_LOG_FACTOR`, `RAIN_LOG_BASE`).
4. **Peak frequency (PFT):** the strongest bin in 100–1500 Hz is tracked; the most frequent peak per
   2 s goes into 30 PFT bins.
5. **FFT energy windows:** two 19-bin windows starting at 300 Hz and 1000 Hz (38 values).

The 100-value histogram (32 DSD + 30 PFT + 38 FFT) is reported each minute. Rain is computed from the
DSD part:

```
energy_i         = ((exp(i·log 1.13) − 1) / 0.6) + 0.6        (reverse binning, bin i)
weighted_dsd_sum = Σ count_i · energy_i                        ("W")
rain             = a·W / (b + c·W),  a = 1.41079931e-03, b = 25.2101903, c = −7.44399129e-06
```

The legacy pipeline's design, assumptions and weaknesses (no windowing, fixed 0.6 threshold, energy per
frame rather than per drop, wind counted as drop energy) are discussed in
[rain_detection_wrm.md](rain_detection_wrm.md) §4 and §9.

### 1.1 Bin edges: firmware vs Python

Firmware computes the bin spacing with integer division (`disdrometer_legacy.h`):

```
DF             = SAMPLING_FREQ / FFT_SAMPLE_LENGTH = 11162 / 512 = 21
FFT_LOW_INDEX  = 400/DF + 1 = 20,   FFT_HIGH_INDEX = 700/DF = 33      → rain band bins 20..33
PFT_LOW_INDEX  = 100/DF + 1 = 5,    PFT_HIGH_INDEX = 1500/DF − 1 = 70
lower window   = 300/DF = 14 .. 32,  upper window = 1000/DF = 47 .. 65
```

`rain_estimator.py` and the DSD emulator on `main` use the true spacing, `fs / 512 = 21.80 Hz`. That
shifts more than the rain band:

| | Firmware | Python on `main` |
|---|---|---|
| Rain band | 20..33 | 19..32 |
| PFT | 5..70 | 5..67 |
| Lower FFT window | 14..32 | 13..31 |
| Upper FFT window | 47..65 | 45..63 |

audio_processing_tools PR #7 (`vo/fix-dsd-band-bins`, open) moves all of these to firmware's integer
DF. Until it merges, Python legacy DSD and window outputs don't line up with the device's.

## 2. Band noise estimation and suppression

A frame-causal noise estimator and suppressor scoped to the primary band, 400–700 Hz, the same band the
DSD energy comes from. It produces a noise-suppressed band magnitude (`M_clean`, `M_clean_fft`) so that
wind and other slowly varying background energy isn't counted as drop energy.

This is a different, narrower module from CM7's spectral noise processor
([rain_detection_ncl.md](rain_detection_ncl.md)), which works across all resonance modes in
400–3500 Hz for frame classification.

**Status** (from reading pluvira `master` source, not observed at runtime).
- *Firmware:* in `disdrometer_process_sample()` CM4 runs `band_noise_dsd_process_frame()` and accumulates
  a second, **noise-cancelled** DSD histogram from `M_clean_fft` (`accumulate_nc_histogram`), next to the
  raw one. It seems not to run on every frame: it is skipped while the audio write pointer is inside the
  scratch area it borrows at the end of the raw-audio buffer (`audio_raw_writep >= BAND_NOISE_SCRATCH_ADDR`),
  until the CM7 model run clears that buffer. Skipped frames add nothing to the noise-cancelled histogram,
  so the noise-cancelled rates may cover less time than the raw one. The estimator is reset at each
  no-rain poll and at the start of each rain-accumulation window (`band_noise_dsd_reset()`).
- *What is reported:* the raw `energy_histogram` is uploaded as the DSD (`add_dsd_sensor_data`). Three
  rain rates go into the rain-classification sensor record (`store_rain_classification_sensor_data`,
  `rain_models.c`): `rain_regression` (raw), `noise_classifier.rain_regression` (noise-cancelled, via
  bins) and `noise_classifier.nc_rain_rate_direct` (noise-cancelled, accumulated `M_clean_fft` without
  binning). The noise-cancelled histogram itself apparently isn't uploaded. Cloud precipitation is
  presumably computed from the raw DSD (not checked).
- *Python:* `BandNoiseEstimator` is standalone. `RainDetectorProcessor` and `estimate_rain_from_audio()`
  don't use it; it is the reference implementation used for parity and offline evaluation.

### 2.1 Frame decision (`NoiseFrameDetector`)

Before noise learning, each frame's four 128-sample subframes are classified as rain or not, so only
non-rain energy feeds the noise estimate. Either of two triggers marks rain:

1. **FFT domain (whole frame):** total rain-band power jumps by ≥ `M_db` (6 dB) **and** the primary
   band (450–650 Hz) by ≥ `N_db` (3 dB) against the previous frame → all four subframes are rain.
2. **Time domain (per subframe, preferred):** the primary-band level rises by ≥ `band_rise_db` (6 dB)
   **and** exceeds the concurrent rise in broadband (high-pass) loudness by ≥ `excess_rise_db` (3 dB).
   A narrowband jump relative to broadband energy separates an impact (which excites the resonance)
   from wind (which raises everything together). A `k_subframes` hold (2) extends the mask over decay
   tails. Working on 128-sample subframes (~11.5 ms) is fine enough to catch individual impacts. The
   assumption can break down for strong gusts that happen to excite the fundamental mode
   preferentially.

The rain bands summed for trigger 1 are the five CM7 mode bands (450–650, 800–1050, 1500–1800,
2350–2550, 3150–3350 Hz), combined into one value: this is a rain/no-rain gate, not a per-band noise
estimate. Two legacy triggers (`use_dE_over_Ehpf`, `use_D_trigger`) exist but are off.

The whole-frame flag `fft_rain_frame` acts through this learning mask and, only when the optional
`smooth_N_E` is on, also picks the wet or dry attack rate of the `N_E` smoothing
(`band_noise_estimator.py`, `is_raining`). `smooth_N_E` is off by default and in firmware (`N_E = N_E_raw`),
so there a change in the rain-frame fraction is not by itself a change in measured rain.

### 2.2 Noise learning

- Non-rain subframe energies go into a ring buffer of length `W` (30). A subframe is only learned if
  it isn't masked as rain (unless `learn_during_rain` / `force_learn_all` are set, for experiments).
- After `W_min` (10) valid entries, the noise estimate is the `q`-quantile (0.30) of the buffer,
  optionally EMA-smoothed with `ema_alpha`. The default `ema_alpha = 1.0` means no smoothing.
- **TTL:** entries older than `noise_buffer_ttl_frames` (200) expire, so the estimate can't stay pinned
  to stale contents during long rain.
- **Replenishment** (off by default): if a frame has no non-rain subframes and the buffer isn't full,
  `noise_replenish_from_all_subframes` pushes a low quantile (`noise_replenish_q` 0.20) of all subframes.
  The effective quantile drifts toward the replenish quantile when this fires and back toward `q` in
  normal learning (`noise_q_replenish_alpha` 0.2, `noise_q_normal_alpha` 0.1).

### 2.3 Suppression

Wiener-like gain from the primary-band energy `Eb` and the noise estimate `N_E`:

```
G_pow       = max(Eb − beta·N_E, 0) / (Eb + eps)
G_mag       = clip(sqrt(G_pow), gain_floor, 1.0)
M_clean     = M_band     · G_mag     # suppressed time-domain BPF magnitude
M_clean_fft = M_band_fft · G_mag     # suppressed FFT magnitude sum → noise-cancelled DSD
```

`gain_floor` (0.10) avoids full silence when over-suppressing. Optional outer smoothing of `N_E`
(`smooth_N_E`, off by default and **not implemented in firmware**) uses an asymmetric attack: slow rise
in rain (`ne_attack_alpha_wet` 0.02), faster rise otherwise (`ne_attack_alpha_dry` 0.15), fast fall
(`ne_release_alpha` 0.25).

### 2.4 Two FFT-band metrics

`M_band_fft` / `E_band_fft` use `legacy_band_bins()`, floor-based bin edges that reproduce
`disdrometer_legacy.h`'s integer macros, rather than the round-based `hz_to_bin()` used for the rain and
primary bands. Firmware does the same (`legacy_low_bin` / `legacy_high_bin` in `band_noise_dsd.c`). This
lines the unsuppressed FFT magnitude sum up with the legacy `drop_energy_level` band, so suppression can
be compared side by side.

The bins match but the input doesn't: `M_band_fft` is computed from the 350 Hz high-passed frame (Python
`band_noise_estimator.py`, firmware `band_noise_dsd.c` `run_rain_detector(x_hpf, …)`), while
`drop_energy_level` uses the raw frame. The 4th-order 350 Hz HPF attenuates about 1.3 dB (power) at
400 Hz and less above it, so `M_band_fft` (and therefore `M_clean_fft`) reads slightly low against the
raw path at the bottom of the band.

### 2.5 Telemetry and the batch adapter

- `BandNoiseEnergyStats`: rolling `total_energy_sum`, `rain_energy_sum`, `noise_energy_sum` and
  buffer-health counters (`noise_buffer_underflow_frame_count`, `frames_since_noise_update`), read and
  cleared once per report interval with `read_and_reset_energy_stats()` (once a minute on the device).
- `BandNoiseEstimatorProcessor`: wraps the estimator for `audio_processing_framework`. It takes a mono
  array and a params dict (nested `det.*` overrides allowed), frames at `hop == frame_len` (required:
  IIR state carries across frames), and returns `(results_dict, state_dict)`. `state_dict` has per-frame
  arrays of the `BandNoiseFrameOut` fields (`M_band`, `E_band`, `N_E`, `subE`, `G_mag`, `M_clean`,
  `M_clean_fft`, …), and `results_dict` has summary medians; the energy stats appear once, as a
  cumulative snapshot under `state["energy_stats"]`.

### 2.6 Configuration (Python default = firmware value)

| Python (`BandNoiseEstimatorConfig` / `.det`) | Firmware (`band_noise_dsd.h`) | Value | Effect |
|---|---|---|---|
| `band_hz` | `BAND_BPF_LO/HI` | 400–700 Hz | Band for energy, noise and suppression |
| `hp_cutoff_hz`, `hp_order`; `bpf_order` | `s_hpf_coeffs`, BPF coefficients in `band_noise_dsd.c` | 350 Hz, 4; 4 | Butterworth HPF (broadband loudness) and 400–700 Hz BPF, precomputed SOS |
| `frame_len`, `subframe_len`, `subhop` | `BAND_NOISE_FRAME_LEN`, `_SUB_LEN`, `_SUBHOP` | 512, 128, 128 | Frame / subframe layout (4 subframes) |
| `W` / `W_min` | `BAND_NOISE_W` / `_W_MIN` | 30 / 10 | Ring buffer length / warm-up |
| `q` | `BAND_NOISE_Q` | 0.30 | Noise quantile |
| `ema_alpha` | `BAND_NOISE_EMA_ALPHA` | 1.0 | Quantile smoothing (1.0 = none) |
| `noise_buffer_ttl_frames` | `BAND_NOISE_BUFFER_TTL_FRAMES` | 200 | Max entry age |
| `beta` | `BAND_NOISE_BETA` | 1.0 | Oversubtraction |
| `gain_floor` | `BAND_NOISE_GAIN_FLOOR` | 0.10 | Minimum gain |
| `noise_replenish_from_all_subframes` | `BAND_NOISE_REPLENISH_FROM_ALL` | off | Replenishment fallback |
| `noise_replenish_q` | `BAND_NOISE_REPLENISH_Q` | 0.20 | Replenish quantile |
| `noise_q_replenish_alpha` / `_normal_alpha` | `BAND_NOISE_Q_REPLENISH_ALPHA` / `_NORMAL_ALPHA` | 0.2 / 0.1 | Effective-q adaptation |
| `smooth_N_E` (+ attack/release alphas) | not implemented | off | Outer N_E smoothing |
| `det.M_db` / `det.N_db` | `BAND_RAIN_M_DB` / `_N_DB` | 6 / 3 dB | FFT whole-frame trigger |
| `det.band_rise_db` / `det.excess_rise_db` | `BAND_RAIN_BAND_RISE_DB` / `_EXCESS_RISE_DB` | 6 / 3 dB | Time-domain onset trigger |
| `det.k_subframes` | `BAND_RAIN_HOLD_SUBFRAMES` | 2 | Hold after a trigger |
| `det.min_Ehpf` / `det.min_Eband` | `BAND_RAIN_MIN_EHPF` / `_MIN_EBAND` | 1e-10 / 1e-12 | Floors for dB rises |

## 3. Evaluated behaviour

Offline results from rain_anomaly_analysis (DS-36 wind work and the CM4 parameter assessment, HW3.0
recordings). They describe the noise estimator's response, not verified loss or gain of true rain: the
recordings carry their own native noise, and there is no clean-rain reference.

- **Firmware defaults vs the old notebook overrides.** The overrides used in early test notebooks
  (rise 8/4 dB, `smooth_N_E`, W 28, EMA 0.25, TTL 28 and others) were never ported to firmware. The two
  that matter, the rise thresholds and `smooth_N_E`, each move results by up to about 3.5–5 percentage
  points, most at −5 dB, and pull in opposite directions; as a set they stay within about 1.3 pp of
  firmware on 10 s clips and about 4 pp in 120 s streams. The recommendation is to keep the firmware
  defaults.
- **Wind response.** At rain-band 0 dB and q = 0.30, the noise estimate rises by a median 0.599 of the
  added wind energy (0.587 in steady state); higher q raises this but also lowers rain output. Across
  levels it falls from about 0.84 at 20 dB to about 0.58 at −5 dB.
- **Tracking.** At 0 dB the estimate reaches 90 % of a wind step in about 0.71 s and releases in about
  0.52 s. The 0.5 s wind ramp alone accounts for about 0.43 s and 0.31 s, so the estimator adds about
  0.3 s and 0.2 s.
- **Possible over-suppression in heavy rain.** Against noise floors taken from the quietest impact-free
  blocks of each clip, CM4's mean noise estimate at q = 0.30 is higher than the 10th-percentile floor in
195/195 clips, the 20th in 193/195 and the 30th in 149/195. In classes 10–12 its
  rain-output reduction (median 12.8 %) is close to the 30th-percentile floor's (12.3 %) but far above
  the 10th-percentile floor's (4.1 %). This is consistent with some rain leaking into the noise
  estimate, but doesn't establish it.
- **Scale against CM7.** CM4's `N_E` ≈ 0.114 × the sum of CM7's noise PSD over the 400–700 Hz bins
  (HW3.0, 250 no-rain clips; 0.122 on white noise).

Sources (rain_anomaly_analysis; files from open PRs appear at these links once merged):
[`learnings/cm4_wind_noise_quantile_findings.md`](https://github.com/Arable/rain_anomaly_analysis/blob/main/learnings/cm4_wind_noise_quantile_findings.md) (PR #6),
[`learnings/cm4_pulse_excluded_noise_findings.md`](https://github.com/Arable/rain_anomaly_analysis/blob/main/learnings/cm4_pulse_excluded_noise_findings.md) (PR #6),
[`notebooks/rain_measurement/cm4_fw_vs_override_params.ipynb`](https://github.com/Arable/rain_anomaly_analysis/blob/main/notebooks/rain_measurement/cm4_fw_vs_override_params.ipynb) (PR #6),
[`presentations/summary_slides/slide_numbers_explained.md`](https://github.com/Arable/rain_anomaly_analysis/blob/main/presentations/summary_slides/slide_numbers_explained.md) (PR #6),
[`learnings/cm4_cm7_noise_scale_handoff.md`](https://github.com/Arable/rain_anomaly_analysis/blob/main/learnings/cm4_cm7_noise_scale_handoff.md) (PR #7).

## 4. Open points

- The estimator is single-band by design (`band_hz`). There is no per-mode-band CM4 noise estimate; a
  per-band version would need `BandNoiseEstimator` itself to be generalised.
- How the three reported rain rates (section 2, Status) are used downstream, and whether cloud
  precipitation uses only the raw DSD, hasn't been checked. Check the reporting path before comparing
  field data against either.
- The band-noise skip while the scratch area is in use (section 2, Status) is from reading the source;
  how often it happens on devices hasn't been measured.
- HW3.5 microphones and gain differ from HW3.0; none of the results above have been repeated on HW3.5.
