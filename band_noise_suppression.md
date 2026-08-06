# Band Noise Suppression for Rain Measurement

**Files:** `audio_processing_tools/edge/band_noise_estimator.py`, `audio_processing_tools/edge/band_noise_processor.py`

## What this is

A frame-causal noise estimator and suppressor scoped to the disdrometer's
primary resonance band (default 400–700 Hz) — the same band the drop-size
distribution (DSD) energy metric is measured from. Its job is to produce a
noise-suppressed version of that band-energy metric (`M_clean` / `M_clean_fft`)
so that wind and other slowly-varying background energy don't get counted as
drop energy.

This is a different, narrower module than the detector/suppressor pipeline in
`spectral_noise_processor.py` (documented in `edge/README.md`) — that one
operates on the full operating band (400–3500 Hz) across all resonance modes
for frame classification. This one is purpose-built for the single primary
band used by the legacy DSD/`drop_energy_level` calculation.

**Current integration status:** standalone. Nothing else in the repo imports
`BandNoiseEstimator` or `BandNoiseEstimatorProcessor` — it is not wired into
`RainDetectorProcessor.run()` or `rain_estimator.py`'s
`estimate_rain_from_audio()`. It exists today as a validated, firmware-parity
diagnostic (`M_clean_fft` vs. `disdrometer_legacy.c`'s un-suppressed
`drop_energy_level`), not yet part of the production rain-rate estimate.

---

## Two-level rain/noise frame decision (`NoiseFrameDetector`)

Before any noise learning happens, each frame's subframes must be classified
as rain or non-rain so that only non-rain energy feeds the noise estimate.
Two independent triggers, either of which marks a frame/subframe as rain:

1. **FFT-domain (whole-frame veto):** total rain-band power jump ≥ `M_db`
   (default 6 dB) **and** primary-band power jump ≥ `N_db` (default 3 dB)
   vs. the previous frame → the *entire* frame's subframes are marked rain.
2. **Time-domain (per-subframe, preferred trigger):** a dB-rise in the
   primary band (`ΔLb ≥ band_rise_db`) that exceeds the concurrent rise in
   overall (high-pass) loudness by `excess_rise_db` — i.e. a narrowband
   energy jump *relative to* broadband energy. This is the discriminator
   that separates a rain impact (excites the resonance band specifically)
   from wind (raises all frequencies together). A `k_subframes`-length hold
   extends the rain mask forward to cover decay tails.

Two legacy/experimental triggers (`use_dE_over_Ehpf`, `use_D_trigger`) exist
for backward compatibility but are off by default.

## Noise learning

- Non-rain subframe energies are pushed into a ring buffer of length `W`
  (default 30 subframes); a subframe only counts as "learned" if the
  detector didn't mark it rain (unless `learn_during_rain`/`force_learn_all`
  override this).
- After `W_min` valid samples, the noise estimate is the `q`-quantile
  (default 0.3) of the buffer, optionally EMA-smoothed via `ema_alpha`.
  The default `ema_alpha=1.0` means no smoothing actually happens out of
  the box (`noise_ema = (1-1)*old + 1*qv = qv`) — smoothing only kicks in
  if `ema_alpha` is lowered below 1.0.
- **TTL expiry** (`noise_buffer_ttl_frames`, default 200): buffer entries
  older than this many frames are expired even without new non-rain
  samples, so the estimate can't stay pinned to stale contents during
  sustained rain.
- **Replenishment fallback:** if a frame produces zero non-rain subframes
  (e.g. heavy sustained rain) and the buffer isn't full,
  `noise_replenish_from_all_subframes` optionally pushes a low-quantile
  (`noise_replenish_q`, default 0.20) value from *all* subframes — betting
  that even in heavy rain the lower-energy tail is background-like. The
  effective quantile (`noise_effective_q`) drifts toward the replenish
  quantile when this fires, and back toward `q` during normal learning.

## Suppression

Wiener-like gain from the primary-band time-domain energy `Eb` and the
estimated noise energy `N_E`:

```
G_pow = max(Eb - beta * N_E, 0) / (Eb + eps)
G_mag = clip(sqrt(G_pow), gain_floor, 1.0)
M_clean     = M_band     * G_mag   # suppressed time-domain BPF amplitude
M_clean_fft = M_band_fft * G_mag   # suppressed FFT magnitude sum
```

`gain_floor` (default 0.10) prevents full silence during over-suppression.
Optional outer EMA smoothing (`smooth_N_E`) uses an asymmetric attack —
slow rise during rain (`ne_attack_alpha_wet`), fast rise otherwise
(`ne_attack_alpha_dry`) — so mechanical noise is tracked quickly without
inflating the noise estimate mid-rain-event.

## Two FFT-band metrics, two different purposes

`M_band_fft`/`E_band_fft` use `legacy_band_bins()` (floor-based bin edges,
matching `disdrometer_legacy.h`'s integer-division macros exactly) rather
than the round-based `hz_to_bin()` used elsewhere in this file. This is
intentional: it makes the *unsuppressed* FFT magnitude sum directly
comparable to the legacy, pre-noise-cancellation `drop_energy_level` path,
for validating suppression's effect side-by-side against firmware. It is
**not** meant to match `band_noise_dsd.c`'s own diagnostics, which use a
different (RMS-style) binning convention.

## Telemetry (`BandNoiseEnergyStats`)

`process_frame()` accumulates rolling energy stats — `total_energy_sum`,
`rain_energy_sum`, `noise_energy_sum`, buffer-health counters
(`noise_buffer_underflow_frame_count`, `frames_since_noise_update`) — meant
to be read once per report interval via `read_and_reset_energy_stats()`
(e.g. once per minute on-device) and then cleared.

## `BandNoiseEstimatorProcessor` (batch adapter)

Thin `audio_processing_framework`-style wrapper: takes a full mono ndarray
+ params dict, builds a `BandNoiseEstimatorConfig` (supports nested
`det.*` overrides), frames the audio at `hop == frame_len` (required,
since the estimator carries IIR filter state across frames), and returns
`(results_dict, state_dict)`. `state_dict` has per-frame arrays for the
per-frame `BandNoiseFrameOut` fields (`M_band`, `E_band`, `N_E`, `subE`,
`G_mag`, `M_clean`, `M_clean_fft`, etc.) plus `results_dict` summary
medians; the `BandNoiseEnergyStats` fields (`noise_energy_sum`,
`rain_frame_count`, etc.) appear only once, as a single cumulative
snapshot under `state["energy_stats"]` — not as per-frame arrays.

## Key config knobs

| Param | Default | Effect |
|---|---|---|
| `band_hz` | (400, 700) | Primary band for energy/noise/suppression |
| `W` / `W_min` | 30 / 10 | Noise ring-buffer length / warmup requirement |
| `q` | 0.3 | Noise quantile |
| `noise_buffer_ttl_frames` | 200 | Max age before a buffer entry expires |
| `M_db` / `N_db` | 6.0 / 3.0 | FFT-domain rain veto thresholds (dB) |
| `band_rise_db` / `excess_rise_db` | 6.0 / 3.0 | TD onset trigger thresholds (dB) |
| `beta` | 1.0 | Oversubtraction factor in Wiener gain |
| `gain_floor` | 0.10 | Minimum suppression gain |
