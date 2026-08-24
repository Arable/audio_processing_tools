# Two bugs around `noise_psd` / per-mode-band noise estimation

**Files:** `edge/rain_signal_processor.py`, `edge/rain_frame_classifier.py`
**Status:** both bugs fixed and committed here (2026-08-24), following a code review. Originally
found and reverted (drafted, verified in isolation, then backed out) during a
`rain_anomaly_analysis` investigation into whether some other mode band (not the primary
400-700Hz band) might have consistently better SNR for rain measurement. See "Resolution" at the
bottom for what actually shipped, which differs from the drafted Bug 2 fix below.

## Context

CM7's rain-detection decision uses a real per-bin causal noise tracker
(`noise_tracker.py`'s `CausalNoiseTracker`) to normalize the spectrum before classification.
It's used in exactly the same causal, frame-by-frame, EMA-smoothed, rain-frame-excluded way in
both of its two call sites:

- `rain_frame_classifier.py`'s streaming `process_frame()` (line ~1995) — produces only an
  aggregate `noise_floor_db` scalar (mean over the whole operating band, not per mode band).
- `rain_signal_processor.py`'s `_estimate_noise_psd_fft()` (line 609) — the real batch-path
  usage. Confirmed by reading it directly: it's a `for t in range(T): tracker.update(...)` loop,
  i.e. the *same* causal sequential update as `process_frame()`, just run once over a whole
  clip's frames rather than truly online. Produces `detector_noise_psd` (line 797, full
  `(F, T)` per-bin PSD over `cfg.operating_band`, default 400-3500Hz) and a lagged copy
  `detector_noise_psd_lag` (line 804, shifted by one frame — this is the one that actually
  builds `P_for_detection`, the real input to the rain/no-rain decision, so it's the one that
  matches genuine causal/streaming semantics: at frame *t* you only had frame *t-1*'s estimate
  available).

So a real, sophisticated, per-bin noise PSD already exists and already drives production
detection. The two bugs below are both about the fact that none of it is actually reachable
outside `keep_debug`/`det_debug`.

## Bug 1 — `detector_noise_psd_lag` is computed but never passed into `_detect_rain_over_time()`

`_detect_rain_over_time()` (`rain_frame_classifier.py:290`) takes `noise_psd: Optional[np.ndarray]
= None` as a parameter — it does **not** compute it internally via `self._noise_tracker`, it
expects the caller to supply it. The one place inside that already reads it, if given
(`rain_energy_summary_enable` block, ~line 978), does:

```python
if noise_psd is not None:
    noise_psd_arr = np.asarray(noise_psd)
    if noise_psd_arr.shape == P.shape:
        noise_band_energy_t = np.sum(noise_psd_arr[rain_band_mask_energy, :], axis=0)
        ...  # stft_noise_band_energy_sum, stft_rain_minus_noise_energy_sum
```

But the real batch call site, `rain_signal_processor.py:826`:

```python
frame_class, rain_conf, det_debug, feature_dump = self._detect_rain_over_time(
    P_for_detection,
    freqs,
    detector_frame_times=np.asarray(times, dtype=work_dtype),
    input_audio=x,
    raw_power=P,
    work_dtype=work_dtype,
)
```

...never passes `noise_psd=...`, even though `detector_noise_psd_lag` is already computed and in
scope at that point (line 804, well before line 826). So `noise_psd` is always `None` in the
real pipeline today — which means `rain_energy_summary_enable`'s `stft_noise_band_energy_sum` /
`stft_rain_minus_noise_energy_sum` have never actually taken a non-zero value in production,
silently. Same dead-wiring shape as the bug documented in
`feature_dump_peak_and_envelope_wiring.md`, just one call-site removed instead of inside the
function body.

**Fix (drafted, reverted, not committed):**

```python
frame_class, rain_conf, det_debug, feature_dump = self._detect_rain_over_time(
    P_for_detection,
    freqs,
    detector_frame_times=np.asarray(times, dtype=work_dtype),
    input_audio=x,
    raw_power=P,
    noise_psd=detector_noise_psd_lag,   # <-- added
    work_dtype=work_dtype,
)
```

Use `detector_noise_psd_lag`, not the unlagged `detector_noise_psd` — the lagged one is what
`P_for_detection` (the actual detection input) is built from, so it's the version faithful to
real causal/streaming semantics (frame *t*'s noise estimate must only reflect frames up to
*t-1*). `detector_noise_psd_lag` defaults to `None` (set at line 773) whenever
`detector_use_noise_norm` is `False`, matching `noise_psd`'s existing `None`-safe default — so
this is safe to pass unconditionally.

This fix on its own is purely additive: it only feeds an already-existing, already-flagged
consumer (`rain_energy_summary_enable`, default `False`) that currently silently no-ops. It does
not change `frame_class`/`rain_conf`/any detection decision, so it shouldn't perturb golden
regression baselines.

## Bug 2 — no way to get *per-mode-band* noise energy into the lean feature dump

Even with Bug 1 fixed, `rain_energy_summary_enable`'s existing consumer only sums noise energy
over one fixed band (`rain_energy_summary_band`, default the primary 400-700Hz band) into a
clip-level scalar. There's no per-mode-band breakdown, and no way to get any form of
`detector_noise_psd`/`detector_noise_psd_lag` into `fd_dense` (the lean, persisted feature-dump
payload) — full-resolution `detector_noise_psd`/`detector_noise_psd_lag` reach only `det_debug`
(gated by `keep_debug`, lines ~977 and ~1166), same expensive-duplication trap as
`feature_dump_peak_and_envelope_wiring.md`'s Fix 1/Fix 2 (~375KB/clip vs a few hundred KB/clip
baseline).

This matters for the investigation that surfaced it: comparing SNR across the five mode bands
(400-700 / 800-1050 / 1500-1800 / 2350-2550 / 3150-3350 Hz) to see whether some non-primary band
has consistently better signal-to-noise than the primary band currently used for rain
measurement — which needs the real per-bin `detector_noise_psd_lag`, aggregated per mode band,
not a proxy.

**Fix (drafted, reverted, not committed)**, same shape as Fix 1/Fix 2 in
`feature_dump_peak_and_envelope_wiring.md` — a new flag,
`feature_dump_include_noise_psd_by_mode` (default `False`), read alongside the other
`feature_dump_include_*` flags (~line 334):

```python
feature_dump_include_noise_psd_by_mode = bool(
    self._dget("feature_dump_include_noise_psd_by_mode", False)
)
```

...and a block inside `if feature_dump_dense_enable:`, right after the existing `td_envelope`
block (~line 1149), aggregating `noise_psd` into the same five ranges as `mode_bands` (mirroring
`peak_count_by_mode_{i}`'s per-index naming convention):

```python
if feature_dump_include_noise_psd_by_mode and noise_psd is not None:
    noise_psd_arr = np.asarray(noise_psd)
    if noise_psd_arr.shape == P.shape:
        for i, (mode_lo, mode_hi) in enumerate(mode_bands):
            mode_mask_full = (freqs >= mode_lo) & (freqs <= mode_hi)
            fd_dense[f"noise_energy_by_mode_{i}"] = np.sum(
                noise_psd_arr[mode_mask_full, :], axis=0
            ).astype(dtype)
```

Note this uses a fresh mask against the full `freqs` array (matching the existing
`rain_band_mask_energy` pattern a few lines above at ~972), not the function's earlier
`mode_masks` (those are relative to `freqs_band`, restricted to `operating_band`, and index into
`P_band` rather than the full-width `noise_psd`/`P` arrays) — `noise_psd_arr.shape == P.shape` is
full-width, so the mask needs to be too.

Depends on Bug 1's fix (`noise_psd` is otherwise always `None` in production) to produce anything
in the real pipeline; the flag alone is a no-op without it.

## Suggested verification, once both are applied here

Same shape as Fix 1's verification in `feature_dump_peak_and_envelope_wiring.md`: a 20-clip smoke
run (`rain_anomaly_analysis`'s `query_rain_peaks_smoke`) with
`feature_dump_include_noise_psd_by_mode=True` (and `detector_use_noise_norm=True`, already the
default) added to detector params, `keep_state_debug` left at its default `False`, confirming
`noise_energy_by_mode_0..4` land in `features` with plausible nonzero values and the dump size
stays at the lean baseline, not the `det_debug`-inflated one.

## Downstream use (once wired)

The eventual per-band SNR comparison this was for: divide each mode band's signal energy
(already available today via `primary_mode_flux`/`support_mode_flux_1..4`, or the raw per-band
sums used in `rain_anomaly_analysis/notebooks/rain_analysis/spectral_leakage_vs_rain_intensity.ipynb`)
by `noise_energy_by_mode_{i}` from this fix, per band, per clip — a production-faithful SNR
signal, replacing the `clip_spectral_occupancy`-based `no_rain_log_power_mean` proxy used as a
stand-in before this fix existed.

## Resolution (2026-08-24) — what actually shipped

**Bug 1** was applied exactly as drafted: `rain_signal_processor.py`'s
`_detect_rain_over_time()` call now passes `noise_psd=detector_noise_psd_lag`.

**Bug 2 shipped as a clip-level summary, not the drafted per-frame `fd_dense` array.** The
downstream use case (per-band SNR, once per clip) doesn't need a `(5, T)` diagnostic — a single
`mode_snr_summary` dict, gated by a new `mode_snr_summary_enable` flag (default `False`), is
computed once per clip and reaches `det_debug["mode_snr_summary"]` and, when
`feature_dump_clip_summary_enable=True`, `feature_dump["mode_snr_summary"]` (the `fd_clip_summary`
tier, alongside `clip_spectral_occupancy` — not `fd_dense`). Per mode band `i`:
`mode_signal_energy_sum_i`, `mode_noise_energy_sum_i`, `mode_snr_i` (linear), `mode_snr_db_i`.

A code review (human + Claude, 2026-08-24) found three issues in the first draft of this fix,
all corrected before commit:

- **`rain_energy_summary` was activated into a dimensionally-inconsistent path.** Once Bug 1's
  fix makes `noise_psd` non-`None` in production, the existing `rain_energy_summary_enable`
  block's `rain_band_energy_t - noise_band_energy_t` mixed `P` (dB-scale once
  `detector_use_noise_norm` normalizes it) with `noise_psd` (linear) — physically meaningless.
  Fixed by switching that block to `raw_power` (linear, matching `noise_psd`'s scale), the same
  reasoning already applied to the new SNR feature; added a `raw_power is not None` guard.
- **`mode_snr_*` was computing `(S+N)/N`, not `S/N`.** `raw_power` is observed signal+noise
  power, not signal power alone. Fixed by recovering signal energy as
  `max(raw_power_sum - noise_energy_sum, 0.0)` before dividing, so a noise-only clip reads near
  the `eps` floor instead of floor-clamped-at-1×.
  (`detector_noise_psd_lag`'s own clamp against `maxr_det * P` means the naive ratio could never
  fall below ~1 by construction — that clamp is fine for the detector-normalization path it was
  built for, but made the naive `raw_power/noise_psd` ratio meaningless as an "SNR".)
- **Missing/invalid `noise_psd` produced an artificially huge (but finite) SNR** via
  `signal_energy_sum / max(0.0, eps)`. Fixed to emit no numeric summary at all in that case —
  `det_debug["mode_snr_summary_error"]` is set instead, and nothing is written to `feature_dump`
  — rather than risk silently contaminating persisted training data with a fake large value.

Verified with synthetic data covering: a real signal+noise clip (SNR reads sensibly above 0dB,
distinct from the old `(S+N)/N` numbers), a noise-only clip (SNR reads at the `eps` floor, not an
inflated ≥1 ratio), and `noise_psd=None` (no `mode_snr_summary` in `det_debug` or `feature_dump`,
`mode_snr_summary_error` present instead).

**Still outstanding:** the real-clip smoke test (`rain_anomaly_analysis`'s
`query_rain_peaks_smoke`) — should specifically check rain, dry/noise, and low-SNR clips, not
just confirm plausible-looking values appear.
