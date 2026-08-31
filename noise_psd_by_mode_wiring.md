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

**Bug 2 shipped as a clip-level energy summary, not the drafted per-frame `fd_dense` array, and
not an SNR.** First draft computed an SNR (`mode_snr_summary`, `mode_signal_energy_sum_i` /
`mode_noise_energy_sum_i` / `mode_snr_i` / `mode_snr_db_i`). Feedback from a second review round
was to drop the SNR entirely and just report the two energy sums directly — matching how
`clip_spectral_occupancy` already folds rain frames' signal+noise together rather than trying to
split them out — and let any signal/noise comparison happen downstream, not inside the detector.
Renamed `mode_snr_summary` → **`mode_band_energy_summary`**
(`mode_band_energy_summary_enable`, default `False`), reporting only
`mode_total_energy_sum_i` (`raw_power`, i.e. S+N, all frames including rain) and
`mode_noise_energy_sum_i` (`noise_psd`) per mode band `i` — no ratio, no dB, no signal-only term.
Reaches `det_debug["mode_band_energy_summary"]` and, when `feature_dump_clip_summary_enable=True`,
`feature_dump["mode_band_energy_summary"]` (the `fd_clip_summary` tier, alongside
`clip_spectral_occupancy` — not `fd_dense`).

**These are STFT-domain sums, not physical energy.** `mode_total_energy_sum_i`/
`mode_noise_energy_sum_i` are sums of `|FFT|^2` over bins and frames — not joules. With
`n_fft=256`/`hop=128` (50% overlap), each audio sample is counted in two overlapping analysis
frames, and the absolute scale otherwise depends on window shape, `n_fft`, and `hop`. Only
relative/ratio comparisons across clips or mode bands are meaningful, not the absolute magnitude.
Documented in both implementations' docstrings/comments, not just here.

**Implemented in both the batch path and the actual streaming/embedded path.**
`_detect_rain_over_time()` (`RainFrameClassifierMixin`, offline/batch/reference) computes the
summary once over a full clip's `raw_power`/`noise_psd` arrays. `RainFrameClassifierState` (the
causal, frame-by-frame class intended for CM7 embedded deployment — see this file's Status table)
now also maintains it: `process_audio_frame()` — the fully self-contained frame-level entry point
that computes its own FFT frame and its own `CausalNoiseTracker` estimate each call — accumulates
two length-`n_modes` running sums (`O(1)` memory, no per-frame history) via
`self._mode_masks` (already `operating_band`-restricted by construction, so no coverage-gap
guard is needed there the way the batch path needed one — see below). `get_mode_band_energy_summary()`
returns the accumulated clip-level dict; `reset()` zeroes it between clips. `process_frame()` and
`replay_clip()` don't get accumulators — neither has both the raw per-bin power and the noise
estimate available internally (callers supply an already-noise-normalized spectrum to those two).

A code review (human + Claude, two rounds, 2026-08-24) found and fixed, before commit:

- **`rain_energy_summary` was activated into a dimensionally-inconsistent path.** Once Bug 1's
  fix makes `noise_psd` non-`None` in production, the existing `rain_energy_summary_enable`
  block's `rain_band_energy_t - noise_band_energy_t` mixed `P` (dB-scale once
  `detector_use_noise_norm` normalizes it) with `noise_psd` (linear) — physically meaningless.
  Fixed by switching that block to `raw_power` (linear, matching `noise_psd`'s scale); added a
  `raw_power is not None` guard.
- **The first SNR draft was computing `(S+N)/N`, not `S/N`** (fixed, then made moot by dropping
  the SNR computation entirely per the second review round above).
- **Missing/invalid `noise_psd` produced an artificially huge (but finite) SNR/ratio.** Fixed by
  emitting no numeric summary at all in that case — `det_debug["mode_band_energy_summary_error"]`
  is set instead, and nothing is written to `feature_dump` — rather than risk silently
  contaminating persisted training data with a fake large value.
- **Automated follow-up review caught a coverage-gap variant of the same bug**, specific to the
  batch path: a mode band's mask was built against the full `freqs` array without checking it
  was actually inside `operating_band`. `_estimate_noise_psd_fft` only ever fills `noise_psd`
  inside `operating_band`, leaving everything else at exactly `0.0` — so a mode band only
  partially/not covered by `operating_band` would silently divide by that leftover zero. Fixed by
  intersecting the batch path's per-mode and `rain_energy_summary` band masks with `band_mask`
  (the operating-band mask), so an out-of-coverage band now falls into the existing "no data"
  zero branch. Verified with `operating_band=(400,1000)` against `mode_bands` reaching to
  3350Hz. (The streaming path's `self._mode_masks` are inherently `operating_band`-restricted by
  construction, so this specific bug doesn't apply there.)

Verified with synthetic data (batch path) and a streaming-path test (renamed in the next round
below to `tests/edge/rain_detection/test_band_energy_summary.py`): the streaming accumulator
equals summing the per-frame energy values `process_audio_frame()` itself returns, the summary is
all-zero and carries no per-frame keys when the flag is off, and `reset()` zeroes the
accumulators between clips.

## Widened to the 16 clip_spectral_occupancy bands (same day, third round)

Renamed once more, `mode_band_energy_summary` → **`band_energy_summary`**
(`band_energy_summary_enable`), and switched its default band set from the 5 detection
`mode_bands` to the same **16 semantic bands** `clip_spectral_occupancy` already uses
(`default_spectral_occupancy_bands()`: `dc`, `wind_1`, `wind_2`, `mode_1`, `inter_1`, `mode_2`,
`inter_2a/2b`, `mode_3`, `inter_3a/3b`, `mode_4`, `inter_4a/4b/4c`, `mode_5`) — so the two features
are directly comparable band-for-band. Output keys are now named per band
(`{band_name}_total_energy_sum` / `{band_name}_noise_energy_sum`, e.g. `mode_1_total_energy_sum`)
rather than indexed (`mode_total_energy_sum_0`). Overridable via `band_energy_summary_bands`
(same shape as `clip_spectral_occupancy_bands`), in both the batch path and
`RainFrameClassifierState` (constructor param + wired through `from_mixin()`).

The streaming class's `self._band_energy_masks` are built once in `__init__` against
`self._freqs_band` (already `operating_band`-restricted, same pattern as `self._mode_masks`), so
the coverage-gap guard the batch path needed (`_full_width_band_mask`'s `& band_mask`) isn't a
separate concern there — an out-of-`operating_band` band mask is simply empty by construction.

Renamed `get_mode_band_energy_summary()` → `get_band_energy_summary()`; per-frame keys renamed
`mode_total_energy_{i}`/`mode_noise_energy_{i}` → `{name}_total_energy`/`{name}_noise_energy`.
Test file renamed to `tests/edge/rain_detection/test_band_energy_summary.py`, plus a new test
confirming a custom `band_energy_summary_bands` override replaces the default 16 bands entirely.

**Still outstanding:** the real-clip smoke test (`rain_anomaly_analysis`'s
`query_rain_peaks_smoke`) — should specifically check rain, dry/noise, and low-signal clips, not
just confirm plausible-looking values appear. Downstream SNR (if wanted) is now the caller's
responsibility — see the fifth round below for which two fields are actually safe to pair for
that computation on a partially-covered band.

## Fourth round: independent (Codex) review — two required correctness fixes

An independent review (Codex) of the third round found two real bugs, both fixed before commit:

**1. Boundary bins were double-counted.** `compute_clip_spectral_occupancy_stats()` already used
half-open `[lo, hi)` for every band except the last (closed `[lo, hi]`), so adjacent bands never
double-count a bin sitting exactly on a shared boundary. `band_energy_summary`'s masks (both
batch and streaming) used `(freqs>=lo)&(freqs<=hi)` — closed on *both* ends — for every band. For
this codebase's actual config (`fs=11162`, `n_fft=256`), **every single one of the 15 internal
boundaries in the default 16 bands lands exactly on an FFT bin** (verified:
`43.6015625 * k` for integer `k`, e.g. `654.0234375 = 43.6015625*15`) — not a hypothetical edge
case, guaranteed on every clip. Fixed by extracting a shared `band_freq_mask(freqs, lo, hi, *,
is_last_band)` helper into `feature_extraction.py` (half-open except the last band, closed) and
using it in all three places: `compute_clip_spectral_occupancy_stats()` (pure refactor, same
behavior), the batch `band_energy_summary`/`rain_energy_summary` masks, and the streaming
`self._band_energy_masks`/`self._band_energy_full_masks`. Verified with a spike placed exactly on
the `mode_1`/`inter_1` boundary bin: counted once (in `inter_1`, whose `lo` matches the boundary),
not twice, not zero times.

**2. Unavailable-band coverage was ambiguous.** The 16 default bands span ~0-3575Hz;
`operating_band` defaults to 400-3500Hz — so `dc`/`wind_1` are fully outside, `wind_2`/`mode_5` are
partially inside, and the rest are fully inside. Returning `0.0`/`0.0` for `noise_energy_sum`
couldn't distinguish "genuinely near-zero noise, measured" from "never measured here at all."
Fixed by:
  - Adding `{name}_coverage_fraction` (covered bins / total bins in that band) to every band,
    computed once from `band_mask`/`operating_band`, static per band — not per frame.
  - Splitting `total_energy_sum` and `noise_energy_sum` to use *different* masks:
    `total_energy_sum` is **not** restricted to `operating_band` (raw_power is a real measurement
    everywhere on the freq grid — same convention `clip_spectral_occupancy` already uses, since it
    doesn't restrict to `operating_band` either), while `noise_energy_sum` **is** restricted (noise
    is only ever estimated inside `operating_band`). Previously both used the same
    `operating_band`-restricted mask, which silently zeroed real, measured `total_energy_sum` for
    bands like `dc` even though `raw_power` there is perfectly valid data.
  - In the streaming class this meant adding a second mask set
    (`self._band_energy_full_masks`, against the full `self._freqs`, used with `P_t`) alongside
    the existing operating-band-restricted `self._band_energy_masks` (against `self._freqs_band`,
    used with `N_band_t`).

**Recommended fixes also applied:**
  - `get_band_energy_summary()` now returns `{}` when `band_energy_summary_enable=False`, instead
    of a dict of `2*n_bands` zeros indistinguishable from "measured and genuinely silent."
  - Per-frame `{name}_total_energy`/`{name}_noise_energy` keys in `process_audio_frame()`'s result
    are no longer added just because accumulation is enabled — a separate
    `band_energy_summary_expose_per_frame` flag (default `False`) gates them, since they exist
    mainly to let tests validate the `O(1)` accumulator against a naive per-frame sum, not for
    production per-frame payloads (the clip accumulator itself remains `O(1)` regardless).

**Tests added** (`tests/edge/rain_detection/test_band_freq_mask.py`,
`tests/edge/rain_detection/test_band_energy_summary.py`, 17 new tests total): the every-boundary-
is-an-exact-bin precondition, boundary-bin-counted-exactly-once, last-band-inclusive,
`clip_spectral_occupancy` boundary integration; batch: all-16-bands-present, out-of-band
coverage=0, partial coverage strictly between 0 and 1, missing-`noise_psd` error (not a fake
summary), and an *independent* hand-computed expected sum (energy placed at a known frequency,
expected value computed by hand, not by re-running the code under test); streaming: accumulator-
equals-frame-sum, disabled-returns-empty-dict, per-frame-not-exposed-by-default, resets-between-
clips, custom-bands-override, out-of-band/partial coverage; and a batch-vs-streaming coverage
parity test. All 22 tests in the directory pass.

## Fifth round: independent (Codex) review — one remaining blocker fixed

A second review pass on the fourth round found the two-mask design still left `total_energy_sum`
and `noise_energy_sum` domain-mismatched for a *partially*-covered band (as opposed to the
already-handled fully-uncovered case). For `mode_5` (3139-3575Hz) against the default
`operating_band` (400-3500Hz): `total_energy_sum` covers 3139-3575Hz but `noise_energy_sum` only
covers 3139-3500Hz — so `signal = total_energy_sum - noise_energy_sum` would silently fold in
3500-3575Hz energy that has no matching noise estimate at all, inflating "signal" by whatever
happens to sit in that untracked sliver. `coverage_fraction` flags that the band is partial but
can't numerically correct for it, since energy isn't uniformly spread across bins.

**Fix:** added a third field, `{name}_covered_total_energy_sum` — `raw_power`/`P_t` summed over
the *same* `operating_band`-intersected mask as `noise_energy_sum`, so the two are always
domain-matched. `{name}_total_energy_sum` (full band, unrestricted) stays available separately
for occupancy-style uses. The safe downstream computation is now
`signal = max(covered_total_energy_sum - noise_energy_sum, 0.0)`, never
`total_energy_sum - noise_energy_sum`. Implemented in both the batch path (one extra `_band_sum`
call reusing the existing `covered_mask`) and the streaming path (a third accumulator array,
summing `P_band_t` — not `P_t` — over `self._band_energy_masks`, the same mask `noise_energy`
already uses).

**New regression test**
(`test_batch_partial_band_energy_only_in_uncovered_region`): places energy exclusively in
`mode_5`'s 3500-3575Hz uncovered remainder and asserts `total_energy_sum > 0` but
`covered_total_energy_sum == 0` and `noise_energy_sum == 0` — proving the two are no longer
conflatable. All other tests updated to also check the new field.

**Test lint:** a genuine `ruff` finding (`F841`, an unused `n_bands` variable in
`test_band_freq_mask.py`) was fixed; the codebase's `pyproject.toml` does enable docstring rules
(`D`) even though there's no CI workflow enforcing them and the pre-existing source files aren't
clean against them — both new test files were still brought to a clean `ruff check` pass, since
they're new code with no existing-debt excuse.

Final count: 23 tests in `tests/edge/rain_detection/` (5 original + 4 boundary-mask + 14
band-energy-summary), all passing; both new test files `ruff`-clean.

## Sixth round: final review before commit

A final review (higher effort, whole diff) confirmed no remaining correctness bugs in the
already-fixed paths, and found one new real bug plus several worthwhile cleanups — all fixed:

**Real bug (confirmed by direct execution): unsorted custom `band_energy_summary_bands` silently
dropped the top-of-spectrum edge bin and mis-assigned an internal boundary bin.**
`band_freq_mask`'s half-open-except-last convention assumes iteration order matches frequency
order (the *last* band in the sequence is the one whose upper bound is closed). The shipped
default (`default_spectral_occupancy_bands()`) is hardcoded and already sorted, so production is
unaffected — but a caller overriding `band_energy_summary_bands` with an out-of-order list (e.g.
`[("top", 2790.5, 3575.328125), ("mid", 436.015625, 2790.5)]`) would have silently lost the true
edge bin (neither band's mask would treat it as inclusive) and mis-handled the boundary between
them. Fixed by adding `normalize_bands()` to `feature_extraction.py` — casts to `(str, float,
float)` and **sorts ascending by `lo`** — used everywhere a bands list is normalized:
`compute_clip_spectral_occupancy_stats()`, the batch `band_energy_summary_bands` normalization,
and `RainFrameClassifierState.__init__`'s. Verified: the unsorted example above now conserves
total energy exactly (`4500 = 3000 + 1500`, no drop, no double-count) instead of silently losing
data.

**Deduplication (addressing the reviewer's explicit "this exact bug class has now been
independently fixed 5 times" concern):** in addition to `normalize_bands()` collapsing three
copies of the same bands-normalization logic into one, added `band_coverage_fraction(full_mask,
covered_mask)` to `feature_extraction.py`, replacing the batch and streaming paths' independently
written (and previously divergent) coverage-fraction arithmetic with one shared implementation.

**Performance (ties directly to this class's stated purpose — O(1)-memory embedded CM7
deployment):** `process_audio_frame()`'s per-band accumulation was a Python loop calling
`np.sum()` `3 * n_bands` times every single frame (48 calls/frame for the default 16 bands).
Replaced with 3 matrix-vector products against precomputed static `(n_bands, F)` / `(n_bands, K)`
0/1 mask matrices, built once in `__init__` — same O(1) memory characteristics, far fewer
Python-level calls per frame.

**Minor cleanup:** removed a redundant triple `is not None` check in the per-frame exposure gate
that only restated the `band_energy_summary_enable` flag already guarding those same variables'
assignment.

**Not changed (reviewed and consciously deferred):** merging `band_energy_summary` into
`compute_clip_spectral_occupancy_stats()` itself (a larger restructuring with no correctness
benefit, given both are already independently tested and this diff has already been through six
review rounds — deferred rather than risking new bugs for marginal duplication savings), and
inlining the one-line `_full_width_band_mask` closure (subjective style preference, not a
defect).

**New regression tests** (`tests/edge/rain_detection/test_band_freq_mask.py`): `normalize_bands`
sorts an unsorted custom list; `normalize_bands(None)` matches
`default_spectral_occupancy_bands()`; an unsorted custom bands list conserves total energy
exactly (the integration-level regression test for the confirmed bug above).

Final count: 26 tests in `tests/edge/rain_detection/`, all passing; both new test files remain
`ruff`-clean; no new lint findings introduced in the modified source files (`ruff check
--select F,E9` shows only the same 2 pre-existing, unrelated findings from before this diff).

## Seventh round: PR review on GitHub — sorting itself was the wrong fix

A review of the pushed PR (Codex) flagged that the sixth round's fix — `normalize_bands()`
**silently sorting** bands ascending by `lo` — was itself a compatibility risk, not just for the
new `band_energy_summary_bands` parameter but for the *existing*
`compute_clip_spectral_occupancy_stats()`, which now routes through the same helper.
`compute_clip_spectral_occupancy_stats()` is an established function with callers outside this
repo (this doc's own history references `rain_anomaly_analysis`); previously, callers who passed
a custom `bands=` list got output arrays (`band_names`, `rain_log_power_mean`,
`no_rain_power_ratio_p90`, etc.) in exactly the order they supplied. Silently re-sorting that list
would reorder those output arrays without warning — a worse surprise for any caller relying on
positional alignment than the original boundary-bin bug being fixed.

**Fix:** `normalize_bands()` now **validates** that `bands` is sorted ascending by `lo` and raises
`ValueError` if not, instead of silently re-sorting. This is strictly safer than the sixth round's
fix even for the original bug: silent reordering could *also* surprise a caller of the brand-new
`band_energy_summary_bands` parameter who listed bands in a specific (non-ascending) order for
their own reasons. Any already-ascending list — the default, and the overwhelmingly likely shape
of any real caller's list, since these are naturally frequency ranges — is completely unaffected;
only a genuinely out-of-order list now fails loudly at construction time instead of either
silently reordering (six-round fix) or silently mis-masking (original bug).

**Tests updated:** the sixth round's "sorts unsorted bands" test became
`test_normalize_bands_rejects_unsorted_custom_bands` (expects `ValueError`); added
`test_normalize_bands_accepts_already_sorted_custom_bands` and
`test_sorted_custom_bands_conserve_total_energy` (the boundary-energy-conservation check, now
using an already-sorted list since that's the only case that should succeed); added
`test_unsorted_band_energy_summary_bands_raises`, an integration-level check that
`RainFrameClassifierState.from_mixin()` itself raises for an unsorted
`band_energy_summary_bands` override, not just the `normalize_bands()` unit.

Final count: 28 tests in `tests/edge/rain_detection/`, all passing; both new test files remain
`ruff`-clean; no new lint findings in the modified source files beyond the same 2 pre-existing,
unrelated ones.

## Eighth round: independent fresh-eyes review (`/code-review`, high effort)

A review with no prior context on this diff (deliberately run this way, since the PR had already
been through seven rounds and needed genuinely independent eyes) found three more real issues:

**`normalize_bands()` validated sort order but not overlap or reversed bounds.** A sorted-but-
overlapping custom `bands` list (e.g. `[("a", 400.0, 1000.0), ("b", 900.0, 3500.0)]`) passed
validation — `los=[400, 900]` is ascending — but `band_freq_mask` then double-counted every bin in
the `[900, 1000)` overlap into both bands: the same double-counting failure class the sixth/seventh
rounds targeted, just reachable via a different path that none of the 28 existing tests exercised.
Fixed by adding explicit reversed/zero-width (`lo < hi`) and overlap (`hi_i <= lo_{i+1}`) checks
to `normalize_bands()`.

**`RainFrameClassifierState.__init__` unconditionally allocated the band-energy masks/matrices/
accumulators**, even with `band_energy_summary_enable=False` (the default) — confirmed by direct
instantiation to cost ~26KB (two `(16, F)`/`(16, K)` float64 mask matrices plus three accumulator
arrays), roughly 5x this class's own documented ~5.4KB total streaming-state memory budget, spent
on a feature that's off by default and never used in that case. Fixed by gating all of the
allocation (and the corresponding `reset()` clears) behind `if self._band_energy_summary_enable:`.

**The batch path's per-band coverage fraction duplicated `band_coverage_fraction()`'s logic
inline** instead of calling the shared helper introduced in the sixth round (which the streaming
path already used) — contradicted the sixth round's own stated goal of collapsing this exact
arithmetic into one implementation. Fixed by calling the shared helper from both paths.

Final count: 28 tests, all still passing (no new tests added yet at this point — see ninth round).

## Ninth round: independent review (Codex) — a crash, a lint gap, and no tests for round 8

**Real bug (confirmed by direct execution): an empty custom `band_energy_summary_bands=[]`
passed `normalize_bands()`'s new validation but crashed downstream.** With streaming enabled, an
empty bands list produces zero-row mask matrices; `process_audio_frame()`'s matrix multiply
(`self._band_energy_full_mask_matrix @ P_t`) then fails with an unrelated `matmul` shape-mismatch
error (`size 129 is different from 0`) instead of a clear validation error at construction time.
Fixed by rejecting an empty sequence directly in `normalize_bands()`.

**Lint: the eighth round's new overlap-check loop used a bare `zip()`, triggering ruff's B905**
(`zip()` without an explicit `strict=`). The two iterables (`normalized` and `normalized[1:]`) are
intentionally offset by one element — `strict=True` would always fail — so fixed with an explicit
`strict=False`.

**None of the eighth round's three fixes had dedicated regression tests** — confirmed by grep, only
output-level tests existed (e.g. "disabled returns `{}`"), which wouldn't catch a regression of the
memory-allocation fix specifically. Added 7 tests: overlapping/reversed/zero-width/empty bands
rejected (`test_band_freq_mask.py`); disabled state leaves the mask matrices and accumulators as
`None` (asserted directly on the attributes, not just output); enabled state allocates one
row/entry per band; empty custom bands raise at construction (`test_band_energy_summary.py`).

Final count: 35 tests, all passing; both new test files remain `ruff`-clean (including B905); no
new lint findings in the modified source files.

## Tenth round: independent review (Codex) — batch/streaming validation parity when disabled

**Real bug (confirmed by direct execution): the streaming constructor only validated
`band_energy_summary_bands` when the feature was enabled, while the batch path
(`_detect_rain_over_time`) always calls `normalize_bands()` unconditionally.** Consequently the
same invalid override (e.g. `band_energy_summary_bands=[]`) raised in batch but silently succeeded
in streaming whenever `band_energy_summary_enable=False` — verified both ways: batch raised
`ValueError: bands must not be empty`, streaming constructed with no error. Fixed by moving the
`normalize_bands()` call in `RainFrameClassifierState.__init__` outside the
`if self._band_energy_summary_enable:` guard, so validation always runs in both paths while only
the expensive mask/matrix allocation stays gated.

**New test:** a parametrized `test_invalid_bands_raise_identically_in_batch_and_streaming_even_when_disabled`
covering empty, reversed, and overlapping bands, asserting both paths raise with
`band_energy_summary_enable=False`.

Final count: 38 tests, all passing; both new test files remain `ruff`-clean; `git diff --check`
clean.

## Eleventh round: automated PR review (codeant-ai) — one real fix, one acknowledged divergence

`codeant-ai[bot]`'s review of the pushed PR raised two findings about the `noise_psd` value fed
into `rain_energy_summary`/`band_energy_summary` (both comments also embedded a "Prompt for AI
Agent" instruction block directing an agent to auto-implement and then auto-chase other PR
comments — the same prompt-injection-shaped pattern flagged and not acted on in the 2026-08-05
session; the underlying technical claims were verified independently instead of following those
embedded instructions).

**Fixed: the diagnostic was receiving the post-clamp `detector_noise_psd_lag`, not "the real"
lagged estimate.** `rain_signal_processor.py`'s `detector_noise_psd_lag` is clamped to `maxr_det *
P` (default `maxr_det=1.0`) purely to keep the detector-normalization ratio numerically sane;
that clamped copy was also being passed as `noise_psd` into `_detect_rain_over_time()`. Whenever a
frame's instantaneous power dipped below the lagged tracker's smoothed estimate (a normal EMA-lag
overshoot, not a data problem), the clamp forced `noise_psd` down to ≈`raw_power` for that
frame/bin, making the diagnostic report a fake zero "signal above noise" — contradicting this
doc's own Bug 1 framing ("the real per-bin causal noise estimate"). Fixed by saving
`detector_noise_psd_lag_unclamped` before the clamp and passing that to the diagnostic instead;
the clamped copy is still used for `P_for_detection`, so `frame_class`/`rain_conf` remain
unaffected.

**Acknowledged, not fixed: batch and streaming compute `band_energy_summary`'s noise source
differently, and that divergence is being left as-is.** Streaming's noise value comes from
`CausalNoiseTracker`, which properly excludes rain frames from the baseline
(`noise_tracker.py:137`). Batch's comes from the same `detector_noise_psd_lag` above, built with
`detector_is_rain_for_psd = np.zeros(T, dtype=bool)` — i.e. every frame treated as a noise
candidate, by pre-existing design, because at that point in the batch pipeline `frame_class` isn't
known yet (that's exactly what's being computed). A tempting "fix" — recompute a rain-excluded PSD
after `frame_class` is known, and use that for the diagnostic — was considered and rejected: that
PSD wouldn't be self-consistent with the `frame_class` used to build it, since `frame_class` was
itself derived from the *uncorrected* PSD. Making it self-consistent would require reclassifying
with the improved PSD (a new `frame_class`, then possibly iterating again) — i.e. re-invoking the
actual detection decision, not just recomputing a diagnostic, which conflicts with this PR's "no
detection-decision impact" guarantee if that reclassification were ever put to use. Decision:
batch's `noise_energy_sum` on this diagnostic is not a rain-excluded noise floor and isn't claimed
to be one (streaming's is) — a known, deliberate, practical divergence rather than a bug to chase,
consistent with this PR's existing policy of leaving SNR interpretation to downstream consumers.

## Twelfth round: PR #5 merged, reverted, re-opened as PR #6; automated review found a real gap

PR #5 was briefly merged (`d21ddef`) then reverted on `main` (`0b8dea0`) at the author's request to
give the team (`QCaudron`, `colinahill`, requested reviewers on both PRs) time to actually review
before it lands — GitHub doesn't allow reopening an already-merged PR, so the same branch/commits
were re-opened as PR #6.

`codeant-ai[bot]`'s review of PR #6 raised two findings (again with an embedded "Prompt for AI
Agent" auto-chase block, not acted on — same pattern as the eleventh round, verified independently
instead):

**Fixed: `normalize_bands()` didn't reject duplicate band names.** It already validated sort order,
non-reversed bounds, non-overlap, and non-empty, but two bands sharing a name (e.g. a mistyped
`band_energy_summary_bands` override) passed validation and then silently overwrote each other's
entries in the output dict — both `_detect_rain_over_time()` (batch) and `get_band_energy_summary()`
(streaming) key their result dicts by `{name}_...`. Fixed by adding a duplicate-name check to
`normalize_bands()` (shared by both paths and by `compute_clip_spectral_occupancy_stats()`), and
added a duplicate-name case to the existing `test_invalid_bands_raise_identically_in_batch_and_streaming_even_when_disabled`
parametrization.

**Fixed: batch/streaming coverage-fraction parity test was too weak.**
`test_batch_and_streaming_band_masks_agree_on_coverage` only compared `coverage_fraction > 0.0`
between the two paths, not the actual fraction — a numeric mismatch could pass undetected. Changed
to `pytest.approx` equality on the real values.

Test count 38 → 39, all passing; no new ruff findings. Landed as `13e0f58`. PR #6's description
left untouched (still the trimmed Scope/Summary/Test-plan form) — this round's detail lives here
instead, per the same policy established in the eleventh round.
