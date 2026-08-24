# Feature-Dump Wiring: Peak Summary + TD Envelope

**File:** `edge/rain_frame_classifier.py`
**Branch:** `main` (both fixes applied; peak-summary verified against real data, td-envelope not yet smoke-tested)

## Scope of this doc

Two config flags in `_detect_rain_over_time()` were read from config but never
actually connected to anything — a "dead flag" bug, same shape in both
cases. This doc covers both: `feature_dump_include_peak_summary` (fixed,
verified against real data) and `feature_dump_include_td_envelope` (same
bug, same fix applied below, not yet smoke-tested). Found while trying to
get `peak_count_by_mode` (per-mode-band spectral peak counts, a candidate
signal for "does energy leak into higher mode bands at higher rain
intensity") into a `rain_anomaly_analysis` feature-dump regeneration without
paying for `det_debug`'s full duplication cost.

## The bug pattern

`_detect_rain_over_time()` returns two separate payloads:

- **`feature_dump`** — built from `fd_dense`/`fd_sparse`/`fd_clip_summary`
  inside `if feature_dump_level > 0:` (~line 1108). This is the lean,
  intentionally-curated payload meant to be persisted (`keep_state_features`,
  on by default in `rain_anomaly_analysis/feature_pipeline/feature_dump.py`).
- **`det_debug`** — a much larger, unfiltered diagnostics dict, gated behind
  `keep_state_debug`/`return_detector_debug`. It duplicates most of
  `feature_dump`'s content at full (non-sparse) resolution, plus everything
  else computed during detection.

Both `feature_dump_include_peak_summary` (line 334) and
`feature_dump_include_td_envelope` (line 333) are read once, near the top of
the function, clearly intended to gate *something* going into
`feature_dump`. Neither one is referenced again anywhere else in the file —
the underlying data (`peak_ratio`/`peak_gate_score`/`peak_valid_count`/
`peak_count_by_mode`, and `td_rise_time_sec`/`td_fall_time_sec`/
`td_rise_slope`/`td_fall_slope`/`td_energy_envelope`/`td_peak_energy`) only
ever reaches `det_debug` (gated correctly there by `peak_features_enable` /
`td_envelope_features_enable` themselves — those two flags work as
intended). So today, the *only* way to persist either feature set is to turn
on `keep_state_debug` and eat the full `det_debug` cost.

**Measured cost of that path** (20-clip smoke test, `rain_anomaly_analysis`
`query_rain_peaks_smoke` run, `keep_state_debug=True` to get
`peak_count_by_mode`): state parquet went from a baseline of a few hundred
KB/clip to **7.5MB / 20 clips ≈ 375KB/clip**, extrapolating to **~11GB**
across the full ~29,813-clip `query_rain_peaks` run — for one field we
actually wanted. `det_debug` also nests a second full copy of
`feature_dump` inside itself (`det_debug["feature_dump"]`), compounding the
duplication.

There's also an unrelated but adjacent trap in this same `keep_state_debug`
path worth knowing about: turning on `return_debug` (which
`rain_signal_processor.py` defaults to `True` whenever `keep_state_debug` is
`True`) builds a `debug["suppressor_params"] = dict(cfg.suppressor or {})`
payload that is always `{}` in `classifier_only_mode` configs (no
`suppressor` section at all) — PyArrow refuses to write a zero-field struct
to parquet (`ArrowNotImplementedError: Cannot write struct type
'suppressor_params' with no child field`). Worked around on the
`rain_anomaly_analysis` side by forcing `return_debug=False` while this repo
still required the full `det_debug` path; no longer needed once the fix
below removes the need for `keep_state_debug` in the first place.

## Fix 1 — `peak_summary` (applied locally, verified)

Added inside `if feature_dump_dense_enable:` (~line 1138, right after the
existing `feature_dump_include_mode_flux_score` block), mirroring the
`support_mode_flux_1..4` per-index naming convention already used elsewhere
in the same dict:

```python
if peak_features_enable and feature_dump_include_peak_summary:
    fd_dense["peak_ratio"] = peak_ratio
    fd_dense["peak_gate_score"] = peak_gate_score
    fd_dense["peak_valid_count"] = peak_valid_count
    for i in range(peak_count_by_mode.shape[0]):
        fd_dense[f"peak_count_by_mode_{i}"] = peak_count_by_mode[i]
```

`peak_ratio`, `peak_gate_score`, and `peak_valid_count` are always
initialized as full-length arrays before the per-frame loop (`np.full`/
`np.zeros`, not conditional on `peak_features_enable`), so referencing them
here is safe regardless of flag state — they just stay at their NaN/0
defaults when `peak_features_enable` is `False`, matching how every other
`feature_dump_include_*` gate in this block behaves.

**Verification** (`rain_anomaly_analysis`, `query_rain_peaks_smoke`, 20
clips, `peak_features_enable=True` + `feature_dump_include_peak_summary=True`,
`keep_state_debug` back to its default `False`):

| Field | Shape | Nonzero count |
|---|---|---|
| `peak_ratio` | (871,) | 140 |
| `peak_gate_score` | (871,) | 42 |
| `peak_valid_count` | (871,) | 283 |
| `peak_count_by_mode_0` | (871,) | 44 |
| `peak_count_by_mode_1` | (871,) | 33 |
| `peak_count_by_mode_2` | (871,) | 32 |
| `peak_count_by_mode_3` | (871,) | 24 |
| `peak_count_by_mode_4` | (871,) | 14 |

State parquet size for the same 20 clips: **1.7MB** (vs. 7.5MB via the old
`det_debug` route) — in line with the pre-existing lean-dump baseline, not
the `det_debug`-inflated one.

**Caveat — deployed only as a venv override, not yet a real fix in this
repo's dependency chain.** `rain_anomaly_analysis` installs this package
from `git = "ssh://git@github.com/Arable/audio_processing_tools"` (pinned,
not editable) — so the smoke-test verification above only worked after
manually copying the locally-edited `rain_frame_classifier.py` into that
venv's `site-packages` copy. That override doesn't survive `uv sync`. The
change needs a real commit + PR here, then a re-pin (`pyproject.toml`/
`uv.lock`) on the `rain_anomaly_analysis` side, before it's durable.

## Fix 2 — `td_envelope` (applied, not yet smoke-tested)

Same bug, same shape, same fix pattern. The data
(`td_rise_time_sec`/`td_fall_time_sec`/`td_rise_slope`/`td_fall_slope`/
`td_energy_envelope`/`td_peak_energy` — drop rise/fall shape, a plausible
extra signal for whether higher-intensity rain drops look different in the
time domain, not just spectrally) previously only reached `det_debug`:

```python
# ~line 1054, det_debug only (still present, unchanged):
if td_envelope_features_enable:
    det_debug.update(
        {
            "td_rise_time_sec": td_rise_time_sec,
            "td_fall_time_sec": td_fall_time_sec,
            "td_rise_slope": td_rise_slope,
            "td_fall_slope": td_fall_slope,
            "td_energy_envelope": td_energy_envelope,
            "td_peak_energy": td_peak_energy,
        }
    )
```

Applied addition, in the same `if feature_dump_dense_enable:` block as Fix
1 (~line 1149-1159, right after it), gated by the previously-dead
`feature_dump_include_td_envelope` flag:

```python
if td_envelope_features_enable and feature_dump_include_td_envelope:
    fd_dense.update(
        {
            "td_rise_time_sec": td_rise_time_sec,
            "td_fall_time_sec": td_fall_time_sec,
            "td_rise_slope": td_rise_slope,
            "td_fall_slope": td_fall_slope,
            "td_energy_envelope": td_energy_envelope,
            "td_peak_energy": td_peak_energy,
        }
    )
```

Same safety argument applies: `td_rise_time_sec` etc. are always
initialized as full-length zero arrays before the per-frame loop in
`_detect_rain_over_time` (not conditional on `td_envelope_features_enable`),
so this is safe to reference unconditionally inside the dense block,
matching the pattern of every other flag there.

**Not yet done for this one:**
- No smoke-test verification run yet (code change is applied; verification
  steps below are what running it should confirm).
- `td_envelope_features_enable` itself defaults `False` in
  `rain_anomaly_analysis/feature_pipeline/feature_dump.py`'s
  `_DETECTOR_PARAMS` — enabling it also means paying for the underlying
  `extract_td_features_inline(..., envelope_features_enable=True)` compute
  (currently skipped whenever the flag is off), not just flipping the dump
  flag.

**Suggested verification, mirroring Fix 1's:** re-run
`rain_anomaly_analysis`'s `query_rain_peaks_smoke` (20-clip throwaway run)
with `td_envelope_features_enable=True` and
`feature_dump_include_td_envelope=True` added to the detector params, then
confirm the six `td_*` fields above land in `features` with plausible
nonzero values and the dump size stays close to the Fix-1-only baseline
(no `det_debug`/`keep_state_debug` involved).

## Related dead-flag pattern, not fixed here

`feature_dump_include_peak_payload` (already wired correctly — it gates
`include_peak_payload`, which does route into `det_debug`, just not into
`feature_dump`) would need the identical treatment if per-peak
frequency/prominence/bandwidth arrays (`peak_valid_freqs_hz`,
`peak_valid_prominences_db`, `peak_valid_bandwidths_hz`) are ever needed in
the lean dump instead of just `peak_count_by_mode`. Flagging for whoever
picks this doc up next — not attempted here since it wasn't needed for the
`rain_anomaly_analysis` use case that prompted this doc, and those are
per-frame variable-length arrays (not fixed-length like everything else in
`fd_dense`), so they'd need `fd_sparse`-style handling (gated by
`sparse_frame_idx`), not a direct `fd_dense` addition like Fixes 1 and 2.
