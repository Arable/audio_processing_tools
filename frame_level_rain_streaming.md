# Frame-Level Streaming Refactor: Noise Estimation + Rain Detection

**Files:** `edge/noise_tracker.py`, `edge/rain_frame_classifier.py`, `edge/feature_extraction.py`, `edge/rain_signal_processor.py`
**Branch:** `feature/frame_level_rain_processing`

## Scope of this doc

`edge/README.md` already documents the detector/suppressor *algorithm* —
lagged noise PSD, noise-normalized detection, adaptive oversubtraction, etc.
That explanation still holds. This doc covers what's **new**: the refactor
from batch/clip-level processing (full spectrogram in memory, O(F×T)) to a
true per-frame streaming path (O(n_fft) working state) for CM7 embedded
deployment — a ~670× memory reduction (~3.6 MB → ~5.4 KB). It does not
re-derive the noise-normalization math; see the README for that.

## `noise_tracker.py` — `CausalNoiseTracker`

The batch path's inline stochastic low-quantile PSD tracker, extracted into
a standalone stateful class so it can be reused frame-by-frame outside
`SpectralNoiseProcessor`. Per-bin state is just three arrays of size K
(operating-band bin count): `_tracker`, `_tracker_scale`, `_prev_N`.

- `update(P_band, is_rain)`: one call per frame. Moves the per-bin quantile
  tracker toward `P_band` at rate `q` (up) / `1-q` (down), asymmetric EMA
  (`ema_up`/`ema_down`) smooths the result, and the whole estimate is
  clamped to `noise_psd_max_ratio * P_band` as a hard safety ceiling.
- `is_rain` gates learning outside a warmup window (`warmup_need`, half the
  tracking window) — rain frames don't move the baseline once warmup is
  done, so rain energy can't leak into the noise floor.
- **Auto-seeding:** `reset()` without a `first_frame` leaves the tracker at
  zero; `update()` now auto-seeds from the first frame it actually observes
  (`_seeded` flag) so a bare `reset()` still starts from a sensible
  baseline instead of biasing the first few frames toward zero noise.

## `feature_extraction.py` — causal-frame TD extraction

`extract_td_features_causal_frame_inline()` computes TD features (crest
factor, kurtosis, block-energy shape) per frame using only that frame's
`n_fft`-sample causal window — no `sosfiltfilt` over future audio. It loops
`extract_td_features_inline()` once per frame on a single-frame slice and
takes the last value of each returned feature array. One consequence:
`frame_times` from that per-call output always restarts at 0 (each call
only ever sees one frame), so the function now computes the absolute
per-frame offset (`t * hop / fs`) directly rather than trusting the
per-call value (fixed in `b70066c`, flagged by PR #4 review).

Also since this branch: `extract_raw_spectral_shape_features_inline()` no
longer has an internal STFT fallback — it requires `raw_power` (F, T) +
`freqs` (F,) from the caller. This removes a second, redundant STFT call
that used to happen inside feature extraction.

## `rain_frame_classifier.py` — `RainFrameClassifierState`

The new streaming counterpart to the batch `RainFrameClassifierMixin`,
exposing three execution modes over the same core decision logic
(`_decide_frame()`):

| Mode | Input | Use |
|---|---|---|
| `replay_clip(P, audio)` | Full precomputed spectrogram + audio | Offline parity validation against the batch mixin |
| `process_frame(frame_spectrum, frame_audio)` | One noise-normalized spectrum column + hop of raw audio | Streaming with caller-supplied noise normalization |
| `process_audio_frame(chunk, is_rain)` | Raw audio chunk only | Fully causal — computes its own FFT + `CausalNoiseTracker` normalization internally |

**`_decide_frame()`** (shared core): computes flux as `frame[t] - frame[t-2]`
(half-wave rectified, causal — needs only two frames of history), sums it
per resonance mode, normalizes each mode's flux against a per-mode causal
low-quantile baseline (`CausalLowQuantileTracker`), applies a TD gate
(`td_crest_factor > td_gate_threshold`, optionally AND'd with
`td_kurtosis <= td_kurtosis_upper_threshold`), then requires the primary
mode to clear `new_rain_primary_flux_min` AND at least
`new_rain_min_support_count` of the 3 support modes to clear their own
thresholds. (Threshold values reconciled to the deployed NoiseCL config —
see `CLAUDE.md` session log.)

**State carried between frames:** two prior spectrogram columns (flux
history), one `CausalLowQuantileTracker` per mode + one for combined flux,
a rolling `n_fft`-sample audio buffer (filtered + raw), and (for
`process_audio_frame`) one `CausalNoiseTracker` over the operating band.
Total: ~5.4 KB regardless of clip length.

**Hop convention** — the rolling buffer must hold exactly `n_fft` samples
aligned to the STFT frame boundary before each FFT:

```
seed_audio(x[0:hop])                  # primes rolling buffer
process_audio_frame(x[1*hop:2*hop])   # frame 0 → buffer = x[0:n_fft]
process_audio_frame(x[2*hop:3*hop])   # frame 1 → buffer = x[hop:n_fft+hop]
```

**Known non-parity vs. `replay_clip()`** (by design, not bugs):
1. Causal `sosfilt` (persistent `zi`) vs. non-causal `sosfiltfilt` over the
   full clip → short filter warm-up transient after `reset()`.
2. TD features come from the same rolling buffer as the detector, so there's
   a 1-frame lag vs. batch's retrospective TD extraction.
3. No per-clip winsorization (flux cap needs future frames) —
   `_total_flux_cap` is always `None` in true streaming.

## `rain_signal_processor.py` — wiring + validation harness

- **`center=False`** on both STFT (line 754) and ISTFT (line 1141): frame
  `t` covers `x[t*hop : t*hop+n_fft]` with no symmetric zero-padding. This
  is what makes frame boundaries match the streaming path's hop convention
  exactly; `center=True`'s padding would otherwise misalign batch vs.
  streaming frame `t`.
- **`CausalNoiseTracker`** replaces the old inline batch PSD-tracking code
  in `_estimate_noise_psd_fft()` (`_make_noise_tracker()` builds one from
  `NoiseProcessorConfig`, seeded from the first frame).
- **Three validation-only flags** on `NoiseProcessorConfig`, each running an
  extra `RainFrameClassifierState` pass alongside the batch detector and
  attaching its own comparison payload to the output — used to validate the
  refactor rather than in production:
  - `run_frame_level_comparison`: `replay_clip()` vs. `_detect_rain_over_time()` → **1.0000** frame-class agreement, **0.000000** flux MAE.
  - `run_streaming_comparison`: true `process_frame()` per hop → **95.5%** agreement with the no-winsor replay reference; **100%** of disagreements attributed to the expected 1-frame TD timing lag.
  - `run_nowinsor_replay`: `replay_clip()` with winsorization forced off, isolating streaming-only differences from winsorization effects.

## Validation status

| Check | Result |
|---|---|
| `replay_clip` vs. `_detect_rain_over_time` | 1.0000 frame-class agreement |
| `process_frame` flux MAE vs. `replay_clip` | 0.000000 |
| `process_frame` vs. no-winsor replay | 0.9552 |
| Cause of streaming disagreements | 100% TD timing (expected causal lag) |

Golden-regression baselines (`set_100`/`set_500`/`set_1000`) were
regenerated against this code and confirm 1.0000 frame-class agreement;
see `CLAUDE.md` for the full reconciliation history and outstanding items
(`golden_regression/scripts/generate_baseline.py` threshold update, still
pending in the `data-science-scratch` repo).
