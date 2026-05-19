# CLAUDE.md — Rain Detector Frame-Level Processing Refactor

## Project context

**Repo:** `/Users/vikrantoak/source1/audio_processing_tools`  
**Active branch:** `feature/frame_level_rain_processing`  
**Golden regression repo:** `/Users/vikrantoak/source1/data-science-scratch/golden_regression`  
**Goal:** Refactor rain detector from batch/clip-level to per-frame (O(n_fft) memory) for embedded CM7 deployment. Target: ≈5 KB working state vs ≈3.6 MB batch → ≈670× reduction.

---

## Key files

| File | Role |
|------|------|
| `audio_processing_tools/edge/rain_signal_processor.py` | Top-level processor. STFT `center=False` at line 753, ISTFT `center=False` at line 1140. Streaming comparison hooks at `run_streaming_comparison`, `run_frame_level_comparison`, `run_nowinsor_replay` flags in `NoiseProcessorConfig`. |
| `audio_processing_tools/edge/rain_frame_classifier.py` | `RainFrameClassifierMixin` (batch path) + `RainFrameClassifierState` (frame-level streaming path). `process_audio_frame()` is the main per-frame entry point. |
| `audio_processing_tools/edge/feature_extraction.py` | `extract_td_features_inline`, `extract_raw_spectral_shape_features_inline`. The RS function no longer accepts `x` (raw audio) — it **requires** `raw_power` (F, T) and `freqs` (F,) from the caller. |
| `audio_processing_tools/edge/noise_tracker.py` | `CausalNoiseTracker` — standalone stateful per-bin noise PSD tracker. State: `_tracker`, `_tracker_scale`, `_prev_N` (each `n_bins` floats). |

---

## Changes made in this session (committed in `7cf3cde`)

### 1. `noise_tracker.py` — `_seeded` flag
`CausalNoiseTracker` previously started from zeros when `reset()` was called without `first_frame`. Added `_seeded: bool` flag so `update()` auto-seeds from the first observed frame instead.

```python
# update() first-frame branch — auto-seed if not already seeded
if self._prev_N is None:
    if not self._seeded:
        self._tracker = np.maximum(P_band.copy(), 0.0)
        self._tracker_scale = np.maximum(np.abs(P_band), self._step_floor)
        self._seeded = True
    ...
```

### 2. `feature_extraction.py` — removed internal STFT
`extract_raw_spectral_shape_features_inline` no longer has `x` (raw audio) parameter or internal `spsig.stft` fallback. Returns empty features if `raw_power` or `freqs` is None.

```python
def extract_raw_spectral_shape_features_inline(
    *, fs, n_fft, hop, operating_band, ...,
    raw_power: Optional[np.ndarray] = None,   # required — (F, T) power spectrum
    freqs: Optional[np.ndarray] = None,        # required — (F,) frequency axis
) -> Dict[str, np.ndarray]:
    if raw_power is None or freqs is None:
        return _empty_raw_spectral_features()
    ...
```

### 3. `rain_frame_classifier.py` — `_extract_frame_features` cleanup
- Always calls `_compute_fft_frame()` at the top when `precomputed_power is None` — no second STFT.
- Hardcodes `td_input_mode="default"`, `td_input_band=None` (applying `sosfiltfilt` on a 256-sample causal buffer is incorrect).
- RS section always passes precomputed power — no if/else branch.

### 4. `replay_clip()` RS power — numpy FFT loop
Raw spectral shape features in `replay_clip()` now use a per-frame `np.fft.rfft` loop (center=False convention) instead of `spsig.stft`.

### 5. `center=False` STFT (already in code before this session)
STFT at line 753 and ISTFT at line 1140 both use `center=False`. This changes frame timestamps and audio content per frame — **baselines generated with `center=True` are stale**.

---

## Streaming path correctness (verified for set_500)

Validated in golden regression notebook (set_500, May 18 16:28 run):

| Check | Result | Target |
|---|---|---|
| `replay_clip` vs `_detect_rain_over_time` | 1.0000 | 1.0 |
| `process_frame` vs `replay_clip` flux MAE | 0.000000 | 0.0 |
| `process_frame` vs no-winsor replay (post warm-up) | 0.9552 | — |
| Streaming disagreement causes | 100% `td_timing` | expected |
| `winsor_or_flux_state` disagreements | 0 | 0 |

The 4.5% streaming disagreement is entirely the expected causal TD gate timing offset (crest-factor computed on rolling buffer lags the batch path by ~1 frame). Flux values are bit-for-bit identical.

---

## Pending task — regenerate golden baselines

The three baseline sets need regeneration with the current code:

| Set | Old parquet date | Status |
|-----|-----------------|--------|
| set_100 | May 18 13:48 | Stale — missing streaming sidecars |
| set_500 | May 18 16:28 | Stale — missing `_seeded` fix (no functional impact on batch path) |
| set_1000 | May 18 11:01 | Stale — missing streaming sidecars |

**Script ready at:**  
`/Users/vikrantoak/source1/data-science-scratch/golden_regression/scripts/generate_baseline.py`

**Run when VPN is connected:**
```bash
VENV=/Users/vikrantoak/source1/data-science-scratch/golden_regression/.venv/bin/python3
SCRIPT=/Users/vikrantoak/source1/data-science-scratch/golden_regression/scripts/generate_baseline.py
LOG=/Users/vikrantoak/source1/data-science-scratch/golden_regression/regression_runs

nohup $VENV "$SCRIPT" 100  > "$LOG/gen_baseline_100.log"  2>&1 &
nohup $VENV "$SCRIPT" 500  > "$LOG/gen_baseline_500.log"  2>&1 &
nohup $VENV "$SCRIPT" 1000 > "$LOG/gen_baseline_1000.log" 2>&1 &
```

Monitor: `tail -f $LOG/gen_baseline_500.log`

Success indicator in each log:
```
[set_NNN] Frame-class agreement (replay vs batch): 1.0000  (target 1.0)
[set_NNN] Done.
```

After all three complete, open the notebook and run the comparison cells (Frame-Level, Streaming, No-Winsor sections) for each SET_SIZE to confirm matching.

---

## Hop convention for streaming

```
seed_audio(x[0:hop])          # primes the rolling buffer
process_audio_frame(x[1*hop:2*hop])  # frame 0  → buffer holds x[0:n_fft]
process_audio_frame(x[2*hop:3*hop])  # frame 1  → buffer holds x[hop:n_fft+hop]
...
```

The rolling buffer always holds exactly `n_fft` samples before the FFT is computed, matching `center=False` STFT frame boundaries.

---

## Memory profile (embedded path)

| Component | State | Size |
|---|---|---|
| `CausalNoiseTracker` | `_tracker`, `_tracker_scale`, `_prev_N` | 3 × K × 4 B |
| `RainFrameClassifierState` | flux history, mode norms, rolling audio buf | ~5.4 KB total |
| Batch path (`SpectralNoiseProcessor`) | full spectrogram O(F×T) | ~3.6 MB |

K = number of band bins ≈ 67 (400–3500 Hz with n_fft=256, fs=11162).
