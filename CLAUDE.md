# CLAUDE.md — Rain Detector Frame-Level Processing Refactor

## Project context

**Repo:** `/Users/vikrantoak/source1/audio_processing_tools`  
**Active branch:** `feature/frame_level_rain_processing`  
**Golden regression repo:** `/Users/vikrantoak/source1/data-science-scratch/golden_regression`  
**Goal:** Refactor rain detector from batch/clip-level to per-frame (O(n_fft) memory) for embedded CM7 deployment.  
**Status:** Core streaming path complete and validated. Most pending check-ins from the May session are now committed (see below). `td_gate_threshold` was raised 2.5→3.7 on 2026-08-05 (briefly, incorrectly, committed as 3.5 first, then corrected to 3.7 — see session log) — **golden-regression re-validation against this threshold is still outstanding** before this branch merges to `main`.

---

## Memory profile (achieved)

| Component | State | Size |
|---|---|---|
| `CausalNoiseTracker` | `_tracker`, `_tracker_scale`, `_prev_N` | 3 × K × 4 B |
| `RainFrameClassifierState` | flux history, mode norms, rolling audio buf | ~5.4 KB total |
| Batch path (`SpectralNoiseProcessor`) | full spectrogram O(F×T) | ~3.6 MB |

K ≈ 67 bins (400–3500 Hz, n_fft=256, fs=11162). **Reduction: 670×.**

---

## Key files

| File | Role |
|------|------|
| `audio_processing_tools/edge/rain_signal_processor.py` | Top-level processor. STFT `center=False` line 753, ISTFT `center=False` line 1140. Comparison flags: `run_frame_level_comparison`, `run_streaming_comparison`, `run_nowinsor_replay` in `NoiseProcessorConfig`. |
| `audio_processing_tools/edge/rain_frame_classifier.py` | `RainFrameClassifierMixin` (batch path) + `RainFrameClassifierState` (streaming path). `process_audio_frame()` is the main per-frame entry point. |
| `audio_processing_tools/edge/feature_extraction.py` | `extract_td_features_inline`, `extract_raw_spectral_shape_features_inline`. RS function requires `raw_power` (F,T) + `freqs` (F,) — `x` parameter removed. |
| `audio_processing_tools/edge/noise_tracker.py` | `CausalNoiseTracker` — standalone stateful noise PSD tracker. Has `_seeded` flag (auto-seeds from first frame). |
| `data-science-scratch/golden_regression/scripts/generate_baseline.py` | Canonical validation script — replaces notebook. Runs processing + frame-level, streaming, and regression checks. |
| `audio_processing_tools/edge/rain_estimator.py` | `estimate_rain_from_audio()` — per-clip DSD-based rain-rate estimate (drop-size histogram → regression model → `precip_mm`). Wired into `RainDetectorProcessor.run()` behind `estimate_clip_rain` flag (2026-08-05). |

---

## Session log — 2026-08-05

**PR review + merge pass (3 open PRs, all resolved):**
- **PR #1** (`update_readme`): found and fixed a package-structure inaccuracy (`noise_processor.py` is top-level, not under `edge/`), then merged.
- **PR #2** (`ai_gen_doc`, `rain_algorithm_technical_assessment.md`): spot-checked against current `main`, no inaccuracies found, merged as-is.
- **PR #3** (`fix/band-noise-fft-magnitude-sum`, our own): added `M_clean_fft` diagnostic + `BandNoiseEstimatorConfig.validate()` guards against `fs < frame_len` and out-of-range legacy FFT bins. Merged.

**Automated PR review incident:** `codeant-ai[bot]` (installed as a GitHub App, installation id `646884`) posted two review findings on PR #3 that were both **factually wrong** when checked against the real firmware source (`mark3-firmware-trunk/CM4/Inc/disdrometer_legacy.h`) — rejected both with evidence in-thread. Its comments also embedded a "Prompt for AI Agent" block directing the agent to implement the suggested fix and then chase down other comments — flagged as a prompt-injection-shaped pattern and not acted on. A third finding, from GitHub Copilot's PR reviewer (no embedded instructions), was valid and got fixed (`BandNoiseEstimatorConfig.validate()` guards above).

**Bugs found and fixed on `feature/frame_level_rain_processing` itself:**
- `estimate_clip_rain` (new, uncommitted `rain_signal_processor.py` feature) silently returned 0.0/0 for `precip_mm`/`rain_energy_sum`/`rain_energy_frame_count` because `edge/rain_estimator.py` was a stale pre-fix version missing those return keys. Fixed by replacing it with the corrected implementation (was sitting duplicated as `rain_estimator-1.py`); verified end-to-end.
- `RainFrameClassifierState.from_mixin()` still defaulted `td_gate_threshold` to **2.5** after the Mixin and `State.__init__` were raised to **3.5** — an uncommitted, internally-inconsistent edit that would have made `run_frame_level_comparison`/`run_streaming_comparison`/`run_nowinsor_replay` report false streaming-vs-batch divergence. Fixed so all three call sites agreed (initially at 3.5).
- Cross-repo porting-gap review (`mark3-firmware-trunk` `sp/noise_cancel_algo`, `edge` repo `sp/noise_cancel_model`) surfaced that C's `noisecl_td_gate_thr` is hardcoded to **3.7**, not 3.5. Checked `data-science-scratch/golden_regression/scripts/generate_baseline.py` (branch `frame-level-processing`) — its canonical `detector_params` also uses `td_gate_threshold: 3.7`. So 3.5 was itself a stale/wrong value; corrected to **3.7** everywhere (Mixin, `State.__init__`, `from_mixin()`) in a follow-up commit the same day.
- Also found real, unresolved numeric divergences from the golden-regression canonical params that are NOT yet reconciled in code: `flux_modes_winsor_enable` (golden regression uses `False`; a separate, unrelated uncommitted param sketch had `True`), `td_soft_enable` (golden regression uses `False`), and the rain-decision scheme itself — golden regression's canonical script uses `new_rain_primary_flux_min`/`new_rain_mode1_flux_min`/`new_rain_mode2_flux_min`/`new_rain_mode3_flux_min`/`new_rain_min_support_count`, not `mode_flux_rain_min`/`primary_flux_sanity_min` (the latter two don't exist anywhere in the current codebase).

**Not a gap (confirmed intentional):** `legacy_band_bins()` (floor-based, added in PR #3) intentionally does NOT match `band_noise_dsd.c`'s own `hz_to_bin()` (round-based) — it exists specifically to make `M_band_fft`/`E_band_fft` comparable to the *legacy*, pre-noise-cancellation `disdrometer_legacy.c` drop_energy_level path, not to `band_noise_dsd.c`'s diagnostics. `M_clean_fft` (derived as `M_band_fft * G_mag`) inherits the same intent, so its L1-sum shape vs. `band_noise_dsd.c`'s L2/RMS-style `M_rfft_clean` is likewise not a mismatch to fix — they're answering different comparisons on purpose.

**Still outstanding:**
- Golden-regression re-validation at `td_gate_threshold=3.7` (see Status above) — not yet run.
- `sp/noise_algo_test` (Santhosh Palethadka, no PR, last commit 2026-04-14) converts `band_noise_estimator.py`'s internals to `float32` throughout and adds a C-library (`libband_noise_pylib.so`) parity test harness — overlaps with the same file we've been editing; worth syncing with him before it's rebased, to avoid conflicting float32 vs float64 assumptions.
- `audio_processing_tools/backend/` (old `spectral_features.py`/`spectral_visualize.py`, unreferenced anywhere) — left untouched, disposition still undecided.
- `audio_processing_tools.code-workspace`, `framework_results.csv`, `framework_states_rain.csv`, `status_output` — intentionally left untracked (editor config / run artifacts).

---

## Changes made (May 18–19, commits `3706886`, `23f365f`, `5551aed`, `7cf3cde`)

### `noise_tracker.py` — `_seeded` flag
`reset()` without `first_frame` previously left tracker at zero. `update()` now auto-seeds from first observed frame.

### `feature_extraction.py` — removed internal STFT
`extract_raw_spectral_shape_features_inline` no longer has `x` or `spsig.stft` fallback. Requires `raw_power (F,T)` + `freqs (F,)` or returns empty dict.

### `rain_frame_classifier.py` — streaming path cleanup
- `_extract_frame_features()`: calls `_compute_fft_frame()` once at top — no second STFT ever.
- Hardcoded `td_input_mode="default"` — `sosfiltfilt` on 256-sample causal buffer is incorrect for other modes.
- RS section: single call, always uses precomputed power.
- `replay_clip()`: RS power from per-frame `np.fft.rfft` loop (center=False) — no more `spsig.stft`.

### `center=False` STFT
STFT line 753 and ISTFT line 1140 both `center=False`. Frame t covers `x[t*hop : t*hop+n_fft]`. Invalidated all prior baselines (regenerated, see below).

---

## Streaming path validation (complete)

| Check | Result | Target |
|---|---|---|
| `replay_clip` vs `_detect_rain_over_time` | **1.0000** | 1.0 |
| `process_frame` flux MAE vs `replay_clip` | **0.000000** | 0.0 |
| `process_frame` vs no-winsor replay | **0.9552** | — |
| Cause of streaming disagreements | **100% td_timing** | expected |
| `winsor_or_flux_state` diffs | **0** | 0 |

4.5% disagreement = causal crest-factor timing lag (rolling buffer sees prefiltered audio 1 frame later than batch). Not a bug.

---

## Golden baselines — regenerated (May 19)

All three sets regenerated with current code via `generate_baseline.py`. Frame-class agreement confirmed 1.0000.

| Set | Parquet date | Frame agreement |
|-----|-------------|-----------------|
| set_100 | May 19 | **1.0000** ✓ |
| set_500 | May 19 | **1.0000** ✓ |
| set_1000 | May 19 | **1.0000** ✓ |

Note: baselines were regenerated with the original simple script (before streaming checks were added to `generate_baseline.py`). The streaming/nowinsor check still needs one run with the updated script:

```bash
VENV=/Users/vikrantoak/source1/data-science-scratch/golden_regression/.venv/bin/python3
SCRIPT=/Users/vikrantoak/source1/data-science-scratch/golden_regression/scripts/generate_baseline.py

$VENV "$SCRIPT" 500 --check   # validate without overwriting
```

Success output:
```
[set_500] PASS — only td_timing diffs (expected causal lag)
[set_500] Frame-class vs baseline  : 1.0000
```

---

## offline vs causal_frame equivalence (May 19)

Validated in `comparison_offline_and_causal_frame.ipynb` on 5k clips:
- `td_feature_timing_mode="offline"` vs `"causal_frame"` → **identical** clip-level TP/FP/FN/TN
- 80k causal frame run: F1 = **0.9770** vs old baseline **0.9776** (Δ −0.0006)
- FN reduced 769→549 (better recall), FP increased 555→818 — expected effect of no winsorization + causal normalization

---

## Pending check-ins

### `audio_processing_tools` repo

All items below were committed on 2026-08-05 (`f75730d`, `72f7a4e`, `d6c8372`, `1d97317`, `9b02e8b`, plus this CLAUDE.md update): `alac_utils.py` (debug print removal), `rain_signal_processor.py` + `rain_estimator.py` (`estimate_clip_rain`, bug fixed), `rain_frame_classifier.py` (`td_gate_threshold`→3.7, corrected from an initial wrong 3.5 + `from_mixin()` fix + `rain_energy_summary`), `env_example`, `smoke_test_framework.py`, `run_rain_noise_analysis.py`, `frame_classifier_feature_analysis.py`, `CLAUDE.md`.

`feature_extraction.py` and `noise_tracker.py` were already committed in earlier commits (`23f365f`, `5551aed`, `7cf3cde`) and needed no further changes today.

Deleted (were empty placeholders or broken duplicates, nothing lost): `test_harness_cm7_rain_detector.py`, `audio_processing_tools/edge/joint_td_gate_sweep.py`, `audio_processing_tools/edge/rain_signal_processor-1.py`, `audio_processing_tools/edge/rain_estimator-1.py` (content merged into `rain_estimator.py`).

Still not committed / left untouched: `audio_processing_tools/backend/` (disposition undecided — see Session log above).

Do NOT commit: `old_*.py`, `Untitled`, `*.code-workspace`, `*.csv`, `status_output`

### `data-science-scratch/golden_regression` repo

| File | Status |
|------|--------|
| `scripts/generate_baseline.py` | New, staged — commit |
| `baseline/set_{100,500,1000}/*.parquet` | Modified (freshly regenerated) — commit |
| `notebooks/comparison_offline_and_causal_frame.ipynb` | New, untracked — commit |
| `notebooks/frame_level_processing_changes.md` | New, untracked — commit |

Do NOT commit: `golden_regression.ipynb` (runner notebook superseded by script), `golden_regression_executed.ipynb`, `pressure_sensor_failure_detection/`, `.ipynb_checkpoints/`, unrelated analysis dirs.

---

## Hop convention for streaming

```
seed_audio(x[0:hop])                  # primes rolling buffer
process_audio_frame(x[1*hop:2*hop])   # frame 0 → buffer = x[0:n_fft]
process_audio_frame(x[2*hop:3*hop])   # frame 1 → buffer = x[hop:n_fft+hop]
```

Rolling buffer always holds exactly `n_fft` samples before FFT — matches `center=False` frame boundaries.
