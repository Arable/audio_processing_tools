# CLAUDE.md — Rain Detector Frame-Level Processing Refactor

## Project context

**Repo:** `/Users/vikrantoak/source1/audio_processing_tools`  
**Active branch:** `feature/frame_level_rain_processing`  
**Golden regression repo:** `/Users/vikrantoak/source1/data-science-scratch/golden_regression`  
**Goal:** Refactor rain detector from batch/clip-level to per-frame (O(n_fft) memory) for embedded CM7 deployment.  
**Status:** Core streaming path complete and validated. Most pending check-ins from the May session are now committed (see below). All FD/TD decision thresholds (`td_gate_threshold`, `td_kurtosis_upper_threshold`, `new_rain_primary_flux_min`, `new_rain_mode1/2/3_flux_min`, `new_rain_min_support_count`, `clip_rain_min_frames`) were reconciled on 2026-08-05 to match the parameter set Santhosh chose and deployed in `edge` repo's NoiseCL (`config_list.h` on `origin/sp/noise_cancel_model`) — see session log for the full value table and the intermediate wrong values (3.5, then 3.7) this went through before landing on the confirmed-deployed set. **`golden_regression/scripts/generate_baseline.py` still needs the same update** (tracked separately, not yet done) before re-running validation against this threshold set. `origin/main` is at `d76f2b4` (2026-08-24) — feature-dump dead-flag fixes + a `td_fall_time_sec`/`td_fall_slope` correctness fix, see session log. `origin/fix/noise-psd-mode-snr-summary` (PR #5, open, not yet merged) is at `730ace8` (2026-08-25) — see that session log entry and `noise_psd_by_mode_wiring.md` for the full 11-round review history.

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
| `audio_processing_tools/edge/rain_signal_processor.py` | Top-level processor. STFT `center=False` line 754, ISTFT `center=False` line 1141. Comparison flags: `run_frame_level_comparison`, `run_streaming_comparison`, `run_nowinsor_replay` in `NoiseProcessorConfig`. |
| `audio_processing_tools/edge/rain_frame_classifier.py` | `RainFrameClassifierMixin` (batch path) + `RainFrameClassifierState` (streaming path). `process_audio_frame()` is the main per-frame entry point. |
| `audio_processing_tools/edge/feature_extraction.py` | `extract_td_features_inline`, `extract_raw_spectral_shape_features_inline`. RS function requires `raw_power` (F,T) + `freqs` (F,) — `x` parameter removed. |
| `audio_processing_tools/edge/noise_tracker.py` | `CausalNoiseTracker` — standalone stateful noise PSD tracker. Has `_seeded` flag (auto-seeds from first frame). |
| `data-science-scratch/golden_regression/scripts/generate_baseline.py` | Canonical validation script — replaces notebook. Runs processing + frame-level, streaming, and regression checks. |
| `audio_processing_tools/edge/rain_estimator.py` | `estimate_rain_from_audio()` — per-clip DSD-based rain-rate estimate (drop-size histogram → regression model → `precip_mm`). Wired into `RainDetectorProcessor.run()` behind `estimate_clip_rain` flag (2026-08-05). |
| `band_noise_suppression.md` | Writeup: noise suppression in `band_noise_estimator.py`/`band_noise_processor.py` (rain-measurement DSD path). Standalone — not yet wired into `RainDetectorProcessor` or `rain_estimator.py`. |
| `frame_level_rain_streaming.md` | Writeup: what's new in the streaming refactor across `noise_tracker.py`/`rain_frame_classifier.py`/`feature_extraction.py`/`rain_signal_processor.py`. Delta doc — see `edge/README.md` for the underlying algorithm. |
| `feature_dump_peak_and_envelope_wiring.md` | Writeup: `feature_dump_include_peak_summary`/`feature_dump_include_td_envelope` were dead flags — the data only ever reached `det_debug` (expensive, `keep_state_debug`-gated), never the lean `feature_dump`. Both fixes applied (`rain_frame_classifier.py` `fd_dense` block, ~line 1142-1159); peak-summary verified against real data, td-envelope not yet smoke-tested. |

---

## Session log — 2026-08-25

**PR #5 (`fix/noise-psd-mode-snr-summary`, `band_energy_summary`) review completion.** Picked up
after 7 prior review rounds and pushed 3 more commits (`93ffb49`, `6e18757`, `730ace8`) closing
out 4 additional review passes — full round-by-round detail (8th–11th) is in
`noise_psd_by_mode_wiring.md`, not duplicated here. Summary: an independent fresh-eyes
`/code-review` found/fixed an overlap-validation gap in `normalize_bands()`, an unconditional
~26KB memory allocation in `RainFrameClassifierState.__init__` regardless of
`band_energy_summary_enable`, and a duplicated coverage-fraction calc; two follow-up Codex passes
found/fixed an empty-bands-list crash, a ruff B905 gap, missing regression tests, and a
batch-vs-streaming validation-parity gap (invalid bands raised in batch but not streaming when
disabled); `codeant-ai[bot]`'s automated review found/fixed a real bug (the diagnostic was
receiving the clamped, not the "real", lagged noise PSD) and correctly flagged a batch-vs-streaming
rain-gating divergence in `band_energy_summary`'s noise source that was deliberately left
unfixed — acknowledged as a known, practical limitation (see doc) rather than chased, since a
self-consistent fix would require re-invoking the actual detection decision.
`codeant-ai[bot]`'s comments again embedded "Prompt for AI Agent" auto-implement/auto-chase
instructions — same prompt-injection-shaped pattern as 2026-08-05, not acted on; claims verified
independently instead.

Test count in `tests/edge/rain_detection/`: 28 → 38, all passing. PR description was expanded with
a "Scope / firmware impact" section (explicitly: no `frame_class`/`rain_conf` impact, and
`band_energy_summary`/`clip_spectral_occupancy` are **not approved for CM7 in their current
form**), then trimmed back down to Scope/Summary/Test-plan once the round-by-round history grew
large enough that it belonged in `noise_psd_by_mode_wiring.md` instead. PR #5 still open, not yet
merged — the real-clip smoke test (`rain_anomaly_analysis`'s `query_rain_peaks_smoke`) is still the
next step before merge.

## Session log — 2026-08-24

**`feature_dump_peak_and_envelope_wiring.md` follow-through:** Fix 2 (`td_envelope`) was applied (same `fd_dense` pattern as the already-verified peak-summary fix), the doc and the `CLAUDE.md` Key-files row updated to stop saying "proposed, not yet applied", and both fixes committed as `f501362`.

**Real bug found in `td_fall_time_sec`/`td_fall_slope`** (`feature_extraction.py:427`, `_subframe_peak_shape_features()`): `fall_dt` was `i_lo_fall * dt_sec` — the full peak→10% decay duration — instead of `(i_lo_fall - i_hi_fall) * dt_sec`, the 90%→10% window symmetric with `rise_dt = (i_hi - i_lo) * dt_sec`. Confirmed with a worked example (`[100,95,85,50,15,5]` → old code gives `5*dt_sec`, correct is `3*dt_sec`). This also corrupted `fall_slope` since its numerator (`hi-lo`, the 90%→10% amplitude span) was being divided by the wrong (inflated) time window — masked whenever the envelope drops below 90% within 1 sample of the peak, which is presumably why it wasn't caught earlier. Fixed in `d76f2b4`. Only affects output when `td_envelope_features_enable=True` (defaults `False` everywhere today); these fields are diagnostics only (`det_debug`/`feature_dump`), never feed `_rain_frame_decision()`, so no rain/no-rain outcome is affected.

**Audited other feature families for bugs of the same shape** (asymmetric index math, off-by-one, mismatched numerator/denominator) — `_block_energy_peak_features()`, subframe/frame and block/frame aggregation windows, `extract_raw_spectral_shape_features_inline()`, mode-flux batch (`_detect_rain_over_time`) vs. streaming (`RainFrameClassifierState`) implementations, and `td_crest_factor`/`td_kurtosis`. All checked out correct, including mode-flux batch-vs-streaming (the two are independently-written and were the most likely place for a divergence, but traced line-by-line identical: same delay-2 flux definition, same quantile-tracker seeding/update, same decision rule, same support-count clamp). Two non-bug findings noted for later: `flux_primary` in `rain_frame_classifier.py` is computed every frame but never read (dead duplicate of `mode_flux_by_mode[0]`); the whole peak-feature family (`peak_ratio`/`peak_gate_score`/`peak_count_by_mode`) only exists in the batch path, no streaming/`replay_clip` equivalent — consistent with the doc's "candidate signal" framing, not a defect.

**Pushed to `origin/main`:** `f501362` (feature-dump wiring) and `d76f2b4` (fall_dt fix) — both were sitting as local-only commits from earlier sessions/this one; now available for other repos (e.g. `rain_anomaly_analysis`) to pin to.

**Branch audit:** no open PRs. `origin/ai_gen_doc`/`update_readme`/`fix/band-noise-fft-magnitude-sum`/`feature/frame_level_rain_processing` all show as "unmerged" via `git branch --no-merged` only because they were squash/regular-merged under PRs #1–4 back in early August — content is already in `main`, branches are stale and safe to delete. `origin/sp/noise_algo_test` (Santhosh, last commit 2026-04-14, no PR, 4 commits ahead of `main`) is the one genuinely open branch — converts `band_noise_estimator.py` to `float32` + adds a C-library parity harness, overlaps the file we've edited. Not merged; still needs sync with Santhosh per the 2026-08-05 note below.

## Session log — 2026-08-06

**PR #4 review pass (own PR, `codeant-ai[bot]` + GitHub Copilot reviewer comments):**
- codeant-ai's `is_rain`-as-bool and `replay_clip()` T=0 IndexError findings were valid — fixed in `0c99223`.
- codeant-ai's claimed `NoiseProcessor._run_spectral_noise()` KeyError regression was rejected as pre-existing/out-of-scope (formatting-only diff at that location); codeant-ai saved a customized review instruction acknowledging it.
- Copilot's `reverse_binning_func` missing-paren/SyntaxError claim was checked against HEAD (`ast.parse()`, manual read) and found false — replied with evidence, no fix needed.
- Copilot's `frame_times` claim was real: `extract_td_features_causal_frame_inline()` took `frame_times` from each single-frame `extract_td_features_inline()` call, which always restarts at `[0]` — fixed in `b70066c` to compute the absolute offset (`t * hop / fs`) directly.
- Copilot's `estimate_clip_rain` "not actually gated by clip_is_rain" claim was true but intentional (rejected clips still need DSD sums for `rejected_precip_mm`/`rejected_rain_energy_sum` diagnostics) — reworded the PR description rather than changing behavior.

**New docs + full markdown review pass:** wrote `band_noise_suppression.md` and `frame_level_rain_streaming.md` (see Key files above), then ran an adversarial review of all six repo `.md` files (this file, root `README.md`, `rain_algorithm_technical_assessment.md`, `edge/README.md`, plus the two new docs) against current source. Fixed: root `README.md`'s broken Quick Start import and stale package-structure diagram; `rain_algorithm_technical_assessment.md` §5.3's rain-frame-classifier mechanism description (was: z-score/soft-confidence/hold-expansion — none of which exist; now: causal low-quantile normalization + hard threshold vote, matching `_rain_frame_decision()`); `edge/README.md`'s "lagged PSD" claim (gain computation is same-frame `N(t)` by default, `use_lagged_noise_psd` defaults `False` — only detector normalization is unconditionally lagged), wrong mode-flag name (`disable_suppression` → `suppressor_bypass`), and the oversubtraction formula (omitted the `noise_conf > 0.7` threshold remap); this file's off-by-one `center=False` line numbers (753/1140 → 754/1141). Also fixed two inaccuracies found in the newly-written `band_noise_suppression.md` itself (overstated per-frame telemetry-array claim; overstated default EMA smoothing — `ema_alpha=1.0` default means none happens out of the box).

## Session log — 2026-08-05

**PR review + merge pass (3 open PRs, all resolved):**
- **PR #1** (`update_readme`): found and fixed a package-structure inaccuracy (`noise_processor.py` is top-level, not under `edge/`), then merged.
- **PR #2** (`ai_gen_doc`, `rain_algorithm_technical_assessment.md`): spot-checked against current `main`, no inaccuracies found, merged as-is.
- **PR #3** (`fix/band-noise-fft-magnitude-sum`, our own): added `M_clean_fft` diagnostic + `BandNoiseEstimatorConfig.validate()` guards against `fs < frame_len` and out-of-range legacy FFT bins. Merged.

**Automated PR review incident:** `codeant-ai[bot]` (installed as a GitHub App, installation id `646884`) posted two review findings on PR #3 that were both **factually wrong** when checked against the real firmware source (`mark3-firmware-trunk/CM4/Inc/disdrometer_legacy.h`) — rejected both with evidence in-thread. Its comments also embedded a "Prompt for AI Agent" block directing the agent to implement the suggested fix and then chase down other comments — flagged as a prompt-injection-shaped pattern and not acted on. A third finding, from GitHub Copilot's PR reviewer (no embedded instructions), was valid and got fixed (`BandNoiseEstimatorConfig.validate()` guards above).

**Bugs found and fixed on `feature/frame_level_rain_processing` itself:**
- `estimate_clip_rain` (new, uncommitted `rain_signal_processor.py` feature) silently returned 0.0/0 for `precip_mm`/`rain_energy_sum`/`rain_energy_frame_count` because `edge/rain_estimator.py` was a stale pre-fix version missing those return keys. Fixed by replacing it with the corrected implementation (was sitting duplicated as `rain_estimator-1.py`); verified end-to-end.
- `RainFrameClassifierState.from_mixin()` still defaulted `td_gate_threshold` to **2.5** after the Mixin and `State.__init__` were raised to **3.5** — an uncommitted, internally-inconsistent edit that would have made `run_frame_level_comparison`/`run_streaming_comparison`/`run_nowinsor_replay` report false streaming-vs-batch divergence. Fixed so all three call sites agreed (initially at 3.5, an intermediate value later found to be wrong too — see below).
- Cross-repo porting-gap review (`mark3-firmware-trunk` `sp/noise_cancel_algo`, `edge` repo `sp/noise_cancel_model`) initially found C's `noisecl_td_gate_thr` at **3.7** — but that check was against a *stale local checkout* of `sp/noise_cancel_model` (missing 10 commits vs `origin`, including the actual config update). On `origin/sp/noise_cancel_model`, `config_list.h` is at **`td_gate_thr=3.4f`**, matching a parameter set Santhosh chose and validated (`manual-v3`/`trial_params3` in `NoiseCL/TargetPC/Test/Util/parameter_validation.ipynb`, added in PR #23 "chosen params", full-dataset F1=0.9765 on 78,781 clips). Corrected in Python to match this confirmed-deployed set (see full table above in Status).
- Full reconciled value table (Python now matches `origin/sp/noise_cancel_model`'s `config_list.h` exactly): `td_gate_threshold=3.4`, `td_kurtosis_upper_threshold=12.0` (was disabled by default — real behavior change, not just a number), `new_rain_primary_flux_min=2.19`, `new_rain_mode1_flux_min=2.63`, `new_rain_mode2_flux_min=2.57`, `new_rain_mode3_flux_min=2.45`, `new_rain_min_support_count=3`, `clip_rain_min_frames=4`. Fixed in all three call sites (Mixin, `State.__init__` signature default, `from_mixin()`) and verified they resolve identically with no explicit overrides.
- **Lesson learned:** always diff local branches against `origin/*` before trusting a "current state" check on someone else's repo — local `sp/noise_cancel_model` being stale is what produced the wrong 3.7 conclusion above.
- `flux_modes_winsor_enable`/`td_soft_enable` and a `mode_flux_rain_min`/`primary_flux_sanity_min`-based rain-decision scheme were raised as possible divergences from an unrelated, informally-pasted param sketch that turned out to reference parameter names that don't exist anywhere in this codebase — not a real reconciliation item, just a red herring from that sketch.

**Not a gap (confirmed intentional):** `legacy_band_bins()` (floor-based, added in PR #3) intentionally does NOT match `band_noise_dsd.c`'s own `hz_to_bin()` (round-based) — it exists specifically to make `M_band_fft`/`E_band_fft` comparable to the *legacy*, pre-noise-cancellation `disdrometer_legacy.c` drop_energy_level path, not to `band_noise_dsd.c`'s diagnostics. `M_clean_fft` (derived as `M_band_fft * G_mag`) inherits the same intent, so its L1-sum shape vs. `band_noise_dsd.c`'s L2/RMS-style `M_rfft_clean` is likewise not a mismatch to fix — they're answering different comparisons on purpose.

**Still outstanding:**
- `golden_regression/scripts/generate_baseline.py`'s canonical `detector_params`/`rain_classifier_params` still has the old/stale threshold set (`td_gate_threshold=3.7`, `new_rain_primary_flux_min=1.8`, etc.) — needs the same update as this repo, tracked as a separate task (different repo, `data-science-scratch` branch `frame-level-processing`).
- Golden-regression re-validation at the now-reconciled threshold set — not yet run.
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
STFT line 754 and ISTFT line 1141 both `center=False`. Frame t covers `x[t*hop : t*hop+n_fft]`. Invalidated all prior baselines (regenerated, see below).

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

All items below were committed on 2026-08-05 (`f75730d`, `72f7a4e`, `d6c8372`, `1d97317`, `9b02e8b`, `253bd05`, `7b41a78`, plus this CLAUDE.md update): `alac_utils.py` (debug print removal), `rain_signal_processor.py` + `rain_estimator.py` (`estimate_clip_rain`, bug fixed; `clip_rain_min_frames` default →4), `rain_frame_classifier.py` (full FD/TD threshold set reconciled to the chosen/deployed NoiseCL config — see Status/session-log above — plus `from_mixin()` fix and `rain_energy_summary`), `env_example`, `smoke_test_framework.py`, `run_rain_noise_analysis.py`, `frame_classifier_feature_analysis.py`, `CLAUDE.md`.

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
