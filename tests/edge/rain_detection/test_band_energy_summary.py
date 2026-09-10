"""Tests for band_energy_summary.

Clip-level per-band STFT energy sums, in both the batch path
(RainFrameClassifierMixin._detect_rain_over_time) and the streaming/embedded
path (RainFrameClassifierState.process_audio_frame).
"""

import librosa
import numpy as np
import pytest

from audio_processing_tools.edge.feature_extraction import default_spectral_occupancy_bands
from audio_processing_tools.edge.rain_frame_classifier import (
    RainFrameClassifierMixin,
    RainFrameClassifierState,
)
from audio_processing_tools.edge.rain_signal_processor import SpectralNoiseProcessor

from .conftest import SAMPLE_RATE

N_FFT = 256
HOP = 128


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _BatchDetector(RainFrameClassifierMixin):
    """Minimal RainFrameClassifierMixin host for direct _detect_rain_over_time() calls."""

    def __init__(self, detector_params):
        cfg = type("_Cfg", (), {})()
        cfg.detector = dict(detector_params)
        self.cfg = cfg


class _StreamMixin:
    """Minimal stand-in for RainFrameClassifierMixin — from_mixin() only needs _dget()."""

    def __init__(self, detector_params):
        self._params = dict(detector_params)

    def _dget(self, name, default=None):
        return self._params.get(name, default)


def _freqs():
    return librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)


def _base_detector_params(detector_params, **overrides):
    params = dict(detector_params)
    params["band_energy_summary_enable"] = True
    params["feature_dump_clip_summary_enable"] = True
    params["feature_dump_level"] = 1
    params.update(overrides)
    return params


def _synthetic_clip(freqs, T, *, operating_band=(400.0, 3500.0), seed=0):
    """Build a synthetic (raw_power, noise_psd) pair.

    Signal is present everywhere; noise is only estimated inside
    operating_band, mirroring _estimate_noise_psd_fft's real behavior (zero
    outside operating_band, not "unknown").
    """
    rng = np.random.default_rng(seed)
    F = len(freqs)
    noise_full = rng.random((F, T)) * 0.1 + 0.01
    signal_only = rng.random((F, T)) * 5.0
    raw_power = noise_full + signal_only
    op_lo, op_hi = operating_band
    op_mask = (freqs >= op_lo) & (freqs <= op_hi)
    noise_psd = noise_full.copy()
    noise_psd[~op_mask, :] = 0.0
    return raw_power, noise_psd


def _run_batch(detector_params, freqs, raw_power, noise_psd=None):
    detector = _BatchDetector(detector_params)
    if noise_psd is not None:
        P_for_detection = 10.0 * np.log10(raw_power + 1e-9) - 10.0 * np.log10(noise_psd + 1e-9)
        kwargs = {"noise_psd": noise_psd}
    else:
        P_for_detection = 10.0 * np.log10(raw_power + 1e-9)
        kwargs = {}
    return detector._detect_rain_over_time(P_for_detection, freqs, raw_power=raw_power, **kwargs)


def _build_state(detector_params, *, enable_summary, expose_per_frame=False, **overrides):
    freqs = _freqs()
    params = dict(detector_params)
    params["band_energy_summary_enable"] = enable_summary
    params["band_energy_summary_expose_per_frame"] = expose_per_frame
    params.update(overrides)
    state = RainFrameClassifierState.from_mixin(_StreamMixin(params), freqs)
    state.reset()
    return state


def _stream_frames(state, audio):
    """Replay audio through process_audio_frame().

    Uses the documented hop convention: seed with the first hop, then feed
    each completing hop.
    """
    state.seed_audio(audio[:HOP])
    # N_FFT == 2*HOP, so the last valid frame is t = len(audio)//HOP - 2
    # (buffer x[t*HOP : t*HOP+N_FFT] must fit in audio) -> n_frames = that + 1.
    n_frames = len(audio) // HOP - 1
    results = []
    for t in range(n_frames):
        c_start = (t + 1) * HOP
        chunk = audio[c_start : c_start + HOP]
        results.append(state.process_audio_frame(chunk))
    return results


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


def test_batch_band_energy_summary_has_all_16_bands_with_coverage(detector_params):
    """All 16 default bands report total/covered-total/noise sums plus coverage_fraction."""
    freqs = _freqs()
    raw_power, noise_psd = _synthetic_clip(freqs, T=20)
    params = _base_detector_params(detector_params)
    _, _, det_debug, feature_dump = _run_batch(params, freqs, raw_power, noise_psd)

    assert det_debug.get("band_energy_summary_error") is None
    summary = det_debug["band_energy_summary"]
    band_names = [name for name, _, _ in default_spectral_occupancy_bands()]
    assert len(band_names) == 16
    for name in band_names:
        assert f"{name}_total_energy_sum" in summary
        assert f"{name}_covered_total_energy_sum" in summary
        assert f"{name}_noise_energy_sum" in summary
        assert f"{name}_coverage_fraction" in summary
    assert summary == feature_dump["band_energy_summary"]


def test_batch_band_outside_operating_band_has_zero_coverage(detector_params):
    """A band fully below operating_band reports zero coverage and zero noise."""
    freqs = _freqs()
    raw_power, noise_psd = _synthetic_clip(freqs, T=20, operating_band=(400.0, 3500.0))
    params = _base_detector_params(detector_params, operating_band=(400.0, 3500.0))
    _, _, det_debug, _ = _run_batch(params, freqs, raw_power, noise_psd)
    summary = det_debug["band_energy_summary"]

    # dc (0-43.6Hz) and wind_1 (43.6-261.6Hz) are fully below operating_band.
    assert summary["dc_coverage_fraction"] == 0.0
    assert summary["wind_1_coverage_fraction"] == 0.0
    assert summary["dc_noise_energy_sum"] == 0.0
    assert summary["dc_covered_total_energy_sum"] == 0.0
    # But total_energy_sum is a real measurement (raw_power is valid everywhere)
    # regardless of operating_band coverage — it is not forced to zero.
    assert summary["dc_total_energy_sum"] > 0.0


def test_batch_partially_covered_band_has_fractional_coverage(detector_params):
    """A band straddling the operating_band edge reports coverage strictly between 0 and 1."""
    freqs = _freqs()
    raw_power, noise_psd = _synthetic_clip(freqs, T=20, operating_band=(400.0, 3500.0))
    params = _base_detector_params(detector_params, operating_band=(400.0, 3500.0))
    _, _, det_debug, _ = _run_batch(params, freqs, raw_power, noise_psd)
    summary = det_debug["band_energy_summary"]

    # mode_5 spans 3139.3-3575.3Hz; operating_band caps at 3500Hz -> partial.
    assert 0.0 < summary["mode_5_coverage_fraction"] < 1.0


def test_batch_partial_band_energy_only_in_uncovered_region(detector_params):
    """Energy placed only in a partial band's uncovered region must not leak into covered/noise sums.

    Regression test for the domain-mismatch bug: total_energy_sum (full band)
    picks up the energy, but covered_total_energy_sum and noise_energy_sum
    (both operating_band-restricted) must read exactly zero — proving
    total_energy_sum alone is not safe to pair with noise_energy_sum for an
    SNR-style computation on a partially-covered band.
    """
    freqs = _freqs()
    T = 5
    raw_power = np.zeros((len(freqs), T))
    noise_psd = np.zeros((len(freqs), T))

    # mode_5 spans 3139.3125-3575.328125 Hz; operating_band caps at 3500 Hz.
    # Place energy strictly above 3500 Hz, inside mode_5's uncovered remainder.
    uncovered_hz = 3538.125
    idx = int(np.argmin(np.abs(freqs - uncovered_hz)))
    assert freqs[idx] > 3500.0
    assert freqs[idx] < 3575.328125
    raw_power[idx, :] = 1000.0

    params = _base_detector_params(detector_params, operating_band=(400.0, 3500.0))
    _, _, det_debug, _ = _run_batch(params, freqs, raw_power, noise_psd)
    summary = det_debug["band_energy_summary"]

    assert summary["mode_5_total_energy_sum"] > 0.0
    assert summary["mode_5_covered_total_energy_sum"] == 0.0
    assert summary["mode_5_noise_energy_sum"] == 0.0
    assert 0.0 < summary["mode_5_coverage_fraction"] < 1.0


def test_batch_missing_noise_psd_emits_error_not_fake_summary(detector_params):
    """Missing noise_psd must produce an explicit error, not a numeric summary."""
    freqs = _freqs()
    raw_power, _ = _synthetic_clip(freqs, T=20)
    params = _base_detector_params(detector_params)
    _, _, det_debug, feature_dump = _run_batch(params, freqs, raw_power, noise_psd=None)

    assert "band_energy_summary" not in det_debug
    assert det_debug["band_energy_summary_error"]
    assert "band_energy_summary" not in feature_dump


def test_batch_band_energy_summary_matches_independent_hand_computed_sum(detector_params):
    """Verify against an independently hand-computed expected sum, not self-consistency.

    Places known energy at a known frequency and checks the exact expected
    values, rather than comparing the code under test against itself.
    """
    freqs = _freqs()
    T = 3
    raw_power = np.zeros((len(freqs), T))
    noise_psd = np.zeros((len(freqs), T))

    # mode_1 band is [436.015625, 654.0234375) Hz (half-open on the right).
    mode_1_bin_hz = 500.0
    idx = int(np.argmin(np.abs(freqs - mode_1_bin_hz)))
    assert 436.015625 <= freqs[idx] < 654.0234375
    raw_power[idx, :] = [10.0, 20.0, 30.0]
    noise_psd[idx, :] = [1.0, 2.0, 3.0]

    params = _base_detector_params(detector_params, operating_band=(400.0, 3500.0))
    _, _, det_debug, _ = _run_batch(params, freqs, raw_power, noise_psd)
    summary = det_debug["band_energy_summary"]

    assert summary["mode_1_total_energy_sum"] == pytest.approx(60.0)
    assert summary["mode_1_covered_total_energy_sum"] == pytest.approx(60.0)
    assert summary["mode_1_noise_energy_sum"] == pytest.approx(6.0)
    for name, _, _ in default_spectral_occupancy_bands():
        if name != "mode_1":
            assert summary[f"{name}_total_energy_sum"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


def test_streaming_band_energy_accumulation_matches_frame_sum(detector_params, deterministic_audio):
    """Verify the O(1) accumulator equals a naive sum over individual frames.

    The running accumulator must equal summing the per-frame energies
    process_audio_frame() itself returns — i.e. streaming accumulation is
    mathematically equivalent to a naive batch sum over individual frames.
    """
    state = _build_state(detector_params, enable_summary=True, expose_per_frame=True)
    frame_results = _stream_frames(state, deterministic_audio)
    assert len(frame_results) > 0

    band_names = [name for name, _, _ in default_spectral_occupancy_bands()]
    summary = state.get_band_energy_summary()

    for name in band_names:
        summed_total = sum(r[f"{name}_total_energy"] for r in frame_results)
        summed_covered = sum(r[f"{name}_covered_total_energy"] for r in frame_results)
        summed_noise = sum(r[f"{name}_noise_energy"] for r in frame_results)
        assert summary[f"{name}_total_energy_sum"] == pytest.approx(summed_total, rel=1e-9)
        assert summary[f"{name}_covered_total_energy_sum"] == pytest.approx(summed_covered, rel=1e-9)
        assert summary[f"{name}_noise_energy_sum"] == pytest.approx(summed_noise, rel=1e-9)

    assert any(v > 0.0 for k, v in summary.items() if k.endswith("_energy_sum"))


def test_band_energy_summary_disabled_returns_empty_dict(detector_params, deterministic_audio):
    """Disabled accumulation returns {}, not a dict of zeros indistinguishable from a silent clip."""
    state = _build_state(detector_params, enable_summary=False)
    frame_results = _stream_frames(state, deterministic_audio)

    assert "mode_1_total_energy" not in frame_results[0]
    assert state.get_band_energy_summary() == {}


def test_band_energy_summary_disabled_leaves_masks_and_accumulators_unallocated(detector_params):
    """band_energy_summary_enable=False must not allocate the mask matrices/accumulators at all.

    Regression test for the fix where __init__ built the (n_bands x n_bins)
    mask matrices and accumulator arrays unconditionally, spending ~5x this
    class's documented streaming-state memory budget even when the feature
    is off (the default) — checking output alone (returns {}) doesn't catch
    that, since the accumulators can exist, be unused, and still return {}.
    """
    state = _build_state(detector_params, enable_summary=False)
    assert state._band_energy_full_mask_matrix is None
    assert state._band_energy_covered_mask_matrix is None
    assert state._band_total_energy_sum is None
    assert state._band_covered_total_energy_sum is None
    assert state._band_noise_energy_sum is None
    assert state._band_energy_coverage_fraction is None


def test_band_energy_summary_enabled_allocates_masks_and_accumulators(detector_params):
    """band_energy_summary_enable=True must allocate one row/entry per configured band."""
    state = _build_state(detector_params, enable_summary=True)
    n_bands = len(state._band_energy_names)
    assert n_bands > 0
    assert state._band_energy_full_mask_matrix.shape[0] == n_bands
    assert state._band_energy_covered_mask_matrix.shape[0] == n_bands
    assert state._band_total_energy_sum.shape == (n_bands,)
    assert state._band_covered_total_energy_sum.shape == (n_bands,)
    assert state._band_noise_energy_sum.shape == (n_bands,)


@pytest.mark.parametrize(
    "bad_bands",
    [
        [],
        [("backwards", 1000.0, 400.0)],
        [("a", 400.0, 1000.0), ("b", 900.0, 3500.0)],
        [("dup", 400.0, 1000.0), ("dup", 1000.0, 3500.0)],
    ],
)
def test_invalid_bands_raise_identically_in_batch_and_streaming_even_when_disabled(detector_params, bad_bands):
    """An invalid band_energy_summary_bands override must be rejected identically in both paths.

    Even while band_energy_summary_enable=False, the batch path
    (_detect_rain_over_time) always calls normalize_bands() unconditionally,
    so the streaming constructor must too, rather than silently accepting
    what batch would reject.
    """
    freqs = _freqs()
    params = dict(detector_params)
    params["band_energy_summary_enable"] = False
    params["band_energy_summary_bands"] = bad_bands

    with pytest.raises(ValueError):
        RainFrameClassifierState.from_mixin(_StreamMixin(params), freqs)

    detector = _BatchDetector(params)
    raw_power = np.random.default_rng(0).random((len(freqs), 3))
    with pytest.raises(ValueError):
        detector._detect_rain_over_time(raw_power, freqs, raw_power=raw_power)


def test_empty_custom_band_energy_summary_bands_raises_at_construction(detector_params):
    """An empty band_energy_summary_bands override must fail loudly at construction.

    Otherwise it crashes later inside process_audio_frame() with an
    unrelated matmul shape error.
    """
    freqs = _freqs()
    params = dict(detector_params)
    params["band_energy_summary_enable"] = True
    params["band_energy_summary_bands"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        RainFrameClassifierState.from_mixin(_StreamMixin(params), freqs)


def test_per_frame_band_energy_not_exposed_unless_requested(detector_params, deterministic_audio):
    """Enabling accumulation alone must not bloat every frame's result dict."""
    state = _build_state(detector_params, enable_summary=True, expose_per_frame=False)
    frame_results = _stream_frames(state, deterministic_audio)
    assert "mode_1_total_energy" not in frame_results[0]
    summary = state.get_band_energy_summary()
    assert any(v > 0.0 for k, v in summary.items() if k.endswith("_energy_sum"))


def test_band_energy_summary_resets_between_clips(detector_params, deterministic_audio):
    """reset() must zero accumulated energy sums (coverage_fraction is static and unaffected)."""
    state = _build_state(detector_params, enable_summary=True)
    _stream_frames(state, deterministic_audio)
    first_summary = state.get_band_energy_summary()
    energy_keys = [k for k in first_summary if k.endswith("_energy_sum")]
    assert any(first_summary[k] > 0.0 for k in energy_keys)

    state.reset()
    zeroed_summary = state.get_band_energy_summary()
    assert all(zeroed_summary[k] == 0.0 for k in energy_keys)


def test_band_energy_summary_bands_are_overridable(detector_params, deterministic_audio):
    """A custom bands list replaces the default 16 semantic bands entirely."""
    state = _build_state(
        detector_params,
        enable_summary=True,
        band_energy_summary_bands=[("low", 400.0, 1000.0), ("high", 1000.0, 3500.0)],
    )
    _stream_frames(state, deterministic_audio)
    summary = state.get_band_energy_summary()
    assert set(summary.keys()) == {
        "low_total_energy_sum",
        "low_covered_total_energy_sum",
        "low_noise_energy_sum",
        "low_coverage_fraction",
        "high_total_energy_sum",
        "high_covered_total_energy_sum",
        "high_noise_energy_sum",
        "high_coverage_fraction",
    }


def test_unsorted_band_energy_summary_bands_raises(detector_params):
    """An unsorted custom bands list must fail loudly at construction, not silently reorder/corrupt."""
    freqs = _freqs()
    params = dict(detector_params)
    params["band_energy_summary_enable"] = True
    params["band_energy_summary_bands"] = [("high", 1000.0, 3500.0), ("low", 400.0, 1000.0)]
    with pytest.raises(ValueError, match="sorted ascending"):
        RainFrameClassifierState.from_mixin(_StreamMixin(params), freqs)


def test_streaming_band_outside_operating_band_has_zero_coverage(detector_params, deterministic_audio):
    """A band fully below operating_band reports zero coverage in the streaming path too."""
    state = _build_state(detector_params, enable_summary=True, operating_band=(400.0, 3500.0))
    _stream_frames(state, deterministic_audio)
    summary = state.get_band_energy_summary()
    assert summary["dc_coverage_fraction"] == 0.0
    assert summary["dc_noise_energy_sum"] == 0.0
    assert summary["dc_covered_total_energy_sum"] == 0.0


def test_streaming_partially_covered_band_has_fractional_coverage(detector_params, deterministic_audio):
    """A band straddling the operating_band edge reports fractional coverage in the streaming path too."""
    state = _build_state(detector_params, enable_summary=True, operating_band=(400.0, 3500.0))
    _stream_frames(state, deterministic_audio)
    summary = state.get_band_energy_summary()
    assert 0.0 < summary["mode_5_coverage_fraction"] < 1.0


# ---------------------------------------------------------------------------
# Batch vs. streaming parity
# ---------------------------------------------------------------------------


def test_batch_and_streaming_band_masks_agree_on_coverage(detector_params):
    """Verify both implementations draw the same coverage conclusion for every band.

    They build masks against different frequency-grid restrictions
    internally (full-width + band_mask in the batch path vs. pre-restricted
    self._freqs_band in the streaming path), but must agree on which bands
    are covered given the same operating_band.
    """
    freqs = _freqs()
    raw_power, noise_psd = _synthetic_clip(freqs, T=20, operating_band=(400.0, 3500.0))
    batch_params = _base_detector_params(detector_params, operating_band=(400.0, 3500.0))
    _, _, det_debug, _ = _run_batch(batch_params, freqs, raw_power, noise_psd)
    batch_summary = det_debug["band_energy_summary"]

    # coverage_fraction is static (computed once, independent of accumulated
    # frames), so it's already valid immediately after construction.
    stream_state = _build_state(detector_params, enable_summary=True, operating_band=(400.0, 3500.0))
    stream_summary = stream_state.get_band_energy_summary()

    for name, _, _ in default_spectral_occupancy_bands():
        assert batch_summary[f"{name}_coverage_fraction"] == pytest.approx(
            stream_summary[f"{name}_coverage_fraction"]
        ), name


# ---------------------------------------------------------------------------
# Processor-level wiring (SpectralNoiseProcessor.process())
# ---------------------------------------------------------------------------


def test_process_band_energy_summary_uses_unclamped_noise_psd(detector_params):
    """band_energy_summary must reflect the real (unclamped) lagged noise PSD.

    Every other test above calls _detect_rain_over_time() directly with an
    explicit noise_psd, so none of them exercise the code in
    rain_signal_processor.py's process() that decides *which* PSD (the
    second-clamped detector_noise_psd_lag vs. the real
    detector_noise_psd_lag_unclamped) reaches the diagnostic — regression test
    for the process()-wiring bug fixed in 730ace8.

    A loud-then-abruptly-quiet clip creates the transient the second clamp
    exists for: detector_noise_psd_lag[:, t] is detector_noise_psd[:, t-1]
    (shifted by one frame), so once the signal drops, the lagged noise
    estimate from the still-loud previous frame exceeds the new, near-silent
    current frame's power and gets clamped down — while the true (unclamped)
    lagged estimate does not. We independently reconstruct both variants from
    the raw per-frame estimate the processor exposes in `debug`, confirm this
    clip actually produces a meaningful divergence between them, then check
    which one the diagnostic used.
    """
    rng = np.random.default_rng(7)
    n = 2 * SAMPLE_RATE
    audio = np.empty(n, dtype=np.float64)
    audio[: n // 2] = rng.normal(0.0, 0.5, n // 2)
    audio[n // 2 :] = rng.normal(0.0, 1e-5, n - n // 2)
    audio = audio.astype(np.float32)

    params = dict(detector_params)
    params["band_energy_summary_enable"] = True
    proc_params = {
        "sample_rate": SAMPLE_RATE,
        "operating_band": (400.0, 3500.0),
        "n_fft": 256,
        "hop": 128,
        "eps": 1e-9,
        "q": 0.30,
        "detector": params,
        "classifier_only_mode": True,
        "return_detector_debug": True,
        "return_debug": True,
    }

    processor = SpectralNoiseProcessor()
    processor.setup(proc_params)
    result = processor.process(audio, sr=SAMPLE_RATE)

    debug = result["debug"]
    raw = np.asarray(debug["detector_noise_psd"])
    clamped_lag = np.asarray(debug["detector_noise_psd_lag"])
    band_mask = np.asarray(debug["band_mask"])

    # Mirrors rain_signal_processor.py's own shift-by-one (lines ~805-808).
    unclamped_lag = raw.copy()
    if unclamped_lag.shape[1] > 1:
        unclamped_lag = np.roll(unclamped_lag, shift=1, axis=1)
        unclamped_lag[:, 0] = raw[:, 0]

    unclamped_total = float(unclamped_lag[band_mask].sum())
    clamped_total = float(clamped_lag[band_mask].sum())
    # The transient must actually make the two clamp variants disagree,
    # or this test would pass regardless of which one process() wires up.
    assert unclamped_total > clamped_total * 1.05

    det_debug = result["det_debug"]
    assert det_debug.get("band_energy_summary_error") is None
    summary = det_debug["band_energy_summary"]
    emitted_total = sum(v for k, v in summary.items() if k.endswith("_noise_energy_sum"))
    assert emitted_total == pytest.approx(unclamped_total, rel=1e-5)
    assert emitted_total != pytest.approx(clamped_total, rel=1e-5)


# ---------------------------------------------------------------------------
# clip_spectral_occupancy_bands validation parity with band_energy_summary_bands
# ---------------------------------------------------------------------------


def test_invalid_clip_spectral_occupancy_bands_raises_like_band_energy_summary_bands(detector_params):
    """An invalid clip_spectral_occupancy_bands override must fail loudly too.

    Before this fix, compute_clip_spectral_occupancy_stats()'s bare
    `except Exception` demoted normalize_bands()'s ValueError (unsorted,
    overlapping, reversed, empty, or duplicate-named bands) to a soft
    clip_spectral_occupancy_error string -- inconsistent with
    band_energy_summary_bands, which raises identically in both batch and
    streaming (test_invalid_bands_raise_identically_in_batch_and_streaming_
    even_when_disabled above). A genuine runtime failure unrelated to bands
    validation should still be caught softly, so this only asserts on the
    bands-validation case.
    """
    freqs = _freqs()
    raw_power, _ = _synthetic_clip(freqs, T=20)
    params = dict(detector_params)
    params["clip_spectral_occupancy_enable"] = True
    params["clip_spectral_occupancy_bands"] = [("high", 1000.0, 3500.0), ("low", 400.0, 1000.0)]
    detector = _BatchDetector(params)
    with pytest.raises(ValueError, match="sorted ascending"):
        detector._detect_rain_over_time(
            10.0 * np.log10(raw_power + 1e-9),
            freqs,
            raw_power=raw_power,
        )
