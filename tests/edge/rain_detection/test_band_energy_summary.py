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
    n_frames = len(audio) // HOP - 2
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
        batch_covered = batch_summary[f"{name}_coverage_fraction"] > 0.0
        stream_covered = stream_summary[f"{name}_coverage_fraction"] > 0.0
        assert batch_covered == stream_covered, name
