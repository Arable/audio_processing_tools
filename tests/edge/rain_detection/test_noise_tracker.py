"""Unit tests for causal PSD tracking invariants."""

import numpy as np

from audio_processing_tools.edge.noise_tracker import CausalNoiseTracker


def test_constant_power_converges_without_exceeding_input():
    """Require a bounded, nonnegative estimate for stationary input power."""
    tracker = CausalNoiseTracker(n_bins=4, q=0.25, fs=11_162, hop=128)
    power = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    estimates = np.asarray([tracker.update(power) for _ in range(100)])
    assert tracker.warmup_complete
    np.testing.assert_allclose(estimates[-1], power, rtol=0.0, atol=1e-4)
    assert np.all(estimates >= 0.0)
    assert np.all(estimates <= power)


def test_reset_restores_first_update_behavior():
    """Require reset to remove all history from the tracker."""
    tracker = CausalNoiseTracker(n_bins=3)
    power = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    first = tracker.update(power).copy()
    tracker.update(power * 0.25)
    tracker.reset()
    np.testing.assert_array_equal(tracker.update(power), first)
