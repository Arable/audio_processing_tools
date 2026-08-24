"""Verify band_freq_mask's half-open-except-last convention.

A bin sitting exactly on a shared boundary between two adjacent bands must be
counted in exactly one of them, never both and never neither. For this
module's actual config (fs=11162, n_fft=256), every boundary in
default_spectral_occupancy_bands() lands exactly on an FFT bin, so this is
not a hypothetical edge case.
"""

import numpy as np
import pytest

from audio_processing_tools.edge.feature_extraction import (
    band_freq_mask,
    compute_clip_spectral_occupancy_stats,
    default_spectral_occupancy_bands,
    normalize_bands,
)

FS = 11162
N_FFT = 256
BIN_HZ = FS / N_FFT  # 43.6015625


def _freqs():
    return np.fft.rfftfreq(N_FFT, d=1.0 / FS)


def test_every_default_band_boundary_lands_exactly_on_a_bin():
    """Check the precondition the rest of this file relies on.

    If this ever stops being true (e.g. fs/n_fft changes), the
    double-counting risk becomes probabilistic rather than guaranteed, and
    these tests should be revisited.
    """
    bands = default_spectral_occupancy_bands()
    boundaries = sorted({b for _, lo, hi in bands for b in (lo, hi)})
    for b in boundaries:
        k = b / BIN_HZ
        assert abs(k - round(k)) < 1e-9, f"{b} Hz is not an exact bin multiple"


def test_boundary_bin_counted_exactly_once_not_zero_not_twice():
    """A bin exactly on a shared boundary must land in exactly one band."""
    freqs = _freqs()
    bands = default_spectral_occupancy_bands()
    n_bands = len(bands)
    # mode_1/inter_1 boundary: 654.0234375 Hz.
    boundary_hz = 654.0234375
    bidx = int(np.argmin(np.abs(freqs - boundary_hz)))
    assert freqs[bidx] == boundary_hz

    masks = [
        band_freq_mask(freqs, lo, hi, is_last_band=(i == n_bands - 1))
        for i, (_, lo, hi) in enumerate(bands)
    ]
    hit_count = sum(int(mask[bidx]) for mask in masks)
    assert hit_count == 1

    # Half-open convention: the boundary belongs to the band whose lo == boundary
    # (inter_1), not the band whose hi == boundary (mode_1).
    names = [name for name, _, _ in bands]
    hit_names = [names[i] for i, mask in enumerate(masks) if mask[bidx]]
    assert hit_names == ["inter_1"]


def test_last_band_boundary_is_inclusive():
    """Check the outer edge of the final band (mode_5's upper bound).

    It must still be included — is_last_band=True must be closed, not
    half-open, or the very top bin would be dropped entirely.
    """
    freqs = _freqs()
    bands = default_spectral_occupancy_bands()
    last_name, last_lo, last_hi = bands[-1]
    assert last_name == "mode_5"
    bidx = int(np.argmin(np.abs(freqs - last_hi)))
    assert freqs[bidx] == last_hi

    mask = band_freq_mask(freqs, last_lo, last_hi, is_last_band=True)
    assert mask[bidx]


def test_clip_spectral_occupancy_does_not_double_count_boundary_energy():
    """Integration check against compute_clip_spectral_occupancy_stats.

    A spike exactly on a shared boundary must contribute to total clip
    energy exactly once, not twice (double-counted) or zero times (dropped).
    """
    freqs = _freqs()
    T = 4
    boundary_hz = 654.0234375
    bidx = int(np.argmin(np.abs(freqs - boundary_hz)))
    raw_power = np.zeros((len(freqs), T), dtype=np.float64)
    raw_power[bidx, :] = 1000.0
    frame_class = np.zeros(T, dtype=np.int8)  # all "not rain" (frame_class != 2)

    stats = compute_clip_spectral_occupancy_stats(
        raw_power=raw_power, freqs=freqs, frame_class=frame_class
    )
    band_names = list(stats["band_names"])
    mode_1_idx = band_names.index("mode_1")
    inter_1_idx = band_names.index("inter_1")

    # log1p(1000) should show up in exactly one of the two adjacent bands' mean
    # log power, and be exactly zero in the other.
    mode_1_val = stats["no_rain_log_power_mean"][mode_1_idx]
    inter_1_val = stats["no_rain_log_power_mean"][inter_1_idx]
    assert mode_1_val == 0.0
    assert inter_1_val > 0.0


def test_normalize_bands_rejects_unsorted_custom_bands():
    """A caller-supplied bands list out of ascending-lo order must be rejected.

    band_freq_mask's is_last_band convention assumes iteration order matches
    frequency order — the last band in the sequence is the one that gets the
    closed upper bound. Silently re-sorting an out-of-order list would risk
    surprising an existing caller of compute_clip_spectral_occupancy_stats()
    who relies on positional alignment between their supplied bands and its
    output arrays (band_names, rain_log_power_mean, etc.) — so an out-of-order
    list is rejected with a clear error instead, rather than silently
    reordered or silently mis-masked.
    """
    unsorted = [("top", 2790.5, 3575.328125), ("mid", 436.015625, 2790.5)]
    with pytest.raises(ValueError, match="sorted ascending"):
        normalize_bands(unsorted)


def test_normalize_bands_accepts_already_sorted_custom_bands():
    """An already-ascending custom bands list must pass through unchanged."""
    sorted_bands = [("mid", 436.015625, 2790.5), ("top", 2790.5, 3575.328125)]
    normalized = normalize_bands(sorted_bands)
    assert normalized == (
        ("mid", 436.015625, 2790.5),
        ("top", 2790.5, 3575.328125),
    )


def test_normalize_bands_defaults_to_spectral_occupancy_bands():
    """None must default to the same 16 bands clip_spectral_occupancy uses."""
    assert normalize_bands(None) == default_spectral_occupancy_bands()


def test_sorted_custom_bands_conserve_total_energy():
    """Integration check: a sorted custom bands list must not drop or double-count energy.

    Regression test for the confirmed bug where an unsorted
    band_energy_summary_bands override would have silently dropped the true
    top-of-spectrum edge bin and double-counted an internal boundary bin —
    now caught by normalize_bands() raising instead, so only the correctly
    sorted (accepted) case needs to conserve energy exactly.
    """
    freqs = _freqs()
    T = 3
    raw_power = np.zeros((len(freqs), T))

    # Top-of-spectrum edge bin (3575.328125 Hz) must be counted exactly once.
    top_idx = int(np.argmin(np.abs(freqs - 3575.328125)))
    assert freqs[top_idx] == 3575.328125
    raw_power[top_idx, :] = 1000.0

    # Internal boundary bin (2790.5 Hz) must be counted exactly once.
    boundary_idx = int(np.argmin(np.abs(freqs - 2790.5)))
    assert freqs[boundary_idx] == 2790.5
    raw_power[boundary_idx, :] = 500.0

    sorted_bands = [("mid", 436.015625, 2790.5), ("top", 2790.5, 3575.328125)]
    normalized = normalize_bands(sorted_bands)
    n_bands = len(normalized)
    total_by_band = {}
    for i, (name, lo, hi) in enumerate(normalized):
        mask = band_freq_mask(freqs, lo, hi, is_last_band=(i == n_bands - 1))
        total_by_band[name] = float(np.sum(raw_power[mask, :]))

    expected_total = float(np.sum(raw_power))
    assert sum(total_by_band.values()) == expected_total
