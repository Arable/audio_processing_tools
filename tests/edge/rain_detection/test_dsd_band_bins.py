"""Verify the legacy DSD band indices match firmware's integer-DF macros.

Firmware (CM4/Inc/disdrometer_legacy.h) computes DF = SAMPLING_FREQ/FFT_SAMPLE_LENGTH
in integer C arithmetic: 11162/512 = 21, not 21.80. With that DF the rain band
(400-700 Hz) is bins 20..33 inclusive, PFT (100-1500 Hz) is 5..70 inclusive, and
the FFT-DSD windows start at 300/21 = 14 and 1000/21 = 47.

Each frame is a pure cosine at an exact bin centre, so with the rectangular
window there is no leakage and only that one bin carries energy.
"""

import numpy as np
import pytest

from audio_processing_tools.edge.rain_estimator import estimate_rain_from_audio
from audio_processing_tools.host_analysis.device_dsd_processing_emulator import DsdProcessingEmualtor

FS = 11162
N = 512
LOUDNESS_BINS = 32


def _tone_at_bin(k: int, n_frames: int = 1) -> np.ndarray:
    n = np.arange(N * n_frames)
    return np.cos(2 * np.pi * k * n / N)


def _emulator() -> DsdProcessingEmualtor:
    return DsdProcessingEmualtor(fs=FS, frame_length=N, hop_length=N, bwindow=False)


def test_emulator_indices_match_firmware_macros():
    emu = _emulator()
    assert emu.dF == 21
    assert (emu.rain_low_idx, emu.rain_high_idx) == (20, 33)
    assert (emu.pft_low_idx, emu.pft_high_idx) == (5, 70)
    assert (emu.lwin_start_idx, emu.lwin_end_idx) == (14, 32)
    assert (emu.hwin_start_idx, emu.hwin_end_idx) == (47, 65)


@pytest.mark.parametrize("k, counted", [(19, False), (20, True), (33, True), (34, False)])
def test_emulator_rain_band_edges(k, counted):
    emu = _emulator()
    emu.process_audio_frame(_tone_at_bin(k))
    assert bool(emu.energy_histogram[:LOUDNESS_BINS].sum() == 1) is counted


def test_emulator_pft_includes_high_index():
    emu = _emulator()
    emu.process_audio_frame(_tone_at_bin(70))
    assert emu.peak_histogram[70] == 1


@pytest.mark.parametrize("k, counted", [(19, False), (20, True), (33, True), (34, False)])
def test_rain_estimator_band_edges(k, counted):
    out = estimate_rain_from_audio(_tone_at_bin(k), fs=FS, frame_length=N)
    assert out["n_frames"] == 1
    assert (out["rain_energy_frame_count"] == 1) is counted
