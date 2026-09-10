"""Fast, self-contained regression checks for the CM7 rain detector."""

from typing import Any

import numpy as np

from audio_processing_tools.edge.rain_signal_processor import SpectralNoiseProcessor

from .conftest import SAMPLE_RATE


def _process(params: dict[str, Any], audio: np.ndarray) -> dict[str, Any]:
    processor = SpectralNoiseProcessor()
    processor.setup(params)
    return processor.process(audio, sr=SAMPLE_RATE)


def test_batch_and_frame_replay_are_identical(processor_params, deterministic_audio):
    """Require batch classification to match its frame replay reference."""
    result = _process(processor_params, deterministic_audio)
    replay = result["frame_level_comparison"]
    assert "error" not in replay
    np.testing.assert_array_equal(result["frame_class"], replay["frame_class"])
    np.testing.assert_array_equal(result["rain_conf"], replay["rain_conf"])


def test_streaming_has_no_unexplained_frequency_domain_drift(processor_params, deterministic_audio):
    """Allow documented TD timing differences but reject FD state drift."""
    result = _process(processor_params, deterministic_audio)
    streaming = result["streaming_comparison"]
    replay = result["nowinsor_replay_comparison"]
    assert "error" not in streaming
    assert "error" not in replay

    warmup = 2
    np.testing.assert_allclose(np.asarray(streaming["primary_mode_flux"])[warmup:],
                               np.asarray(replay["primary_mode_flux"])[warmup:],
                               rtol=0.0, atol=1e-5)
    disagreements = (np.asarray(streaming["frame_class"])[warmup:]
                     != np.asarray(replay["frame_class"])[warmup:])
    stream_gate = np.asarray(streaming["td_gate_mask"], dtype=bool)[warmup:]
    replay_gate = np.asarray(replay["td_gate_mask"], dtype=bool)[warmup:]
    # Every class difference must be explained by documented causal TD timing.
    assert not np.any(disagreements & (stream_gate == replay_gate))


def test_detector_is_deterministic(processor_params, deterministic_audio):
    """Require repeat runs with identical input and configuration to match."""
    first = _process(processor_params, deterministic_audio)
    second = _process(processor_params, deterministic_audio)
    np.testing.assert_array_equal(first["frame_class"], second["frame_class"])
    np.testing.assert_array_equal(first["rain_conf"], second["rain_conf"])
    np.testing.assert_array_equal(first["frame_level_comparison"]["primary_mode_flux"],
                                  second["frame_level_comparison"]["primary_mode_flux"])
