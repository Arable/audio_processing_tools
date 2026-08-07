import numpy as np

RAIN_ENERGY_THRESHOLD = 0.6
RAIN_LOG_FACTOR = 0.6
RAIN_LOG_BASE = 1.13
DSD_BINS = 32


def binning_func(energy, threshold=RAIN_ENERGY_THRESHOLD):
    return np.floor(
        np.log(1 + ((energy - threshold) * RAIN_LOG_FACTOR))
        / np.log(RAIN_LOG_BASE)
    )


def reverse_binning_func(drop_bin, threshold=RAIN_ENERGY_THRESHOLD):
    return (((np.e ** (drop_bin * np.log(RAIN_LOG_BASE))) - 1) / RAIN_LOG_FACTOR) + threshold


DSD_WEIGHTS = np.array([reverse_binning_func(i) for i in range(DSD_BINS)])


def rain_regression_model(weighted_dsd_sum):
    a = 1.41079931e-03
    b = 2.52101903e01
    c = -7.44399129e-06
    return (a * weighted_dsd_sum) / (b + c * weighted_dsd_sum)


def estimate_rain_from_audio(
    audio_data,
    fs=11162,
    frame_length=512,
    rain_low_freq=400,
    rain_high_freq=700,
    extrapolate_to_seconds=60.0,
):
    audio_data = np.asarray(audio_data, dtype=np.float64).reshape(-1)

    dF = fs / frame_length
    rain_low_idx = int(rain_low_freq // dF) + 1
    rain_high_idx = int(rain_high_freq // dF)

    n_frames = len(audio_data) // frame_length
    dsd_hist = np.zeros(DSD_BINS, dtype=np.float64)
    rain_energy_sum = 0.0
    rain_energy_frame_count = 0

    for frame_idx in range(n_frames):
        start = frame_idx * frame_length
        frame = audio_data[start : start + frame_length]

        spectrum = np.abs(np.fft.fft(frame))
        rain_energy = np.sum(spectrum[rain_low_idx : rain_high_idx + 1])

        if rain_energy > RAIN_ENERGY_THRESHOLD:
            rain_energy_sum += float(rain_energy)
            rain_energy_frame_count += 1

            bin_idx = int(binning_func(rain_energy))
            bin_idx = int(np.clip(bin_idx, 0, DSD_BINS - 1))
            dsd_hist[bin_idx] += 1

    processed_samples = n_frames * frame_length
    duration_seconds = processed_samples / fs if fs > 0 else 0.0

    weighted_dsd_sum = float(np.sum(dsd_hist * DSD_WEIGHTS))
    precip_mm = float(rain_regression_model(weighted_dsd_sum))

    return {
        "duration_seconds": duration_seconds,
        "n_frames": n_frames,
        "dsd_hist": dsd_hist,
        "weighted_dsd_sum": weighted_dsd_sum,
        "precip_mm": precip_mm,
        "rain_energy_sum": float(rain_energy_sum),
        "rain_energy_frame_count": int(rain_energy_frame_count),
    }