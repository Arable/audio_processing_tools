📄 Noise Suppressor & Rain Detector

Overview
This system enables accurate rain detection and measurement in environments where acoustic signals are heavily contaminated by wind and slowly varying background noise.
This module implements a real-time rain detection and spectral noise suppression pipeline for acoustic disdrometer signals (Fs ≈ 11.16 kHz).

It is designed for noisy outdoor environments, particularly where signals are affected by wind and slowly varying background noise.

The system is intended to be robust to:
	•	Wind-induced noise
	•	Slowly varying environmental and structural noise

Note:
	This design has not been exhaustively validated across all possible deployment conditions. Performance may vary in environments with strong mechanical or highly structured vibrations.

⸻

🧱 Architecture

Input Audio
   │
   ▼
[Pre-filter]
   │
   ▼
[STFT → Power Spectrum]
   │
   ├── Detector Path
   │     ├── FD features (mode flux)
   │     ├── TD features (crest, kurtosis)
   │     ├── Noise-normalized spectrum (optional)
   │     └── Frame classification
   │
   └── Suppressor Path
         ├── Noise PSD estimation (causal)
         ├── Adaptive gain computation
         ├── Spectral suppression
         └── ISTFT (optional)
   │
   ▼
[Clip Aggregation]
   │
   ▼
Outputs (metrics + state)


The system consists of two tightly coupled components:

1. Detector Path
   - Extracts FD + TD features
   - Classifies frames as rain or noise

2. Suppressor Path
   - Estimates noise PSD
   - Applies adaptive spectral suppression

The detector and suppressor are interdependent:
- The suppressor provides a time-lagged noise estimate used to normalize detector inputs
- Detector and suppressor operate on the same frame-level spectral representation

⸻

🎯 Design Goals

Robust Detection
	•	Detect impulsive rain signatures
	•	Reject wind / mechanical noise

Noise-Robust Measurement
	•	Suppress non-rain energy
	•	Preserve rain spectral structure

Edge Deployment
	•	Causal / low latency
	•	Float32 support
	•	Configurable + modular

⸻
🧠 Core Design Principles

1. Frame-Level Causal Processing

The pipeline operates in a frame-causal manner (with small optional latency depending on STFT configuration):
	•	Frame decisions depend only on:
	•	Current frame t
	•	Past frames ≤ t
	•	No future information is used in:
	•	Noise PSD estimation
	•	Gain computation
	•	Frame classification

This ensures:
	•	✅ Real-time deployability (CM4 / CM7)
	•	✅ Stable behavior in streaming conditions
	•	✅ No look-ahead bias

⚠️ Note:
	•	STFT with center=True introduces symmetric latency
	•	Setting center=False enables strictly causal execution for firmware deployment

⸻

2. Time-Lagged Noise Estimate (Key Innovation)

A critical design feature is the use of lagged noise PSD:

N_lag(t) = N(t-1)

This is used in two places:

A. Detector input normalization

log(P(t)) - log(N_lag(t))

B. Gain computation

G(t) uses N_lag(t) instead of N(t)


⸻

🎯 Why Lagged PSD Matters

Problem (without lag)

If current frame PSD is used:
	•	Rain energy leaks into noise estimate
	•	Suppression becomes self-referential
	•	Detector sees distorted spectrum

⸻

Solution (with lag)

Using N(t-1) ensures:
	•	✅ No leakage of current frame signal into noise estimate
	•	✅ Stable normalization (true “above-noise” signal)
	•	✅ Cleaner separation between:
	•	rain transients
	•	slowly varying noise floor

💡 This effectively converts the system into a causal, frame-level SNR estimator

⸻

🌧️ Handling Noisy Conditions (Key Strength)

This design is explicitly optimized for non-stationary, real-world noise.

⸻

1. Noise-Normalized Detection

Instead of absolute energy:

Detection ∝ P(t) / N(t-1)

➡ Rain is detected as energy above noise baseline, not raw amplitude.

Effect:
	•	Wind bursts → suppressed
	•	Mechanical noise → reduced when it behaves like slowly varying background energy
	•	Rain impulses → preserved

⸻

2. Stable Noise PSD Estimation

Noise PSD is estimated using a causal low-quantile style tracker on the spectral power, without relying on detector frame decisions in the current configuration.

Effect:
	•	Avoids feedback from detector errors into the PSD estimate
	•	Provides a stable background-noise estimate for normalization and suppression

⸻

3. Adaptive Suppression via Confidence

oversub ∝ noise_conf

	•	Rain-like frames → low suppression
	•	Noise-like frames → strong suppression

Effect:
	•	Preserves rain structure
	•	Aggressively removes noise

⸻

4. SNR-Based Protection Layer

Optional gating:

High SNR → reduce suppression

Effect:
	•	Strong rain impulses are protected
	•	Prevents over-suppression

⸻

5. Asymmetric PSD Tracking

ema_up  (fast)
ema_down (slow)

Effect:
	•	Quickly adapts to increasing noise
	•	Slowly decays → avoids instability

⸻

📌 Combined Effect

These mechanisms together create:

A system that detects relative energy changes (rain)
while tracking slow background noise (wind/mechanical)

⸻

⚠️ Known Failure Modes (Important for Deployment)

- Rapidly changing noise:
  - Noise PSD may lag, causing temporary under- or over-suppression

- Very low rain intensity:
  - May fall below noise floor and be missed

- Strong impulsive mechanical noise:
  - Can resemble rain in TD features

- Frame misclassification: Incorrect rain/noise labels can affect clip-level decisions and suppression behavior

⸻

🆚 Why This Works Better in Noisy Environments

| Challenge             | Typical System               | This Design                              |
|----------------------|------------------------------|------------------------------------------|
| Wind bursts          | Misclassified as rain        | Normalized out                           |
| Mechanical vibration | Strong false positives       | Filtered by TD + normalization           |
| Non-stationary noise | PSD drift or slow adaptation | Quantile tracking + lagged normalization |
| Rain + noise overlap | Suppressed                   | Preserved via confidence + SNR           |


⸻

✅ Suggested One-Line Summary (for slides / exec)

“Rain is detected as energy above a continuously learned noise baseline, enabling robust performance in highly noisy environments.”

⸻


🔊 Processing Pipeline

1. Pre-filter

pre_filter_mode = "highpass" | "bandpass" | "none"

	•	High-pass → removes structural noise
	•	Band-pass → restricts to operating band (400–3500 Hz)

⸻


2. STFT

n_fft = 256
hop = 128
window = "hann"

Outputs:
	•	S → complex spectrum
	•	P = |S|² → power

⸻

3. Rain Detection

Implemented via RainFrameClassifierMixin.

Input Modes
	•	Absolute spectrum (dB)
	•	Noise-normalized spectrum:

log(P) - log(N_lag)

➡ Acts like instantaneous SNR

⸻

3.1 Features

Frequency-domain
	•	Mode bands (e.g. 450–650 Hz, etc.)
	•	Mode flux score (primary feature)
	•	Optional peak features

Time-domain
	•	Crest factor (strong)
	•	Kurtosis (very strong)
	•	Time flux (observed to be less discriminative in current datasets)	

⸻

3.2 Classification

Outputs:

frame_class ∈ {RAIN, NOISE}
rain_conf ∈ [0,1]
noise_conf = 1 - rain_conf


⸻

🌊 Noise PSD Estimation

Method: Causal Quantile Tracking

Tracks low quantile of power spectrum:

if P > tracker:
    increase slowly
else:
    decrease faster

Key Parameters

q = 0.25
win_sec = 0.5
ema_up / ema_down

Properties
	•	Causal
	•	Handles non-stationary noise
	•	Independent of detector decisions in current configuration

⸻

🔉 Gain Computation

Adaptive Oversubtraction

oversub = base + noise_conf * (max - base)

Typical:
	•	base = 1.0
	•	max = 3.0

⸻

Gain Modes

sqrt subtraction (default)

G = 1 - α * sqrt(N / P)

Wiener

G = max(P - αN, 0) / P


⸻

Stabilization
	•	Frequency smoothing → reduces musical noise
	•	Temporal smoothing → disabled for rain frames

⸻

SNR Gating (Optional)

Reduces suppression when SNR is high:

if SNR high → reduce suppression


⸻

🔁 Suppression

S_hat = G * S

Optional:

y = ISTFT(S_hat)


⸻

📊 Clip-Level Aggregation

Clip-level aggregation is used to improve robustness by requiring consistent evidence of rain across multiple frames, rather than relying on isolated detections.

Rain Decision

clip_is_rain = rain_frame_count >= clip_rain_min_frames

This ensures that short-duration noise bursts or isolated misclassifications do not trigger rain detection at the clip level.

⸻

Confidence

median_conf = median(rain_conf on rain frames)
abundance_conf = rain_frame_count / (2 * min_frames)

clip_rain_conf = max(median_conf, abundance_conf)


⸻

📦 Outputs

Metrics

{
  "clip_is_rain": bool,
  "clip_rain_conf": float,
  "rain_frame_count": int,
  "clip_rain_fraction": float,
  "mean_noise_floor_db": float,
}

State

{
  "frame_class": [...],
  "rain_conf": [...],
  "noise_conf": [...],
  "features": {...},
  "debug": {...},
}


⸻

⚙️ Configuration

{
  "suppressor": {...},
  "detector": {...}
}

Precedence:

flat params > nested params > defaults


⸻

🧪 Modes

Mode	Description
Full	Detection + suppression
classifier_only_mode	Feature extraction only
disable_suppression	Detector only


⸻

✅ Strengths
	•	Noise-normalized detection (core differentiator)
	•	Strong TD features (crest, kurtosis)
	•	Confidence-driven suppression
	•	Modular design (detector vs suppressor)
	•	Edge-friendly implementation

⸻

⚠️ Limitations
	•	FD features are highly correlated
	•	Threshold-based decision logic
	•	PSD tracking may lag under rapidly changing noise conditions
	•	Potential overfitting on small datasets

⸻

🔧 Improvement Opportunities

Detector
	•	Replace thresholds with learned model
	•	Feature decorrelation

Suppressor
	•	Multi-timescale PSD tracking
	•	Band-wise adaptive suppression

System
	•	Secondary microphone (noise reference)
	•	Event-level detection
	•	Distance-aware modeling

⸻

📌 Summary

This pipeline provides a robust baseline for rain detection in wind-affected and slowly varying noisy environments, combining:
	•	Spectral + temporal features
	•	Adaptive noise modeling
	•	Confidence-aware suppression
