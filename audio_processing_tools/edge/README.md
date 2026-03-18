
# 🌧️ CM7 Rain Detection Algorithm

## Overview

The **CM7 Rain Detection Algorithm** is a real-time acoustic signal processing pipeline designed to:

- Detect rain events from audio signals
- Suppress environmental noise (wind, mechanical, pivot noise)
- Provide stable rain/noise classification for downstream measurement

The system operates on audio sampled at **11,162 Hz** and is optimized for **Cortex-M7 (CM7)** deployment.

---

## 🧠 High-Level Architecture (Corrected Flow)

Raw Audio Input
│
▼
[ Preprocessing ]
│
├───────────────► [ Time-Domain Detector ]
│                         │
│                         ▼
│                  (Impulse Features)
│
▼
[ STFT → Spectrum S(f,t) ]
│
▼
[ Noise Estimation N(f,t) ]
│
▼
[ Spectral Noise Suppression / Normalization ]
│
▼
[ Noise-Suppressed Spectrum Ŝ(f,t) ]
│
▼
[ Spectral Feature Extraction ]
│
▼
[ Rain Frame Classifier (Fusion) ]
│
▼
Final Rain Decision

---

## ⚠️ Critical Design Insight

> **All spectral rain detection features are computed on the noise-suppressed (or noise-normalized) spectrum, NOT on the raw spectrum.**

This is essential because:

- Raw spectrum is dominated by wind/mechanical noise
- Rain signatures become separable only after noise compensation
- Prevents systematic false positives during noisy periods

---

## ⚙️ Processing Pipeline

---

### 1. Preprocessing

- High-pass filtering (~350 Hz) to remove low-frequency noise
- Frame segmentation:
  - Frame length: `512 samples` (~46 ms)
  - Sub-frame: ~11–12 ms

---

### 2. Time-Domain Detector

Detects impulsive rain signatures directly from waveform.

#### Features

- Sub-frame energy
- Time flux (rapid rise detection)
- Crest factor (`peak / RMS`)
- Kurtosis (impulsiveness)

#### Output

- `time_flux_score`
- `crest_factor`
- `kurtosis`
- `td_soft_label`

---

### 3. Spectral Representation

- Compute STFT:

S(f,t)

- Focus on rain-sensitive bands:
- 400–700 Hz (primary resonance)
- Harmonics (800 Hz, 1.6 kHz, etc.)

---

### 4. Noise Estimation

Tracks background noise in frequency domain.

#### Method

- Slow adaptive tracking:
- Quantile-based OR EMA
- Updated only using **non-rain frames**

#### Output

N(f,t)

---

### 5. Spectral Noise Suppression / Normalization

Transforms raw spectrum into a **rain-relevant representation**.

#### Gain Function

G(f,t) = max(gain_floor, (S(f,t) - N(f,t)) / S(f,t))

#### Suppressed Spectrum

Ŝ(f,t) = G(f,t) * S(f,t)

---

### 🔑 Interpretation

This stage effectively computes:

- Signal above noise floor
- Noise-normalized energy
- Improves rain vs noise separability

---

### 6. Spectral Feature Extraction (ON Ŝ, NOT S)

All spectral features are computed on:

Ŝ(f,t)

#### Features

- `z_primary` → normalized band energy
- `flux_primary` → spectral flux
- `z_modes`, `flux_modes`
- SNR-like ratios

---

### 7. Rain Frame Classifier (Fusion Logic)

Combines:

#### Time-Domain Features

- `time_flux_score`
- `crest_factor`
- `kurtosis`

#### Spectral Features (from Ŝ)

- `z_primary`, `z_modes`
- `flux_primary`, `flux_modes`
- Band energy ratios

---

### Decision Logic (Simplified)

Rain = (Time-domain impulse detected)
AND
(Spectral energy consistent with rain AFTER noise suppression)

---

### Soft Labelling

- Combines features into:

soft_score ∈ [0, 1]

- Used for:
- Threshold tuning
- Visualization
- Continuous confidence

---

## 📊 Outputs

Per frame:

- `is_rain`
- `rain_conf`, `noise_conf`
- `soft_score`
- `noise_psd`
- `gain G(f,t)`
- Debug features:
- `flux_primary`
- `z_modes`
- `crest_factor`
- `kurtosis`

---

## 🧩 Key Design Principles

---

### 1. Detection on Noise-Suppressed Domain

> Spectral detection operates on **Ŝ(f,t)**, not raw **S(f,t)**

---

### 2. Separation of Roles

| Component | Role |
|----------|------|
| Time-domain detector | Impulse detection |
| Noise estimator | Background tracking |
| Suppressor | Signal conditioning |
| Spectral features | Rain validation |
| Classifier | Decision fusion |

---

### 3. Robustness to Real-World Noise

Handles:

- Wind
- Pivot noise
- Mechanical vibrations

Through:

- Band-limited processing
- Noise tracking
- Multi-feature fusion

---

### 4. Real-Time Embedded Design

- Frame-based processing
- Low memory footprint
- Optimized for CM7

---

## 🔧 Key Parameters

### Time-Domain

| Parameter | Typical Value |
|----------|--------------|
| `crest_factor_min` | ~4 |
| `kurtosis_min` | ~6 |
| `time_flux_score_min` | tuned |

---

### Spectral

| Parameter | Description |
|----------|------------|
| `band_hz` | (400–700 Hz) |
| `z_primary` | normalized energy (post suppression) |
| `flux_primary` | spectral change |

---

### Noise Estimation

| Parameter | Description |
|----------|------------|
| `n_hist` | history length |
| `q` | quantile |
| `alpha` | smoothing |

---

## 📈 Debug & Visualization

Recommended plots:

- Waveform + rain decisions
- Time flux vs threshold
- Crest factor vs threshold
- Kurtosis vs threshold
- Soft score vs decision
- Spectrogram:
- Raw `S(f,t)`
- Suppressed `Ŝ(f,t)`

---

## ⚠️ Known Challenges

- Noise modulation artifacts
- Threshold tuning across devices
- Over-suppression risk
- Sensitivity to extreme wind

---

## 🚀 Future Improvements

- Adaptive thresholds
- Multi-band fusion
- Better noise normalization (local vs global)
- Kalman-based noise tracking
- Temporal smoothing

---

## 📝 Summary

The CM7 rain detection system is a:

> **Noise-aware, hybrid time-domain + spectral detection pipeline**

that:

- Detects impulsive rain signatures
- Suppresses noise BEFORE spectral decision-making
- Extracts features from noise-conditioned spectrum
- Produces robust rain detection in noisy environments

---

## 📂 Related Modules

- `time_domain_detector.py`
- `rain_frame_classifier.py`
- `spectral_noise_processor.py`
- `rain_detector.py`

---

## 👨‍🔬 Author Notes

Designed for **acoustic disdrometers in agricultural environments**, where noise is highly dynamic and device-specific.

---
