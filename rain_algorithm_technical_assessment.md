# Technical Assessment: Arable Mark Rain Detection and Quantification Algorithm

**Prepared for the Arable Data Science Team**
**March 2026**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview and Physical Principles](#2-system-overview-and-physical-principles)
3. [Algorithm Walkthrough: Rain Detection](#3-algorithm-walkthrough-rain-detection)
4. [Algorithm Walkthrough: Rain Quantification (DSD)](#4-algorithm-walkthrough-rain-quantification-dsd)
5. [The Noise Suppression System (In Development)](#5-the-noise-suppression-system-in-development)
6. [Literature Context and State of the Art](#6-literature-context-and-state-of-the-art)
7. [Assumptions and Their Validity](#7-assumptions-and-their-validity)
8. [Strengths](#8-strengths)
9. [Weaknesses and Risks](#9-weaknesses-and-risks)
10. [Improvement Opportunities](#10-improvement-opportunities)
11. [References](#11-references)

---

## 1. Executive Summary

The Arable Mark rain measurement system is an **acoustic disdrometer**: it uses a microphone inside a resonance chamber to detect and quantify rainfall from the sound of drops impacting the dome. The system operates on 10 seconds of audio every minute, sampled at 11,162 Hz.

There are two distinct algorithms:

1. **Rain Detection** (`dsp_rain_detection.py`): Determines *whether* it is raining. This is a gate — if it says no rain, the system reports zero precipitation.
2. **Rain Quantification** (`device_dsd_processing_emulator.py`): Determines *how much* rain, by building a 32-bin drop size distribution (DSD) histogram. A cloud-side regression model converts the DSD into millimeters of precipitation.

These algorithms are complementary but architecturally independent. The detection algorithm uses spectral novelty analysis across resonant harmonics with time-domain validation. The quantification algorithm uses per-frame energy binning in the fundamental resonance band.

This assessment walks through each algorithm step by step, explains the signal processing concepts involved, evaluates the assumptions, and identifies opportunities for improvement.

---

## 2. System Overview and Physical Principles

### 2.1 The Resonance Chamber

The Mark device has a rounded dome with a microphone mounted underneath. This dome acts as a **resonance chamber** — a structure with natural vibration modes at specific frequencies. The fundamental resonance frequency is approximately **500–600 Hz**, with harmonics at integer multiples (roughly 1000, 1500, 2000 Hz, etc.).

**Why this matters:** When any force disturbs the dome — a raindrop impact, wind pressure, mechanical vibration — the dome responds most strongly at its natural frequencies. The microphone picks up this filtered version of the disturbance. The resonance acts as a natural amplifier at specific frequencies, which concentrates the signal energy and makes it easier to detect above broadband electronic noise.

### 2.2 Rain Drops as Impulse Excitations

A raindrop hitting the dome produces a very brief, sharp force — in signal processing terms, something close to a **Dirac delta function** (an idealized instantaneous impulse). The key property of an impulse is that it excites *all* frequencies simultaneously. When an impulse hits a resonant system, the response is a burst of energy at *every* resonant mode — the fundamental and all its harmonics light up together, then decay exponentially.

This means rain should produce:
- **Broadband harmonic excitation** — energy appearing simultaneously at f₀, 2f₀, 3f₀, etc.
- **Transient, impulsive temporal pattern** — short bursts with rapid onset and exponential decay.
- **High kurtosis** — the amplitude distribution has heavy tails (sharp spikes on a quiet background).
- **High crest factor** — the peak amplitude is much larger than the RMS (average) level.

### 2.3 Wind as Continuous Narrowband Excitation

Wind interacts with the dome differently. Turbulent wind pressure fluctuations are continuous (not impulsive) and have a smooth spectrum that rolls off at higher frequencies — most wind energy is below 500 Hz (Strasberg, 1988; Nelke & Vary, 2014). When wind excites the resonance chamber, it preferentially drives the **fundamental mode** (and perhaps the second harmonic), because these are the lowest-frequency modes where wind energy is strongest.

This means wind should produce:
- **Narrowband excitation** — energy concentrated at the fundamental, with much less at higher harmonics.
- **Slowly varying, continuous temporal pattern** — energy changes on the timescale of wind gusts (seconds), not milliseconds.
- **Lower kurtosis** — the amplitude distribution is closer to Gaussian (no sharp spikes).
- **Lower crest factor** — no isolated peaks, just continuous variation.

### 2.4 The Core Discrimination Challenge

The central challenge of the algorithm is **distinguishing rain-excited resonance from wind-excited resonance**. Both produce energy in the same frequency bands (because both excite the same physical structure), but they differ in:

1. **How many harmonics are excited** (rain: many; wind: few)
2. **How the energy varies over time** (rain: impulsive transients; wind: slow modulation)
3. **The statistical character of the waveform** (rain: super-Gaussian; wind: Gaussian)

The current algorithm exploits all three of these differences, though not equally. The primary detection relies on (1), with (2) and (3) used for validation.

### 2.5 Audio Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 11,162 Hz | Nyquist frequency = 5,581 Hz |
| Recording duration | 10 seconds | Per minute, gated by 1-second amplitude check |
| FFT size (detection) | 256 samples | → 43.6 Hz frequency resolution |
| Hop size (detection) | 128 samples | → 11.5 ms time resolution |
| FFT size (DSD) | 512 samples | → 21.8 Hz frequency resolution |
| Operating band | 400–3,500 Hz | Covers fundamental through ~7th harmonic |
| Fundamental resonance | ~500–600 Hz | Empirical range; varies slightly by device |

**A note on sample rate:** The 11,162 Hz sample rate is somewhat unusual. At a Nyquist limit of ~5.6 kHz, the system can capture up to approximately the 9th–11th harmonic (for f₀ = 500–600 Hz). This is sufficient for harmonic analysis but limits the ability to capture very high-frequency content that might help with drop size characterization (some literature suggests useful information extends to 10+ kHz). The origin of this specific rate is unknown.

---

## 3. Algorithm Walkthrough: Rain Detection

The rain detection algorithm (`dsp_rain_detection.py`, function `analyse_raw_audio`) processes 10 seconds of audio and returns a binary rain/no-rain decision plus a drop count. It has five stages.

### 3.1 Stage 1: Pre-processing

**What happens:**
1. The raw 10-second audio (float32, normalized to [-1, 1]) is **bandpass filtered** to [400, 3500] Hz using a 4th-order Butterworth IIR filter.
2. The **Short-Time Fourier Transform (STFT)** is computed: 256-sample windows, 128-sample hop, Hann window. This produces a spectrogram of shape (129 frequency bins × ~870 time frames).
3. The magnitude spectrum `Y = |STFT|` is extracted.

**Why we do this:**
- The bandpass filter removes energy outside the operating range. Below 400 Hz, wind noise dominates and there are no useful resonance modes. Above 3500 Hz, the signal-to-noise ratio degrades and the higher harmonics are weak.
- The STFT converts the time-domain signal into a time-frequency representation, allowing us to analyze how energy is distributed across frequencies at each moment. The 256-sample FFT gives ~43.6 Hz resolution, which is sufficient to resolve individual harmonics (spaced ~500–600 Hz apart). The 128-sample hop gives ~11.5 ms time resolution, fast enough to resolve individual raindrop impacts (which have rise times of ~1–5 ms).
- The Hann window reduces spectral leakage — without windowing, the sharp edges of each FFT frame create artifacts that spread energy across frequency bins.

**Assessment:** The pre-processing is sound and well-parameterized for the system's acoustic properties. The bandpass lower cutoff at 400 Hz is just below the fundamental, which is appropriate — you want to see the fundamental but don't need the sub-400 Hz range where wind noise dominates.

### 3.2 Stage 2: Fundamental Frequency Estimation

**What happens:**
1. Within the [400, 700] Hz band (the expected range of the dome's fundamental resonance), the algorithm finds the **spectral peak** in each time frame using `scipy.signal.find_peaks`.
2. The peak frequencies across all frames are averaged (using a frequency-weighted or mode-based average) to produce `frain_mean` — the estimated fundamental resonance frequency for this 10-second window.

**Why we do this:**
- The dome's fundamental frequency is not perfectly constant — it varies slightly with temperature, humidity, mounting stress, and manufacturing tolerances. Rather than hardcoding a fixed frequency, the algorithm estimates it from the data. This is important because the harmonic search (Stage 3) needs to know where to look for harmonics at 2×f₀, 3×f₀, etc.

**Assessment:** Adaptive fundamental estimation is a good approach. However, there is a chicken-and-egg problem: we're looking for the peak frequency in a band that might contain *wind* energy, not just rain. If wind is exciting the fundamental strongly, `frain_mean` will reflect the wind-driven resonance, which is fine (it's still the dome's natural frequency). But if there is no rain and only ambient noise, the estimated frequency may be unreliable or meaningless. The algorithm handles this implicitly by gating later stages on novelty thresholds — if there's no significant spectral activity, the novelty will be zero regardless of the estimated frequency.

### 3.3 Stage 3: Spectral Novelty at Fundamental + Harmonics

This is the **core detection mechanism**. It answers: "Are there sudden energy bursts happening at the dome's resonant frequencies?"

#### 3.3.1 Per-Band Spectral Novelty

For each harmonic band (fundamental, 2nd harmonic, ..., up to 6th), the algorithm:

1. **Isolates the frequency band**: Copies the magnitude STFT and zeros out all bins except a ~300 Hz band centered on the expected harmonic frequency. For the fundamental, this is [F_natural, F_natural + 300]. For the nth harmonic, approximately [n × frain_mean - 100, n × frain_mean + 200].

2. **Computes spectral gradient energy** (`compute_novelty_spectrum_new`):
   - Takes the first difference of the magnitude spectrum **along the frequency axis** (not time): `Y_diff = diff(Y, axis=0)`. This computes the slope of the spectrum between adjacent frequency bins.
   - **Half-wave rectifies**: Keeps only positive differences (spectrum rising with frequency), zeros out negative differences.
   - **Sums over frequency** for each time frame → a single scalar per frame.

   **What this measures:** For each time frame, this produces a number that is large when the isolated band contains a **spectral peak** (a resonance being excited) and small when the band is flat or quiet. The positive-gradient-only approach means it captures the rising edge of spectral peaks. When the band has been zeroed outside the region of interest, the sum is dominated by the magnitude of spectral energy at the lower edge of the band plus any peak structure within it. In effect, this is a rough measure of "how much resonant energy is present in this harmonic band at this moment."

3. **Normalizes by local noise floor** (`calculate_snr`):
   - Estimates the noise floor using `compute_local_average`: for each frame, takes a window of ±M frames (M ≈ 20, corresponding to ~470 ms), sorts the spectral novelty values, and averages the **lowest M/6 values** (approximately the bottom ~3 values).
   - Divides the novelty by this noise floor estimate → an SNR-like ratio.

   **Why:** This is a form of **minimum statistics noise estimation** (cf. Martin, 2001). By taking the mean of the quietest samples in a local window, we estimate the background level of spectral energy in this band when nothing interesting is happening. Dividing by this gives us a measure of "how many times louder is this frame compared to the quiet background?" This makes the detection adaptive to the current noise environment — a windy period will have a higher baseline, so rain needs to be proportionally louder to trigger detection. This is conceptually similar to the MCRA (Minima Controlled Recursive Averaging) approach used in speech enhancement.

4. **Peak selection and thresholding**:
   - Only local peaks (detected by `scipy.signal.find_peaks`) in the SNR signal are kept; non-peak frames are zeroed. This ensures we detect discrete **onset events** rather than sustained energy.
   - Values below the threshold (e.g., 4.5 for the fundamental) are zeroed. Values above 1.5× threshold are clipped to 1.5× threshold.

   **Why the clipping:** This prevents a single very loud event (e.g., a nearby gunshot or a large raindrop) from dominating the novelty sum. It effectively normalizes "strength of detection" so that one enormous peak doesn't count more than ~1.5× a normal detection.

5. **Peak gating**: For each frame, separately check whether a spectral peak actually exists in the expected frequency range using `find_peaks_in_frequency_range`. If no peak is found, the novelty is zeroed.

   **Why:** This is a sanity check. Even if the novelty metric fires, if there isn't actually a visible peak at the expected harmonic frequency in the raw spectrum, it's probably a false alarm.

#### 3.3.2 Harmonic Combination

After computing per-band novelty for the fundamental and up to 5 additional harmonics:

1. **Fundamental gates all harmonics**: If the fundamental novelty is zero for a frame, the novelty for all harmonics in that frame is also set to zero.

   **Why we do this:** A rain drop should excite the fundamental and harmonics simultaneously. If the fundamental isn't active, harmonic activity is probably noise or an artifact. This is a physically motivated constraint.

   **A concern:** This makes the algorithm entirely dependent on detecting activity at the fundamental. If the fundamental is masked by wind noise (which concentrates energy there), or if the fundamental happens to be weak for a particular drop size, the algorithm will miss the event even if the harmonics clearly show it.

2. **Sum across harmonics**: `nov_hn = sum(nov[0], nov[1], ..., nov[N])` per frame.

3. **Binary rain decision**: Frames where `nov_hn > rain_thr_hn` (the sum of the top 3 harmonic thresholds, = 12.0) are classified as "raining." All other frames are classified as "not raining."

4. **Rain drop count**: `rain_drops = number of frames where raining >= 1`.

**Assessment of Stage 3:**
- The multi-harmonic approach is well-motivated by the physics: rain excites all harmonics, so requiring activity at multiple harmonics should reduce false positives from narrowband noise sources.
- The minimum-statistics noise normalization is a solid adaptive approach from the speech processing literature.
- However, the computation labeled "novelty" is somewhat unusual — it's a frequency-domain gradient measure, not a time-domain onset detector. It captures spectral *structure* (peakedness in a band) rather than temporal *transience*. This means a sustained tonal interference at a harmonic frequency could fire this detector just as readily as a rain impact, as long as it exceeds the local noise floor. The time-domain validation in Stage 4 partially addresses this.

### 3.4 Stage 4: Time-Domain Validation

When false-positive handling is enabled (default: `handle_fp = True`), the algorithm computes time-domain statistics to validate the spectral detection.

**What happens:**
1. The audio (already bandpass filtered to [400, 3500] Hz) is used.
2. Per-frame (256 samples, 128-sample hop), three features are computed:
   - **Kurtosis** (Fisher): Measures the "tailedness" of the amplitude distribution within the frame. A Gaussian signal has kurtosis = 0 (Fisher convention). Impulsive signals (like rain impacts) have kurtosis >> 0 because they have sharp peaks and a quiet background. The threshold is 2.5.
   - **Crest factor**: The ratio of peak absolute amplitude to RMS amplitude. Rain impacts produce isolated peaks much larger than the average level. The threshold is 3.75.
   - **Differential energy**: The ratio of the current frame's energy to the minimum of the previous two frames' energies. This captures sudden energy jumps — the signature of an impulsive event arriving. The threshold is 6.5 (i.e., current energy must be 6.5× the recent minimum).

3. A frame is a **"rain peak"** only if ALL three conditions are simultaneously met. The count of such frames is `rain_peaks_count`.

**Why we do this:** The spectral novelty in Stage 3 detects "is there energy at the resonant frequencies?" but doesn't directly check whether that energy has the impulsive, transient character expected of rain. Wind can also produce resonant energy. The time-domain statistics provide an independent check: "Does the waveform *look like* rain impacts?" Kurtosis and crest factor capture the statistical shape of the signal; differential energy captures sudden onsets.

**Assessment:** This is a strong complementary check. The literature strongly supports kurtosis and crest factor as discriminative features for impulsive vs. continuous sounds (Sharma et al., 2005). However, there is a subtlety: during heavy rain, individual impacts merge into a continuous "hiss," which lowers kurtosis and crest factor. The algorithm may undercount rain peaks during heavy rain, when ironically the signal is clearest in the frequency domain. The thresholds (2.5, 3.75, 6.5) appear to be empirically tuned. For reference, a pure Gaussian signal would have Fisher kurtosis of 0 and crest factor of ~3–4 (probabilistically), so the crest factor threshold of 3.75 is very close to the Gaussian value — this threshold may not be very discriminative.

### 3.5 Stage 5: Final Decision

The spectral `rain_drop_count` and time-domain `rain_peaks_count` are combined via hard thresholds:

**False Positive Suppression** (`handle_fp = True`):
- If the spectral detector says "raining" but `rain_peaks_count < 9` OR `rain_drop_count < threshold`, override to **not raining** and set count to 0.

**False Negative Recovery** (`handle_fn = True`):
- If the spectral detector says "not raining" but `rain_drop_count > 50` or `rain_peaks_count > 30`, override to **raining**.

**Why:** The FP suppression says: "Even if the frequency analysis sees resonant energy, if the time-domain statistics don't confirm impulsive events, it's probably wind." The FN recovery says: "Even if the frequency analysis is uncertain, if there's overwhelming time-domain evidence of impulsive events, call it rain."

**Assessment:** This hard-thresholded logic is brittle. The transitions at these exact thresholds are cliff edges — 8 rain peaks = no rain, 9 rain peaks = rain. There's also a logical issue: the FP condition uses an OR (either low peaks OR low drops → suppress), which means a single weak criterion can override strong evidence from the other. A more robust approach would combine these scores with a softer decision boundary.

Additionally, `if (True) or (rain_drop_count < rain_drop_max_thr)` on line 2669 — the `(True)` means the second condition is always ignored. This appears to be a debugging artifact that has made its way into the logic, effectively making the FP suppression more aggressive than perhaps intended.

### 3.6 The 2-Second Chunking

The 10-second audio is processed in **2-second chunks** (`MAX_DURATION_FW = 2`), with results accumulated:
- Novelty arrays are concatenated across chunks.
- Rain drop counts are summed.

**Why:** This mimics the firmware's memory-constrained processing. The Cortex M4/M7 cannot hold 10 seconds of FFT data in memory, so it processes the audio in smaller blocks.

**Assessment:** This is a pragmatic firmware constraint. However, it has algorithmic consequences: the noise floor estimation (`compute_local_average` with M=20 frames) has a window of ~470 ms. Within a 2-second chunk, this window is well-contained. But at chunk boundaries, the noise estimate may be inaccurate because it can't see across the boundary. The merge logic (`merge_algo_state`) concatenates arrays across chunks, but the noise normalization has already happened within each chunk independently.

---

## 4. Algorithm Walkthrough: Rain Quantification (DSD)

The DSD quantification algorithm (`device_dsd_processing_emulator.py`, class `DsdProcessingEmualtor`) is a separate, simpler system that runs on the same audio. It produces a 32-bin drop size distribution histogram.

### 4.1 Frame Processing

For each audio frame (512 samples, no overlap, no window function by default):

1. **FFT** of the frame → magnitude spectrum.
2. **Sum magnitude in [400, 700] Hz** — this is the fundamental resonance band. The sum of FFT magnitudes across bins in this range gives `drop_energy_level`.

   **Why this band:** This is where raindrop impacts produce the strongest response (the dome's fundamental resonance). The energy in this band is a proxy for the total impact energy.

3. **Energy binning**: If `drop_energy_level > 0.6` (the energy threshold):
   - Compute: `bin = floor(log(1 + (energy - 0.6) × 0.6) / log(1.13))`
   - Clip to range [0, 31]
   - Increment `energy_histogram[bin]`

   **What this does:** The logarithmic binning maps energy levels to 32 bins on a logarithmic scale. Small energies (just above threshold) go in bin 0; large energies go in higher bins. The base of the logarithm (1.13) determines the bin width — each successive bin represents energy that is 13% higher than the previous. This is conceptually similar to how decibels work: it compresses a wide dynamic range into a manageable number of bins.

   **Why logarithmic:** Drop size distributions span a wide range — drizzle drops are ~0.5 mm, heavy rain drops are ~5 mm, and their kinetic energy spans even more (roughly D³ × v²(D)). A logarithmic scale ensures that both small and large drops get reasonable bin resolution.

### 4.2 Peak Frequency Tracking

In addition to the DSD, the algorithm tracks:
- **Peak frequency per 2-second window** (30 bins covering the 60-second period): For each frame, the FFT bin with the highest magnitude in the [100, 1500] Hz band is recorded. A histogram accumulates which bin index is the peak most often. Every 2 seconds, the most frequent peak index is stored.
- **Accumulated FFT energies** (38 bins): Energies in two frequency windows (starting at 300 Hz and 1000 Hz, each 19 bins wide) are log-compressed and stored.

**Why:** These provide spectral shape metadata alongside the DSD. The peak frequency tracking indicates the dominant resonance being excited, and the FFT energies provide a compact spectral fingerprint. These could be used downstream for quality control or alternative estimation.

### 4.3 Cloud-Side Rain Volume Estimation

The DSD histogram from the edge is transmitted to the cloud, where it's converted to rain volume:

1. **Reverse binning**: Each bin index is mapped back to an energy level using the inverse of the binning function: `energy = ((e^(bin × log(1.13)) - 1) / 0.6) + 0.6`
2. **Weighted DSD**: Each bin's count is multiplied by its representative energy → `weighted_dsd_sum = Σ (count_i × energy_i)`
3. **Regression model**: `rain_rate = (a × weighted_dsd_sum) / (b + c × weighted_dsd_sum)` with fitted coefficients a = 1.41×10⁻³, b = 25.2, c = -7.44×10⁻⁶.
4. **Accumulation**: 1-minutely rain rates are summed to 5-minute totals (the official output).

**Why this model form:** The function `f(t) = at/(b + ct)` is a saturating function — it increases with weighted DSD sum but levels off at high values. This is physically reasonable: at very high rain rates, the sensor may saturate (overlapping impacts, ringing) and the DSD underestimates. The saturating model compensates.

**Assessment of the DSD pipeline:**
- The per-frame approach (512-sample, no overlap, no window) is simple and firmware-friendly, but has limitations:
  - **No windowing** means significant spectral leakage. Energy from frequencies outside the [400, 700] Hz band leaks into the measurement. A Hann window would reduce this.
  - **No overlap** means some drop impacts may straddle frame boundaries and be partially captured in two frames, underestimating their energy.
  - **Energy threshold of 0.6** is a fixed value, not adaptive. In a noisy environment, the ambient energy in the fundamental band may routinely exceed this threshold, causing wind energy to be counted as drops.
- **The [400, 700] Hz band measures total energy, not individual drops.** At moderate-to-high rain rates, multiple drops may impact within one frame (~46 ms at 512 samples / 11162 Hz). The energy reflects their combined impact, not individual sizes. The DSD bins therefore represent "energy per frame" rather than "individual drop energy."
- **The cloud regression was fit against cal/val data** and performs well in low-noise conditions. Wind noise causing overprediction is the primary known failure mode — wind energy in the resonance band gets counted as drop energy.

---

## 5. The Noise Suppression System (In Development)

A noise suppression pipeline is under active development across three modules. This system is designed to estimate and compensate for background noise, primarily wind, before or alongside rain detection.

### 5.1 Spectral Noise Processor (`spectral_noise_processor.py`)

This implements a **spectral subtraction** approach (Boll, 1979) with adaptive confidence weighting:

1. **Pre-filtering**: High-pass at 350 Hz (or bandpass to the operating band).
2. **STFT**: 256-sample FFT, 128-sample hop.
3. **Rain frame classification** (via `RainFrameClassifierMixin`): Each frame is classified as RAIN, NOISE, or UNCERTAIN using spectral flux in configured "mode bands" (resonance harmonics).
4. **Noise PSD estimation**: Tracks the noise power spectral density using:
   - A ring buffer of noise-only frames (frames not classified as rain)
   - Quantile-based estimation (default q=0.25, i.e., 25th percentile of recent noise frames)
   - Asymmetric EMA smoothing: fast-attack (ema_up=0.6) when noise increases, slow-release (ema_down=0.95) when it decreases
5. **Gain computation**: Spectral subtraction with confidence-weighted oversubtraction:
   - `G = 1 - α × sqrt(N/P)` (sqrt subtraction) or Wiener-like `G = max(P - αN, 0) / (P + ε)`
   - Oversubtraction factor α varies from 1.0 (rain-likely) to 3.0 (noise-likely)
   - Temporal smoothing is reduced on rain-like frames to preserve transient character
6. **ISTFT**: Reconstruct the denoised waveform.

**Assessment:** This is a well-structured noise suppression system consistent with established approaches. The adaptive oversubtraction based on frame classification is a good idea — suppress more aggressively when you're confident it's noise, preserve signal when it might be rain. The main risk is the chicken-and-egg problem: you need to classify rain vs. noise before you suppress noise, but your classification may be unreliable in high-noise conditions. The warmup logic (learning from all frames until the buffer is half-filled) is a practical solution to cold-start issues.

### 5.2 Band Noise Estimator (`band_noise_estimator.py`)

This is a complementary, more targeted noise estimator focused on the [400, 700] Hz fundamental band:

1. **Per-subframe processing**: 512-sample frames are divided into 128-sample subframes (4 per frame).
2. **Bandpass filtering** to [400, 700] Hz for the target metric, plus high-pass for overall energy.
3. **Rain detection** at two levels:
   - **FFT-level**: Compares total rain-band energy and primary-band energy frame-to-frame (6 dB and 3 dB jump thresholds, respectively).
   - **Subframe-level**: Detects sudden band energy rises (≥6 dB) that exceed overall loudness rises (by ≥3 dB). This is clever: it detects narrowband energy jumps *relative to broadband energy*, which is a good discriminator for rain (narrowband resonance response) vs. wind (broadband energy increase).
4. **Noise estimation**: Non-rain subframe energies are pushed into a ring buffer; noise is estimated as the 30th percentile via quantile estimation with EMA smoothing.
5. **Wiener-like gain**: `G = sqrt(max(E_band - β×N_E, 0) / (E_band + ε))`, with gain floor of 0.10.

**Assessment:** The subframe-level processing at 128 samples (~11.5 ms) provides fine temporal resolution, appropriate for catching individual raindrop transients. The "excess rise" trigger (band rise minus overall rise ≥ 3 dB) is a thoughtful feature that captures the physics: a rain drop excites the resonance band specifically, while wind raises all frequencies. However, this assumption breaks down for strong gusts that happen to excite the fundamental mode preferentially.

### 5.3 Rain Frame Classifier (`rain_frame_classifier.py`)

The classifier used within the noise processor is distinct from the detection algorithm's approach. It uses:

1. **Spectral flux in configured mode bands**: Frame-to-frame positive differences (half-wave rectified) in log-compressed power, computed separately for the primary mode and all modes combined.
2. **Rolling robust z-scores** (median + MAD): The flux values are normalized using a rolling median and median absolute deviation, producing z-scores that are robust to outliers.
3. **Soft confidence**: Z-scores are scaled to [0, 1] confidences. The final confidence is the minimum of the primary-mode and multi-mode confidences (an AND gate: both must agree).
4. **Peak structure check**: Spectral peaks in the frame must fall within mode bands (minimum 2 of the top 6 peaks).
5. **Hold expansion**: Rain-classified frames are extended forward by a configurable number of frames to protect the decay tail of resonance ringing.

**Assessment:** This classifier is more sophisticated than the detection algorithm's novelty-based approach. The z-score normalization is a substantial improvement over the fixed-threshold SNR in the detection algorithm. The AND-gate requirement (both primary and multi-mode flux must be high) is a good safeguard against narrowband interference. The peak structure check directly tests the "rain excites resonance modes" hypothesis. This classifier could potentially replace or complement the detection algorithm's spectral novelty stage.

---

## 6. Literature Context and State of the Art

### 6.1 Acoustic Disdrometry

The foundational work in acoustic rainfall measurement comes from Nystuen (1986, 1994, 1999), who established that sound produced by rainfall has measurable spectral signatures that vary with drop size. Joss and Waldvogel (1967) created the original impact disdrometer (JWD), establishing the paradigm of inferring drop size from impact transients. Marshall and Palmer (1948) provided the canonical exponential DSD model.

The Arable system differs from traditional acoustic disdrometers in that it uses a **resonance chamber** rather than a flat plate. This adds complexity (the resonance structure modulates the signal) but also provides natural amplification and a distinctive spectral signature.

### 6.2 Key Physical Relationships

- **Drop kinetic energy**: KE = (π/12) × ρ × D³ × v(D)², where v(D) is the terminal velocity (Gunn & Kinzer, 1949; Atlas et al., 1973).
- **Impact duration**: Approximately 0.1–1 ms for drops 0.5–5 mm, decreasing with drop size.
- **Acoustic response**: The dome responds with a decaying harmonic series. Larger drops produce higher total energy across all harmonics. **Whether drop size shifts the spectral shape** (e.g., exciting higher harmonics more strongly for larger drops) is an open question that merits investigation through controlled drop experiments.

### 6.3 Noise Discrimination

The literature supports the approach taken here. Spectral flux is cited as one of the most discriminative features for transient sound detection (Chu, 2006; Piczak, 2015). Kurtosis and crest factor are well-established for impulsive vs. continuous discrimination. The spectral subtraction and minimum statistics approaches used in the noise processor follow Boll (1979), Martin (2001), and Cohen & Berdugo (2001).

### 6.4 Machine Learning Approaches

Recent work (Avanzato & Beritelli, 2020; Gupta et al., 2021) has demonstrated that even small CNNs on mel-spectrograms can achieve >95% rain detection accuracy. The DCASE community and pre-trained audio models (PANNs, Kong et al., 2020) represent the current state of the art. These approaches could be applied on the cloud side using the DSD histogram or spectral features as inputs, without requiring raw audio.

---

## 7. Assumptions and Their Validity

### A1: "Rain drop impacts excite all resonant harmonics simultaneously"
**Validity: Strong.** This follows directly from the physics of impulse excitation. A Dirac-like impact produces a broadband impulse response, and the resonance chamber filters this into a harmonic series. Well-supported by theory and literature (Pumphrey & Crum, 1990).

### A2: "Wind excites primarily the fundamental mode"
**Validity: Moderate.** Low-frequency wind turbulence predominantly excites the lowest modes. However, strong gusts, vortex shedding, and turbulent eddies at the dome's edges can excite higher modes as well. The degree to which this assumption holds depends on wind speed, angle of attack, and the dome's aerodynamic properties. This assumption is the weakest link in the wind discrimination strategy.

### A3: "The fundamental frequency is stable within a 10-second window"
**Validity: Strong.** Structural resonance frequencies depend on material properties and geometry, which change very slowly (thermal expansion, aging). 10 seconds is far too short for significant drift.

### A4: "The fundamental frequency is in [400, 700] Hz for all devices"
**Validity: Moderate.** Manufacturing tolerances, aging, and field conditions (temperature, mounting) can shift the resonance. If a device's fundamental drifts outside this range, the algorithm will estimate an incorrect `frain_mean` and search for harmonics in the wrong places. The 300 Hz search width provides some tolerance, but systematic validation across the fleet of 500+ devices would be valuable.

### A5: "Spectral novelty above a fixed threshold indicates rain"
**Validity: Weak.** Fixed thresholds (`rain_thr = [4.5, 4.0, 3.5, ...]`) don't account for varying noise conditions, drop intensity, or device-to-device variation. The SNR normalization helps adapt to local noise levels, but the thresholds themselves were likely tuned on specific test data and may not generalize well.

### A6: "If the fundamental novelty is zero, there is no rain"
**Validity: Moderate.** This is the gate where fundamental activity is required before harmonics are considered. It could fail if: (a) the fundamental is masked by coincident wind energy, (b) the fundamental's response is weak for certain drop sizes, or (c) the local noise floor in `compute_local_average` is poorly estimated. False negatives from this gate would be invisible — the algorithm would confidently say "no rain" even though harmonics might show clear activity.

### A7: "Per-frame FFT energy in [400, 700] Hz is proportional to drop impact energy"
**Validity: Moderate.** This underlies the DSD. It's approximately true for isolated impacts in the center of a frame. It breaks down when: (a) multiple impacts occur in one frame (energies add), (b) impacts straddle frame boundaries, (c) wind energy contributes to the band, or (d) resonance ringing from a previous frame leaks into the current one.

### A8: "The 1-second amplitude check is a good gate for rain"
**Validity: Problematic.** This gate means the system *only records audio when something loud is happening*. This creates severe selection bias: there are almost no "no rain" audio samples, making it impossible to characterize the false positive rate. It also means the system may miss the beginning of light rain events that start quietly.

---

## 8. Strengths

**S1: Physics-motivated design.** The algorithm is grounded in the physical properties of the resonance chamber. Multi-harmonic detection, impulsiveness checking, and adaptive noise estimation all follow from the physics of how rain and noise interact with the dome. This is not a black box — each step has a physical rationale.

**S2: Adaptive noise floor.** The minimum-statistics noise estimation (`compute_local_average`) adapts to changing noise conditions within each 10-second window. This makes the detection more robust across environments than a fixed threshold would.

**S3: Multi-harmonic verification.** Requiring simultaneous activity at multiple harmonics is a strong discriminator against narrowband interference. A random tonal noise source is unlikely to coincide with multiple harmonics of the dome's fundamental.

**S4: Independent time-domain validation.** The kurtosis/crest-factor/differential-energy check provides an orthogonal validation of the spectral detection. This catches false positives where wind produces resonant energy but not impulsive time-domain characteristics.

**S5: Firmware-feasible design.** The algorithm uses standard, computationally inexpensive operations (FFT, thresholding, simple statistics) suitable for implementation on resource-constrained embedded processors.

**S6: Logarithmic DSD binning.** The logarithmic energy-to-bin mapping is appropriate for the wide dynamic range of raindrop energies.

---

## 9. Weaknesses and Risks

**W1: The "novelty" computation is mislabeled and potentially misleading.**
The function `compute_novelty_spectrum_new` computes the sum of positive frequency-domain gradients — a measure of spectral peak structure, not temporal novelty (onset detection). This is not inherently wrong, but the name "novelty" suggests temporal change detection, which this is not. The function actually measures "how prominent are the spectral peaks within this band at each frame." This matters because sustained tonal interference at a resonance frequency would score high on this measure indefinitely, not just at onset.

**W2: Fundamental-gating creates a single point of failure.**
The requirement that the fundamental must be active before harmonics are considered means that any masking of the fundamental — by wind, by coincident interference, or by a weak fundamental response — vetoes the entire detection, even if harmonics 2–5 clearly show rain. This is overly conservative and likely contributes to false negatives.

**W3: Fixed thresholds throughout.**
Almost every decision point uses hard-coded thresholds: novelty thresholds per harmonic, kurtosis/crest/energy thresholds, rain peak count thresholds, the FP/FN override thresholds. These are presumably tuned on specific test data and may not generalize across the device fleet, seasons, or geographic regions. There is no mechanism for per-device calibration or adaptation.

**W4: Wind noise directly corrupts the DSD.**
The DSD emulator uses raw energy in [400, 700] Hz without any noise compensation. Wind energy in this band is counted as drop energy, inflating the DSD and causing overprediction. This is the primary known failure mode and is the motivation for the noise suppression work. Until noise suppression is integrated into the DSD pipeline, wind will cause systematic positive bias in rain estimates.

**W5: No broadband analysis.**
The algorithm focuses almost exclusively on the resonance bands (fundamental + harmonics). It does not use energy *between* harmonics or *outside* the resonance bands as a diagnostic. This is a missed opportunity: wind should produce relatively more inter-harmonic energy (smooth broadband spectrum) while rain should produce relatively less (energy concentrated at harmonics). A "harmonic-to-noise ratio" or "spectral flatness within the operating band" could be a powerful discriminator.

**W6: The 2-second chunking may cause boundary artifacts.**
Noise normalization happens independently within each 2-second chunk. At chunk boundaries, the noise estimate resets, potentially causing inconsistent detection behavior. Events that straddle chunk boundaries may be split and undercounted.

**W7: The crest factor threshold (3.75) may be too low.**
A Gaussian random signal has a crest factor of approximately 3–4 (depending on the window size). A threshold of 3.75 is barely above the Gaussian baseline, meaning this criterion may not effectively distinguish rain from noise. For comparison, a clear rain impact might have a crest factor of 10–20+.

**W8: Global state via `configure_parameters`.**
The detection algorithm uses extensive global variables set by `configure_parameters`. This makes the code fragile (concurrent calls would interfere), hard to test, and hard to reason about. Multiple global variables (`Fs`, `frame_length`, `sos`, `process_fp`, `natural_freq_range`, etc.) are set as side effects of calling this function. This is an engineering concern, not an algorithmic one, but it increases the risk of subtle bugs.

**W9: Debugging artifacts in production code.**
The `if (True) or (...)` pattern on line 2669 disables a condition that was presumably meant to limit FP suppression to moderate rain counts. The `(True)` overrides the check entirely. Combined with extensive commented-out code, print statements, and hardcoded file paths, this suggests the code has evolved from an interactive notebook without rigorous review.

**W10: No use of temporal structure across frames.**
Individual drop impacts have a characteristic temporal signature: a sharp onset followed by an exponential decay (determined by the Q factor of the resonance). The algorithm doesn't model or detect this shape — it only looks at per-frame statistics. Matched filtering or template-based detection could exploit this structure for more robust individual drop detection.

---

## 10. Improvement Opportunities

### 10.1 Edge-Side Improvements

These could be implemented in the Python algorithm and handed off to firmware:

**E1: Add a "spectral flatness" or "harmonic-to-noise ratio" feature.**
Compute the ratio of energy at harmonic frequencies to energy at inter-harmonic frequencies within the operating band. Rain (broadband impulse → strong harmonics, weak inter-harmonics) would have a high ratio; wind (broadband continuous → uniform energy distribution) would have a low ratio. This directly exploits the rain/wind spectral distinction without requiring additional computational resources.

**E2: Remove the fundamental-gating requirement.**
Instead of requiring the fundamental to be active before considering harmonics, use a weighted vote across all available harmonics. The fundamental could receive a higher weight, but harmonics alone should still be able to trigger detection. This addresses weakness W2.

**E3: Increase the crest factor threshold.**
Consider raising the crest factor threshold from 3.75 to 5.0–6.0, based on empirical analysis of the rain/noise distributions. The current threshold is barely above the Gaussian baseline.

**E4: Integrate noise compensation into the DSD energy measurement.**
Before binning frame energy into the DSD, subtract the estimated noise floor in the [400, 700] Hz band. The `BandNoiseEstimator` already produces a per-frame noise estimate `N_E` that could be applied: `adjusted_energy = max(drop_energy - N_E, 0)`. This would directly address the wind overprediction problem (W4).

**E5: Use temporal onset detection.**
Instead of (or in addition to) the frequency-gradient-based "novelty," implement a true temporal onset detector: detect frames where energy in the resonance bands increases suddenly compared to the immediately preceding frame. This would be more directly sensitive to the impulsive nature of rain impacts.

### 10.2 Cloud-Side Improvements

These leverage the DSD histogram, metadata, and computational resources available on the cloud:

**C1: Wind speed correction using anemometer data.**
For devices with collocated anemometers, build a model that predicts the expected wind-induced DSD bias as a function of wind speed. Subtract this bias before applying the rain regression. This could be a simple lookup table or a fitted model: `corrected_dsd = raw_dsd - f(wind_speed)`.

**C2: Machine learning for rain classification.**
Train a classifier (random forest or gradient-boosted tree) on the 100-element output vector (32 DSD bins + 30 peak-frequency bins + 38 FFT energy bins) to predict rain/no-rain, using tipping bucket data as labels. This leverages the rich spectral information already being transmitted without requiring raw audio. The model could run alongside the existing regression for comparison.

**C3: Temporal consistency filtering.**
Rain is a continuous weather phenomenon — it doesn't start and stop every minute. Apply temporal smoothing or a hidden Markov model to the per-minute rain/no-rain decisions: require multiple consecutive "rain" minutes before declaring a rain event, and don't declare rain cessation until multiple consecutive "no rain" minutes. This would reduce isolated false positives.

**C4: Cross-device validation.**
For clusters of nearby devices, compare rain detections. If one device reports rain but its neighbors don't, flag it as a potential false positive. This exploits the spatial coherence of precipitation.

**C5: Re-examine the energy-to-rain regression.**
The current regression `rain_rate = at/(b + ct)` was fitted against cal/val data. Given that wind noise systematically inflates the weighted DSD sum, the regression coefficients may have been fitted on partially contaminated data. Re-fitting on a carefully curated, low-wind subset — or using a two-stage model that first classifies confidence and then applies different regressions — could improve accuracy.

**C6: Use the DSD shape, not just the sum.**
The current cloud model sums the weighted DSD into a single scalar before applying the regression. The *shape* of the DSD carries additional information: wind noise tends to inflate low bins uniformly, while rain produces a characteristic DSD shape (exponential or gamma distribution). A model that uses the full 32-bin DSD as input features could learn to distinguish rain-shaped DSDs from noise-shaped DSDs.

### 10.3 Research / Validation Priorities

**R1: Controlled drop experiments.**
Use a drop generator to produce drops of known size onto the Mark dome, and characterize the relationship between drop diameter and acoustic response (energy, spectral shape, harmonic ratios). This would ground-truth assumptions A1 and A7, and potentially enable physics-based DSD inversion rather than the current empirical binning.

**R2: Collect "no rain" audio samples.**
The 1-second amplitude gate means almost no quiet/no-rain audio exists. Temporarily disabling this gate on a few devices (or deploying a dedicated test device that always records) would provide the negative samples needed to properly characterize and improve false positive rates.

**R3: Characterize the fundamental frequency across the fleet.**
Run the fundamental frequency estimator across a large sample of devices and conditions to validate assumption A4. If significant inter-device variation exists, per-device calibration may be necessary.

**R4: Drop size vs. frequency content investigation.**
The literature is sparse on whether raindrop size shifts the spectral content of a resonance chamber response (beyond just amplitude). This could be explored through controlled experiments or analysis of existing data where independent DSD measurements are available.

---

## 11. References

1. Atlas, D., Srivastava, R.C., & Sekhon, R.S. (1973). Doppler Radar Characteristics of Precipitation at Vertical Incidence. *Rev. Geophys.*, 11(1), 1–35.
2. Avanzato, R. & Beritelli, F. (2020). An Innovative Acoustic Rain Sensor Using Convolutional Neural Networks. *Information*, 11(4), 183.
3. Boll, S. (1979). Suppression of Acoustic Noise in Speech Using Spectral Subtraction. *IEEE Trans. ASSP*, 27(2), 113–120.
4. Chu, S. (2006). Environmental Sound Recognition with Time-Frequency Audio Features. *IEEE Trans. ASAL*.
5. Cohen, I. & Berdugo, B. (2001). Noise Estimation by Minima Controlled Recursive Averaging for Robust Speech Enhancement. *IEEE Signal Process. Lett.*, 9(1), 12–15.
6. Gunn, R. & Kinzer, G.D. (1949). The Terminal Velocity of Fall for Water Droplets in Stagnant Air. *J. Meteor.*, 6, 243–248.
7. Gupta, A. et al. (2021). Rain Detection and Intensity Estimation from Audio Using Deep Learning.
8. Joss, J. & Waldvogel, A. (1967). Ein Spektrograph für Niederschlagstropfen mit automatischer Auswertung. *Pure Appl. Geophys.*, 68, 240–246.
9. Kong, Q. et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, 28, 2880–2894.
10. Martin, R. (2001). Noise Power Spectral Density Estimation Based on Optimal Smoothing and Minimum Statistics. *IEEE Trans. SAP*, 9(5), 504–512.
11. Marshall, J.S. & Palmer, W.M.K. (1948). The Distribution of Raindrops with Size. *J. Meteor.*, 5, 165–166.
12. Nelke, M. & Vary, P. (2014). Measurement, Analysis and Simulation of Wind Noise Signals for Mobile Communication Devices.
13. Nystuen, J.A. (1986). Rainfall Measurements Using Underwater Sound. *J. Acoust. Soc. Am.*, 79(4), 972–982.
14. Nystuen, J.A. (1999). Relative Performance of Automatic Rain Gauges under Different Rainfall Conditions. *J. Atmos. Ocean. Tech.*, 16, 1025–1043.
15. Piczak, K.J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE MLSP*.
16. Pumphrey, H.C. & Crum, L.A. (1990). Underwater Sound Produced by Individual Drop Impacts and Rainfall. *J. Acoust. Soc. Am.*, 87(4), 1518–1526.
17. Sharma, G. et al. (2005). Impulsive Noise Detection and Identification.
18. Strasberg, M. (1988). Dimensional Analysis of Windscreen Noise. *J. Acoust. Soc. Am.*
19. Ulbrich, C.W. (1983). Natural Variations in the Analytical Form of the Raindrop Size Distribution. *J. Climate Appl. Meteor.*, 22, 1764–1775.

---

*This assessment is based on the codebase at commit 32183ae. The literature review draws on both the published academic literature and the author's knowledge of signal processing and acoustic sensing principles.*
