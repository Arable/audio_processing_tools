from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.signal as spsig

EPS = 1e-12


# ----------------------------
# helpers
# ----------------------------
def hz_to_bin(f_hz: float, fs: float, n_fft: int) -> int:
    return int(np.round(f_hz * n_fft / fs))


def db_to_ratio(db: float) -> float:
    # power ratio
    return 10.0 ** (db / 10.0)


def _frame_view(sig: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if sig.size < frame_len:
        return np.empty((0, frame_len), dtype=np.float64)
    T = 1 + (sig.size - frame_len) // hop
    stride = sig.strides[0]
    return np.lib.stride_tricks.as_strided(
        sig,
        shape=(T, frame_len),
        strides=(hop * stride, stride),
        writeable=False,
    )

@dataclass
class NoiseFrameDetectorConfig:
    fs: int = 11162
    n_fft: int = 512

    # FFT rain decision:
    # (1) total rain bands sum jump by M dB, AND (2) primary jump by N dB
    M_db: float = 6.0
    N_db: float = 3.0

    # Primary band (Hz)
    primary_hz: Tuple[float, float] = (450.0, 650.0)

    # Rain-mode band list (Hz) used for "total sum"
    rain_bands_hz: Tuple[Tuple[float, float], ...] = (
        (450.0, 650.0),
        (800.0, 1050.0),
        (1500.0, 1800.0),
        (2350.0, 2550.0),
        (3150.0, 3350.0),
    )

    # Hold length (subframes)
    k_subframes: int = 2

    # NEW (preferred): trigger on metric = Δ(E_band)/E_hpf
    dE_over_Ehpf_thr: float = 0.08  # tune: ~0.03–0.20 typical depending on scaling

    # Optional legacy trigger: Eb jump threshold (dB)
    use_D_trigger: bool = False
    D_db: float = 6.0


class NoiseFrameDetector:
    """
    Produces per-subframe rain mask using:
      - FFT-domain: total rain-band sum jump >= M dB AND primary jump >= N dB => whole frame rain
      - Time-domain: metric = max(Eb - Eb_prev, 0)/Ehpf >= thr => start/refresh hold
        Optional fallback: Eb > Eb_prev * D_ratio
    """
    def __init__(self, cfg: NoiseFrameDetectorConfig, *, subframes_per_frame: int):
        self.cfg = cfg
        self.S = int(subframes_per_frame)

        # FFT bins
        self._rain_bins = []
        for f0, f1 in cfg.rain_bands_hz:
            self._rain_bins.append(
                (hz_to_bin(f0, cfg.fs, cfg.n_fft), hz_to_bin(f1, cfg.fs, cfg.n_fft))
            )
        self._p0 = hz_to_bin(cfg.primary_hz[0], cfg.fs, cfg.n_fft)
        self._p1 = hz_to_bin(cfg.primary_hz[1], cfg.fs, cfg.n_fft)

        self._M_ratio = db_to_ratio(cfg.M_db)
        self._N_ratio = db_to_ratio(cfg.N_db)
        self._D_ratio = db_to_ratio(cfg.D_db)

        # prev FFT energies
        self._prev_rain_sum: Optional[float] = None
        self._prev_primary: Optional[float] = None

        # time-domain state across subframes
        self._prev_Eb: Optional[float] = None
        self._hold = 0

    @staticmethod
    def _band_sum_power(P: np.ndarray, b0: int, b1: int) -> float:
        b0 = max(0, min(b0, len(P) - 1))
        b1 = max(0, min(b1, len(P) - 1))
        if b1 < b0:
            return 0.0
        return float(np.sum(P[b0:b1 + 1]))

    def fft_rain(self, x: np.ndarray) -> bool:
        X = np.fft.rfft(x, n=self.cfg.n_fft)
        P = (X.real * X.real + X.imag * X.imag)

        rain_sum = 0.0
        for (b0, b1) in self._rain_bins:
            rain_sum += self._band_sum_power(P, b0, b1)

        primary = self._band_sum_power(P, self._p0, self._p1)

        if self._prev_rain_sum is None:
            self._prev_rain_sum = rain_sum
            self._prev_primary = primary
            return False

        cond1 = rain_sum > (self._prev_rain_sum + EPS) * self._M_ratio
        cond2 = primary > (self._prev_primary + EPS) * self._N_ratio

        self._prev_rain_sum = rain_sum
        self._prev_primary = primary
        return bool(cond1 and cond2)

    def time_rain_mask_from_subE(
        self,
        subE: np.ndarray,
        subEhpf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        subE:    (S,) band energies (400–700 BPF) per subframe
        subEhpf: (S,) HPF energies per subframe (proxy for total S+N energy)
        """
        subE = np.asarray(subE, dtype=np.float64).reshape(-1)
        if subE.size != self.S:
            raise ValueError(f"subE must have shape ({self.S},), got {subE.shape}")

        if subEhpf is not None:
            subEhpf = np.asarray(subEhpf, dtype=np.float64).reshape(-1)
            if subEhpf.size != self.S:
                raise ValueError(f"subEhpf must have shape ({self.S},), got {subEhpf.shape}")

        mask = np.zeros(self.S, dtype=bool)
        thr = float(self.cfg.dE_over_Ehpf_thr)

        for s in range(self.S):
            Eb = float(max(subE[s], EPS))

            # hold logic unchanged
            if self._hold > 0:
                mask[s] = True
                self._hold -= 1

            triggered = False

            # Preferred trigger: Δ(Eb)/Ehpf
            if (subEhpf is not None) and (self._prev_Eb is not None):
                Eh = float(max(subEhpf[s], EPS))
                dE = max(Eb - self._prev_Eb, 0.0)
                metric = dE / (Eh + EPS)

                if metric >= thr:
                    triggered = True

            # Optional legacy trigger
            if (not triggered) and bool(self.cfg.use_D_trigger) and (self._prev_Eb is not None):
                if Eb > (self._prev_Eb + EPS) * self._D_ratio:
                    triggered = True

            if triggered:
                mask[s] = True
                self._hold = max(self._hold, max(0, int(self.cfg.k_subframes) - 1))

            self._prev_Eb = Eb

        return mask

    def process_frame(
        self,
        x: np.ndarray,
        subE: np.ndarray,
        *,
        subEhpf: Optional[np.ndarray] = None,
    ) -> tuple[bool, np.ndarray]:
        """
        Returns:
          fft_rain_frame, rain_submask(S)
        """
        fft_rain_frame = self.fft_rain(x)
        time_mask = self.time_rain_mask_from_subE(subE, subEhpf=subEhpf)

        if fft_rain_frame:
            return True, np.ones(self.S, dtype=bool)

        return False, time_mask

    def reset(self) -> None:
        # prev FFT energies
        self._prev_rain_sum = None
        self._prev_primary = None

        # time-domain state
        self._prev_Eb = None
        self._hold = 0

@dataclass
class BandNoiseFrameOut:
    M_band: float
    E_band: float
    # noise estimate (power) in band for this frame
    N_E: float
    N_E_raw: float
    # per-subframe noise estimates (power)
    N_sub: np.ndarray  # shape (S,)
    subE: np.ndarray  # shape (S,)
    # rain mask used for estimator (True = rain => excluded from noise learning)
    rain_submask: np.ndarray  # shape (S,)
    # gain + cleaned magnitude
    G_mag: float
    M_clean: float
    # diagnostics
    fft_rain_frame: bool


@dataclass
class BandNoiseEstimatorConfig:
    fs: int = 11162
    frame_len: int = 512

    # Optional HPF
    hp_cutoff_hz: float = 350.0
    hp_order: int = 4

    # Band used for BPF energy + suppression metric
    band_hz: Tuple[float, float] = (400.0, 700.0)
    bpf_order: int = 4

    # Subframes
    subframe_len: int = 128
    subhop: int = 128

    # Noise estimation
    W: int = 30           # quantile window length (per-subframe lane) (~342 ms @ 11.4 ms/subframe)
    W_min: int = 10       # warmup valid samples required (per lane) 114 msec
    q: float = 0.3       # quantile
    ema_alpha: float = 1  # smoothing on quantile result

    # Suppression (Wiener-like)
    beta: float = 1.0
    gain_floor: float = 0.10
    eps: float = 1e-12

    # Optional extra smoothing on final noise estimate (frame-level)
    # Use different "attack" (rise) speeds depending on whether we think it's raining.
    # This helps track mechanical noise quickly while avoiding over-estimating noise during rain.
    ne_attack_alpha_dry: float = 0.15   # fast rise when NOT raining (0.05–0.30 typical)
    ne_attack_alpha_wet: float = 0.02   # slow rise during rain (0.005–0.05 typical)
    ne_release_alpha: float = 0.25      # fast fall (0.10–0.60 typical)
    smooth_N_E: bool = False

    # detector config
    det: NoiseFrameDetectorConfig = field(default_factory=NoiseFrameDetectorConfig)
    def validate(self) -> None:
        if self.frame_len % self.subframe_len != 0:
            raise ValueError("subframe_len must divide frame_len")
        if not (0.0 < self.q < 1.0):
            raise ValueError("q must be in (0,1)")
        if self.W <= 0 or self.W_min < 0 or self.W_min > self.W:
            raise ValueError("Need W>0 and 0<=W_min<=W")
        lo, hi = self.band_hz
        if not (0 < lo < hi < 0.5 * self.fs):
            raise ValueError("band_hz out of range")

# ----------------------------
# estimator
# ----------------------------
class BandNoiseEstimator:
    """
    - Computes band metrics (FFT magnitude sum and power sum in band_hz)
    - Computes BPF subframe energies
    - Uses detector to decide which subframes are rain (exclude from noise learning)
    - Keeps per-lane ring buffers of length W for noise-only subframe energies
    - After W_min valid samples per lane, outputs quantile+EMA noise estimate per lane
    - Noise per frame = sum of lane noise estimates
    - Wiener-like gain computed from E_band and N_E
    """
    def __init__(self, cfg: BandNoiseEstimatorConfig):
        cfg.validate()
        self.cfg = cfg
        self.N = int(cfg.frame_len)
        self.S = self.N // int(cfg.subframe_len)

        # FFT band mask (for M_band/E_band)
        freqs = np.fft.rfftfreq(self.N, d=1.0 / cfg.fs)
        lo, hi = cfg.band_hz
        self.band_mask = (freqs >= lo) & (freqs <= hi)

        # Filters
        self.hpf_sos = self._design_hpf(cfg)
        self.hpf_zi: Optional[np.ndarray] = None

        self.bpf_sos = self._design_bpf(cfg)
        self.bpf_zi: Optional[np.ndarray] = None

        self._need_zi_seed = True

        self.N_E_smooth = 0.0

        # Detector
        self.det = NoiseFrameDetector(cfg.det, subframes_per_frame=self.S)

        # Noise buffer: single continuous subframe stream
        self.W = int(cfg.W)  # now interpreted as "number of subframes in window"
        self.buf = np.zeros(self.W, dtype=np.float64)
        self.valid = np.zeros(self.W, dtype=bool)
        self.wr = 0
        self.count_valid = 0

        # EMA state for scalar noise estimate (per-subframe)
        self.noise_ema = 0.0

    @staticmethod
    def _design_hpf(cfg: BandNoiseEstimatorConfig) -> Optional[np.ndarray]:
        if cfg.hp_cutoff_hz <= 0:
            return None
        nyq = 0.5 * cfg.fs
        w = np.clip(cfg.hp_cutoff_hz / nyq, 1e-6, 0.999)
        return spsig.butter(cfg.hp_order, w, btype="highpass", output="sos")

    @staticmethod
    def _design_bpf(cfg: BandNoiseEstimatorConfig) -> np.ndarray:
        lo, hi = cfg.band_hz
        nyq = 0.5 * cfg.fs
        w1 = np.clip(lo / nyq, 1e-6, 0.999)
        w2 = np.clip(hi / nyq, 1e-6, 0.999)
        if w2 <= w1:
            w2 = min(0.999, w1 + 1e-3)
        return spsig.butter(cfg.bpf_order, [w1, w2], btype="bandpass", output="sos")

    def reset(self) -> None:
        self.hpf_zi = None
        self.bpf_zi = None
        self._need_zi_seed = True

        # reset noise buffers
        self.buf[:] = 0.0
        self.valid[:] = False
        self.wr = 0
        self.count_valid = 0
        self.noise_ema = 0.0
        self.N_E_smooth = 0.0

        # reset detector state
        self.det.reset()

    def _push_stream(self, v: float) -> None:
        j = int(self.wr)
        was_valid = bool(self.valid[j])

        self.buf[j] = float(v)
        self.valid[j] = True

        if not was_valid:
            self.count_valid += 1

        self.wr = (j + 1) % self.W
    
    def _estimate_noise_scalar(self) -> float:
        if int(self.count_valid) < int(self.cfg.W_min):
            return 0.0

        vals = self.buf[self.valid]
        if vals.size == 0:
            return 0.0

        qv = float(np.quantile(vals, self.cfg.q))
        a = float(self.cfg.ema_alpha)
        self.noise_ema = (1.0 - a) * self.noise_ema + a * qv
        return float(self.noise_ema)

    def process_frame(self, frame: np.ndarray) -> BandNoiseFrameOut:
        cfg = self.cfg
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim != 1 or x.size != self.N:
            raise ValueError(f"frame must be 1-D length {self.N}")

        # seed filter states
        if self._need_zi_seed:
            x0 = float(x[0]) if x.size else 0.0
            if self.hpf_sos is not None:
                self.hpf_zi = spsig.sosfilt_zi(self.hpf_sos).astype(np.float64) * x0
            self.bpf_zi = spsig.sosfilt_zi(self.bpf_sos).astype(np.float64) * x0
            self._need_zi_seed = False

        # HPF
        if self.hpf_sos is not None:
            assert self.hpf_zi is not None
            x, self.hpf_zi = spsig.sosfilt(self.hpf_sos, x, zi=self.hpf_zi)

        # ---- NEW: HPF subframe energies (Ehpf per subframe)
        subs_hpf = _frame_view(x, int(cfg.subframe_len), int(cfg.subhop))
        if subs_hpf.shape[0] == 0:
            subEhpf = np.asarray([float(np.sum(x * x))], dtype=np.float64)
            subEhpf = np.pad(subEhpf, (0, self.S - subEhpf.size), mode="edge")
        else:
            subEhpf = np.sum(subs_hpf * subs_hpf, axis=1).astype(np.float64)
            if subEhpf.size != self.S:
                subEhpf = np.resize(subEhpf, self.S)

        # FFT band metrics (diagnostics only)
        # NOTE: Not used for suppression because FFT-domain scaling is not directly
        # comparable to time-domain (BPF) energy without careful Parseval normalization.
        X = np.fft.rfft(x)
        mag = np.abs(X)
        Mb_fft = float(np.sum(mag[self.band_mask]))
        Eb_fft = float(np.sum(mag[self.band_mask] ** 2))

        # BPF for time-domain band subframe energies (subE)
        assert self.bpf_zi is not None
        x_bp, self.bpf_zi = spsig.sosfilt(self.bpf_sos, x, zi=self.bpf_zi)

        # Time-domain band energy for this FFT frame (this is the scale used by subE/N_E)
        # This makes Eb directly comparable to N_E_raw (both are sum(x^2) energies).
        Eb = float(np.sum(x_bp * x_bp))

        # A magnitude-like companion metric for reporting/cleaned magnitude.
        # Keep it consistent with Eb's scale (avoid mixing FFT magnitude sums with time energy).
        Mb = float(np.sqrt(max(Eb, 0.0)))

        subs = _frame_view(x_bp, int(cfg.subframe_len), int(cfg.subhop))
        if subs.shape[0] == 0:
            subE = np.asarray([float(np.sum(x_bp * x_bp))], dtype=np.float64)
            subE = np.pad(subE, (0, self.S - subE.size), mode="edge")
        else:
            subE = np.sum(subs * subs, axis=1).astype(np.float64)
            if subE.size != self.S:
                subE = np.resize(subE, self.S)

        # detector (NEW: pass subEhpf)
        fft_rain_frame, rain_submask = self.det.process_frame(x, subE, subEhpf=subEhpf)

        for s in range(self.S):
            # keep your bypass / or revert to not bool(rain_submask[s])
            if 1:  # or: if not bool(rain_submask[s]):
                self._push_stream(float(max(subE[s], cfg.eps)))

        N_sub_scalar = self._estimate_noise_scalar()  # noise per-subframe energy
        N_sub = np.full(self.S, N_sub_scalar, dtype=np.float64)
        N_E_raw = float(self.S * N_sub_scalar)

        # Optional: outer smoothing on totalx noise (asymmetric EMA)
        #  - Rising (attack): fast when NOT raining, slow when raining
        #  - Falling (release): fast
        if bool(cfg.smooth_N_E):
            dn = float(cfg.ne_release_alpha)

            # Treat as "raining" if FFT says rain or any subframe is marked as rain
            is_raining = bool(fft_rain_frame) or bool(np.any(rain_submask))
            up = float(cfg.ne_attack_alpha_wet if is_raining else cfg.ne_attack_alpha_dry)

            # choose alpha based on direction
            a = up if N_E_raw > self.N_E_smooth else dn

            # EMA
            self.N_E_smooth = (1.0 - a) * self.N_E_smooth + a * N_E_raw
            N_E = float(self.N_E_smooth)
        else:
            N_E = N_E_raw

        # Wiener-like gain (use band energy Eb and noise N_E)
        num = max(Eb - cfg.beta * N_E, 0.0)
        den = Eb + cfg.eps
        G_pow = num / den
        G_mag = float(np.sqrt(np.clip(G_pow, 0.0, 1.0)))
        G_mag = float(np.clip(G_mag, cfg.gain_floor, 1.0))
        M_clean = float(Mb * G_mag)

        return BandNoiseFrameOut(
            M_band=Mb,
            E_band=Eb,
            N_E=N_E,
            N_E_raw=N_E_raw,
            N_sub=N_sub,
            subE=subE.copy(),
            rain_submask=rain_submask,
            G_mag=G_mag,
            M_clean=M_clean,
            fft_rain_frame=bool(fft_rain_frame),
        )