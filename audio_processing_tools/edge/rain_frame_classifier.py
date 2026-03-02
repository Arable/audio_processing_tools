from __future__ import annotations


from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
import scipy.signal as spsig

@dataclass
class FrameFeatures:
    """Compact per-frame diagnostics for tuning/visualization.

    Keep this small: per-frame scalars + a light peak summary + final decision.
    Heavy arrays stay in `det_debug`.
    """

    # indexing / time
    t: int
    time_s: float

    # operating-band loudness (dB)
    Ltot: float
    dLtot: float

    # key evidence (already robust-normalized where applicable)
    primary_flux_z: float
    mode_count_z: int

    # must-have gates
    gate_primary_rise_ok: bool
    gate_primary_peak_ok: bool
    gate_top_peaks_mode_ok: bool
    gate_multi_mode_ok: bool

    # light peak debug
    top_peaks_hz: Optional[List[float]] = None
    top_peaks_db: Optional[List[float]] = None
    top_peaks_in_mode: Optional[List[bool]] = None
    top_peaks_in_primary: Optional[List[bool]] = None

    # final
    rain_score_raw: float = 0.0
    label: Optional[str] = None        # "rain" | "noise" | "uncertain"
    use_for_noise_psd: Optional[bool] = None

class RainFrameClassifierMixin:
    # ----------------------- band-energy helper -----------------------

    # Keep the truly-required knobs here; everything else should have a safe default.
    REQUIRED_CFG_FIELDS = [
        "eps",
        "operating_band",
        "mode_bands",
        "primary_mode_idx",
        "primary_rise_db_thresh",
        "rain_hi",
        "noise_hi",
        "noise_update_thresh",
        "rain_hold_frames",
        "noise_confirm_frames",
    ]

    # Optional tunables (all have defaults inside the implementation)
    OPTIONAL_CFG_FIELDS = [
        "peak_top_p", "peak_primary_q", "peaks_in_mode_min",
        "fs", "hop",
        "flux_win_frames", "flux_low_frac", "flux_use_log",
        "primary_flux_z_thresh", "mode_flux_z_thresh", "modes_flux_min_count",
        "peak_smooth_bins", "peak_prominence_db", "peak_min_distance_hz",
    ]

    def _validate_rain_cfg(self):
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise AttributeError("self.cfg is missing in processor")

        missing = [f for f in self.REQUIRED_CFG_FIELDS if not hasattr(cfg, f)]
        if missing:
            raise AttributeError(f"RainFrameClassifier config missing required fields: {missing}")

        # ---- structural validation ----
        eps = float(getattr(cfg, "eps"))
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError("cfg.eps must be a positive finite float")

        op = getattr(cfg, "operating_band")
        if not (isinstance(op, (list, tuple)) and len(op) == 2):
            raise ValueError("cfg.operating_band must be a (f_lo, f_hi) tuple")
        op_lo, op_hi = float(op[0]), float(op[1])
        if not (np.isfinite(op_lo) and np.isfinite(op_hi) and op_lo < op_hi):
            raise ValueError("cfg.operating_band must satisfy f_lo < f_hi")

        mb = getattr(cfg, "mode_bands")
        if not isinstance(mb, (list, tuple)) or len(mb) == 0:
            raise ValueError("cfg.mode_bands must be a non-empty list/tuple of (f_lo, f_hi)")
        # validate each band tuple
        for i, b in enumerate(mb):
            if not (isinstance(b, (list, tuple)) and len(b) == 2):
                raise ValueError(f"cfg.mode_bands[{i}] must be a (f_lo, f_hi) tuple")
            lo, hi = float(b[0]), float(b[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"cfg.mode_bands[{i}] must satisfy f_lo < f_hi")

        pidx = int(getattr(cfg, "primary_mode_idx"))
        if not (0 <= pidx < len(mb)):
            raise ValueError("cfg.primary_mode_idx must be a valid index into cfg.mode_bands")

        thr = float(getattr(cfg, "primary_rise_db_thresh"))
        if not np.isfinite(thr):
            raise ValueError("cfg.primary_rise_db_thresh must be finite")

    @staticmethod
    def _band_db(P_frame: np.ndarray, freqs: np.ndarray, band: Tuple[float, float], eps: float) -> float:
        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return -np.inf
        val = float(np.sum(P_frame[mask]))
        return 10.0 * np.log10(val + eps)

    @staticmethod
    def _soft_score(value: float, low: float, high: float) -> float:
        if not np.isfinite(value):
            return 0.0
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    @staticmethod
    def _peakiness_db(
        P_frame: np.ndarray,
        freqs: np.ndarray,
        band: Tuple[float, float],
        eps: float,
        surround_hz: float = 150.0,
    ) -> float:
        """
        Intra-frame “bump”: peak in band vs local surround floor.

        peakiness_db = 10log10(max_in_band / median_in_surround)

        Surround is (band expanded by surround_hz) minus the band itself.
        """
        lo, hi = band
        band_mask = (freqs >= lo) & (freqs < hi)
        if not np.any(band_mask):
            return -np.inf

        peak = float(np.max(P_frame[band_mask]) + eps)

        lo2 = max(0.0, lo - surround_hz)
        hi2 = hi + surround_hz
        sur_mask = (freqs >= lo2) & (freqs < hi2) & (~band_mask)
        if not np.any(sur_mask):
            # fallback: within-band median if no surround bins
            floor = float(np.median(P_frame[band_mask]) + eps)
        else:
            floor = float(np.median(P_frame[sur_mask]) + eps)

        return 10.0 * np.log10(peak / floor)

    @staticmethod
    def _band_lin(
        P_frame: np.ndarray,
        freqs: np.ndarray,
        band: Tuple[float, float],
    ) -> float:
        """Linear power in band."""
        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(np.sum(P_frame[mask]))

    @staticmethod
    def _band_flux_pos(
        P_t: np.ndarray,
        P_prev: Optional[np.ndarray],
        freqs: np.ndarray,
        band: Tuple[float, float],
        eps: float,
        *,
        use_log: bool = True,
    ) -> float:
        """Positive spectral flux in a band vs previous frame.

        flux = sum(max(0, x_t - x_prev)) over bins in band, where x is either
        log-power (default) or linear power.
        """
        if P_prev is None:
            return 0.0
        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        a = np.asarray(P_t[mask], dtype=float)
        b = np.asarray(P_prev[mask], dtype=float)
        if use_log:
            a = np.log10(a + eps)
            b = np.log10(b + eps)
        d = a - b
        d[d < 0.0] = 0.0
        return float(np.sum(d))

    @staticmethod
    def _robust_z_at(
        x: np.ndarray,
        t: int,
        win: int,
        low_frac: float,
        eps: float,
    ) -> Tuple[float, float, float]:
        """Robust z-score at index t using a *symmetric* window.

        We estimate the baseline from the bottom `low_frac` fraction of values
        inside the window to reduce contamination from rain-like outliers.

        Returns (z, baseline, scale).
        """
        T = int(x.size)
        if T == 0:
            return 0.0, 0.0, 0.0

        w = int(max(0, win))
        t0 = max(0, int(t) - w)
        t1 = min(T, int(t) + w + 1)
        vals = np.asarray(x[t0:t1], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 3:
            return 0.0, float(vals[0]) if vals.size else 0.0, 0.0

        # bottom-k subset for baseline
        lf = float(np.clip(low_frac, 0.05, 0.8))
        k = max(3, int(np.floor(lf * vals.size)))
        srt = np.sort(vals)
        bot = srt[:k]

        # robust center = median of low-energy subset
        med = float(np.median(bot))

        # MAD-based scale (robust std estimate)
        mad = float(np.median(np.abs(bot - med)))
        scale = float(1.4826 * mad + eps)

        # use median consistently as baseline
        baseline = med

        z = float((x[t] - med) / scale) if np.isfinite(x[t]) else 0.0
        return z, baseline, scale

    @staticmethod
    def _top_k_peaks_in_band(
        P_frame: np.ndarray,
        freqs: np.ndarray,
        band: Tuple[float, float],
        k: int,
        eps: float,
        *,
        smooth_bins: int = 3,
        prominence_db: float = 3.0,
        min_distance_hz: float = 80.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (peak_freqs_hz, peak_db) for top-k *spectral peaks* within a band.

        Uses `scipy.signal.find_peaks` on a lightly-smoothed log-power spectrum.

        Parameters
        ----------
        smooth_bins:
            Moving-average window in FFT bins for stabilizing peak finding.
        prominence_db:
            Peak prominence threshold in dB.
        min_distance_hz:
            Minimum separation between peaks (Hz). Converted to bins.
        """
        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask) or k <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        idx = np.flatnonzero(mask)
        vals = np.asarray(P_frame[idx], dtype=float)
        # log-power in dB for peak finding
        y_db = 10.0 * np.log10(vals + eps)

        # optional light smoothing to reduce bin-to-bin jitter
        sb = int(max(1, smooth_bins))
        if sb > 1:
            kernel = np.ones(sb, dtype=float) / float(sb)
            y_db = np.convolve(y_db, kernel, mode="same")

        # convert min_distance_hz to bins (roughly)
        df = float(np.median(np.diff(freqs[idx]))) if idx.size > 1 else 1.0
        dist_bins = int(max(1, round(float(min_distance_hz) / max(1e-9, df))))

        peaks, props = spsig.find_peaks(
            y_db,
            prominence=float(prominence_db),
            distance=dist_bins,
        )
        if peaks.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        # rank by peak height (or prominence) descending
        heights = y_db[peaks]
        order = np.argsort(heights)[::-1]
        peaks = peaks[order]

        k_eff = min(int(k), int(peaks.size))
        peaks = peaks[:k_eff]

        peak_idx = idx[peaks]
        peak_freq = freqs[peak_idx]
        peak_db = 10.0 * np.log10(P_frame[peak_idx] + eps)

        return peak_freq.astype(float), peak_db.astype(float)

    @staticmethod
    def _in_any_band(f: float, bands: List[Tuple[float, float]]) -> bool:
        for lo, hi in bands:
            if (f >= lo) and (f < hi):
                return True
        return False

    @staticmethod
    def _as_list(x) -> List[float]:
        """Convert numpy arrays / lists / tuples to a plain python list of floats."""
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            return [float(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        return [float(x)]

    @staticmethod
    def _as_bool_list(x) -> List[bool]:
        """Convert numpy arrays / lists / tuples to a plain python list of bools."""
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            return [bool(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [bool(v) for v in x]
        return [bool(x)]

    # ----------------------- per-frame scoring -----------------------

    def _detect_rain_frame_v2(
        self,
        P_frame: np.ndarray,
        freqs: np.ndarray,
        P_prev1: Optional[np.ndarray],
        P_prev2: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Per-frame raw features for flux-based rain/noise classification.

        This function intentionally returns *raw* (unnormalized) features.
        Robust normalization (MAD z-scores) and labeling happen in
        `_detect_rain_over_time`, where a symmetric time window is available.
        """
        cfg = self.cfg
        eps = float(getattr(cfg, "eps", 1e-9))

        op_lo, op_hi = cfg.operating_band
        op_band = (float(op_lo), float(op_hi))

        mode_bands: List[Tuple[float, float]] = list(getattr(cfg, "mode_bands", []))
        n_modes = len(mode_bands)
        if n_modes == 0:
            raise AttributeError("cfg.mode_bands must be defined for flux-based classifier")

        pidx = int(getattr(cfg, "primary_mode_idx", 0))
        if not (0 <= pidx < n_modes):
            pidx = 0
        primary_band = mode_bands[pidx]

        # peak-gate knobs
        top_p = int(getattr(cfg, "peak_top_p", 6))
        top_q = int(getattr(cfg, "peak_primary_q", 3))
        peaks_in_mode_min = int(getattr(cfg, "peaks_in_mode_min", 2))

        peak_smooth_bins = int(getattr(cfg, "peak_smooth_bins", 3))
        peak_prominence_db = float(getattr(cfg, "peak_prominence_db", 3.0))
        peak_min_distance_hz = float(getattr(cfg, "peak_min_distance_hz", 80.0))

        # flux knobs
        use_log_flux = bool(getattr(cfg, "flux_use_log", True))

        # operating-band loudness (for debug only)
        Ltot = self._band_db(P_frame, freqs, op_band, eps)

        # raw positive flux (t vs t-1 and t vs t-2)
        primary_flux_1 = self._band_flux_pos(P_frame, P_prev1, freqs, primary_band, eps, use_log=use_log_flux)
        primary_flux_2 = self._band_flux_pos(P_frame, P_prev2, freqs, primary_band, eps, use_log=use_log_flux)
        primary_flux = float(max(primary_flux_1, primary_flux_2))

        mode_flux = np.zeros(n_modes, dtype=float)
        for i, b in enumerate(mode_bands):
            f1 = self._band_flux_pos(P_frame, P_prev1, freqs, b, eps, use_log=use_log_flux)
            f2 = self._band_flux_pos(P_frame, P_prev2, freqs, b, eps, use_log=use_log_flux)
            mode_flux[i] = max(f1, f2)

        # --- peaks gate (in operating band) ---
        top_freq, top_db = self._top_k_peaks_in_band(
            P_frame,
            freqs,
            op_band,
            top_p,
            eps,
            smooth_bins=peak_smooth_bins,
            prominence_db=peak_prominence_db,
            min_distance_hz=peak_min_distance_hz,
        )
        in_mode = [self._in_any_band(float(f), mode_bands) for f in top_freq]
        in_primary = [self._in_any_band(float(f), [primary_band]) for f in top_freq]

        gate_top_peaks_mode_ok = (sum(in_mode) >= peaks_in_mode_min) if top_freq.size else False

        gate_primary_peak_ok = False
        if top_freq.size:
            q_eff = min(int(top_q), int(top_freq.size))
            gate_primary_peak_ok = any(in_primary[:q_eff])

        return {
            "Ltot": float(Ltot),
            "primary_flux": float(primary_flux),
            "mode_flux": mode_flux,
            "top_peaks_hz": top_freq,
            "top_peaks_db": top_db,
            "top_peaks_in_mode": np.asarray(in_mode, dtype=bool),
            "top_peaks_in_primary": np.asarray(in_primary, dtype=bool),
            "gate_primary_peak_ok": bool(gate_primary_peak_ok),
            "gate_top_peaks_mode_ok": bool(gate_top_peaks_mode_ok),
        }

    # ----------------------- 3-state classification -----------------------

    def _classify_3state_over_time(
        self,
        rain_score_raw: np.ndarray,   # (T,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          label: array of strings {"rain","noise","uncertain"}
          use_for_noise_psd: bool (True only for high-confidence noise frames)
        """
        cfg = self.cfg
        T = len(rain_score_raw)

        labels = np.empty(T, dtype="<U9")  # "uncertain" is 9 chars
        use_for_noise = np.zeros(T, dtype=bool)

        rain_hold = 0
        noise_run = 0

        for t in range(T):
            r = float(rain_score_raw[t])

            # strong rain => immediate rain + hold
            if r >= float(cfg.rain_hi):
                labels[t] = "rain"
                rain_hold = int(cfg.rain_hold_frames)
                noise_run = 0

            elif rain_hold > 0:
                labels[t] = "rain"
                rain_hold -= 1
                noise_run = 0

            else:
                # high-confidence noise condition: rain score very low
                noise_score = 1.0 - r
                if noise_score >= float(cfg.noise_hi):
                    noise_run += 1
                    if noise_run >= int(cfg.noise_confirm_frames):
                        labels[t] = "noise"
                    else:
                        labels[t] = "uncertain"
                else:
                    noise_run = 0
                    labels[t] = "uncertain"

            # PSD update only for *high-confidence* noise
            noise_score = 1.0 - r
            use_for_noise[t] = (labels[t] == "noise") and (noise_score >= float(cfg.noise_update_thresh))

        return labels, use_for_noise

    # ----------------------- time evolution (public) -----------------------

    def _detect_rain_over_time(
        self,
        P: np.ndarray,      # (F,T)
        freqs: np.ndarray,  # (F,)
    ) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
        """
        NEW outputs added:
          - label (T,) in {"rain","noise","uncertain"}
          - use_for_noise_psd (T,) bool (what your PSD estimator should use)
          - rain_score_raw (T,)
          - mode debug arrays

        Back-compat outputs:
          - is_rain (T,) bool = label == "rain"
          - rain_conf (T,) = rain_score_raw  (no EMA)
          - noise_conf (T,) = 1 - rain_score_raw
        """
        self._validate_rain_cfg()

        F, T = P.shape
        cfg = self.cfg

        # ---- raw flux features (per frame) ----
        mode_bands: List[Tuple[float, float]] = list(getattr(cfg, "mode_bands", []))
        n_modes = len(mode_bands)
        if n_modes == 0:
            raise AttributeError("cfg.mode_bands must be defined")

        primary_flux = np.full(T, np.nan)
        mode_flux = np.full((n_modes, T), np.nan)
        Ltot = np.full(T, np.nan)
        total_mode_flux = np.full(T, np.nan)

        # peak-gate traces
        peaks_in_mode_count = np.full(T, 0, dtype=int)
        peaks_in_primary_topq = np.full(T, 0, dtype=int)
        gate_primary_peak_ok = np.zeros(T, dtype=bool)
        gate_top_peaks_mode_ok = np.zeros(T, dtype=bool)

        debug_frames: List[FrameFeatures] = []

        # config knobs for robust normalization + decision
        win = int(getattr(cfg, "flux_win_frames", 20))
        low_frac = float(getattr(cfg, "flux_low_frac", 0.30))

        primary_z_thresh = float(getattr(cfg, "primary_flux_z_thresh", 6.0))
        mode_z_thresh = float(getattr(cfg, "mode_flux_z_thresh", 3.0))
        modes_min_count = int(getattr(cfg, "modes_flux_min_count", 2))

        # for debug time axis
        fs = float(getattr(cfg, "fs", 1.0))
        hop = float(getattr(cfg, "hop", 0.0))

        # store per-frame peak lists temporarily
        _top_peaks_hz: List[List[float]] = []
        _top_peaks_db: List[List[float]] = []
        _top_peaks_in_mode: List[List[bool]] = []
        _top_peaks_in_primary: List[List[bool]] = []

        for t in range(T):
            P_prev1 = P[:, t - 1] if t - 1 >= 0 else None
            P_prev2 = P[:, t - 2] if t - 2 >= 0 else None

            det = self._detect_rain_frame_v2(
                P[:, t],
                freqs,
                P_prev1,
                P_prev2,
            )

            Ltot[t] = float(det.get("Ltot", np.nan))
            primary_flux[t] = float(det.get("primary_flux", np.nan))
            mf = det.get("mode_flux")
            if isinstance(mf, np.ndarray) and mf.shape[0] == n_modes:
                mode_flux[:, t] = mf
            if isinstance(mf, np.ndarray):
                total_mode_flux[t] = float(np.sum(mf))

            gate_primary_peak_ok[t] = bool(det.get("gate_primary_peak_ok", False))
            gate_top_peaks_mode_ok[t] = bool(det.get("gate_top_peaks_mode_ok", False))

            # peak counts for quick tuning
            in_mode = det.get("top_peaks_in_mode")
            in_primary = det.get("top_peaks_in_primary")
            if isinstance(in_mode, np.ndarray):
                peaks_in_mode_count[t] = int(np.sum(in_mode.astype(int)))
            if isinstance(in_primary, np.ndarray):
                top_q = int(getattr(cfg, "peak_primary_q", 3))
                q_eff = min(int(top_q), int(in_primary.size))
                peaks_in_primary_topq[t] = int(np.sum(in_primary[:q_eff].astype(int)))

            # stash lists for compact per-frame records
            _top_peaks_hz.append(self._as_list(det.get("top_peaks_hz")))
            _top_peaks_db.append(self._as_list(det.get("top_peaks_db")))
            _top_peaks_in_mode.append(self._as_bool_list(det.get("top_peaks_in_mode")))
            _top_peaks_in_primary.append(self._as_bool_list(det.get("top_peaks_in_primary")))

        # operating-band loudness rise (debug)
        dLtot = np.full(T, np.nan)
        for t in range(T):
            if t - 1 >= 0 and np.isfinite(Ltot[t]) and np.isfinite(Ltot[t - 1]):
                dLtot[t] = float(Ltot[t] - Ltot[t - 1])
            else:
                dLtot[t] = 0.0

        # ---- robust z-scores (MAD) using symmetric window ----
        primary_flux_z = np.full(T, np.nan)
        primary_flux_base = np.full(T, np.nan)

        mode_flux_z = np.full((n_modes, T), np.nan)
        mode_flux_base = np.full((n_modes, T), np.nan)
        total_mode_flux_z = np.full(T, np.nan)
        total_mode_flux_base = np.full(T, np.nan)

        for t in range(T):
            z, base, _ = self._robust_z_at(primary_flux, t, win, low_frac, float(cfg.eps))
            primary_flux_z[t] = z
            primary_flux_base[t] = base
            zt, bt, _ = self._robust_z_at(total_mode_flux, t, win, low_frac, float(cfg.eps))
            total_mode_flux_z[t] = zt
            total_mode_flux_base[t] = bt
            for i in range(n_modes):
                zi, bi, _ = self._robust_z_at(mode_flux[i, :], t, win, low_frac, float(cfg.eps))
                mode_flux_z[i, t] = zi
                mode_flux_base[i, t] = bi

        # ---- gates + score ----
        pidx = int(getattr(cfg, "primary_mode_idx", 0))
        if not (0 <= pidx < n_modes):
            pidx = 0

        gate_primary_rise_ok = np.isfinite(primary_flux_z) & (primary_flux_z >= primary_z_thresh)

        mode_count_z = np.sum(np.isfinite(mode_flux_z) & (mode_flux_z >= mode_z_thresh), axis=0).astype(int)
        gate_multi_mode_ok = mode_count_z >= modes_min_count

        # minimal score in [0,1]
        # (1) primary z, (2) multi-mode count, (3) peak gates
        # peak gates are treated as hard constraints.
        rain_score_raw = np.zeros(T, dtype=float)
        for t in range(T):
            if not (bool(gate_primary_peak_ok[t]) and bool(gate_top_peaks_mode_ok[t])):
                rain_score_raw[t] = 0.0
                continue
            if not bool(gate_primary_rise_ok[t]):
                rain_score_raw[t] = 0.0
                continue

            # soft contributions
            s_primary = float(np.clip((primary_flux_z[t] - primary_z_thresh) / max(1e-9, primary_z_thresh), 0.0, 1.0))
            s_modes_count = float(np.clip(mode_count_z[t] / max(1, n_modes), 0.0, 1.0))

            # combined mode flux (more stable than per-mode count alone)
            total_z = float(total_mode_flux_z[t]) if np.isfinite(total_mode_flux_z[t]) else 0.0
            s_modes_total = float(np.clip(total_z / max(1e-9, primary_z_thresh), 0.0, 1.0))

            # combine both mode evidences
            s_modes = 0.5 * s_modes_count + 0.5 * s_modes_total

            rain_score_raw[t] = float(np.clip(0.65 * s_primary + 0.35 * s_modes, 0.0, 1.0))
            if not bool(gate_multi_mode_ok[t]):
                rain_score_raw[t] = min(rain_score_raw[t], 0.35)

        # ---- build compact per-frame debug records ----
        for t in range(T):
            time_s = float(t * hop / fs) if (hop > 0 and fs > 0) else float(t)
            ff = FrameFeatures(
                t=int(t),
                time_s=time_s,
                Ltot=float(Ltot[t]),
                dLtot=float(dLtot[t]),
                primary_flux_z=float(primary_flux_z[t]),
                mode_count_z=int(mode_count_z[t]),
                gate_primary_rise_ok=bool(gate_primary_rise_ok[t]),
                gate_primary_peak_ok=bool(gate_primary_peak_ok[t]),
                gate_top_peaks_mode_ok=bool(gate_top_peaks_mode_ok[t]),
                gate_multi_mode_ok=bool(gate_multi_mode_ok[t]),
                top_peaks_hz=_top_peaks_hz[t],
                top_peaks_db=_top_peaks_db[t],
                top_peaks_in_mode=_top_peaks_in_mode[t],
                top_peaks_in_primary=_top_peaks_in_primary[t],
                rain_score_raw=float(rain_score_raw[t]),
            )
            debug_frames.append(ff)

        # 3-state labeling + PSD gating
        label, use_for_noise_psd = self._classify_3state_over_time(rain_score_raw)

        # stamp labels/psd-use into compact frame records
        for t in range(T):
            debug_frames[t].label = str(label[t])
            debug_frames[t].use_for_noise_psd = bool(use_for_noise_psd[t])

        # back-compat
        is_rain = (label == "rain")
        rain_conf = np.clip(rain_score_raw, 0.0, 1.0)
        noise_conf = 1.0 - rain_conf

        det_debug = {
            "operating_band": tuple(getattr(cfg, "operating_band")),
            "times_s": np.asarray([ff.time_s for ff in debug_frames], dtype=float),

            # raw + robust-normalized flux
            "primary_flux": primary_flux,
            "primary_flux_z": primary_flux_z,
            "primary_flux_base": primary_flux_base,

            "mode_flux": mode_flux,
            "mode_flux_z": mode_flux_z,
            "mode_flux_base": mode_flux_base,
            "mode_count_z": mode_count_z,
            "total_mode_flux": total_mode_flux,
            "total_mode_flux_z": total_mode_flux_z,
            "total_mode_flux_base": total_mode_flux_base,

            # peak gates
            "peaks_in_mode_count": peaks_in_mode_count,
            "peaks_in_primary_topq": peaks_in_primary_topq,
            "gate_primary_peak_ok": gate_primary_peak_ok,
            "gate_top_peaks_mode_ok": gate_top_peaks_mode_ok,

            # decision outputs
            "rain_score_raw": rain_score_raw,
            "label": label,
            "use_for_noise_psd": use_for_noise_psd,

            # loudness debug
            "Ltot": Ltot,
            "dLtot": dLtot,

            # compact per-frame records for interactive viz
            "frames": debug_frames,
        }

        return is_rain, det_debug, rain_conf, noise_conf