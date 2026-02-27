from __future__ import annotations


from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
import scipy.signal as spsig

@dataclass
class FrameFeatures:
    """Compact per-frame diagnostics for tuning/visualization."""

    # indexing / time
    t: int
    time_s: float

    # operating-band loudness (dB) + rise (dB)
    Ltot: float
    dLtot: float

    # must-have gates
    gate_primary_rise_ok: bool
    gate_primary_peak_ok: bool
    gate_top_peaks_mode_ok: bool

    # hybrid / consensus gates
    gate_multi_mode_ok: bool = False
    mode_count: int = 0
    mode_ok: Optional[List[bool]] = None
    primary_rise_db: Optional[float] = None
    margin_db: Optional[float] = None
    margin_score: Optional[float] = None

    # peak debug
    top_peaks_hz: Optional[List[float]] = None
    top_peaks_db: Optional[List[float]] = None
    top_peaks_in_mode: Optional[List[bool]] = None
    top_peaks_in_primary: Optional[List[bool]] = None

    # mode evidence
    mode_L: Optional[List[float]] = None
    mode_rise_db: Optional[List[float]] = None
    mode_peak_db: Optional[List[float]] = None
    mode_score: Optional[List[float]] = None

    mode_score_joint: Optional[float] = None
    primary_ok: Optional[bool] = None

    # final
    rain_score_raw: Optional[float] = None
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
        "mode_weights",
        "mode_rise_db_low", "mode_rise_db_high",
        "mode_peak_db_low", "mode_peak_db_high",
        "mode_w_rise", "mode_w_peak",
        "primary_min_score",
        "peak_top_p", "peak_primary_q", "peaks_in_mode_min",
        "mode_overall_margin_db_low", "mode_overall_margin_db_high",
        "prob_w_mode_rise", "prob_w_mode_peak", "prob_w_margin",
        # used only for FrameFeatures time axis (safe defaults if missing)
        "fs", "hop",
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
        prev_mode_db1: Optional[np.ndarray],
        prev_mode_db2: Optional[np.ndarray],
        prev_Ltot1: Optional[float] = None,
        prev_Ltot2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Mode-centric per-frame rain evidence with hybrid (hard-gate + soft-evidence) logic.

        Must-have (hard gates):
          (G1) Primary-mode inter-frame rise must exceed cfg.primary_rise_db_thresh.
          (G2) One of the top-Q peaks must lie in the primary band.

        Consensus gate:
          (G3) At least K mode-bands must show evidence (rise/peak) in the current frame.

        Soft evidence (only used when the hard gates pass):
          - mode consensus strength (fraction/weighted fraction of modes that are ok)
          - per-mode rise/peak soft scores
          - margin: mode-rise vs overall-loudness-rise (reject “everything got louder”)

        Returns a rain_score_raw in [0,1] and rich debug fields.
        """
        cfg = self.cfg
        eps = float(getattr(cfg, "eps", 1e-9))

        # ---------------- config with safe defaults ----------------
        op_lo, op_hi = cfg.operating_band
        op_band = (float(op_lo), float(op_hi))

        top_p = int(getattr(cfg, "peak_top_p", 6))
        top_q = int(getattr(cfg, "peak_primary_q", 3))
        # default: require ~half the peaks to land in some mode band
        peaks_in_mode_min = int(getattr(cfg, "peaks_in_mode_min", max(2, int(np.ceil(top_p / 2)))))

        # multi-mode consensus requirement
        mode_count_min = int(getattr(cfg, "mode_count_min", 2))

        # peak finder knobs
        peak_smooth_bins = int(getattr(cfg, "peak_smooth_bins", 3))
        peak_prominence_db = float(getattr(cfg, "peak_prominence_db", 3.0))
        peak_min_distance_hz = float(getattr(cfg, "peak_min_distance_hz", 80.0))

        primary_rise_thresh = float(getattr(cfg, "primary_rise_db_thresh", 6.0))

        # margin thresholds
        margin_low = float(getattr(cfg, "mode_overall_margin_db_low", 1.5))
        margin_high = float(getattr(cfg, "mode_overall_margin_db_high", 6.0))

        # NEW: hard-ish margins to enforce "not just overall loudness" and "multi-mode rise"
        primary_overall_margin_db_thresh = float(getattr(cfg, "primary_overall_margin_db_thresh", 2.0))
        topk_overall_margin_db_thresh = float(getattr(cfg, "topk_overall_margin_db_thresh", 1.0))
        topk_margin_k = int(getattr(cfg, "topk_margin_k", 2))

        # soft-evidence weights
        w_mode_consensus = float(getattr(cfg, "prob_w_mode_consensus", 0.45))
        w_mode_rise = float(getattr(cfg, "prob_w_mode_rise", 0.35))
        w_mode_peak = float(getattr(cfg, "prob_w_mode_peak", 0.10))
        w_margin = float(getattr(cfg, "prob_w_margin", 0.10))
        wsum = max(1e-12, (w_mode_consensus + w_mode_rise + w_mode_peak + w_margin))
        w_mode_consensus /= wsum
        w_mode_rise /= wsum
        w_mode_peak /= wsum
        w_margin /= wsum

        mode_bands: List[Tuple[float, float]] = list(getattr(cfg, "mode_bands", []))
        n_modes = len(mode_bands)
        if n_modes == 0:
            raise AttributeError("cfg.mode_bands must be defined for mode-centric classifier")

        w_mode = np.asarray(getattr(cfg, "mode_weights", [1.0] * n_modes), dtype=float)
        if w_mode.size != n_modes:
            w_mode = np.ones(n_modes, dtype=float)

        pidx = int(getattr(cfg, "primary_mode_idx", 0))
        if not (0 <= pidx < n_modes):
            pidx = 0
        primary_band = mode_bands[pidx]

        rise_low = float(getattr(cfg, "mode_rise_db_low", 2.0))
        rise_high = float(getattr(cfg, "mode_rise_db_high", 6.0))
        peak_low = float(getattr(cfg, "mode_peak_db_low", 2.0))
        peak_high = float(getattr(cfg, "mode_peak_db_high", 8.0))

        # optional per-mode hard thresholds (if provided)
        rise_thresh_low = float(getattr(cfg, "mode_rise_db_thresh_low", rise_low))
        rise_thresh_high = float(getattr(cfg, "mode_rise_db_thresh_high", rise_high))
        peak_thresh_low = float(getattr(cfg, "mode_peak_db_thresh_low", peak_low))
        peak_thresh_high = float(getattr(cfg, "mode_peak_db_thresh_high", peak_high))

        primary_min_score = float(getattr(cfg, "primary_min_score", 0.5))

        # ---------------- operating-band loudness ----------------
        Ltot = self._band_db(P_frame, freqs, op_band, eps)
        dLtot1 = (Ltot - prev_Ltot1) if (prev_Ltot1 is not None and np.isfinite(Ltot) and np.isfinite(prev_Ltot1)) else 0.0
        dLtot2 = (Ltot - prev_Ltot2) if (prev_Ltot2 is not None and np.isfinite(Ltot) and np.isfinite(prev_Ltot2)) else 0.0
        dLtot = float(max(dLtot1, dLtot2))

        # ---------------- peak logic (hard gates + auxiliary) ----------------
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

        # ---------------- mode energies / rises / peakiness ----------------
        mode_db = np.zeros(n_modes, dtype=float)
        mode_peak_db = np.zeros(n_modes, dtype=float)
        for i, b in enumerate(mode_bands):
            mode_db[i] = self._band_db(P_frame, freqs, b, eps)
            mode_peak_db[i] = self._peakiness_db(P_frame, freqs, b, eps)

        mode_rise = np.zeros(n_modes, dtype=float)
        if prev_mode_db1 is not None:
            mode_rise = np.maximum(mode_rise, mode_db - prev_mode_db1)
        if prev_mode_db2 is not None:
            mode_rise = np.maximum(mode_rise, mode_db - prev_mode_db2)

        primary_rise = float(mode_rise[pidx]) if (0 <= pidx < n_modes) else 0.0
        gate_primary_rise_ok = (primary_rise >= primary_rise_thresh)

        # ---------------- per-mode soft scores ----------------
        mode_sr = np.zeros(n_modes, dtype=float)
        mode_sp = np.zeros(n_modes, dtype=float)
        mode_score_each = np.zeros(n_modes, dtype=float)
        for i in range(n_modes):
            mode_sr[i] = self._soft_score(float(mode_rise[i]), rise_low, rise_high)
            mode_sp[i] = self._soft_score(float(mode_peak_db[i]), peak_low, peak_high)
            mode_score_each[i] = 0.8 * mode_sr[i] + 0.2 * mode_sp[i]

        primary_ok = bool(mode_score_each[pidx] >= primary_min_score)

        # ---------------- NEW: consensus (per-mode ok + count) ----------------
        # must-have inter-frame bump: even "peak-only" evidence must include at least some rise
        mode_ok = (
            (mode_rise >= rise_thresh_high)
            | ((mode_rise >= rise_thresh_low) & (mode_peak_db >= peak_thresh_low))
            | ((mode_rise >= rise_thresh_low) & (mode_peak_db >= peak_thresh_high))
        )
        mode_count = int(np.sum(mode_ok))
        gate_multi_mode_ok = (mode_count >= mode_count_min)

        # Consensus strength in [0,1]
        denom_w = float(np.sum(w_mode) + eps)
        consensus_wfrac = float(np.sum(w_mode * mode_ok.astype(float)) / denom_w)
        consensus_frac = float(mode_count / max(1, n_modes))
        consensus_score = float(0.5 * consensus_frac + 0.5 * consensus_wfrac)

        # ---------------- margin: mode rise vs overall rise ----------------
        margin_score = 0.0
        margin_db = 0.0
        primary_margin_ok = False
        topk_margin_ok = False

        if prev_mode_db1 is not None:
            # margin uses "mode-rise" vs "overall loudness" rise
            dL_total = float(dLtot)

            # primary margin
            primary_margin_db = float(primary_rise - dL_total)
            primary_margin_ok = bool(primary_margin_db >= primary_overall_margin_db_thresh)

            # top-k modes margin (robust to one-mode domination)
            k_eff = int(max(1, min(topk_margin_k, n_modes)))
            topk_rises = np.sort(mode_rise)[-k_eff:]
            topk_mean_rise = float(np.mean(topk_rises)) if topk_rises.size else 0.0
            topk_margin_db = float(topk_mean_rise - dL_total)
            topk_margin_ok = bool(topk_margin_db >= topk_overall_margin_db_thresh)

            # soft margin score based on weighted-average mode rise
            dL_mode = float(np.sum(w_mode * mode_rise) / (float(np.sum(w_mode)) + eps))
            margin_db = float(dL_mode - dL_total)
            margin_score = self._soft_score(margin_db, margin_low, margin_high)

        # ---------------- hard gating ----------------
        must_ok = bool(gate_primary_rise_ok and gate_primary_peak_ok and gate_top_peaks_mode_ok)

        # ---------------- final score ----------------
        if must_ok:
            # Use consensus as primary driver; keep soft rise/peak as refinement.
            rise_strength = float(np.sum(w_mode * mode_sr) / (float(np.sum(w_mode)) + eps))
            peak_strength = float(np.sum(w_mode * mode_sp) / (float(np.sum(w_mode)) + eps))

            rain_score_raw = float(
                np.clip(
                    (w_mode_consensus * consensus_score)
                    + (w_mode_rise * rise_strength)
                    + (w_mode_peak * peak_strength)
                    + (w_margin * margin_score),
                    0.0,
                    1.0,
                )
            )

            # If multi-mode consensus fails, strongly cap the score (prevents single-mode override)
            if not gate_multi_mode_ok:
                rain_score_raw = min(rain_score_raw, 0.35)

        else:
            rain_score_raw = 0.0

        return {
            "Ltot": float(Ltot), "dLtot": float(dLtot),

            "top_peaks_hz": top_freq,
            "top_peaks_db": top_db,
            "top_peaks_in_mode": np.asarray(in_mode, dtype=bool),
            "top_peaks_in_primary": np.asarray(in_primary, dtype=bool),

            "gate_primary_rise_ok": bool(gate_primary_rise_ok),
            "gate_primary_peak_ok": bool(gate_primary_peak_ok),
            "gate_top_peaks_mode_ok": bool(gate_top_peaks_mode_ok),
            "gate_multi_mode_ok": bool(gate_multi_mode_ok),

            "primary_rise_db": float(primary_rise),

            "mode_db": mode_db,
            "mode_rise": mode_rise,
            "mode_peak_db": mode_peak_db,
            "mode_score_each": mode_score_each,

            # NEW debug
            "mode_ok": mode_ok.astype(bool),
            "mode_count": int(mode_count),
            "consensus_score": float(consensus_score),
            "margin_db": float(margin_db),
            "margin_score": float(margin_score),

            "primary_ok": bool(primary_ok),
            "rain_score_raw": float(rain_score_raw),
            "primary_margin_ok": bool(primary_margin_ok),
            "topk_margin_ok": bool(topk_margin_ok),
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

        # mode arrays
        n_modes = len(cfg.mode_bands)
        mode_db_all = np.full((n_modes, T), np.nan)
        mode_rise_all = np.full((n_modes, T), np.nan)
        mode_peak_all = np.full((n_modes, T), np.nan)
        mode_score_each_all = np.full((n_modes, T), np.nan)
        mode_joint = np.full(T, np.nan)
        primary_ok = np.zeros(T, dtype=bool)

        rain_score_raw = np.full(T, np.nan)

        # New: operating-band loudness history
        Ltot = np.full(T, np.nan)
        dLtot = np.full(T, np.nan)
        prev_Ltot1: Optional[float] = None
        prev_Ltot2: Optional[float] = None

        prev_mode1: Optional[np.ndarray] = None
        prev_mode2: Optional[np.ndarray] = None
        
        debug_frames: List[FrameFeatures] = []
        for t in range(T):
            det = self._detect_rain_frame_v2(
                P[:, t],
                freqs,
                prev_mode1,
                prev_mode2,
                prev_Ltot1,
                prev_Ltot2,
            )

            rain_score_raw[t] = det["rain_score_raw"]

            mode_db_all[:, t] = det["mode_db"]
            mode_rise_all[:, t] = det["mode_rise"]
            mode_peak_all[:, t] = det["mode_peak_db"]
            mode_score_each_all[:, t] = det["mode_score_each"]
            # mode_joint and primary_ok are for v1; hybrid logic may not return those
            mode_joint[t] = float(det.get("mode_score_joint", np.nan))
            primary_ok[t] = bool(det.get("primary_ok"))

            # Track Ltot/dLtot
            Ltot[t] = det.get("Ltot", np.nan)
            dLtot[t] = det.get("dLtot", np.nan)
            prev_Ltot2, prev_Ltot1 = prev_Ltot1, Ltot[t]

            # shift mode history
            prev_mode2, prev_mode1 = prev_mode1, det["mode_db"].copy()

            # --- compact per-frame debug record (for interactive viz / tuning) ---
            fs = float(getattr(cfg, "fs", 1.0))
            hop = float(getattr(cfg, "hop", 0.0))
            time_s = float(t * hop / fs) if (hop > 0 and fs > 0) else float(t)

            ff = FrameFeatures(
                t=int(t),
                time_s=time_s,
                Ltot=float(Ltot[t]),
                dLtot=float(dLtot[t]),
                gate_primary_rise_ok=bool(det.get("gate_primary_rise_ok", False)),
                gate_primary_peak_ok=bool(det.get("gate_primary_peak_ok", False)),
                gate_top_peaks_mode_ok=bool(det.get("gate_top_peaks_mode_ok", False)),
                gate_multi_mode_ok=bool(det.get("gate_multi_mode_ok", False)),
                mode_count=int(det.get("mode_count", 0)),
                mode_ok=self._as_bool_list(det.get("mode_ok")),
                primary_rise_db=float(det.get("primary_rise_db", np.nan)),
                margin_db=float(det.get("margin_db", np.nan)),
                margin_score=float(det.get("margin_score", np.nan)),
                top_peaks_hz=self._as_list(det.get("top_peaks_hz")),
                top_peaks_db=self._as_list(det.get("top_peaks_db")),
                top_peaks_in_mode=self._as_bool_list(det.get("top_peaks_in_mode")),
                top_peaks_in_primary=self._as_bool_list(det.get("top_peaks_in_primary")),
                mode_L=self._as_list(det.get("mode_db")),
                mode_rise_db=self._as_list(det.get("mode_rise")),
                mode_peak_db=self._as_list(det.get("mode_peak_db")),
                mode_score=self._as_list(det.get("mode_score_each")),
                mode_score_joint=float(det.get("mode_score_joint", np.nan)),
                primary_ok=bool(det.get("primary_ok", False)),
                rain_score_raw=float(det.get("rain_score_raw", np.nan)),
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
            "Ltot": Ltot,
            "dLtot": dLtot,

            "rain_score_raw": rain_score_raw,

            "mode_db": mode_db_all,
            "mode_rise": mode_rise_all,
            "mode_peak_db": mode_peak_all,
            "mode_score_each": mode_score_each_all,
            "mode_score_joint": mode_joint,
            "primary_ok": primary_ok,

            "label": label,
            "use_for_noise_psd": use_for_noise_psd,

            "frames": debug_frames,
        }

        return is_rain, det_debug, rain_conf, noise_conf