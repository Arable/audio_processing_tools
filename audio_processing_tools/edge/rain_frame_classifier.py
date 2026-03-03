from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal as spsig



# -----------------------------
# Debug / tuning payload
# -----------------------------

@dataclass
class FrameFeatures:
    """Compact per-frame diagnostics for tuning/visualization."""

    t: int
    time_s: float

    # Operating-band loudness (dB) and delta vs previous frame (dB)
    Ltot_db: float
    dLtot_db: float

    # Flux (raw) and robust z-scores
    flux_primary: float
    flux_modes: float
    z_primary: float
    z_modes: float

    # Peak structure (within operating band)
    n_peaks_total: int
    n_peaks_mode: int
    n_peaks_primary: int
    peak_ratio_mode: float

    top_peaks_hz: Optional[List[float]] = None
    top_peaks_db: Optional[List[float]] = None
    top_peaks_in_mode: Optional[List[bool]] = None
    top_peaks_in_primary: Optional[List[bool]] = None

    # Gates / score / outputs
    gate_primary_flux_ok: bool = False
    gate_primary_peak_ok: bool = False
    gate_top_peaks_mode_ok: bool = False

    rain_score_raw: float = float("nan")
    label: Optional[str] = None  # "rain" | "noise" | "uncertain"
    use_for_noise_psd: Optional[bool] = None



class RainFrameClassifierMixin:
    REQUIRED_DET_KEYS = [
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

    DET_DEFAULTS: Dict[str, Any] = {
        "zscore_mode": "mad",     # "mad" | "iqr"
        "z_win_sec": 1.0,

        "peak_top_p": 6,
        "peaks_in_mode_min": 2,
        "peak_prominence_db": 3.0,
        "peak_min_db_above_floor": 6.0,

        "primary_flux_z_thresh": 2.5,
        "modes_flux_z_thresh": 2.0,
        "peak_ratio_thresh": 0.5,

        "prob_w_primary": 0.05,
        "prob_w_modes": 0.90,
        "prob_w_peaks": 0.05,

        # optional plotting helpers
        "fs": 11162,
        "hop": 128,
    }

    def _detector_params(self) -> Dict[str, Any]:
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise AttributeError("self.cfg is missing")

        det = dict(self.DET_DEFAULTS)

        det_dict = getattr(cfg, "detector", None)
        if isinstance(det_dict, dict):
            det.update(det_dict)

        # Back-compat fallback: if someone still sets cfg.<field>, accept it.
        for k in self.REQUIRED_DET_KEYS:
            if k not in det and hasattr(cfg, k):
                det[k] = getattr(cfg, k)

        # normalize types
        if "operating_band" in det and isinstance(det["operating_band"], (list, tuple)) and len(det["operating_band"]) == 2:
            det["operating_band"] = (float(det["operating_band"][0]), float(det["operating_band"][1]))

        if "mode_bands" in det:
            det["mode_bands"] = [(float(a), float(b)) for (a, b) in det["mode_bands"]]

        det["eps"] = float(det.get("eps", 1e-9))
        det["primary_mode_idx"] = int(det.get("primary_mode_idx", 0))
        det["primary_rise_db_thresh"] = float(det.get("primary_rise_db_thresh", 0.0))

        det["rain_hold_frames"] = int(det.get("rain_hold_frames", 0))
        det["noise_confirm_frames"] = int(det.get("noise_confirm_frames", 0))
        det["peaks_in_mode_min"] = int(det.get("peaks_in_mode_min", det["peaks_in_mode_min"]))
        det["peak_top_p"] = int(det.get("peak_top_p", det["peak_top_p"]))

        return det


    # -----------------------------
    # Validation / cfg access
    # -----------------------------

def _validate_rain_cfg(self):
    cfg = getattr(self, "cfg", None)
    if cfg is None:
        raise AttributeError("self.cfg is missing in processor")

    det = self._detector_params()
    missing = [k for k in self.REQUIRED_DET_KEYS if k not in det]
    if missing:
        raise AttributeError(f"Detector config missing required keys: {missing}")

    op_lo, op_hi = det["operating_band"]
    if not (np.isfinite(op_lo) and np.isfinite(op_hi) and 0.0 < op_lo < op_hi):
        raise ValueError(f"Invalid operating_band: {det['operating_band']!r}")

    mb = det["mode_bands"]
    if not mb:
        raise ValueError("mode_bands must be non-empty")

    pidx = det["primary_mode_idx"]
    if pidx < 0 or pidx >= len(mb):
        raise ValueError(f"primary_mode_idx out of range: {pidx} for {len(mb)} mode_bands")

    def _cfg(self) -> Any:
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise AttributeError("self.cfg is missing in processor")
        return cfg

    # -----------------------------
    # Small helpers
    # -----------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray, k: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-k * x))

    @staticmethod
    def _db10(x: np.ndarray, eps: float) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(x, 0.0) + eps)

    @staticmethod
    def _robust_z_mad(x: np.ndarray, med: np.ndarray, mad: np.ndarray, eps: float) -> np.ndarray:
        # 1.4826 makes MAD consistent with std for Gaussian
        return (x - med) / (1.4826 * mad + eps)

    @staticmethod
    def _robust_z_iqr(x: np.ndarray, q25: np.ndarray, q75: np.ndarray, eps: float) -> np.ndarray:
        iqr = np.maximum(q75 - q25, 0.0)
        scale = 0.7413 * iqr  # ~std for Gaussian
        center = 0.5 * (q25 + q75)
        return (x - center) / (scale + eps)

    @staticmethod
    def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        lo, hi = float(band[0]), float(band[1])
        return (freqs >= lo) & (freqs <= hi)

    def _time_axis_s(self, T: int) -> np.ndarray:
        cfg = self._cfg()
        fs = float(getattr(cfg, "fs", 11162))
        hop = float(getattr(cfg, "hop", 128))
        return (np.arange(T, dtype=np.float64) * hop) / fs

    # -----------------------------
    # Feature extraction
    # -----------------------------

    def _compute_flux(self, X_db: np.ndarray, band_mask: np.ndarray, t: int, t_prev: int) -> float:
        """Positive spectral flux (in dB units) inside a band."""
        if t_prev < 0:
            return 0.0
        d = X_db[band_mask, t] - X_db[band_mask, t_prev]
        return float(np.sum(np.maximum(d, 0.0)))

    def _find_top_peaks(
        self,
        X_db_t: np.ndarray,
        freqs: np.ndarray,
        op_mask: np.ndarray,
        peak_top_p: int,
        prominence_db: float,
        min_db_above_floor: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (peak_freqs, peak_heights_db) for top peaks in operating band."""
        x = X_db_t.copy()
        x[~op_mask] = -np.inf

        finite = np.isfinite(x)
        if not finite.any():
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        floor = np.nanpercentile(x[finite], 20)
        height_min = float(floor + min_db_above_floor)

        peaks, _props = spsig.find_peaks(x, prominence=prominence_db)
        if peaks.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        heights = x[peaks]
        keep = heights >= height_min
        peaks = peaks[keep]
        heights = heights[keep]
        if peaks.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        order = np.argsort(heights)[::-1][: max(1, int(peak_top_p))]
        peaks = peaks[order]
        heights = heights[order]

        return freqs[peaks].astype(np.float64), heights.astype(np.float64)

    # -----------------------------
    # Robust stats over symmetric window
    # -----------------------------

    def _robust_stats_symmetric(self, x: np.ndarray, W: int) -> Dict[str, np.ndarray]:
        """Compute median/MAD and q25/q75 for each t using a symmetric window."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        T = x.size

        med = np.full(T, np.nan, dtype=np.float64)
        mad = np.full(T, np.nan, dtype=np.float64)
        q25 = np.full(T, np.nan, dtype=np.float64)
        q75 = np.full(T, np.nan, dtype=np.float64)

        for t in range(T):
            a = max(0, t - W)
            b = min(T, t + W + 1)
            w = x[a:b]
            if w.size < 3:
                w = x[: max(1, min(T, 5))]

            m = np.nanmedian(w)
            med[t] = m
            mad[t] = np.nanmedian(np.abs(w - m))
            q25[t] = np.nanpercentile(w, 25)
            q75[t] = np.nanpercentile(w, 75)

        return {"med": med, "mad": mad, "q25": q25, "q75": q75}

    # -----------------------------
    # Main API used by SpectralNoiseProcessor
    # -----------------------------

    def _detect_rain_over_time(
        self,
        P: np.ndarray,      # (F, T) power
        freqs: np.ndarray,  # (F,)
        use_robust_z: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Returns:
          is_rain (T,) bool
          det_debug (dict)
          rain_conf (T,) float in [0,1]
          noise_conf (T,) float in [0,1]
        """
        self._validate_rain_cfg()
        cfg = self._cfg()
        eps = float(getattr(cfg, "eps", 1e-9))

        F, T = P.shape
        times_s = self._time_axis_s(T)

        op_lo, op_hi = cfg.operating_band
        op_mask = self._band_mask(freqs, (op_lo, op_hi))

        mode_bands: Sequence[Tuple[float, float]] = list(getattr(cfg, "mode_bands"))
        primary_idx = int(getattr(cfg, "primary_mode_idx"))
        primary_band = mode_bands[primary_idx]

        # Optional tunables
        peak_top_p = int(getattr(cfg, "peak_top_p", 6))
        peak_primary_q = int(getattr(cfg, "peak_primary_q", 3))
        peaks_in_mode_min = int(getattr(cfg, "peaks_in_mode_min", 2))
        peak_prominence_db = float(getattr(cfg, "peak_prominence_db", 3.0))
        peak_min_db_above_floor = float(getattr(cfg, "peak_min_db_above_floor", 6.0))

        z_win_frames = getattr(cfg, "z_win_frames", None)
        z_win_sec = float(getattr(cfg, "z_win_sec", 1.0))
        if z_win_frames is None:
            fs = float(getattr(cfg, "fs", 11162))
            hop = float(getattr(cfg, "hop", 128))
            frames_per_sec = fs / hop
            W = max(2, int(0.5 * z_win_sec * frames_per_sec))
        else:
            W = max(2, int(z_win_frames))

        zscore_mode = str(getattr(cfg, "zscore_mode", "mad")).lower()

        # Score thresholds / weights
        primary_flux_z_thresh = float(getattr(cfg, "primary_flux_z_thresh", 2.5))
        modes_flux_z_thresh = float(getattr(cfg, "modes_flux_z_thresh", 2.0))
        peak_ratio_thresh = float(getattr(cfg, "peak_ratio_thresh", 0.5))

        w_primary = float(getattr(cfg, "prob_w_primary", 0.05))
        w_modes = float(getattr(cfg, "prob_w_modes", 0.90))
        w_peaks = float(getattr(cfg, "prob_w_peaks", 0.05))
        w_sum = max(1e-9, (w_primary + w_modes + w_peaks))
        w_primary, w_modes, w_peaks = w_primary / w_sum, w_modes / w_sum, w_peaks / w_sum

        # dB spectra for flux/peaks
        X_db = self._db10(P, eps=eps)  # (F, T)

        # Loudness in operating band
        Ltot_db = np.full(T, np.nan, dtype=np.float64)
        for t in range(T):
            Ltot_db[t] = float(10.0 * np.log10(np.sum(P[op_mask, t]) + eps))
        dLtot_db = np.r_[0.0, np.diff(Ltot_db)]

        # Flux features
        prim_mask = self._band_mask(freqs, primary_band)

        # Masks for all modes except primary
        other_mode_masks = [
            self._band_mask(freqs, b)
            for i, b in enumerate(mode_bands)
            if i != primary_idx
        ]

        flux_primary = np.zeros(T, dtype=np.float64)
        flux_modes = np.zeros(T, dtype=np.float64)

        for t in range(T):
            # --- Primary mode flux ---
            f1 = self._compute_flux(X_db, prim_mask, t, t - 1)
            f2 = self._compute_flux(X_db, prim_mask, t, t - 2)
            primary_flux_t = max(f1, f2)
            flux_primary[t] = primary_flux_t

            # --- Other modes flux ---
            fm1 = 0.0
            fm2 = 0.0
            for mm in other_mode_masks:
                fm1 += self._compute_flux(X_db, mm, t, t - 1)
                fm2 += self._compute_flux(X_db, mm, t, t - 2)

            other_modes_flux_t = max(fm1, fm2)

            # z_modes will now represent TOTAL mode activity including primary
            flux_modes[t] = primary_flux_t + other_modes_flux_t

        # Robust z-scores
        if use_robust_z:
            st_p = self._robust_stats_symmetric(flux_primary, W=W)
            st_m = self._robust_stats_symmetric(flux_modes, W=W)
            if zscore_mode == "iqr":
                z_primary = self._robust_z_iqr(flux_primary, st_p["q25"], st_p["q75"], eps)
                z_modes = self._robust_z_iqr(flux_modes, st_m["q25"], st_m["q75"], eps)
            else:
                z_primary = self._robust_z_mad(flux_primary, st_p["med"], st_p["mad"], eps)
                z_modes = self._robust_z_mad(flux_modes, st_m["med"], st_m["mad"], eps)
        else:
            z_primary = (flux_primary - np.median(flux_primary)) / (np.std(flux_primary) + eps)
            z_modes = (flux_modes - np.median(flux_modes)) / (np.std(flux_modes) + eps)

        # Peak structure
        n_peaks_total = np.zeros(T, dtype=np.int32)
        n_peaks_mode = np.zeros(T, dtype=np.int32)
        n_peaks_primary = np.zeros(T, dtype=np.int32)
        peak_ratio_mode = np.zeros(T, dtype=np.float64)

        top_peaks_hz_all: List[List[float]] = []
        top_peaks_db_all: List[List[float]] = []
        top_peaks_in_mode_all: List[List[bool]] = []
        top_peaks_in_primary_all: List[List[bool]] = []

        def in_any_mode(f: float) -> bool:
            for (a, b) in mode_bands:
                if float(a) <= f <= float(b):
                    return True
            return False

        pa, pb = float(primary_band[0]), float(primary_band[1])

        for t in range(T):
            pk_hz, pk_db = self._find_top_peaks(
                X_db[:, t],
                freqs,
                op_mask,
                peak_top_p=peak_top_p,
                prominence_db=peak_prominence_db,
                min_db_above_floor=peak_min_db_above_floor,
            )

            n_peaks_total[t] = int(pk_hz.size)
            in_mode = [in_any_mode(float(f)) for f in pk_hz]
            in_primary = [(pa <= float(f) <= pb) for f in pk_hz]

            n_peaks_mode[t] = int(np.sum(in_mode))
            n_peaks_primary[t] = int(np.sum(in_primary))
            peak_ratio_mode[t] = float(n_peaks_mode[t]) / float(max(1, n_peaks_total[t]))

            top_peaks_hz_all.append([float(x) for x in pk_hz])
            top_peaks_db_all.append([float(x) for x in pk_db])
            top_peaks_in_mode_all.append([bool(x) for x in in_mode])
            top_peaks_in_primary_all.append([bool(x) for x in in_primary])

        # Primary flux must-have gate (new name primary_flux_thresh; legacy primary_rise_db_thresh)
        primary_flux_thresh = float(
            getattr(cfg, "primary_flux_thresh", getattr(cfg, "primary_rise_db_thresh", 6.0))
        )
        gate_primary_flux_ok = flux_primary >= primary_flux_thresh

        gate_primary_peak_ok = np.zeros(T, dtype=bool)
        for t in range(T):
            flags = top_peaks_in_primary_all[t]
            gate_primary_peak_ok[t] = bool(np.any(flags[: max(1, peak_primary_q)])) if flags else False

        gate_top_peaks_mode_ok = n_peaks_mode >= peaks_in_mode_min

        # Scoring
        s_primary = self._sigmoid(z_primary - primary_flux_z_thresh, k=1.0)
        s_modes = self._sigmoid(z_modes - modes_flux_z_thresh, k=1.0)
        s_peaks = self._sigmoid(peak_ratio_mode - peak_ratio_thresh, k=8.0)

        rain_score_raw = w_primary * s_primary + w_modes * s_modes + w_peaks * s_peaks
        rain_score_raw = np.clip(rain_score_raw, 0.0, 1.0)

        # Must-have gates clamp the score
        must_ok = gate_primary_flux_ok & gate_primary_peak_ok & gate_top_peaks_mode_ok
        rain_score_raw = np.where(must_ok, rain_score_raw, np.minimum(rain_score_raw, 0.49))

        # 3-state classification
        rain_hi = float(getattr(cfg, "rain_hi", 0.80))
        noise_hi = float(getattr(cfg, "noise_hi", 0.85))
        noise_update_thresh = float(getattr(cfg, "noise_update_thresh", 0.85))

        label = np.full(T, "uncertain", dtype=object)
        label[rain_score_raw >= rain_hi] = "rain"
        label[(1.0 - rain_score_raw) >= noise_hi] = "noise"

        # Persistence / hysteresis
        rain_hold = int(getattr(cfg, "rain_hold_frames", 0))
        noise_confirm = int(getattr(cfg, "noise_confirm_frames", 1))

        if rain_hold > 0:
            is_r = (label == "rain")
            for t in range(T):
                if is_r[t]:
                    t1 = min(T, t + 1 + rain_hold)
                    label[t + 1 : t1] = "rain"

        if noise_confirm > 1:
            is_n = (label == "noise")
            run = 0
            for t in range(T):
                run = (run + 1) if is_n[t] else 0
                if run < noise_confirm:
                    label[t] = "uncertain"

        # PSD update mask: only strong noise
        noise_score = 1.0 - rain_score_raw
        use_for_noise_psd = (label == "noise") & (noise_score >= noise_update_thresh)

        # Outputs
        is_rain = (label == "rain")
        rain_conf = rain_score_raw.copy()
        noise_conf = 1.0 - rain_conf

        # FrameFeatures list
        frame_features: List[FrameFeatures] = []
        for t in range(T):
            frame_features.append(
                FrameFeatures(
                    t=int(t),
                    time_s=float(times_s[t]),
                    Ltot_db=float(Ltot_db[t]),
                    dLtot_db=float(dLtot_db[t]),
                    flux_primary=float(flux_primary[t]),
                    flux_modes=float(flux_modes[t]),
                    z_primary=float(z_primary[t]),
                    z_modes=float(z_modes[t]),
                    n_peaks_total=int(n_peaks_total[t]),
                    n_peaks_mode=int(n_peaks_mode[t]),
                    n_peaks_primary=int(n_peaks_primary[t]),
                    peak_ratio_mode=float(peak_ratio_mode[t]),
                    top_peaks_hz=top_peaks_hz_all[t],
                    top_peaks_db=top_peaks_db_all[t],
                    top_peaks_in_mode=top_peaks_in_mode_all[t],
                    top_peaks_in_primary=top_peaks_in_primary_all[t],
                    gate_primary_flux_ok=bool(gate_primary_flux_ok[t]),
                    gate_primary_peak_ok=bool(gate_primary_peak_ok[t]),
                    gate_top_peaks_mode_ok=bool(gate_top_peaks_mode_ok[t]),
                    rain_score_raw=float(rain_score_raw[t]),
                    label=str(label[t]),
                    use_for_noise_psd=bool(use_for_noise_psd[t]),
                )
            )

        det_debug: Dict[str, Any] = {
            "times_s": times_s,
            "Ltot_db": Ltot_db,
            "dLtot_db": dLtot_db,
            "flux_primary": flux_primary,
            "flux_modes": flux_modes,
            "z_primary": z_primary,
            "z_modes": z_modes,
            "n_peaks_total": n_peaks_total,
            "n_peaks_mode": n_peaks_mode,
            "n_peaks_primary": n_peaks_primary,
            "peak_ratio_mode": peak_ratio_mode,
            "peak_ratio": peak_ratio_mode,  # alias for downstream plotting
            "rain_score_raw": rain_score_raw,
            "label": label,
            "rain_conf": rain_conf,  # alias for downstream plotting
            "noise_conf": noise_conf,  # alias for downstream plotting
            "use_for_noise_psd": use_for_noise_psd,
            "gate_primary_flux_ok": gate_primary_flux_ok,
            "gate_primary_peak_ok": gate_primary_peak_ok,
            "gate_top_peaks_mode_ok": gate_top_peaks_mode_ok,
            # legacy alias for older plots
            "gate_primary_rise_ok": gate_primary_flux_ok,
            "frame_features": frame_features,
            # metadata / thresholds for plotting
            "operating_band": (float(op_lo), float(op_hi)),
            "mode_bands": [(float(a), float(b)) for (a, b) in mode_bands],
            "primary_mode_idx": int(primary_idx),
            "primary_band": (float(pa), float(pb)),
            "z_win_half_frames": int(W),
            "zscore_mode": zscore_mode,
            "primary_flux_thresh": primary_flux_thresh,
            # legacy alias
            "primary_rise_db_thresh": primary_flux_thresh,
            "primary_flux_z_thresh": primary_flux_z_thresh,
            "modes_flux_z_thresh": modes_flux_z_thresh,
            "peak_ratio_thresh": peak_ratio_thresh,
            "peak_top_p": peak_top_p,
            "peak_primary_q": peak_primary_q,
            "peaks_in_mode_min": peaks_in_mode_min,
            "peak_prominence_db": peak_prominence_db,
            "peak_min_db_above_floor": peak_min_db_above_floor,
            "rain_hi": rain_hi,
            "noise_hi": noise_hi,
            "noise_update_thresh": noise_update_thresh,
            "prob_w_primary": w_primary,
            "prob_w_modes": w_modes,
            "prob_w_peaks": w_peaks,
        }

        return is_rain, det_debug, rain_conf, noise_conf


__all__ = ["FrameFeatures", "RainFrameClassifierMixin"]