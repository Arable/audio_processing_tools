from __future__ import annotations


from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

@dataclass
class FrameFeatures:
    """
    Per-frame diagnostic features for rain / noise classification.

    Intended for:
      - debugging
      - visualization
      - detector parameter tuning

    NOT part of the public framework API.
    """

    # indexing / time
    t: int
    time_s: float

    # core band energies (dB or log-power domain)
    LA: float
    LB: float
    dAB: float

    # inter-frame temporal features
    drise1: float
    drise2: float
    drise: float

    # ---------- mode-frequency evidence (optional) ----------
    mode_L: Optional[List[float]] = None
    mode_rise_db: Optional[List[float]] = None
    mode_peak_db: Optional[List[float]] = None
    mode_score: Optional[List[float]] = None

    mode_score_joint: Optional[float] = None
    primary_ok: Optional[bool] = None

    # ---------- decisions ----------
    rain_score_raw: Optional[float] = None
    label: Optional[str] = None        # "RAIN" | "NOISE" | "UNCERTAIN"
    is_rain_for_psd: Optional[bool] = None

class RainFrameClassifierMixin:
    # ----------------------- band-energy helper -----------------------

    REQUIRED_CFG_FIELDS = [
        "rain_band", "adj_band", "eps",
        "N_db", "M_db",
        "dab_conf_low", "dab_conf_high",
        "drise_conf_low", "drise_conf_high",
        "rain_conf_w_spec", "rain_conf_w_rise",
        # NOTE: we will NOT use rain_conf_smooth_alpha for decisions anymore,
        # but we keep compatibility if you still want a viz-only EMA.
        "rain_conf_smooth_alpha",
        # --- new fields ---
        "mode_bands",
        "mode_weights",
        "mode_rise_db_low", "mode_rise_db_high",
        "mode_peak_db_low", "mode_peak_db_high",
        "mode_w_rise", "mode_w_peak",
        "primary_mode_idx", "primary_min_score",
        "rain_hi", "noise_hi", "noise_update_thresh",
        "rain_hold_frames", "noise_confirm_frames",
    ]

    def _validate_rain_cfg(self):
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise AttributeError("self.cfg is missing in processor")

        missing = [f for f in self.REQUIRED_CFG_FIELDS if not hasattr(cfg, f)]
        if missing:
            raise AttributeError(f"SpectralFeatureConfig missing required fields: {missing}")

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

    # ----------------------- per-frame scoring -----------------------

    def _detect_rain_frame_v2(
        self,
        P_frame: np.ndarray,
        freqs: np.ndarray,
        prev_LA1: Optional[float],
        prev_LA2: Optional[float],
        prev_mode_db1: Optional[np.ndarray],
        prev_mode_db2: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Per-frame detector that combines:
          - legacy rain_band vs adj_band (dAB, drise)
          - mode evidence: per-mode rise + per-mode peakiness
        """
        cfg = self.cfg
        eps = cfg.eps

        # --- legacy features ---
        LA = self._band_db(P_frame, freqs, cfg.rain_band, eps)
        LB = self._band_db(P_frame, freqs, cfg.adj_band, eps)

        if np.isneginf(LA) or np.isneginf(LB):
            # return safe defaults
            n_modes = len(cfg.mode_bands)
            return {
                "LA": LA, "LB": LB, "dAB": 0.0,
                "drise1": 0.0, "drise2": 0.0, "drise": 0.0,
                "mode_db": np.full(n_modes, -np.inf),
                "mode_rise": np.zeros(n_modes),
                "mode_peak_db": np.full(n_modes, -np.inf),
                "mode_score_each": np.zeros(n_modes),
                "mode_score_joint": 0.0,
                "primary_ok": False,
                "rain_score_raw": 0.0,
            }

        dAB = LA - LB
        drise1 = (LA - prev_LA1) if prev_LA1 is not None else 0.0
        drise2 = (LA - prev_LA2) if prev_LA2 is not None else 0.0
        drise_max = max(drise1, drise2)

        # --- legacy soft scores (raw, no EMA here) ---
        s_spec = self._soft_score(dAB, cfg.dab_conf_low, cfg.dab_conf_high)
        s_rise = self._soft_score(drise_max, cfg.drise_conf_low, cfg.drise_conf_high)
        legacy_score = float(cfg.rain_conf_w_spec * s_spec + cfg.rain_conf_w_rise * s_rise)

        # --- mode evidence ---
        mode_bands: List[Tuple[float, float]] = list(cfg.mode_bands)
        n_modes = len(mode_bands)

        mode_db = np.zeros(n_modes, dtype=float)
        mode_peak_db = np.zeros(n_modes, dtype=float)
        for i, b in enumerate(mode_bands):
            mode_db[i] = self._band_db(P_frame, freqs, b, eps)
            mode_peak_db[i] = self._peakiness_db(P_frame, freqs, b, eps)

        # inter-frame rises per mode: use max of t-1/t-2 like legacy
        mode_rise = np.zeros(n_modes, dtype=float)
        if prev_mode_db1 is not None:
            mode_rise = np.maximum(mode_rise, mode_db - prev_mode_db1)
        if prev_mode_db2 is not None:
            mode_rise = np.maximum(mode_rise, mode_db - prev_mode_db2)

        # score each mode, but *ignore* modes that don’t meet minimal criteria
        w_mode = np.asarray(cfg.mode_weights, dtype=float)
        if w_mode.size != n_modes:
            # be forgiving
            w_mode = np.ones(n_modes, dtype=float)

        mode_score_each = np.zeros(n_modes, dtype=float)
        mode_active = np.zeros(n_modes, dtype=bool)

        for i in range(n_modes):
            sr = self._soft_score(mode_rise[i], cfg.mode_rise_db_low, cfg.mode_rise_db_high)
            sp = self._soft_score(mode_peak_db[i], cfg.mode_peak_db_low, cfg.mode_peak_db_high)

            score_i = float(cfg.mode_w_rise * sr + cfg.mode_w_peak * sp)

            # "don’t use modes where criteria is not met"
            # -> require at least some evidence in rise OR peak
            active_i = (sr > 0.0) or (sp > 0.0)

            mode_active[i] = active_i
            mode_score_each[i] = score_i if active_i else 0.0

        # primary mode must be present
        pidx = int(cfg.primary_mode_idx)
        primary_ok = (0 <= pidx < n_modes) and (mode_score_each[pidx] >= float(cfg.primary_min_score))

        # joint mode score (weighted average over *active* modes)
        w_eff = w_mode * mode_active.astype(float)
        if np.sum(w_eff) > 0:
            mode_score_joint = float(np.sum(w_eff * mode_score_each) / (np.sum(w_eff) + eps))
        else:
            mode_score_joint = 0.0

        # Combine legacy + mode evidence:
        # - If primary mode missing => cap score (prevents “rain” from weak/false cues)
        if not primary_ok:
            mode_score_joint = min(mode_score_joint, 0.25)

        # Final raw rain score: keep mode evidence as an additional term
        # (Tune weights later; start conservative so you don’t wrongly exclude noise frames.)
        rain_score_raw = float(np.clip(0.6 * legacy_score + 0.4 * mode_score_joint, 0.0, 1.0))

        return {
            "LA": LA, "LB": LB, "dAB": dAB,
            "drise1": drise1, "drise2": drise2, "drise": drise_max,
            "s_spec": s_spec, "s_rise": s_rise, "legacy_score": legacy_score,
            "mode_db": mode_db,
            "mode_rise": mode_rise,
            "mode_peak_db": mode_peak_db,
            "mode_score_each": mode_score_each,
            "mode_score_joint": mode_score_joint,
            "mode_active": mode_active,
            "primary_ok": primary_ok,
            "rain_score_raw": rain_score_raw,
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

        labels = np.empty(T, dtype=object)
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
    ):
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

        # legacy debug arrays
        LA = np.full(T, np.nan)
        LB = np.full(T, np.nan)
        dAB = np.full(T, np.nan)
        drise1 = np.full(T, np.nan)
        drise2 = np.full(T, np.nan)
        drise = np.full(T, np.nan)

        # mode arrays
        n_modes = len(cfg.mode_bands)
        mode_db_all = np.full((n_modes, T), np.nan)
        mode_rise_all = np.full((n_modes, T), np.nan)
        mode_peak_all = np.full((n_modes, T), np.nan)
        mode_score_each_all = np.full((n_modes, T), np.nan)
        mode_joint = np.full(T, np.nan)
        primary_ok = np.zeros(T, dtype=bool)

        legacy_score = np.full(T, np.nan)
        rain_score_raw = np.full(T, np.nan)

        prev_LA1: Optional[float] = None
        prev_LA2: Optional[float] = None
        prev_mode1: Optional[np.ndarray] = None
        prev_mode2: Optional[np.ndarray] = None
        
        debug_frames: List[FrameFeatures] = []
        for t in range(T):
            det = self._detect_rain_frame_v2(
                P[:, t],
                freqs,
                prev_LA1,
                prev_LA2,
                prev_mode1,
                prev_mode2,
            )

            LA[t] = det["LA"]; LB[t] = det["LB"]; dAB[t] = det["dAB"]
            drise1[t] = det["drise1"]; drise2[t] = det["drise2"]; drise[t] = det["drise"]

            legacy_score[t] = det.get("legacy_score", np.nan)
            rain_score_raw[t] = det["rain_score_raw"]

            mode_db_all[:, t] = det["mode_db"]
            mode_rise_all[:, t] = det["mode_rise"]
            mode_peak_all[:, t] = det["mode_peak_db"]
            mode_score_each_all[:, t] = det["mode_score_each"]
            mode_joint[t] = det["mode_score_joint"]
            primary_ok[t] = bool(det["primary_ok"])

            # shift history
            prev_LA2, prev_LA1 = prev_LA1, LA[t]
            prev_mode2, prev_mode1 = prev_mode1, det["mode_db"].copy()

        # 3-state labeling + PSD gating
        label, use_for_noise_psd = self._classify_3state_over_time(rain_score_raw)

        # back-compat
        is_rain = (label == "rain")
        rain_conf = np.clip(rain_score_raw, 0.0, 1.0)
        noise_conf = 1.0 - rain_conf

        det_debug = {
            "LA": LA, "LB": LB, "dAB": dAB,
            "drise1": drise1, "drise2": drise2, "drise": drise,

            "legacy_score": legacy_score,
            "rain_score_raw": rain_score_raw,

            "mode_db": mode_db_all,
            "mode_rise": mode_rise_all,
            "mode_peak_db": mode_peak_all,
            "mode_score_each": mode_score_each_all,
            "mode_score_joint": mode_joint,
            "primary_ok": primary_ok,

            "label": label,
            "use_for_noise_psd": use_for_noise_psd,
        }

        return is_rain, det_debug, rain_conf, noise_conf