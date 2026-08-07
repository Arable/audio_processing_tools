# edge/noise_tracker.py
from __future__ import annotations

from typing import Optional

import numpy as np


class CausalNoiseTracker:
    """
    Frame-by-frame causal noise PSD tracker.

    Implements the stochastic low-quantile tracking algorithm used inside
    SpectralNoiseProcessor, but as a standalone stateful object that can be
    embedded in any frame-level processor or called independently.

    The tracker maintains a per-bin running low-quantile estimate of the noise
    PSD. Rain frames are excluded from baseline updates (outside warm-up) so
    that the noise floor tracks only the ambient noise, not rain events.

    Usage
    -----
    tracker = CausalNoiseTracker(n_bins=K, q=0.25, fs=11162, hop=128)
    tracker.reset()
    for t, P_band in enumerate(frames):
        N_band = tracker.update(P_band, is_rain=prev_frame_was_rain)

    Notes
    -----
    n_bins should equal the number of frequency bins inside the operating band,
    not the full FFT length.  The caller is responsible for slicing the band
    before passing to update() and expanding back to the full grid afterwards.
    """

    def __init__(
        self,
        n_bins: int,
        *,
        q: float = 0.25,
        fs: int = 11162,
        hop: int = 128,
        win_sec: float = 0.5,
        ema_up: float = 0.6,
        ema_down: float = 0.95,
        eps: float = 1e-9,
        noise_psd_max_ratio: float = 1.0,
        adaptive_q_enable: bool = False,
        adaptive_q_min: float = 0.10,
        adaptive_q_alpha: float = 0.95,
        dtype=None,
    ):
        if dtype is None:
            dtype = np.float32
        self._dtype = dtype
        self._n_bins = int(n_bins)
        self._q = float(q)
        self._ema_up = float(ema_up)
        self._ema_down = float(ema_down)
        self._eps = float(eps)
        self._maxr = float(
            np.clip(
                noise_psd_max_ratio if np.isfinite(noise_psd_max_ratio) else 1.0,
                0.0,
                1.0,
            )
        )
        self._adaptive_q_enable = bool(adaptive_q_enable)
        self._adaptive_q_min = float(np.clip(adaptive_q_min, 1e-4, q))
        self._adaptive_q_alpha = float(np.clip(adaptive_q_alpha, 0.0, 1.0))

        frames_per_sec = float(fs) / float(hop)
        W = max(10, int(win_sec * frames_per_sec))
        self._eta = float(np.clip(2.0 / max(W + 1, 2), 1e-4, 1.0))
        self._warmup_need = max(10, W // 2)
        self._step_floor = max(float(eps), 1e-9)

        self._tracker: np.ndarray = np.zeros(self._n_bins, dtype=dtype)
        self._tracker_scale: np.ndarray = np.full(self._n_bins, self._step_floor, dtype=dtype)
        self._prev_N: Optional[np.ndarray] = None
        self._warmup_count: int = 0
        self._rain_prev_ema: float = 0.0
        self._seeded: bool = False

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, first_frame: Optional[np.ndarray] = None) -> None:
        """
        Reset all tracker state.

        Parameters
        ----------
        first_frame : array, shape (n_bins,), optional
            Seed the tracker with the first frame's band power.  Matches the
            batch-path behaviour where the tracker initialises from P[:, 0].
            If None the tracker starts from zeros.
        """
        dtype = self._dtype
        if first_frame is not None:
            ff = np.asarray(first_frame, dtype=dtype).reshape(-1)
            self._tracker = np.maximum(ff.copy(), 0.0)
            self._tracker_scale = np.maximum(np.abs(ff), self._step_floor)
            self._seeded = True
        else:
            self._tracker = np.zeros(self._n_bins, dtype=dtype)
            self._tracker_scale = np.full(self._n_bins, self._step_floor, dtype=dtype)
            self._seeded = False
        self._prev_N = None
        self._warmup_count = 0
        self._rain_prev_ema = 0.0

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, P_band: np.ndarray, is_rain: bool = False) -> np.ndarray:
        """
        Update the tracker with one frame's band power and return the noise estimate.

        Parameters
        ----------
        P_band : array, shape (n_bins,)
            Instantaneous power in the tracked frequency band for this frame.
        is_rain : bool, optional
            Whether this frame is classified as rain.  Rain frames do not
            update the noise baseline outside the initial warm-up period.

        Returns
        -------
        N_band : array, shape (n_bins,)
            Noise PSD estimate for this frame, clamped to [0, maxr * P_band].
        """
        dtype = self._dtype
        P_band = np.asarray(P_band, dtype=dtype).reshape(-1)

        allow_update = (self._warmup_count < self._warmup_need) or (not bool(is_rain))

        if self._prev_N is None:
            if not self._seeded:
                # Auto-seed from first observed frame so reset() without a
                # first_frame argument still starts from a sensible baseline.
                self._tracker = np.maximum(P_band.copy(), 0.0)
                self._tracker_scale = np.maximum(np.abs(P_band), self._step_floor)
                self._seeded = True
            if allow_update:
                self._warmup_count += 1
            raw_q = self._tracker
        else:
            err = P_band - self._tracker
            self._tracker_scale = (
                self._ema_down * self._tracker_scale
                + (1.0 - self._ema_down) * np.abs(err)
            )
            step = self._eta * np.maximum(self._tracker_scale, self._step_floor)

            if self._adaptive_q_enable:
                q_eff = self._q - (self._q - self._adaptive_q_min) * self._rain_prev_ema
                q_eff = float(np.clip(q_eff, self._adaptive_q_min, self._q))
            else:
                q_eff = self._q

            delta = np.where(
                P_band >= self._tracker,
                q_eff * step,
                -(1.0 - q_eff) * step,
            )
            candidate = np.maximum(self._tracker + delta, 0.0)

            if allow_update:
                self._tracker = candidate
                self._warmup_count += 1

            raw_q = self._tracker

        # EMA smoothing of the raw quantile estimate
        if self._prev_N is None:
            N_band = raw_q.copy()
        else:
            up = raw_q > self._prev_N
            lam = np.where(up, self._ema_up, self._ema_down)
            N_band = lam * self._prev_N + (1.0 - lam) * raw_q

        N_band = np.minimum(N_band, self._maxr * P_band)
        N_band = np.maximum(N_band, 0.0).astype(dtype, copy=False)

        if self._adaptive_q_enable:
            self._rain_prev_ema = (
                self._adaptive_q_alpha * self._rain_prev_ema
                + (1.0 - self._adaptive_q_alpha) * float(bool(is_rain))
            )

        self._prev_N = N_band
        return N_band

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_estimate(self) -> Optional[np.ndarray]:
        """Last noise PSD estimate, or None before the first update() call."""
        return self._prev_N

    @property
    def warmup_complete(self) -> bool:
        """True once the tracker has processed at least warmup_need frames."""
        return self._warmup_count >= self._warmup_need
