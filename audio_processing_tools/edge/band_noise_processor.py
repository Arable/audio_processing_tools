# audio_processing_tools/edge/band_noise_processor.py

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import numpy as np

from .band_noise_estimator import (
    BandNoiseEstimator,
    BandNoiseEstimatorConfig,
    NoiseFrameDetectorConfig,
)

class BandNoiseEstimatorProcessor:
    """
    Offline / batch adapter for BandNoiseEstimator.

    Fits into audio_processing_framework:
      - input: full ndarray (mono)
      - output: (results_dict, state_dict)
    """
    _printed_params_global: bool = False

    def __init__(self, name: str = "band_noise", mode: str = "fft"):
        # Keep mode for backward compatibility with harness configs,
        # but it is not used by the new estimator.
        self.name = name
        self.mode = (mode or "fft").lower().strip()

        # Print params once per processor instance (not once per file)
        self._printed_params_once = False

        # Optional runtime verbosity (can also be overridden via params["verbose"]).
        self.verbose: bool = False

    # ------------------------
    # Config builder
    # ------------------------
    def _build_config(self, params: Dict[str, Any]) -> BandNoiseEstimatorConfig:
        """
        Build BandNoiseEstimatorConfig from params dict.
        We accept both 'fs' and framework-style 'sample_rate'.

        Also supports nested detector overrides via dotted keys:
          det.M_db, det.N_db, det.D_db, det.k_subframes, det.primary_hz, det.rain_bands_hz
        """
        cfg = BandNoiseEstimatorConfig()
        # Backward compat: older BandNoiseEstimatorConfig may not have det yet.
        if not hasattr(cfg, "det") or cfg.det is None:
            cfg.det = NoiseFrameDetectorConfig()

        # 1) Apply direct overrides for known cfg attributes
        for k, v in params.items():
            if k.startswith("det."):
                # support nested detector overrides like det.M_db, det.primary_hz, etc.
                subk = k.split(".", 1)[1]
                if hasattr(cfg.det, subk):
                    setattr(cfg.det, subk, v)
                continue

            if hasattr(cfg, k):
                setattr(cfg, k, v)

        # 2) Prefer framework global sample_rate if provided
        if "sample_rate" in params:
            cfg.fs = int(params["sample_rate"])
        elif "fs" in params:
            cfg.fs = int(params["fs"])

        # keep detector consistent with estimator
        cfg.det.fs = int(cfg.fs)
        cfg.det.n_fft = int(cfg.frame_len)

        cfg.validate()
        return cfg

    # ------------------------
    # Framework entry point
    # ------------------------
    def run(
        self,
        audio_data: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        x = np.asarray(audio_data, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("audio_data must be non-empty mono ndarray")

        cfg = self._build_config(params)

        # Optional: print merged params once (per processor instance)
        if bool(params.get("print_params_once", False)) and (not self._printed_params_once) and (not self.__class__._printed_params_global):
            merged_view = dict(params)
            merged_view["mode"] = self.mode  # keep visible for old harnesses
            print("\n=====================================")
            print(f"MERGED PARAMS (global overridden by proc) — {self.name}")
            print("=====================================")
            for k in sorted(merged_view.keys()):
                print(f"{k}: {merged_view[k]}")
            print("=====================================\n")
            self._printed_params_once = True
            self.__class__._printed_params_global = True

        # NOTE: run() is called once per input vector (file/test-vector)
        # so we must reset estimator state for each new stream.
        est = BandNoiseEstimator(cfg)
        if hasattr(est, "reset"):
            est.reset()
        elif hasattr(est, "reset_for_new_stream"):
            est.reset_for_new_stream()

        fs = int(cfg.fs)
        N = int(cfg.frame_len)

        # framework often uses hop = frame_len; allow override
        hop = int(params.get("hop", N))
        if hop <= 0:
            raise ValueError("hop must be positive")

        # Subframe count must match estimator logic: S = 1 + (N - sub_len)//subhop
        sub_len = int(cfg.subframe_len)
        subhop = int(cfg.subhop)
        if sub_len <= 0 or subhop <= 0:
            raise ValueError("subframe_len and subhop must be positive")
        if sub_len > N:
            raise ValueError("subframe_len cannot exceed frame_len")
        if (N - sub_len) % subhop != 0:
            raise ValueError(
                f"Subframes do not tile frame cleanly: (frame_len - subframe_len) % subhop must be 0; got "
                f"(N={N} - sub_len={sub_len}) % subhop={subhop} = {(N - sub_len) % subhop}"
            )
        S = 1 + (N - sub_len) // subhop

        # Optional runtime verbosity
        verbose = bool(params.get("verbose", self.verbose))

        # Determine full frames only (no padding). This matches typical embedded framing.
        n_frames = 1 + (len(x) - N) // hop if len(x) >= N else 0
        times_s = (np.arange(n_frames, dtype=np.float64) * hop) / fs

        if n_frames == 0:
            results = {
                "processor": self.name,
                "mode": self.mode,
                "n_frames": 0,
                "M_clean_med": np.nan,
                "noise_E_med": np.nan,
                "gain_med": np.nan,
                "fft_rain_frac": np.nan,
            }
            state: Dict[str, Any] = {
                "processor": self.name,
                "mode": self.mode,
                "times_s": times_s,
                "M_band": np.zeros(0, dtype=np.float64),
                "E_band": np.zeros(0, dtype=np.float64),
                "N_E": np.zeros(0, dtype=np.float64),
                "N_E_raw": np.zeros(0, dtype=np.float64),
                "subE": np.zeros((0, S), dtype=np.float64),
                "N_sub": np.zeros((0, S), dtype=np.float64),
                "rain_submask": np.zeros((0, S), dtype=bool),
                "fft_rain_frame": np.zeros(0, dtype=bool),
                "G_mag": np.zeros(0, dtype=np.float64),
                "M_clean": np.zeros(0, dtype=np.float64),
                "config": cfg,
            }
            if bool(params.get("include_audio_in_state", False)):
                state["x_in"] = x.copy()
            return results, state

        # Allocate outputs (new API)
        M_band = np.zeros(n_frames, dtype=np.float64)
        E_band = np.zeros(n_frames, dtype=np.float64)
        subE = np.zeros((n_frames, S), dtype=np.float64)
        N_E = np.zeros(n_frames, dtype=np.float64)
        N_E_raw = np.zeros(n_frames, dtype=np.float64)
        G_mag = np.zeros(n_frames, dtype=np.float64)
        M_clean = np.zeros(n_frames, dtype=np.float64)

        fft_rain_frame = np.zeros(n_frames, dtype=bool)
        rain_submask = np.zeros((n_frames, S), dtype=bool)
        N_sub = np.zeros((n_frames, S), dtype=np.float64)

        # Streaming loop
        for i in range(n_frames):
            frame = x[i * hop : i * hop + N]

            out = est.process_frame(frame)

            M_band[i] = out.M_band
            E_band[i] = out.E_band
            N_E[i] = out.N_E
            N_E_raw[i] = out.N_E_raw
            G_mag[i] = out.G_mag
            M_clean[i] = out.M_clean

            # Defensive shape checks (helps catch config / estimator mismatches)
            if out.subE.shape[0] != S:
                raise ValueError(f"Estimator returned subE length {out.subE.shape[0]} but expected S={S}")
            if out.rain_submask.shape[0] != S:
                raise ValueError(f"Estimator returned rain_submask length {out.rain_submask.shape[0]} but expected S={S}")
            if out.N_sub.shape[0] != S:
                raise ValueError(f"Estimator returned N_sub length {out.N_sub.shape[0]} but expected S={S}")

            subE[i, :] = out.subE
            fft_rain_frame[i] = bool(out.fft_rain_frame)
            rain_submask[i, :] = out.rain_submask
            N_sub[i, :] = out.N_sub

            if verbose and i < int(params.get("debug_first_n", 0)):
                # count_valid may be a scalar (stream buffer) or an array (legacy lane buffers)
                cv = getattr(est, "count_valid", None)
                try:
                    cv_disp = cv.copy() if hasattr(cv, "copy") else cv
                except Exception:
                    cv_disp = cv
                #print("band_noise debug:", {"i": i, "count_valid": cv_disp, "N_E_raw": float(out.N_E_raw)})

        # Summary (used by results_df)
        results = {
            "processor": self.name,
            "mode": self.mode,  # kept for backward compatibility
            "n_frames": int(n_frames),
            "M_clean_med": float(np.median(M_clean)) if n_frames else np.nan,
            "noise_E_med": float(np.median(N_E)) if n_frames else np.nan,
            "gain_med": float(np.median(G_mag)) if n_frames else np.nan,
            "fft_rain_frac": float(np.mean(fft_rain_frame)) if n_frames else np.nan,
        }

        # Full debug state (used by tuning / plots)
        state: Dict[str, Any] = {
            "processor": self.name,
            "mode": self.mode,
            "times_s": times_s,

            "M_band": M_band,
            "E_band": E_band,
            "N_E": N_E,
            "N_E_raw": N_E_raw,
            "subE": subE,
            
            "N_sub": N_sub,
            "rain_submask": rain_submask,
            "fft_rain_frame": fft_rain_frame,

            "G_mag": G_mag,
            "M_clean": M_clean,

            "config": cfg,
        }

        # Optional: include the (possibly truncated) input audio in state for plotting
        if bool(params.get("include_audio_in_state", False)):
            state["x_in"] = x.copy()

        return results, state