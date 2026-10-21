# Rain detector regression tests

This fast suite uses deterministic synthetic audio and requires no database,
local audio cache, or DVC data. Run it from the repository root:

```bash
pytest tests/edge/rain_detection
```

The large field-data evaluations, tuning, and notebooks remain in
`rain_anomaly_analysis`. Those measure statistical performance; this suite
protects deterministic behavior and batch/streaming parity on every code
change. The CM7 parameters in `conftest.py` match NCL release 1
(`trial_params3`/`manual-v3`), the set the larger harness's golden baselines
still use; `edge` `master`/2.2.203.14 hardcode the same set in
`noise_cl_config()`. NCL release 2 (`td_gate_threshold=3.7`,
`new_rain_mode3_flux_min=3.5`, `clip_rain_min_frames=2`) is hardcoded only in
NCL build 2.2.204.14 (= `edge` PR #30), reportedly on four devices as of
2026-10-12, so most NCL pre-release devices probably still run release 1. When release 2 goes live, update `conftest.py` together with the code
defaults and the golden baselines, not on its own. `conftest.py` uses
`q = 0.30`; firmware uses 0.25. See `docs/rain_detection_ncl.md`.
