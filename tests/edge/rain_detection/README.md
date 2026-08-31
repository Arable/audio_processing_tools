# Rain detector regression tests

This fast suite uses deterministic synthetic audio and requires no database,
local audio cache, or DVC data. Run it from the repository root:

```bash
pytest tests/edge/rain_detection
```

The large field-data evaluations, tuning, and notebooks remain in
`rain_anomaly_analysis`. Those measure statistical performance; this suite
protects deterministic behavior and batch/streaming parity on every code
change. The CM7 parameters in `conftest.py` intentionally match the deployed
configuration used by that larger harness.
