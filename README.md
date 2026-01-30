# audio_processing_tools

A Python package for processing raw audio signals from Mark-3 devices, with a focus on rain detection and noise estimation. This package provides a flexible framework for batch processing audio files from local storage or remote S3 buckets, applying DSP algorithms, and analyzing results.

## Overview

This package is designed to handle the complete audio processing pipeline for Mark-3 device audio data:

- **Audio I/O**: Load and parse Mark-3 binary audio files (`.bin`) and WAV files from local filesystems or remote S3 storage
- **Batch Processing Framework**: Orchestrate multiple audio processors across large datasets with configurable batching
- **Rain Detection**: Advanced algorithms for detecting rain events in audio signals, including false positive/negative handling
- **Noise Estimation**: SNR calculation and noise floor estimation
- **Edge Processing**: DSP algorithms optimized for edge devices, including C library integration
- **Parameter Tuning**: Grid search and visualization tools for optimizing DSP parameters
- **Database Integration**: Query and store audio metadata using PostgreSQL


## Key Features

### Modular Processor Architecture
The framework uses a protocol-based design where processors implement a simple interface:
- Each processor receives standardized audio buffers (float32, mono, fixed sample rate)
- Processors return metrics (scalar values) and internal state (arrays, diagnostics)
- Results are automatically namespaced to avoid conflicts when running multiple processors

### Flexible Data Sources
Support for multiple input types:
- **LocalPath**: Recursively scan local directories for audio files
- **RemotePath**: Query database for audio keys and fetch from S3
- **CsvInput**: Load source file list from CSV and hydrate metadata from database
- **KeyList**: Process a provided list of source files

### Audio Format Support
- **Mark-3 Binary Format**: Custom binary format with magic bytes (`0xADFBCADE`)
- **WAV Files**: Standard WAV format with automatic resampling and channel conversion
- Automatic conversion to normalized float32 mono audio at a specified sample rate

### Batch Processing
- Process thousands of audio files efficiently with configurable batch sizes
- Memory-efficient design with automatic garbage collection between batches
- Progress tracking and error handling


## Quick Start

```python
from audio_processing_tools import process_audio_batches_v2
from audio_processing_tools.processors import RainProcessor, NoiseProcessor
from audio_processing_tools.postprocess.rain import postprocess_rain

# Define processors
rain_proc = RainProcessor(name="rain", fn=your_rain_detection_function)
noise_proc = NoiseProcessor(name="noise", fn=your_noise_estimation_function)

# Configure global parameters
params_global = {
    "sample_rate": 11162,
    "check_duration": 10.0,
    "rain_drop_min_thr": 3,
}

# Process audio files
results_df, states_df_by_proc = process_audio_batches_v2(
    processors=[rain_proc, noise_proc],
    params_global=params_global,
    InputType="LocalPath",
    test_vector_path="/path/to/audio/files",
    batch_size=1000,
)

# Post-process results
test_results_df, feature_df = postprocess_rain(
    results_df,
    states_df_by_proc["rain"],
    params_global,
)
```

## Package Structure

```
audio_processing_tools/
├── audio_processing_framework.py  # Batch orchestration framework
├── processors.py                   # Base processor classes (RainProcessor, NoiseProcessor)
├── audio_io.py                     # Audio loading and key discovery utilities
├── parse.py                        # Mark-3 binary format parser
├── fetch.py                        # S3/remote audio fetching
├── db_tools.py                     # Database utilities
├── edge/                           # Edge device processing
│   ├── dsp_rain_detection.py      # Rain detection DSP algorithms
│   ├── noise_processor.py         # Noise processing
│   └── parameter_tuning/          # Parameter optimization tools
└── postprocess/                    # Result formatting utilities
    ├── rain.py                     # Rain detection post-processing
    └── noise.py                    # Noise estimation post-processing
```

## Documentation

### AudioProcessor Protocol
All processors must implement:
- `name` property: Short identifier (e.g., "rain", "noise")
- `run(audio_data, params)` method: Returns `(results_dict, state_dict)`

### Input Data Format
- Audio buffers are 1-D float32 NumPy arrays
- Values normalized to [-1, 1] range
- Fixed sample rate and duration as specified in `params_global`

### Output Format
- **Results DataFrame**: One row per file with namespaced metrics (e.g., `rain__rain_drops`, `noise__snr_db`)
- **States DataFrames**: Per-processor DataFrames with internal state and diagnostics
