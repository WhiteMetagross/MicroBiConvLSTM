# ESP32 Native Deployment

This directory contains the native ESP-IDF deployment project used for classic ESP32 benchmarking.

The runtime is intentionally separated from the generated model artifacts. The following files are produced by the preparation scripts and are not tracked:

- `main/edge_model_data.h`
- `main/edge_model_data.cc`
- `main/edge_fixture.h`
- `main/edge_ops_config.h`

Use:

- `scripts/prepareEsp32Bundle.py`
- `scripts/runSingleEsp32Bundle.py`
- `scripts/runEsp32NativeDeploymentSweep.py`

before building the project with `idf.py`.
