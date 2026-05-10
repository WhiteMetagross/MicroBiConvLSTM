# ESP32 Model Artifacts:

This directory contains the ESP32 facing deployment artifacts that were derived from the paper retraining sweep and the validated native ESP-IDF runs. The connected hardware target for the reported study was the classic `ESP32-D0WD-V3` board without PSRAM.

The directory was arranged in three parts.

- `paperRetrainSeed29/`, which stores the INT8 deployment models and parity reports grouped by family and dataset.
- `Results/`, which stores the consolidated ESP32 deployment JSON summary used by the report.
- `ValidatedRuns/`, which stores the selected native run logs and result files copied from the final validated artifact directories.

The runtime implementation remains in `embedded/esp32NativeDeployment/`. The files stored here should be interpreted as board-facing deployment artifacts and validation records.
