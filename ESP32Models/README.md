# ESP32 Model Artifacts:

This directory contains the ESP32 facing deployment artifacts that were derived from the paper retraining sweep and the validated native ESP-IDF runs. The connected hardware target for the reported study was the classic `ESP32-D0WD-V3` board without PSRAM.

The quantized baseline study now includes a repaired subset of bundles in which `INT16` activations and `INT8` weights were used to recover parity on rows that had previously collapsed under a pure full-`INT8` export. These repaired bundles replaced the earlier canonical quantized artifacts only where the recovery was clearly measured and recorded in the corresponding `parity_report.json` file.

The directory was arranged in three parts.

- `paperRetrainSeed29/`, which stores the canonical quantized deployment models and parity reports grouped by family and dataset.
- `Results/`, which stores the consolidated ESP32 deployment JSON summaries used by the report, including the corrected combined matrix in `esp32Fp32Int8Results.json`.
- `ValidatedRuns/`, which stores the selected native run logs and result files copied from the final validated artifact directories.

The runtime implementation remains in `embedded/esp32NativeDeployment/`. The files stored here should be interpreted as board-facing deployment artifacts and validation records.
