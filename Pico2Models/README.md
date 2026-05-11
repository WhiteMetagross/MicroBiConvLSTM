# Pico 2 Model Artifacts:

This directory contains the Pico 2 facing deployment artifacts that were derived from the paper retraining sweep. The committed files were organised so that the board-ready TFLite Micro arrays could be inspected alongside the hardware result JSON files.

The directory was split into two layers.

- `paperRetrainSeed29/`, which stores the per-model and per-dataset TFLite Micro C and C++ arrays together with the corresponding INT8 deployment files and parity reports.
- `Results/`, which stores the consolidated Pico 2 hardware summaries used by the deployment report.

The corrected final Pico 2 interpretation should be taken from `Results/pico2Fp32Int8Results.json` and `Results/pico2Fp32Int8Results.md`. The older `microbiPico2Metrics.json` and `baselinePico2Metrics.json` files are still preserved because they remain the raw family-level result sources, but the final paper-facing conclusion was based on the merged `FP32` and `INT8` matrix.

The runtime project itself is kept in `embedded/pico2EdgeRuntime/`. The files stored here should be interpreted as deployment artifacts rather than as the runtime implementation.
