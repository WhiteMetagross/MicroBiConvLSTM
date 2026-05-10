# Converted Paper Models:

This directory contains the generic exported model bundles obtained from the paper retraining sweep with seed `29`. The files here are hardware-agnostic conversion outputs and should be treated as the canonical export layer for reproducibility.

Each model and dataset bundle contains the following elements where available.

- The original checkpoint used for export.
- The ONNX graph.
- The FP32 TensorFlow Lite model.
- The INT8 TensorFlow Lite model.
- The TFLite Micro C and C++ array representation.
- The parity report generated during export validation.

Board-specific deployment folders are stored separately in `Pico2Models/` and `ESP32Models/`.
