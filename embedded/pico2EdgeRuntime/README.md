# Pico 2 Edge Runtime

This sketch is a reusable TFLite Micro runtime for Raspberry Pi Pico 2.

The tracked sources provide the runtime and the vendored support code. The actual model blob and verification fixture are generated on demand by:

- `scripts/generatePicoFixture.py`
- `scripts/runPico2DeploymentSweep.py`

Generated files that are intentionally not tracked:

- `edge_model.h`
- `edge_model.cpp`
- `edge_fixture.h`

Run the deployment scripts before compiling the sketch.
