# ESP32 Deployment Results Report:

## Scope:

This report summarizes the native `ESP-IDF` deployment study for `MicroBiConvLSTM` and the baseline families `TinyHAR`, `TinierHAR`, and `DeepConvLSTM` on the connected classic `ESP32-D0WD-V3` board with `4 MB` flash and no PSRAM.

Three result layers are preserved in the repository.

- `ESP32Models/Results/esp32PaperResults.json`.
- `ESP32Models/Results/esp32Fp32Int8Results.json`.
- `ESP32Models/Results/baselineInt8QuantFix20260511.json`.

The second file should now be treated as the canonical merged matrix, because it includes the corrected `FP32` study and the repaired quantized baseline rows.

## Main Findings:

`MicroBiConvLSTM` remained the strongest ESP32 family. The native `FP32` path preserved essentially exact PyTorch parity on `7/8` datasets, and the quantized path retained full `8/8` coverage on the connected board.

The baseline families were more conditional. The earlier severe quantized collapses were traced to activation-range loss in selected exports. For the affected rows, a repaired quantized recipe with `INT16` activations and `INT8` weights was evaluated and promoted where the resulting deployment artifact materially improved parity.

The repaired quantized outcomes were mixed.

- `TinyHAR / motionsense` improved to `88.106%`.
- `TinyHAR / daphnet` improved from `0%` to `69.743%`.
- `TinierHAR / motionsense` improved to `91.593%`.
- `TinierHAR / daphnet` improved from `0%` to `64.089%`.
- `TinyHAR / ucihar` recovered strongly on desktop parity, but the repaired quantized graph no longer fit the classic ESP32 internal-SRAM envelope and therefore became a real board failure.

## Coverage Summary:

| Model | Quantized Runs | Quantized Fails | FP32 Runs | FP32 Fails | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `MicroBiConvLSTM` | 8 | 0 | 7 | 1 | Best overall ESP32 family. |
| `TinyHAR` | 5 | 3 | 3 | 5 | Partly deployable. Quantized repairs helped, but one repaired row became SRAM-limited. |
| `TinierHAR` | 6 | 2 | 3 | 5 | Quantized repairs materially improved selected rows. |
| `DeepConvLSTM` | 0 | 8 | 0 | 8 | Not deployable on the tested no-PSRAM board. |

## Repaired Quantized Baseline Rows:

The repaired quantized baseline rows should be interpreted carefully. They are not pure full-`INT8` exports. They are the promoted canonical quantized bundles chosen by the exporter after comparing the full-`INT8` and mixed-quantized parity outcomes.

| Model | Dataset | Quant Recipe | Desktop Parity (%) | ESP32 Parity (%) | Status | Arena Used (B) | Latency (ms) | Comment |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `TinyHAR` | `ucihar` | `INT16 activations + INT8 weights` | 88.585 | N/A | Fails. | N/A | N/A | The repaired graph exceeded the classic ESP32 internal-SRAM envelope. |
| `TinyHAR` | `motionsense` | `INT16 activations + INT8 weights` | 88.107 | 88.106 | Runs. | 148108 | 735.682 | A real parity recovery was obtained. |
| `TinyHAR` | `daphnet` | `INT16 activations + INT8 weights` | 69.732 | 69.743 | Runs. | 100684 | 518.130 | The collapse was repaired, but fidelity remained limited. |
| `TinierHAR` | `motionsense` | `INT16 activations + INT8 weights` | 91.595 | 91.593 | Runs. | 173996 | 231.314 | This became a strong deployable quantized row. |
| `TinierHAR` | `daphnet` | `INT16 activations + INT8 weights` | 64.089 | 64.089 | Runs. | 97644 | 154.479 | The degenerate result was repaired, but the row remained weak. |

## FP32 Context:

The repaired quantized baseline overlay does not change the broader `FP32` conclusion.

- `MicroBiConvLSTM` remained a high-fidelity `FP32` ESP32 family with `7/8` successful datasets.
- `TinyHAR` `FP32` remained successful on `wisdm`, `unimib`, and `daphnet`.
- `TinierHAR` `FP32` remained successful on `wisdm`, `unimib`, and `daphnet`.
- `DeepConvLSTM` remained outside the practical memory envelope of this target.

## Saved Artifacts:

- `ESP32Models/Results/esp32Fp32Int8Results.json`.
- `ESP32Models/Results/esp32Fp32Int8Results.md`.
- `ESP32Models/Results/baselineInt8QuantFix20260511.json`.
- `ESP32Models/paperRetrainSeed29/`.

## Conclusion:

The ESP32 results now tell a cleaner story.

`MicroBiConvLSTM` remains the strongest practical deployment family on the connected classic ESP32 target. The repaired baseline quantized study shows that several catastrophic parity collapses were export-path artifacts rather than unavoidable hardware behavior. At the same time, the repaired rows also show the platform limit clearly. On the no-PSRAM `ESP32-D0WD-V3`, improved quantized fidelity sometimes comes with a real SRAM cost, and not every repaired baseline row remains fully deployable.
