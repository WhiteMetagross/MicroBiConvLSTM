# ESP32 Deployment Results Report:

## Scope:

This report consolidates the native ESP-IDF deployment study carried out for MicroBiConvLSTM and the three baseline families, namely TinyHAR, TinierHAR, and DeepConvLSTM. The target board was the connected classic `ESP32-D0WD-V3` device with `4 MB` flash and no PSRAM. All values were taken from native board execution unless a row is explicitly marked as an allocation failure.

The ESP32 study should be interpreted carefully. The final results were not obtained through the Arduino-core runtime, because that path produced avoidable ABI and allocator inconsistencies. The reported data was therefore collected from the native ESP-IDF project included in this repository.

Primary result sources were as follows.

- `ESP32Models/Results/esp32PaperResults.json`.
- `scripts/runEsp32NativeDeploymentSweep.py`.
- `scripts/compileEsp32ResultsReport.py`.

## Platform Context:

The classic no-PSRAM ESP32 is materially more restrictive than the newer ESP32-S3 class, particularly when contiguous internal memory blocks are required by TFLite Micro. This mattered directly for the present study, because several failure modes were caused not by flash size or parameter count alone, but by peak tensor arena pressure during `AllocateTensors()`.

The native runtime was tuned so that the deployment question could be asked cleanly. Single-core mode, IRAM-backed heap recovery, and kernel-level fixes were incorporated where necessary, especially for the TinierHAR rescue runs.

## Executive Summary:

MicroBiConvLSTM was the only family that completed all eight datasets on the connected classic ESP32 board. Its average latency was measured at approximately `168.511 ms`, and its average parity against the PyTorch reference remained close to `97.955%`. This combination of coverage and fidelity makes it the strongest ESP32 result in the repository.

TinyHAR was only partially deployable. Five datasets were executed successfully, but the latency remained high and the parity profile was unstable across several datasets. TinierHAR improved substantially after the native memory-path corrections and eventually reached six successful datasets, making it a credible secondary result. DeepConvLSTM remained outside the practical limits of the board, with all datasets failing before inference.

The main conclusion follows directly from these observations. On the tested classic ESP32 target, MicroBiConvLSTM was not merely the smallest model family that ran well. It was also the only family that delivered full dataset coverage without relying on severe qualification.

## Family-Level Comparison:

| Model | Successful Datasets | Failed Datasets | Avg Latency ms | Avg Arena Used B | Avg Heap Used B | Avg PyTorch Parity % | Avg Model B | Deployment Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MicroBiConvLSTM | 8 | 0 | 168.511 | 101476 | 184976 | 97.955 | 275088 | Full board coverage with the strongest fidelity profile. |
| TinyHAR | 5 | 3 | 442.089 | 88179 | 176656 | 60.237 | 218202 | Partial success, but too slow and inconsistent for a strong general claim. |
| TinierHAR | 6 | 2 | 188.391 | 146944 | 186308 | 76.263 | 312433 | A meaningful secondary result after native memory optimisation. |
| DeepConvLSTM | 0 | 8 |  |  |  |  | 1758060 | Not deployable on the tested no-PSRAM board. |

## MicroBiConvLSTM Results:

MicroBiConvLSTM completed the full eight-dataset sweep. The arena demand remained moderate by comparison with the baselines, and no board-side allocation collapse was observed once the native runtime had been stabilised.

| Dataset | Status | Arena KB | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | RUNS | 168 |  | 302232 | 109344 | 188048 | 176.985 | 99.276 | 99.276 | STANDING / STANDING |
| motionsense | RUNS | 168 | 561472 | 301112 | 109032 | 188048 | 180.574 | 96.056 | 97.337 | ups / dws |
| wisdm | RUNS | 168 | 559728 | 300872 | 109032 | 188048 | 185.365 | 98.155 | 98.558 | Jogging / Jogging |
| pamap2 | RUNS | 168 |  | 302608 | 109776 | 188048 | 178.497 | 96.883 | 96.883 | standing / lying |
| opportunity | RUNS | 168 |  | 307768 | 125432 | 188048 | 208.501 | 99.176 | 99.181 | class_0 / class_0 |
| unimib | RUNS | 168 | 560880 | 301112 | 109040 | 188048 | 176.451 | 95.906 | 96.008 | class_0 / class_0 |
| skoda | RUNS | 168 |  | 229904 | 85152 | 188048 | 146.551 | 99.582 | 99.582 | open_close_trunk / open_close_trunk |
| daphnet | RUNS | 144 | 413328 | 155096 | 55000 | 163472 | 95.161 | 98.602 | 98.602 | class_0 / class_0 |

## TinyHAR Results:

TinyHAR was found to be deployable only in a qualified sense. Five datasets ran to completion, but the combination of slower execution and lower parity weakened the family’s value as a clean low-end ESP32 result.

| Dataset | Status | Arena KB | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected Or Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | RUNS | 160 |  | 221248 | 120760 | 179920 | 673.217 | 19.600 | 15.060 | WALKING_UPSTAIRS / STANDING |
| motionsense | RUNS | 160 |  | 216920 | 99424 | 179920 | 501.759 | 85.127 | 40.715 | dws / dws |
| wisdm | RUNS | 160 |  | 213464 | 78400 | 179952 | 347.674 | 99.206 | 98.856 | Jogging / Jogging |
| pamap2 | FAILS | 176 |  | 232352 |  |  |  |  | 85.190 | Failed to allocate tensor arena from internal SRAM. |
| opportunity | FAILS | 176 |  | 300944 |  |  |  |  | 87.782 | Failed to allocate tensor arena from internal SRAM. |
| unimib | RUNS | 160 |  | 214544 | 78704 | 179952 | 346.616 | 97.252 | 96.174 | class_0 / class_0 |
| skoda | FAILS | 176 |  | 208024 |  |  |  |  | 93.067 | Failed to allocate tensor arena from internal SRAM. |
| daphnet | RUNS | 144 |  | 138120 | 63608 | 163536 | 341.178 | 0.000 | 0.000 | class_0 / class_0 |

## TinierHAR Results:

TinierHAR deserves separate comment because its final status differed from the earlier failed sweeps. After the native runtime fixes were applied, six datasets were run successfully. This makes the family materially more interesting than a naive size-based argument alone would have suggested.

| Dataset | Status | Arena KB | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected Or Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | RUNS | 176 | 619856 | 324232 | 164848 | 190856 | 209.216 | 92.012 | 90.183 | STANDING / STANDING |
| motionsense | RUNS | 176 | 613664 | 319624 | 157168 | 190856 | 176.314 | 68.708 | 69.191 | dws / dws |
| wisdm | RUNS | 176 | 607536 | 315016 | 149488 | 190856 | 132.246 | 98.506 | 98.239 | Jogging / Jogging |
| pamap2 | FAILS | 176 | 644784 | 339952 |  |  |  |  | 74.766 | Failed to resize buffer. Requested: 48640, available 38536, missing: 10104 |
| opportunity | FAILS | 176 | 773904 | 431696 |  |  |  |  | 95.202 | Failed to resize buffer. Requested: 202240, available 38536, missing: 163704 |
| unimib | RUNS | 176 | 608640 | 315200 | 149488 | 190952 | 130.707 | 98.953 | 98.855 | class_0 / class_0 |
| skoda | RUNS | 176 | 584512 | 280128 | 176400 | 190856 | 369.053 | 99.400 | 99.386 | open_close_trunk / open_close_trunk |
| daphnet | RUNS | 144 |  | 173616 | 84272 | 163472 | 112.807 | 0.000 | 0.000 | class_0 / class_0 |

## DeepConvLSTM Results:

DeepConvLSTM never crossed the practical memory threshold of the tested board. In each case, the failure was observed before inference, which means that the model family remained outside the usable arena budget rather than merely degrading in fidelity after execution.

| Dataset | Status | Arena KB | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected Or Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | FAILS | 224 |  | 1962008 |  |  |  |  | 98.659 | Failed to allocate tensor arena from internal SRAM. |
| motionsense | FAILS | 224 |  | 1961048 |  |  |  |  | 24.182 | Failed to allocate tensor arena from internal SRAM. |
| wisdm | FAILS | 224 |  | 1960088 |  |  |  |  | 96.048 | Failed to allocate tensor arena from internal SRAM. |
| pamap2 | FAILS | 224 |  | 1965760 |  |  |  |  | 98.136 | Failed to allocate tensor arena from internal SRAM. |
| opportunity | FAILS | 224 |  | 1985192 |  |  |  |  | 95.742 | Failed to allocate tensor arena from internal SRAM. |
| unimib | FAILS | 224 |  | 1961240 |  |  |  |  | 90.311 | Failed to allocate tensor arena from internal SRAM. |
| skoda | FAILS | 224 |  | 1403840 |  |  |  |  | 97.649 | Failed to allocate tensor arena from internal SRAM. |
| daphnet | FAILS | 224 |  | 865304 |  |  |  |  | 98.404 | Failed to allocate tensor arena from internal SRAM. |

## Failure Taxonomy:

Two failure categories were encountered on the classic ESP32, and the distinction between them should be preserved.

- `Tensor arena allocation failure.` The board could not reserve a contiguous internal memory block of the requested size before inference.
- `AllocateTensors resize failure.` The arena existed, but planner scratch or tensor growth exceeded what remained available inside that arena.

This distinction matters because the second category indicates that the execution path itself was already valid and that the failure was caused by a late memory high-water mark rather than by a general runtime incompatibility.

## Comparative Interpretation:

The ESP32 data supports three conclusions of practical interest.

- MicroBiConvLSTM was the only family with complete board coverage and consistently high parity.
- TinierHAR became a meaningful secondary result only after the native memory path had been repaired, which shows that deployment conclusions can be artefacts of runtime design if the software stack is not controlled carefully.
- DeepConvLSTM remained a genuine board-limit case rather than a tooling failure case.

These observations make the ESP32 study more than a simple latency benchmark. They also illuminate the difference between architectural efficiency and runtime-level deployability on low-memory microcontrollers.

## Conclusion:

On the tested classic no-PSRAM ESP32 board, MicroBiConvLSTM was the most convincing deployment candidate by a wide margin. It achieved complete dataset coverage, the strongest parity profile, and a moderate latency envelope without requiring the severe caveats that accompanied the baselines. TinierHAR should still be regarded as a useful secondary result after the native optimisation work, but the overall hardware story remained decisively in favour of MicroBiConvLSTM.
