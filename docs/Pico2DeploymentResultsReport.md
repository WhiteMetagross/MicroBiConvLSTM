# Pico 2 Deployment Results Report:

## Scope:

This report consolidates the Raspberry Pi Pico 2 deployment study carried out for the paper-aligned MicroBiConvLSTM retraining sweep and the three baseline families, namely TinyHAR, TinierHAR, and DeepConvLSTM. All values reported below were taken from on-device execution unless a row is explicitly marked as an allocation failure.

The report should be read as a deployment document rather than as a training summary. The emphasis was placed on what was actually sustained by the board, including flash footprint, tensor arena usage, heap pressure, latency, and fidelity relative to the desktop reference outputs.

Primary result sources were as follows.

- `Pico2Models/Results/microbiPico2Metrics.json`.
- `Pico2Models/Results/baselinePico2Metrics.json`.
- `Pico2Models/Results/baselineValidRunsSummary.json`.
- `scripts/runPico2DeploymentSweep.py`.
- `scripts/generatePicoFixture.py`.

## Executive Summary:

MicroBiConvLSTM was found to be the strongest Pico 2 deployment candidate in this study. Full dataset coverage was achieved, memory behaviour remained stable across all eight datasets, and the latency profile stayed materially below the baseline families. The average latency was measured at approximately `72.813 ms`, while the average arena usage remained close to `109.6 KB`.

The baseline picture was more mixed. TinyHAR ran on most datasets, but one dataset failed at the allocator stage and several successful runs exhibited substantial prediction drift. TinierHAR fit more broadly in memory, yet its parity behaviour was unstable on several datasets despite successful execution. DeepConvLSTM behaved as the clearest negative control, with most runs failing before inference could be completed.

From a paper-quality deployment standpoint, the Pico 2 data supports a straightforward conclusion. MicroBiConvLSTM offered the best joint balance of board compatibility, runtime efficiency, and output fidelity.

## Family-Level Comparison:

| Model | Successful Datasets | Failed Datasets | Avg Latency ms | Avg Arena Used B | Avg Heap Used B | Avg PyTorch Parity % | Avg Flash B | Avg Model B | Deployment Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MicroBiConvLSTM | 8 | 0 | 72.813 | 109630 | 432050 | 85.665 | 647069 | 275088 | Full board coverage with the strongest overall deployment profile. |
| TinyHAR | 7 | 1 | 354.053 | 123188 | 432050 | 60.447 | 572644 | 195484 | Broadly runnable, but fidelity was inconsistent. |
| TinierHAR | 8 | 0 | 229.326 | 183974 | 432050 | 54.230 | 691539 | 312433 | Memory fit was good, but parity drift remained a concern. |
| DeepConvLSTM | 1 | 7 | 768.780 | 244084 | 432048 | 25.482 | 1239620 | 865304 | The board was generally overmatched by this family. |

## MicroBiConvLSTM Results:

MicroBiConvLSTM completed all eight datasets on-device. The board-level behaviour was notably steady. Arena usage changed with dataset shape, as expected, but no catastrophic allocator instability was observed once the model had been embedded successfully.

| Dataset | Status | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | RUNS | 669436 | 302232 | 118340 | 432048 | 78.914 | 89.874 | 99.276 | STANDING / STANDING |
| motionsense | RUNS | 666740 | 301112 | 117988 | 432048 | 79.033 | 75.346 | 97.337 | ups / dws |
| wisdm | RUNS | 664988 | 300872 | 117988 | 432048 | 76.634 | 91.356 | 98.558 | Jogging / Jogging |
| pamap2 | RUNS | 675060 | 302608 | 118724 | 432048 | 76.969 | 89.002 | 96.883 | standing / lying |
| opportunity | RUNS | 710748 | 307768 | 134436 | 432052 | 89.878 | 94.709 | 99.181 | class_0 / class_0 |
| unimib | RUNS | 665260 | 301112 | 117988 | 432052 | 77.154 | 77.169 | 96.008 | class_0 / class_0 |
| skoda | RUNS | 604444 | 229904 | 91972 | 432052 | 63.656 | 95.906 | 99.582 | open_close_trunk / open_close_trunk |
| daphnet | RUNS | 519876 | 155096 | 59604 | 432048 | 40.267 | 71.953 | 98.602 | class_0 / class_0 |

## TinyHAR Results:

TinyHAR was found to be memory-compatible on most datasets, but the fidelity story was much less convincing than the raw success count first suggests. On several datasets, the board did run the model, but the output class and parity metrics drifted enough to weaken the value of the deployment from a scientific reporting standpoint.

| Dataset | Status | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected Or Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | RUNS | 597924 | 221248 | 126916 | 432048 | 328.974 | 6.124 | 15.060 | WALKING_UPSTAIRS / STANDING |
| motionsense | RUNS |  |  |  |  | 249.543 | 66.585 |  | dws / dws |
| wisdm | RUNS |  |  |  |  | 171.717 | 96.965 |  | Jogging / Jogging |
| pamap2 | RUNS |  |  |  |  | 641.389 | 80.223 |  | standing / lying |
| opportunity | FAILS | 713412 | 300944 |  |  |  |  | 87.782 | Failed to resize buffer. Requested: 492960, available 302012, missing: 190948 |
| unimib | RUNS | 588180 | 214544 | 84868 | 432052 | 170.132 | 88.440 | 96.174 | class_0 / class_0 |
| skoda | RUNS | 592052 | 208024 | 213380 | 432052 | 742.740 | 84.792 | 93.067 | open_close_trunk / open_close_trunk |
| daphnet | RUNS | 512420 | 138120 | 67588 | 432048 | 173.879 | 0.000 | 0.000 | class_0 / class_0 |

## TinierHAR Results:

TinierHAR produced an interesting contrast. Board fit was not the principal issue here, because all eight datasets were run successfully. The weaker point was prediction fidelity, which varied sharply across datasets and in a few cases fell to levels that would not support a strong deployment claim without qualification.

| Dataset | Status | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | RUNS | 691444 | 324232 | 172468 | 432048 | 121.846 | 77.517 | 90.183 | STANDING / STANDING |
| motionsense | RUNS | 685252 | 319624 | 164788 | 432048 | 100.396 | 66.702 | 69.191 | dws / dws |
| wisdm | RUNS | 688620 | 315016 | 157108 | 432048 | 76.412 | 79.044 | 98.239 | Jogging / Jogging |
| pamap2 | RUNS | 721908 | 339952 | 198068 | 432048 | 205.091 | 16.485 | 74.766 | running / lying |
| opportunity | RUNS | 844172 | 431696 | 351668 | 432052 | 930.591 | 27.469 | 95.202 | class_4 / class_0 |
| unimib | RUNS | 688836 | 315200 | 157108 | 432052 | 77.248 | 81.975 | 98.855 | class_0 / class_0 |
| skoda | RUNS | 664164 | 280128 | 182244 | 432052 | 257.794 | 84.647 | 99.386 | open_close_trunk / open_close_trunk |
| daphnet | RUNS | 547916 | 173616 | 88340 | 432048 | 65.231 | 0.000 | 0.000 | class_0 / class_0 |

## DeepConvLSTM Results:

DeepConvLSTM was found to be largely incompatible with the Pico 2 memory envelope in the tested configuration. The repeated failure mode was allocator exhaustion, which is scientifically useful because it defines a meaningful boundary condition for the platform.

| Dataset | Status | Flash B | Model B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Predicted / Expected Or Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ucihar | FAILS | 2338700 | 1962008 |  |  |  |  | 98.659 | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 |
| motionsense | FAILS | 2336148 | 1961048 |  |  |  |  | 24.182 | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 |
| wisdm | FAILS | 2333692 | 1960088 |  |  |  |  | 96.048 | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 |
| pamap2 | FAILS | 2347716 | 1965760 |  |  |  |  | 98.136 | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 |
| opportunity | FAILS | 2397676 | 1985192 |  |  |  |  | 95.742 | Failed to allocate tail memory. Requested: 60, available 44, missing: 16 |
| unimib | FAILS | 2334892 | 1961240 |  |  |  |  | 90.311 | Failed to allocate tail memory. Requested: 60, available 44, missing: 16 |
| skoda | FAILS | 1787884 | 1403840 |  |  |  |  | 97.649 | Failed to allocate temp memory. Requested: 99576, available 77308, missing: 22268 |
| daphnet | RUNS | 1239620 | 865304 | 244084 | 432048 | 768.780 | 25.482 | 98.404 | class_0 / class_0 |

## Comparative Interpretation:

Three practical observations were supported by the Pico 2 data.

- Full deployment coverage was achieved only by MicroBiConvLSTM.
- Broad memory fit was not sufficient for a strong result, because TinierHAR showed that a model may run while still drifting severely from the reference output.
- Hardware failure was not the only negative outcome of interest, because DeepConvLSTM established a useful upper bound for what should not be expected from this class of microcontroller.

It should therefore be noted that deployment quality in this study was determined by a compound criterion. A good deployment required successful flashing, successful tensor allocation, acceptable latency, and reasonable parity against the desktop references.

## Conclusion:

The Pico 2 results support MicroBiConvLSTM as the most credible deployment choice among the tested families. The model was lighter than the baselines in practice, it remained consistently executable across all datasets, and its board-level behaviour was the least pathological when latency, memory use, and fidelity were examined together. The baseline results remain informative, but they mostly reinforce the efficiency advantage claimed for MicroBiConvLSTM in the paper.
