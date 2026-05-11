# Pico 2 Deployment Results Report:

## Scope:

This report consolidates the corrected Raspberry Pi Pico 2 deployment study for `MicroBiConvLSTM` and the three baseline families, namely `TinyHAR`, `TinierHAR`, and `DeepConvLSTM`. Both `FP32` and `INT8` deployment variants were examined on the board wherever the runtime could be brought to completion. When a deployment did not complete, the row has been preserved as a real allocator or buffer-sizing failure rather than being left as an unresolved gap.

The present report supersedes the earlier Pico summary that relied too heavily on the first `INT8` runs alone. The final repository state should therefore be read through the consolidated deployment matrix in `Pico2Models/Results/pico2Fp32Int8Results.json`, together with the per-family raw summaries retained in the same directory.

Primary result sources were as follows.

- `Pico2Models/Results/pico2Fp32Int8Results.json`.
- `Pico2Models/Results/pico2Fp32Int8Results.md`.
- `Pico2Models/Results/microbiPico2Metrics.json`.
- `Pico2Models/Results/baselinePico2Metrics.json`.
- `scripts/runPico2DeploymentSweep.py`.
- `scripts/compilePico2VariantResults.py`.

## Executive Summary:

The corrected Pico 2 picture is now much clearer than in the earlier draft. `MicroBiConvLSTM` completed all `8/8` datasets in both `INT8` and `FP32`, but the fidelity story differed sharply between the two variants. The `FP32` path reached effectively exact PyTorch agreement across every dataset, whereas the `INT8` path exhibited substantial parity loss on several datasets despite successful execution.

The baselines behaved less uniformly. `TinyHAR` improved materially under `FP32`, but three heavier datasets still exceeded the practical Pico memory envelope. `TinierHAR` proved to be a much stronger `FP32` candidate than the initial `INT8` study suggested, completing `7/8` datasets with effectively exact parity. `DeepConvLSTM` remained the negative control. Only `daphnet` completed on-device in either numeric mode, and parity stayed poor even in `FP32`.

The paper-facing conclusion is therefore straightforward. On Pico 2, `MicroBiConvLSTM` should be interpreted as a high-fidelity deployment family when the `FP32` bundles are used. The low-parity issue is not a universal microcontroller property of the model, but mainly an `INT8` deployment-path effect for this TFLite Micro stack.

## Variant-Level Summary:

| Model | INT8 Runs | INT8 Fails | Avg INT8 Latency (ms) | Avg INT8 Parity (%) | FP32 Runs | FP32 Fails | Avg FP32 Latency (ms) | Avg FP32 Parity (%) | Deployment Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `MicroBiConvLSTM` | 8 | 0 | 72.813 | 85.665 | 8 | 0 | 83.002 | 99.999992 | Full coverage in both modes. `FP32` is the research-grade Pico path. |
| `TinyHAR` | 7 | 1 | 354.053 | 60.447 | 5 | 3 | 486.132 | 89.584 | Partial recovery in `FP32`, but several datasets remain memory-limited. |
| `TinierHAR` | 8 | 0 | 229.326 | 54.230 | 7 | 1 | 231.524 | 99.999986 | `FP32` becomes highly faithful, with one heavy dataset still exceeding the buffer budget. |
| `DeepConvLSTM` | 1 | 7 | 768.780 | 25.482 | 1 | 7 | 2034.567 | 26.312 | The family is not practically deployable on Pico 2 in this configuration. |

## MicroBiConvLSTM:

`MicroBiConvLSTM` is the strongest Pico 2 result in this repository. Every dataset bundle ran to completion in both variants. The important scientific distinction is that `INT8` execution was not sufficiently faithful on several datasets, whereas `FP32` execution retained essentially exact PyTorch agreement throughout.

| Dataset | INT8 Latency (ms) | INT8 Parity (%) | FP32 Latency (ms) | FP32 Parity (%) | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `ucihar` | 78.914 | 89.874 | 64.800 | 99.999992 | `FP32` removes the remaining fidelity gap. |
| `motionsense` | 79.033 | 75.346 | 62.331 | 99.999992 | A large `INT8` drift is fully corrected in `FP32`. |
| `wisdm` | 76.634 | 91.356 | 60.475 | 99.999992 | `INT8` was already acceptable, but `FP32` becomes exact. |
| `pamap2` | 76.969 | 89.002 | 69.473 | 99.999992 | `FP32` recovers the deployment cleanly. |
| `opportunity` | 89.878 | 94.709 | 249.506 | 99.999992 | `FP32` is slower, but fidelity remains exact. |
| `unimib` | 77.154 | 77.169 | 60.455 | 99.999992 | This is another strong correction from the `FP32` path. |
| `skoda` | 63.656 | 95.906 | 61.657 | 99.999992 | Both modes run well, with `FP32` retaining full agreement. |
| `daphnet` | 40.267 | 71.953 | 35.320 | 99.999992 | The `INT8` gap disappears completely under `FP32`. |

Two label rows, namely `motionsense` and `pamap2`, still preserve a top-1 mismatch against the stored fixture label even in `FP32`. Because parity versus the PyTorch logits remains effectively exact, this disagreement should be interpreted as a reference-label issue for the chosen fixture rather than as a board-side divergence.

## TinyHAR:

`TinyHAR` showed that a baseline can be memory-compatible in some settings while still being scientifically weak as an edge result. The `INT8` path was often poor. The `FP32` path repaired several datasets, but not all of them, and three datasets remained true Pico memory failures.

| Dataset | INT8 Status | INT8 Parity (%) | FP32 Status | FP32 Parity (%) | Readout |
| --- | --- | ---: | --- | ---: | --- |
| `ucihar` | runs | 6.124 | runs | 94.892 | `FP32` converts a failed `INT8` fidelity case into a usable result. |
| `motionsense` | runs | 66.585 | runs | 88.628 | Improved, but still below the preferred `90%` threshold. |
| `wisdm` | runs | 96.965 | runs | 99.639 | Strong in both modes, with `FP32` slightly cleaner. |
| `pamap2` | runs | 80.223 | fails | - | `FP32` exceeds the available buffer budget. |
| `opportunity` | fails | - | fails | - | Both modes exceed Pico memory limits. |
| `unimib` | runs | 88.440 | runs | 99.859 | `FP32` restores research-grade fidelity. |
| `skoda` | runs | 84.792 | fails | - | The wider `FP32` tensors exceed the board budget. |
| `daphnet` | runs | 0.000 | runs | 64.902 | The family remains weak on this dataset even in `FP32`. |

## TinierHAR:

`TinierHAR` produced the most dramatic correction after the final `FP32` completion pass. The earlier `INT8` study gave the impression of a broadly drifting family. Once the `FP32` bundles were executed on the board, `7/8` datasets became effectively exact against PyTorch, and only the largest `opportunity` case remained a true memory failure.

| Dataset | INT8 Parity (%) | FP32 Parity (%) | FP32 Status | Readout |
| --- | ---: | ---: | --- | --- |
| `ucihar` | 77.517 | 99.999992 | runs | Fully corrected in `FP32`. |
| `motionsense` | 66.702 | 99.999985 | runs | Fully corrected in `FP32`. |
| `wisdm` | 79.044 | 99.999992 | runs | Fully corrected in `FP32`. |
| `pamap2` | 16.485 | 99.999992 | runs | A severe `INT8` failure becomes exact in `FP32`. |
| `opportunity` | 27.469 | - | fails | The family exceeds the Pico buffer budget on the heaviest case. |
| `unimib` | 81.975 | 99.999992 | runs | Fully corrected in `FP32`. |
| `skoda` | 84.647 | 99.999992 | runs | Fully corrected in `FP32`. |
| `daphnet` | 0.000 | 99.999954 | runs | Fully corrected in `FP32`. |

## DeepConvLSTM:

`DeepConvLSTM` remains the clearest negative control in this study. The family is too large for the Pico 2 memory envelope in nearly every case. Most datasets failed before inference under both modes, and the single successful `daphnet` execution remained low-parity even in `FP32`.

| Dataset | INT8 Status | FP32 Status | FP32 Failure Or Parity | Interpretation |
| --- | --- | --- | --- | --- |
| `ucihar` | fails | fails | Failed to allocate tail memory. Requested `60`, available `12`, missing `48`. | Real allocator failure. |
| `motionsense` | fails | fails | Failed to allocate tail memory. Requested `60`, available `12`, missing `48`. | Real allocator failure. |
| `wisdm` | fails | fails | Failed to allocate tail memory. Requested `60`, available `12`, missing `48`. | Real allocator failure. |
| `pamap2` | fails | fails | Failed to allocate tail memory. Requested `60`, available `12`, missing `48`. | Real allocator failure. |
| `opportunity` | fails | fails | Failed to allocate tail memory. Requested `60`, available `12`, missing `48`. | Real allocator failure. |
| `unimib` | fails | fails | Failed to allocate tail memory. Requested `60`, available `12`, missing `48`. | Real allocator failure. |
| `skoda` | fails | fails | Failed to allocate temp memory. Requested `99456`, available `79532`, missing `19924`. | Real allocator failure. |
| `daphnet` | runs | runs | `26.312%` parity in `FP32`. | Runtime mismatch persists even after quantization is removed. |

## Practical Conclusion:

The final Pico 2 record supports four paper-grade statements.

- `MicroBiConvLSTM` is a faithful Pico 2 deployment family when the `FP32` bundles are used.
- `TinyHAR` can be partially rescued in `FP32`, but it remains less convincing because fidelity and memory behaviour are still inconsistent across datasets.
- `TinierHAR` is much stronger than the initial `INT8` sweep suggested, and its `FP32` path is highly faithful on every dataset that fits.
- `DeepConvLSTM` should remain framed as an impractical Pico 2 target in this deployment configuration.

The canonical machine-readable summary for this conclusion is `Pico2Models/Results/pico2Fp32Int8Results.json`. The per-family `INT8` summaries are still preserved, but they should be interpreted as lower-level artifacts rather than as the final paper-facing deployment view.
