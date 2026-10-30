# Cross-Paradigm Efficiency Results:

All cycle measurements cover one complete raw-window inference. MicroBiConvLSTM includes its full neural forward pass. Classical pipelines include handcrafted feature extraction, scaling, PCA where applicable, and final prediction. Measurements are invariant-TSC reference cycles on one pinned WSL CPU thread.

Model-state size is a compact FP32-equivalent representation, not framework container size. XGBoost uses compact split and leaf arrays.

- CPU affinity: `0`.
- Samples per estimator: `24`.
- Repeats per sample: `5`.
- Measurements per estimator: `120`.

## Cross-Dataset Summary:

| Model | Mean Macro-F1 (%) | Mean State (KiB) | Mean Mcycles/window | F1 points/KiB | F1 points/Mcycle |
|---|---:|---:|---:|---:|---:|
| microbi_convlstm | 84.68 | 44.68 | 0.510 | 1.895 | 166.055 |
| logistic_regression | 77.82 | 14.52 | 1.207 | 5.361 | 64.446 |
| rbf_svm | 82.02 | 1213.95 | 1.782 | 0.068 | 46.021 |
| xgboost | 81.29 | 752.69 | 2.370 | 0.108 | 34.304 |

## Per-Dataset Results:

| Model | Dataset | Macro-F1 (%) | State (KiB) | Mcycles/window | F1 points/KiB | F1 points/Mcycle |
|---|---|---:|---:|---:|---:|---:|
| microbi_convlstm | ucihar | 94.02 | 41.10 | 0.524 | 2.287 | 179.580 |
| microbi_convlstm | motionsense | 90.44 | 40.16 | 0.545 | 2.252 | 166.039 |
| microbi_convlstm | wisdm | 76.93 | 39.23 | 0.484 | 1.961 | 158.845 |
| microbi_convlstm | pamap2 | 62.49 | 45.38 | 0.504 | 1.377 | 124.056 |
| microbi_convlstm | opportunity | 87.81 | 62.79 | 0.524 | 1.399 | 167.587 |
| microbi_convlstm | unimib | 77.25 | 39.80 | 0.595 | 1.941 | 129.766 |
| microbi_convlstm | skoda | 98.14 | 48.62 | 0.459 | 2.018 | 213.752 |
| microbi_convlstm | daphnet | 90.32 | 40.34 | 0.445 | 2.239 | 203.110 |
| logistic_regression | ucihar | 92.70 | 5.93 | 1.096 | 15.633 | 84.574 |
| logistic_regression | motionsense | 89.43 | 3.96 | 1.066 | 22.578 | 83.904 |
| logistic_regression | wisdm | 75.12 | 1.99 | 0.977 | 37.709 | 76.893 |
| logistic_regression | pamap2 | 59.20 | 21.87 | 1.177 | 2.707 | 50.308 |
| logistic_regression | opportunity | 84.92 | 45.38 | 2.156 | 1.871 | 39.382 |
| logistic_regression | unimib | 62.16 | 2.74 | 0.969 | 22.667 | 64.156 |
| logistic_regression | skoda | 96.01 | 32.04 | 1.233 | 2.997 | 77.894 |
| logistic_regression | daphnet | 63.01 | 2.22 | 0.987 | 28.399 | 63.864 |
| rbf_svm | ucihar | 92.71 | 493.66 | 1.489 | 0.188 | 62.252 |
| rbf_svm | motionsense | 90.47 | 870.36 | 1.527 | 0.104 | 59.263 |
| rbf_svm | wisdm | 80.11 | 480.56 | 1.329 | 0.167 | 60.252 |
| rbf_svm | pamap2 | 77.23 | 1965.10 | 2.003 | 0.039 | 38.563 |
| rbf_svm | opportunity | 88.29 | 1225.11 | 2.643 | 0.072 | 33.400 |
| rbf_svm | unimib | 67.55 | 711.19 | 1.389 | 0.095 | 48.637 |
| rbf_svm | skoda | 96.92 | 2154.43 | 2.130 | 0.045 | 45.498 |
| rbf_svm | daphnet | 62.90 | 1811.18 | 1.748 | 0.035 | 35.990 |
| xgboost | ucihar | 92.81 | 408.63 | 2.058 | 0.227 | 45.098 |
| xgboost | motionsense | 93.51 | 803.35 | 2.149 | 0.116 | 43.509 |
| xgboost | wisdm | 84.61 | 623.82 | 1.911 | 0.136 | 44.264 |
| xgboost | pamap2 | 56.49 | 897.25 | 2.715 | 0.063 | 20.809 |
| xgboost | opportunity | 89.26 | 666.57 | 3.230 | 0.134 | 27.630 |
| xgboost | unimib | 65.27 | 1113.53 | 2.503 | 0.059 | 26.076 |
| xgboost | skoda | 99.53 | 1279.90 | 2.996 | 0.078 | 33.219 |
| xgboost | daphnet | 68.83 | 228.49 | 1.394 | 0.301 | 49.380 |

## Interpretation:

- State size includes every learned or fitted numeric array required for inference.
- Cycle counts are host-implementation measurements and should not be presented as Pico or ESP32 cycles.
- Cross-device claims require rerunning every complete pipeline on the same target hardware and compiler configuration.
