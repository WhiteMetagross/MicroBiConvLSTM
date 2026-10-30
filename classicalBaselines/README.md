# Classical HAR Baselines and Cross-Paradigm Efficiency:

Three strong classical baselines were evaluated against MicroBiConvLSTM: multinomial logistic regression, PCA-RBF SVM, and XGBoost. The implementation accepts the same raw sensor windows used by the neural models and extracts 21 deterministic time-domain, frequency-domain, and distributional features per sensor channel. Scaling, PCA, and estimator fitting are learned from the training split only.

Five experiment seeds were deterministically generated from master seed 17. The complete dataset split was used unless an explicit SVM training cap was requested. No held-out labels were used for fitting preprocessing transforms or estimator hyperparameters.

## Implemented Programs:

- `trainClassicalBaselines.py` extracts or reuses cached features, fits all three estimators, evaluates macro-F1, and writes resumable per-run records.
- `profileClassicalComplexity.py` derives compact learned-state sizes and analytical inference-complexity descriptions from the fitted estimators.
- `benchmarkCrossParadigmEfficiency.py` measures complete raw-window inference using one pinned WSL CPU thread and an invariant timestamp counter.
- `readTsc.c` provides the serialized x86 timestamp-counter reader used by the host benchmark.
- `tests/testClassicalBaselines.py` checks the feature schema, deterministic seed generation, and stratified sample capping.

## Classical Training:

```bash
python classicalBaselines/trainClassicalBaselines.py \
  --models all \
  --datasets all \
  --seeds 5 \
  --master-seed 17 \
  --run-name classicalMaster17FiveSeeds
```

The command can be checked without fitting estimators by appending `--dry-run`. Generated model containers and feature caches remain runtime products and are not duplicated in this research-results directory.

## Cross-Paradigm Measurement:

The timestamp-counter helper is compiled under WSL before the benchmark is run.

```bash
gcc -O3 -shared -fPIC classicalBaselines/readTsc.c -o results/crossParadigm/libReadTsc.so

python classicalBaselines/benchmarkCrossParadigmEfficiency.py \
  --classical-run results/classicalBaselines/classicalMaster17FiveSeeds \
  --micro-root results/paperRetraining/microBiConvLstm \
  --tsc-library results/crossParadigm/libReadTsc.so \
  --output-dir results/crossParadigm
```

Classical cycle measurements include feature extraction, FFTs, scaling, PCA where applicable, and prediction. MicroBiConvLSTM includes its complete neural forward pass. State sizes are compact FP32-equivalent inference representations rather than `.pth` or `.joblib` container sizes.

## Cross-Dataset Results:

| Model | Macro-F1 | Learned state size | Mcycles/window | F1 points/KiB | F1 points/Mcycle |
|---|---:|---:|---:|---:|---:|
| MicroBiConvLSTM | 84.68% | 44.68 KiB | 0.510 | 1.895 | 166.055 |
| Logistic Regression | 77.82% | 14.52 KiB | 1.207 | 5.361 | 64.446 |
| PCA-RBF SVM | 82.02% | 1,213.95 KiB | 1.782 | 0.068 | 46.021 |
| XGBoost | 81.29% | 752.69 KiB | 2.370 | 0.108 | 34.304 |

The classical values are means across eight datasets and five seeds. The MicroBiConvLSTM values are means across the eight available seed-29 checkpoints. These are host implementation cycles and must not be presented as Raspberry Pi Pico 2 or ESP32 cycles.

## Saved Results:

- `results/data/classicalBaselineSummary.json` contains aggregate accuracy, macro-F1, balanced accuracy, and fit-time measurements.
- `results/data/classicalBaselinePerRunResults.json` and `.csv` contain all 120 model, dataset, and seed records.
- `results/data/classicalComplexity.md`, `.json`, and `.csv` preserve the estimator state-size and complexity analysis.
- `results/data/crossParadigmEfficiency.md`, `.json`, and `.csv` preserve the complete host-cycle measurements.
- `results/data/crossParadigmEfficiencySummary.csv` contains the four-model cross-dataset table shown above.

Classical estimators do not produce epoch-wise training and validation loss curves. Their convergence and generalization should be studied through estimator-specific diagnostics or learning curves, not by manufacturing neural-style epoch histories.
