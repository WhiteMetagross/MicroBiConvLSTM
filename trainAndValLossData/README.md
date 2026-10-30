# Seed-29 Training and Validation Loss Artifacts:

This directory preserves the loss histories and publication-ready figures produced by the paper-aligned retraining of `MicroBiConvLSTM`, `DeepConvLSTM`, `TinyHAR`, and `TinierHAR`. Every model was trained once on each of the eight HAR datasets with seed 29, a maximum of 200 epochs, and an early-stopping patience of 10 epochs.

The term validation loss is retained in the figures because it identifies the loss monitored during model selection. In the paper-aligned trainers, this value was calculated on the existing held-out `testLoader`. It must therefore not be described as loss from a newly introduced or independent validation split.

## Directory Layout:

- `images/` contains 32 loss-curve PNG files, one for every model and dataset combination.
- `data/` contains the corresponding JSON and CSV histories, the complete run manifest, the execution status, and a compact cross-run summary.
- `data/seed29TrainingSummary.csv` provides one row per completed training run.
- `data/seed29TrainingSummary.json` preserves the same summary with provenance metadata.

## Experimental Matrix:

| Model | UCI-HAR | MotionSense | WISDM | PAMAP2 | Opportunity | UniMiB | Skoda | Daphnet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MicroBiConvLSTM | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| DeepConvLSTM | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| TinyHAR | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| TinierHAR | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

## Reproduction:

The full sequential run is started from the repository root as follows.

```bash
python scripts/runPaperRetraining.py \
  --models all \
  --datasets all \
  --seed-list 29 \
  --epochs 200 \
  --patience 10
```

The model trainers call `scripts/trainingCurveArtifacts.py` after the best checkpoint has been restored. JSON and CSV histories are written for numerical analysis, while the PNG curve is rendered with a deterministic non-interactive Matplotlib backend. Existing checkpoint and result filenames are deliberately retained so that the deployment conversion programs remain compatible.

## Interpretation Notes:

- The dashed vertical line identifies the epoch with the best monitored macro-F1 score.
- Early stopping may leave fewer than 200 observations in a history.
- Training loss and held-out loss are cross-entropy losses recorded after each completed epoch.
- The archived figures are evidence from the completed seed-29 experiment. They are not regenerated when this repository is installed.
