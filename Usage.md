# MicroBiConvLSTM Usage Guide.

**Author:** Mridankan Mandal

**Paper:** [MicroBiConvLSTM on arXiv](https://arxiv.org/abs/2602.06523)

This document provides detailed instructions for using the MicroBiConvLSTM codebase. It covers training, hyperparameter optimization, ablation studies, benchmarking, and visualization.

---

## Table of Contents.

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Training MicroBiConvLSTM](#training-microbiconvlstm)
4. [Training Baseline Models](#training-baseline-models)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Ablation Studies](#ablation-studies)
7. [Memory Footprint Benchmarking](#memory-footprint-benchmarking)
8. [Generating Publication Figures](#generating-publication-figures)
9. [Sensor Data Visualization](#sensor-data-visualization)
10. [Model Creation and Inference](#model-creation-and-inference)
11. [Understanding the Results](#understanding-the-results)

---

## Prerequisites.

Before using the scripts, ensure the following are in place.

- Python 3.8 or higher is installed.
- All dependencies from `requirements.txt` are installed (see [InstallationAndSetup.md](InstallationAndSetup.md)).
- A CUDA-compatible GPU is recommended but not required (CPU training is supported).
- The benchmark datasets are downloaded and placed in a `datasets/` folder at the repository root.

---

## Dataset Preparation.

The data loaders expect the datasets to be organized in the following structure relative to the repository root.

```
datasets/
|-- UCI HAR Dataset/
|   |-- train/
|   |   |-- Inertial Signals/
|   |   |-- X_train.txt
|   |   |-- y_train.txt
|   |-- test/
|       |-- Inertial Signals/
|       |-- X_test.txt
|       |-- y_test.txt
|-- motion-sense-master/
|-- WISDM_ar_v1.1/
|-- PAMAP2_Dataset/
|-- Opportunity/
|-- UniMiB-SHAR/
|-- Skoda/
|-- Daphnet/
```

### Dataset Download Sources.

| Dataset | Source |
|---------|--------|
| UCI-HAR | https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones |
| MotionSense | https://github.com/mmalekzadeh/motion-sense |
| WISDM | https://www.cis.fordham.edu/wisdm/dataset.php |
| PAMAP2 | https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring |
| Opportunity | https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition |
| UniMiB-SHAR | http://www.sal.disco.unimib.it/technologies/unimib-shar/ |
| Skoda | https://sensor.informatik.uni-mannheim.de/ |
| Daphnet | https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait |

Download each dataset manually and extract it into the corresponding folder under `datasets/`.

---

## Training MicroBiConvLSTM.

The main training script is `scripts/trainMicroBiConvLstm.py`. It trains the MicroBiConvLSTM model with HPO-tuned hyperparameters, multi-seed evaluation, early stopping, and AMP/FP16 acceleration.

### Training on a Single Dataset.

```bash
python scripts/trainMicroBiConvLstm.py --dataset ucihar --seeds 5
```

This command trains MicroBiConvLSTM on UCI-HAR with 5 random seeds derived from the master seed (17). Results are saved to `results/training/` and checkpoints are saved to `checkpoints/`.

### Training on All Datasets.

```bash
python scripts/trainMicroBiConvLstm.py --dataset all --seeds 5
```

This sequentially trains the model on all eight datasets.

### Command-Line Arguments for Training.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ucihar` | Dataset name or `all` for all datasets. Valid options: `ucihar`, `motionsense`, `wisdm`, `pamap2`, `opportunity`, `unimib`, `skoda`, `daphnet`. |
| `--seeds` | `5` | Number of random seeds for statistical evaluation. |
| `--epochs` | `200` | Maximum number of training epochs. |
| `--patience` | `10` | Early stopping patience (epochs without improvement). |

### Training Protocol Details.

- The optimizer is AdamW with dataset-specific learning rate and weight decay (from HPO).
- The learning rate scheduler is CosineAnnealingLR with minimum LR of 1e-6.
- Gradient clipping is applied with a maximum norm of 1.0.
- Early stopping monitors the macro F1 score on the test set.
- For imbalanced datasets (PAMAP2, Opportunity, Skoda, Daphnet), class weights are applied to the CrossEntropyLoss.
- Mixed precision training (AMP/FP16) is automatically enabled when a CUDA GPU is available.

### Output Files.

After training, the following files are produced.

- `checkpoints/microBiConvLstm_{dataset}_seed{seed}.pt` -- Model checkpoint with the best weights.
- `results/training/microBiConvLstm_{dataset}.json` -- Training results including accuracy, F1 score, training time, and epoch history.

---

## Training Baseline Models.

The baseline training script is `scripts/trainBaselines.py`. It supports training DeepConvLSTM, TinyHAR, and TinierHAR with the same training protocol used for MicroBiConvLSTM.

### Training a Specific Baseline on a Single Dataset.

```bash
python scripts/trainBaselines.py --dataset ucihar --model tinyhar --seeds 5
```

### Training All Baselines on a Single Dataset.

```bash
python scripts/trainBaselines.py --dataset ucihar --model all --seeds 5
```

### Training All Baselines on All Datasets.

```bash
python scripts/trainBaselines.py --dataset all --model all --seeds 5
```

### Command-Line Arguments for Baselines.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ucihar` | Dataset name or `all`. |
| `--model` | `all` | Model name: `tinyhar`, `tinierhar`, `deepconvlstm`, or `all`. |
| `--seeds` | `5` | Number of random seeds. |
| `--epochs` | `200` | Maximum training epochs. |
| `--patience` | `10` | Early stopping patience. |

### Output Files.

- `results/baselines/{model}/{dataset}/seed{seed}_checkpoint.pt` -- Model checkpoint.
- `results/baselines/{model}/{dataset}/seed{seed}_results.json` -- Per-seed result file.

---

## Hyperparameter Optimization.

HPO is performed using Optuna with a TPE (Tree-structured Parzen Estimator) sampler. The architecture is frozen during HPO and only three training hyperparameters are tuned: learning rate, weight decay, and dropout.

### HPO for MicroBiConvLSTM.

```bash
python scripts/hpoMicroBiConvLstm.py --dataset ucihar --n-trials 50
```

### HPO for All Datasets.

```bash
python scripts/hpoMicroBiConvLstm.py --dataset all --n-trials 50
```

### HPO for Baseline Models.

```bash
python scripts/hpoBaselines.py --model tinyhar --dataset ucihar --n-trials 50
```

### HPO for All Baselines on All Datasets.

```bash
python scripts/hpoBaselines.py --model all --dataset all --n-trials 50
```

### HPO Command-Line Arguments.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ucihar` | Dataset name or `all`. |
| `--model` | (baselines only) | Model name or `all`. |
| `--n-trials` | `50` | Number of Optuna trials. |
| `--epochs` | `50` | Training epochs per trial. |
| `--patience` | `5` | Early stopping patience per trial. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--save-dir` | `./hpo_results` | Directory to save HPO results. |

### HPO Search Space.

| Hyperparameter | Range | Scale |
|----------------|-------|-------|
| Learning Rate | 1e-4 to 1e-2 | Log-uniform |
| Weight Decay | 1e-5 to 0.05 | Log-uniform |
| Dropout | 0.0 to 0.5 | Uniform |

### Output Files.

- `hpo_results/microBiConvLstm_{dataset}_hpo.json` -- HPO results including best trial, optimal parameters, and all trial data.
- `hpo_results/{model}_{dataset}_hpo.json` -- Baseline HPO results.

---

## Ablation Studies.

The ablation study runner supports five distinct study types. All studies use a fixed learning rate of 1e-3 by default for controlled comparison.

### Architectural Ablation (A0-A4).

Trains and evaluates the five architectural variants.

```bash
python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study arch --seeds 5
```

### Pareto Grid Search.

Explores the parameter space by varying convFilters and lstmHidden sizes.

```bash
python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study pareto --seeds 1
```

The default grid searches over `convFilters` in {8, 16, 24} and `lstmHidden` in {16, 24, 32}. These can be customized.

```bash
python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study pareto --conv-grid 8,16,24,32 --lstm-grid 16,24,32,48
```

### Complexity Scaling Verification.

Verifies O(N) complexity by computing MACs at different sequence lengths.

```bash
python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study complexity
```

### INT8 Post-Training Quantization.

Evaluates accuracy retention after dynamic INT8 quantization.

```bash
python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study quant
```

This requires a trained A0 checkpoint. Run `--study arch` first.

### Sensitivity and Robustness.

Evaluates model robustness under channel dropout, sampling jitter, and preprocessing bypass.

```bash
python scripts/ablationStudiesMicroBiConvLstm.py --dataset pamap2 --study sensitivity
```

### Running All Studies on All Datasets.

```bash
for dataset in ucihar motionsense wisdm pamap2 opportunity unimib skoda daphnet; do
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset $dataset --study arch --seeds 5
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset $dataset --study pareto --seeds 1
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset $dataset --study complexity
done
```

### Ablation Command-Line Arguments.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ucihar` | Dataset name. |
| `--study` | `arch` | Study type: `arch`, `pareto`, `complexity`, `quant`, `sensitivity`, `gradcam`. |
| `--seeds` | `3` | Number of random seeds. |
| `--epochs` | `200` | Maximum training epochs. |
| `--patience` | `10` | Early stopping patience. |
| `--lr` | `0.001` | Fixed learning rate for ablation training. |
| `--conv-grid` | `8,16,24` | Comma-separated conv filter sizes for Pareto study. |
| `--lstm-grid` | `16,24,32` | Comma-separated LSTM hidden sizes for Pareto study. |

### Output Files.

Results are saved to `results/ablations/{dataset}/{study}/` with JSON artifacts and optional plots.

---

## Memory Footprint Benchmarking.

Measures and compares model sizes across FP32 and INT8 formats.

### Benchmark a Single Dataset.

```bash
python scripts/benchmarkMemoryFootprint.py --dataset ucihar
```

### Benchmark All Datasets.

```bash
python scripts/benchmarkMemoryFootprint.py --all-datasets
```

### CPU-Only Mode.

```bash
python scripts/benchmarkMemoryFootprint.py --all-datasets --cpu-only
```

### Markdown Output.

```bash
python scripts/benchmarkMemoryFootprint.py --dataset ucihar --output-format markdown
```

---

## Generating Publication Figures.

The `create_paper_figures.py` script generates all publication-quality figures.

```bash
python scripts/create_paper_figures.py
```

This creates the following figures in the output directory.

- Benchmark comparison grid (all models across all datasets and metrics).
- Ablation study grid (variants A0-A4 across datasets).
- Pareto efficiency frontier plot.
- Radar comparison chart.
- Efficiency heatmap.
- Quantization comparison chart.

---

## Sensor Data Visualization.

The `visualize_sensor_waves.py` script creates publication-quality visualizations of raw sensor waveforms from the UCI-HAR dataset.

```bash
python scripts/visualize_sensor_waves.py
```

This produces vertical wave plots showing accelerometer and gyroscope channels for different activity types, illustrating the signal patterns that the model learns to distinguish.

---

## Model Creation and Inference.

### Creating a MicroBiConvLSTM Model in Python.

```python
from models import createMicroBiConvLstm

# Create model for UCI-HAR dataset.
model = createMicroBiConvLstm('ucihar', dropout=0.15)

# Print model info.
info = model.getModelInfo()
for key, value in info.items():
    print(f"  {key}: {value}")
```

### Running Inference.

```python
import torch
from models import createMicroBiConvLstm

# Create and load trained model.
model = createMicroBiConvLstm('ucihar', dropout=0.15)
checkpoint = torch.load('checkpoints/microBiConvLstm_ucihar_seed42.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference on a single sample.
# Input shape: [batch_size, sequence_length, channels]
x = torch.randn(1, 128, 9)
with torch.no_grad():
    logits = model(x)
    predicted_class = logits.argmax(dim=-1).item()
    print(f"Predicted class: {predicted_class}")
```

### Creating Baseline Models.

```python
from baselines import DeepConvLSTM, TinyHAR, TinierHAR
from baselines.deepConvLstm import createDeepConvLstm
from baselines.tinyHar import createTinyHar
from baselines.tinierHar import createTinierHar

# Create baseline models for UCI-HAR.
dcl = createDeepConvLstm('ucihar')
thar = createTinyHar('ucihar')
tinier = createTinierHar('ucihar')
```

### Creating Ablation Variants.

```python
from models import makeVariantSpec, createVariantModel

# Create the A2 (Unidirectional) variant for UCI-HAR.
spec = makeVariantSpec(
    variantId='A2',
    numClasses=6,
    inChannels=9,
    seqLen=128,
    dropout=0.15,
)
model = createVariantModel(spec)
```

---

## Understanding the Results.

### Training Results.

The training results in `results/TrainingResults.md` and the JSON files in `results/training/` contain.

- **Test Accuracy:** Percentage of correctly classified samples.
- **Test F1 Score (Macro):** Macro-averaged F1 score across all classes. This is the primary evaluation metric.
- **Training Time:** Wall-clock training time in seconds.
- **Best Epoch:** The epoch at which the best F1 score was achieved.

All results report the mean and standard deviation across 5 random seeds.

### HPO Results.

The HPO results in `results/HPOResults.md` and the JSON files in `results/hpo/` contain.

- **Best F1 Score:** The highest F1 score achieved during HPO.
- **Optimal Hyperparameters:** The learning rate, weight decay, and dropout values that produced the best result.
- **Trial History:** All trial outcomes for reproducibility.

### Ablation Results.

The ablation results in `results/ablations/AblationsResults.md` contain.

- **Per-variant F1 scores** with standard deviations across seeds.
- **Parameter counts** and **MAC estimates** for each variant.
- **Pareto frontier data** showing optimal configurations.
- **Complexity scaling data** verifying O(N) behavior.
- **Quantization accuracy retention** for INT8 evaluation.

### Interpreting Figures.

- **Benchmark Comparison Grid:** Shows all models side by side across eight datasets. Use this to compare MicroBiConvLSTM against baselines on each metric.
- **Ablation Study Grid:** Shows how removing each component affects performance. The base model (A0) is the reference point.
- **Pareto Efficiency:** Points on the frontier represent configurations that cannot be improved in one metric without degrading the other.
- **Radar Chart:** Provides a holistic view of each model across multiple dimensions. Larger area generally indicates better overall performance.
