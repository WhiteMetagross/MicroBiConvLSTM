# MicroBiConvLSTM Codebase Index.

**Author:** Mridankan Mandal

**Paper:** [MicroBiConvLSTM on arXiv](https://arxiv.org/abs/2602.06523)

This document provides a complete index of every file in the repository, organized by directory. Each entry includes the file name, a description of its purpose, and the key classes or functions it exports.

---

## Table of Contents.

1. [Root Files](#root-files)
2. [models/ Directory](#models-directory)
3. [baselines/ Directory](#baselines-directory)
4. [data/ Directory](#data-directory)
5. [scripts/ Directory](#scripts-directory)
6. [results/ Directory](#results-directory)
7. [docs/ Directory](#docs-directory)

---

## Root Files.

| File | Description |
|------|-------------|
| `README.md` | Main project documentation with architecture overview, benchmark results, and quick start guide. |
| `CodeBaseIndex.md` | This file. Complete index of all source files and their purposes. |
| `Usage.md` | Detailed usage guide for training, HPO, ablation studies, and benchmarking. |
| `InstallationAndSetup.md` | Step-by-step installation and environment setup instructions. |
| `requirements.txt` | Python package dependencies required to run the project. |

---

## models/ Directory.

This directory contains the MicroBiConvLSTM model implementation and its ablation variants. The architecture is frozen for the research paper, meaning that structural hyperparameters (conv filters, LSTM hidden size, number of layers) are fixed. Only training hyperparameters (learning rate, weight decay, dropout) are tuned.

| File | Description | Key Exports |
|------|-------------|-------------|
| `__init__.py` | Package initialization. Imports and exposes all model classes and factory functions. | `MicroBiConvLSTM`, `createMicroBiConvLstm`, `MICRO_BI_CONV_LSTM_CONFIG`, `MicroBiConvLSTMVariant`, `MicroBiConvLSTMVariantSpec`, `createVariantModel`, `makeVariantSpec` |
| `microBiConvLstm.py` | The main MicroBiConvLSTM model implementation. Contains the frozen architecture specification with two Conv1D blocks, a bidirectional LSTM, temporal aggregation, and a linear classification head. Includes weight initialization, parameter counting, and a factory function for dataset-specific model creation. | `MicroBiConvLSTM` (class), `createMicroBiConvLstm` (factory), `MICRO_BI_CONV_LSTM_CONFIG` (dict) |
| `microBiConvLstmVariants.py` | Ablation variant implementations for controlled architectural studies. Provides variants A0 through A4 without modifying the frozen reference model. Each variant modifies a specific architectural component (pooling, directionality, number of conv blocks, aggregation method). | `MicroBiConvLSTMVariant` (class), `MicroBiConvLSTMVariantSpec` (dataclass), `makeVariantSpec` (function), `createVariantModel` (function) |

### Model Architecture Details.

The frozen configuration used across all datasets is:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `convFilters` | 16 | Number of convolutional filters per layer. |
| `convKernel` | 5 | Convolution kernel size (captures approximately 50ms context at 100Hz). |
| `convPadding` | 2 | Same padding to preserve temporal dimension through convolutions. |
| `lstmHidden` | 24 | LSTM hidden dimension per direction. |
| `lstmLayers` | 1 | Single LSTM layer. |
| `bidirectional` | True | Bidirectional LSTM for full temporal context. |
| `poolSize` | 2 | Max pooling kernel and stride (applied twice for 4x compression). |

### Ablation Variants.

| Variant ID | Name | Modification | Requires Retraining |
|------------|------|-------------|---------------------|
| A0 | Base Model | No modification (control). | No |
| A1 | No MaxPool | Removes pool1 and pool2. | Yes |
| A2 | Unidirectional | Sets bidirectional to False. | Yes |
| A3 | Single Conv | Removes the second conv block (conv2, bn2, pool2). | Yes |
| A4 | Mean Pooling | Changes aggregation from last timestep to mean pooling. | No (eval-only using A0 weights) |

---

## baselines/ Directory.

This directory contains reimplementations of three baseline HAR models used for comparison in the research paper. These do not include any Mamba-based models.

| File | Description | Key Exports |
|------|-------------|-------------|
| `__init__.py` | Package initialization. Imports TinyHAR, DeepConvLSTM, and TinierHAR. | `TinyHAR`, `DeepConvLSTM`, `TinierHAR` |
| `deepConvLstm.py` | DeepConvLSTM baseline. Four Conv1D layers followed by a two-layer unidirectional LSTM, matching the original Ordonez and Roggen (Sensors 2016) architecture. Approximately 132K parameters. | `DeepConvLSTM` (class), `createDeepConvLstm` (factory) |
| `tinyHar.py` | TinyHAR baseline. Channel-wise Conv2D feature extraction with cross-channel self-attention, channel fusion, temporal LSTM, and temporal weighted aggregation, following Zhou et al. (ISWC 2022). Approximately 42K parameters. | `TinyHAR` (class), `createTinyHar` (factory) |
| `tinierHar.py` | TinierHAR baseline. Ultra-lightweight architecture using depthwise separable convolutions with residual shortcuts and a bidirectional GRU, following the TinierHAR paper (UbiComp/ISWC 2025). Approximately 17K parameters. | `TinierHAR` (class), `createTinierHar` (factory) |

### Baseline Model Comparison.

| Model | Conv Type | Temporal Unit | Parameters | Reference |
|-------|----------|---------------|------------|-----------|
| DeepConvLSTM | 4x Conv1D (64 filters) | 2-layer unidirectional LSTM (64 hidden) | Approximately 132K | Ordonez and Roggen, Sensors 2016 |
| TinyHAR | 4x Conv2D (24 filters) + Self-Attention | 1-layer LSTM + Temporal Weighted Aggregation | Approximately 42K | Zhou et al., ISWC 2022 |
| TinierHAR | 4x Depthwise Separable Conv2D (8 filters) | 1-layer Bidirectional GRU (16 hidden) | Approximately 17K | TinierHAR, UbiComp/ISWC 2025 |

---

## data/ Directory.

This directory contains dataset loader implementations for all eight HAR benchmark datasets. Each loader handles downloading (where applicable), preprocessing, windowing, normalization, and creation of PyTorch DataLoader objects. Some datasets include Signal Rescue preprocessing (low-pass filtering) for noisy sensor data.

| File | Description | Key Exports |
|------|-------------|-------------|
| `__init__.py` | Package initialization. Imports all dataset classes and loader functions. | All dataset classes and loader functions listed below. |
| `uciHar.py` | UCI-HAR dataset loader. 9 channels (accelerometer + gyroscope), 128 timesteps, 6 activity classes. Uses the pre-segmented train/test split from the original dataset. | `UciHarDataset`, `getUciHarLoaders` |
| `motionSense.py` | MotionSense dataset loader. 6 channels (attitude + rotation rate), 128 timesteps, 6 activity classes. | `MotionSenseDataset`, `getMotionSenseLoaders` |
| `wisdm.py` | WISDM dataset loader. 3 channels (tri-axial accelerometer), 128 timesteps, 6 activity classes. | `WisdmDataset`, `getWisdmLoaders` |
| `pamap2.py` | PAMAP2 dataset loader. 19 channels (compact multi-IMU), 128 timesteps, 12 activity classes. Includes Signal Rescue low-pass filtering at 10Hz cutoff and class weight computation for imbalanced classes. | `Pamap2Dataset`, `getPamap2Loaders` |
| `opportunity.py` | Opportunity dataset loader. 79 channels (full body sensor network), 128 timesteps, 5 gesture classes. Includes class weight computation. | `OpportunityDataset`, `getOpportunityLoaders` |
| `unimib.py` | UniMiB-SHAR dataset loader. 3 channels (smartphone accelerometer), 128 timesteps, 9 activity classes (ADL and falls). | `UniMiBDataset`, `getUnimibLoaders` |
| `skoda.py` | Skoda dataset loader. 30 channels (body-worn accelerometers), 98 timesteps, 11 gesture classes. Uses Signal Rescue filtering at 5Hz cutoff and a stratified shuffle split for reproducibility. Includes class weight computation. | `SkodaDataset`, `getSkodaLoaders`, `computeSkodaClassWeights` |
| `daphnet.py` | Daphnet dataset loader. 9 channels (shank and thigh accelerometers), 64 timesteps, 2 classes (freeze/no-freeze). Includes Signal Rescue filtering at 12Hz cutoff and aggressive class weighting for extreme imbalance. | `DaphnetDataset`, `getDaphnetLoaders` |

### Dataset Summary Table.

| Dataset | Channels | Seq Length | Classes | Normalization | Signal Rescue Filter | Class Weights |
|---------|----------|-----------|---------|---------------|---------------------|---------------|
| UCI-HAR | 9 | 128 | 6 | Z-Score | No | No |
| MotionSense | 6 | 128 | 6 | Z-Score | No | No |
| WISDM | 3 | 128 | 6 | Z-Score | No | No |
| PAMAP2 | 19 | 128 | 12 | Robust Scaling | 10Hz LPF | Yes |
| Opportunity | 79 | 128 | 5 | Z-Score | No | Yes |
| UniMiB-SHAR | 3 | 128 | 9 | Z-Score | No | No |
| Skoda | 30 | 98 | 11 | Z-Score | 5Hz LPF | Yes |
| Daphnet | 9 | 64 | 2 | Z-Score | 12Hz LPF | Yes (aggressive) |

---

## scripts/ Directory.

This directory contains all executable scripts for training, hyperparameter optimization, ablation studies, benchmarking, and visualization.

| File | Description | Key Functions |
|------|-------------|---------------|
| `__init__.py` | Package initialization for scripts module. | N/A |
| `trainMicroBiConvLstm.py` | Training script for MicroBiConvLSTM. Supports multi-seed training with early stopping, AMP/FP16 acceleration, cosine annealing LR scheduling, and comprehensive logging. Uses dataset-specific HPO-tuned hyperparameters. | `trainModel`, `trainMultiSeed` (via CLI) |
| `trainBaselines.py` | Training script for baseline models (TinyHAR, TinierHAR, DeepConvLSTM). Same training protocol as MicroBiConvLSTM for fair comparison. | `trainModel`, `runMultiSeed` |
| `hpoMicroBiConvLstm.py` | Hyperparameter optimization for MicroBiConvLSTM using Optuna with TPE sampler. Tunes learning rate, weight decay, and dropout while keeping the architecture frozen. 50 trials per dataset with median pruning. | `runHPO`, `objective` |
| `hpoBaselines.py` | Hyperparameter optimization for baseline models using the same Optuna configuration as MicroBiConvLSTM HPO for fairness. | `runHPO`, `runAllHPO`, `objective` |
| `ablationStudiesMicroBiConvLstm.py` | Ablation study runner supporting five study types: architectural ablations (A0-A4), Pareto grid search (convFilters x lstmHidden), complexity scaling analysis, INT8 post-training quantization simulation, and sensitivity/robustness evaluations. | `study_arch`, `study_pareto`, `study_complexity`, `study_quant` |
| `benchmarkMemoryFootprint.py` | Memory footprint benchmarking script. Measures parameter count, FP32/INT8 model file sizes, state dictionary sizes, and peak inference memory for MicroBiConvLSTM and all baselines. | `benchmark_model`, `create_all_models` |
| `create_paper_figures.py` | Publication figure generation script. Creates TinyHAR-style benchmark comparison grids, ablation result plots, Pareto efficiency charts, radar comparisons, and efficiency heatmaps using matplotlib. | Multiple figure creation functions |
| `visualize_sensor_waves.py` | Sensor time series visualization script. Creates publication-quality plots of accelerometer and gyroscope data from UCI-HAR showing different activity patterns. | `create_vertical_wave_visualization` |

### Training Configuration.

All models use the following training protocol for fair comparison.

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR Scheduler | CosineAnnealingLR |
| Max Epochs | 200 |
| Early Stopping Patience | 10 epochs |
| Batch Size | 64 (512 for Skoda and Daphnet) |
| Gradient Clipping | Max norm 1.0 |
| AMP/FP16 | Enabled on CUDA |
| Optimization Target | Macro F1 Score |
| Seeds | 5 seeds from master seed 17 |

### HPO Configuration.

| Parameter | Value |
|-----------|-------|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Trials | 50 per dataset |
| Epochs per Trial | 50 |
| Pruner | Median Pruner (5 startup trials, 10 warmup steps) |
| Early Stopping | 5 epochs patience |
| Tuned Parameters | learning_rate [1e-4, 1e-2], weight_decay [1e-5, 0.05], dropout [0.0, 0.5] |

---

## results/ Directory.

This directory contains all experiment results in both Markdown summary format and raw JSON format.

### Markdown Result Files.

| File | Description |
|------|-------------|
| `TrainingResults.md` | Training results for MicroBiConvLSTM across all eight datasets with per-dataset accuracy, F1 score, and HPO-tuned hyperparameters. |
| `HPOResults.md` | Hyperparameter optimization results showing the best trial, optimal hyperparameters, and top-5 trials per dataset. |

### results/ablations/ Subdirectory.

| File | Description |
|------|-------------|
| `AblationsResults.md` | Complete ablation study results including architectural ablations (A0-A4), Pareto study, complexity scaling, INT8 quantization, and sensitivity analysis across all eight datasets. |
| `PublicationTables.md` | Auto-generated publication-ready tables for the research paper, including architectural ablation results, preprocessing summary, memory footprint comparison, and deployment suitability matrix. |
| `MemoryFootprintResults.md` | Detailed memory footprint benchmarks comparing FP32 and INT8 model sizes, state dictionary sizes, inference memory, and deployment recommendations for various target platforms. |

### results/hpo/ Subdirectory.

Contains JSON files with HPO trial results for each model-dataset combination. File naming convention: `hpo_{model}_{dataset}.json`.

Models covered: `MicroBiConvLSTM`, `deepconvlstm`, `tinyhar`, `tinierhar`.
Datasets covered: `ucihar`, `motionsense`, `wisdm`, `pamap2`, `opportunity`, `unimib`, `skoda`, `daphnet`.

### results/training/ Subdirectory.

Contains JSON files with multi-seed training results for each model-dataset combination. File naming convention: `{model}_{dataset}.json`.

Models covered: `MicroBiConvLSTM`, `deepconvlstm`, `tinyhar`, `tinierhar`.
Datasets covered: All eight benchmark datasets.

---

## docs/ Directory.

| File | Description |
|------|-------------|
| `Architecture.md` | Detailed technical architecture specification including stage-by-stage descriptions, parameter budget analysis, computational complexity analysis, LSTM mathematics, design rationale, and baseline comparison. |

### docs/figures/ Subdirectory.

Contains all publication-quality figures used in the research paper and documentation.

| File | Description |
|------|-------------|
| `MicroBiConvLSTM_Architecture.png` | Architecture diagram showing the five stages of MicroBiConvLSTM (Conv Stem, Conv Block, BiLSTM, Aggregation, Classification Head). |
| `benchmark_comparison_grid.png` | Grid figure comparing all models across all datasets on multiple metrics (parameters, MACs, FLOPs, F1, model size, efficiency ratios). |
| `ablation_study_grid.png` | Grid figure showing ablation variant results (A0-A4) across all eight datasets. |
| `ablation_absolute.png` | Absolute ablation results comparing each variant against the base model. |
| `pareto_efficiency.png` | Pareto efficiency frontier plot showing the trade-off between model size and F1 score. |
| `efficiency_heatmap.png` | Heatmap visualization of F1-per-parameter and F1-per-MAC efficiency ratios across models and datasets. |
| `radar_comparison.png` | Radar chart comparing MicroBiConvLSTM against baselines across multiple evaluation dimensions. |
| `quantization_comparison.png` | Comparison of FP32 and INT8 model sizes and accuracy retention after dynamic quantization. |
