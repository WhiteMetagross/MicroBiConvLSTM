# MicroBiConvLSTM: Complete Technical Architecture Specification:

**An Ultra-Lightweight Convolutional LSTM Architecture for Efficient Human Activity Recognition**.

---

## Table of Contents:

1. [Model Overview](#model-overview).
2. [Architecture Diagram](#architecture-diagram).
3. [Frozen Configuration](#frozen-configuration).
4. [Stage-by-Stage Architecture](#stage-by-stage-architecture).
5. [Parameter Budget Analysis](#parameter-budget-analysis).
6. [Computational Complexity Analysis](#computational-complexity-analysis).
7. [LSTM Mathematics](#lstm-mathematics).
8. [Design Rationale](#design-rationale).
9. [Dataset Preprocessing](#dataset-preprocessing).
10. [Baseline Comparison](#baseline-comparison).
11. [Implementation Details](#implementation-details).

---

## Model Overview:

### Identity:

| Property | Value |
|----------|-------|
| **Official Name** | MicroBiConvLSTM |
| **Architecture Family** | Convolutional LSTM |
| **Version** | 1.0 (Frozen Architecture) |
| **Target Application** | Human Activity Recognition (HAR) |
| **Deployment Target** | Edge devices, MCUs, Wearables, Mobile |
| **Core Design** | 2× Conv1d + Bidirectional LSTM |

### Key Features:

- **Ultra-Lightweight**: ~10.5K parameters (frozen across all datasets).
- **Linear Complexity**: O(N) time complexity with LSTM sequential processing.
- **Temporal Compression**: MaxPool reduces sequence length by 4×.
- **Bidirectional Context**: Full forward and backward temporal modeling.
- **Efficient Convolutions**: 16 filters with kernel size 5.
- **No Attention Required**: Pure convolutional + recurrent architecture.
- **Edge-Ready**: ~10.5 KB INT8 model fits in 64KB Flash.

### Comparison to DeepConvLSTM:

| Property | MicroBiConvLSTM | DeepConvLSTM |
|----------|-------------------|--------------|
| Conv Layers | 2 | 4 |
| Conv Filters | 16 | 64 |
| LSTM Layers | 1 | 2 |
| LSTM Hidden | 24 | 64 |
| Bidirectional | ✓ | ✗ (Unidirectional) |
| Pooling | MaxPool(2) × 2 | None |
| Parameters | ~10.5K | ~135K |
| MACs (avg) | ~485K | ~15.5M |

---

## Architecture Diagram:

Diagram image file: `docs/OriginalImg/MicroBiConvLSTM_Architecture.png`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MicroBiConvLSTM                                  │
│                  Target: ~10,454 params | O(N) Inference                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input[B, T, C]  ──────────────────────────────────────────────────────►    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE I: CONVOLUTIONAL STEM                        ~768 params      │    │
│  │   Transpose: (B, T, C) → (B, C, T)                                  │    │
│  │   Conv1D(C→16, k=5, padding=2) → BatchNorm → ReLU                   │    │
│  │   MaxPool1D(kernel=2, stride=2)                                     │    │
│  │   Purpose: Local edge detection (spikes/drops) for HAR inductive    │    │
│  │            bias                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼ (B, 16, T/2)                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE II: SECOND CONV BLOCK                        ~1,328 params    │    │
│  │   Conv1D(16→16, k=5, padding=2) → BatchNorm → ReLU                  │    │
│  │   MaxPool1D(kernel=2, stride=2)                                     │    │
│  │   Purpose: Feature refinement + temporal compression                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼ (B, 16, T/4)                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE III: BIDIRECTIONAL LSTM                      ~8,064 params    │    │
│  │   Transpose: (B, 16, T/4) → (B, T/4, 16)                            │    │
│  │   LSTM(input=16, hidden=24, layers=1, bidirectional=True)           │    │
│  │                                                                     │    │
│  │   Forward:  h_fwd ∈ ℝ^(B × T/4 × 24)                                │    │
│  │   Backward: h_bwd ∈ ℝ^(B × T/4 × 24)                                │    │
│  │   Output:   h_out ∈ ℝ^(B × T/4 × 48) [concatenated]                 │    │
│  │                                                                     │    │
│  │   Purpose: Full temporal context modeling with bidirectional scan   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE IV: TEMPORAL AGGREGATION                     0 params         │    │
│  │   Option A: Last timestep h[:, -1, :]                               │    │
│  │   Option B: Mean pooling (alternative)                              │    │
│  │   Output: (B, 48)                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE V: CLASSIFICATION HEAD                       ~294 params      │    │
│  │   Dropout(p) → Linear(48, num_classes)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  Output[B, num_classes]                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Frozen Configuration:

The architecture uses a **frozen configuration** that remains constant across all datasets. Only training hyperparameters (learning rate, weight decay, dropout) are tuned.

```python
MICRO_BI_CONV_LSTM_CONFIG = {
    # Convolutional Parameters
    'convFilters': 16,        # Number of convolutional filters
    'convKernel': 5,          # Convolution kernel size
    'convPadding': 2,         # Same padding for conv layers
    'numConvLayers': 2,       # Number of conv blocks
    
    # Pooling Parameters
    'poolSize': 2,            # Max pool kernel size
    'poolStride': 2,          # Pooling stride (2× compression per layer)
    
    # LSTM Parameters
    'lstmHidden': 24,         # LSTM hidden dimension
    'lstmLayers': 1,          # Number of LSTM layers
    'bidirectional': True,    # Bidirectional LSTM
    
    # Output Dimension
    'lstmOutput': 48,         # 24 × 2 (bidirectional)
}
```

### Why These Values?:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `convFilters=16` | 16 | Minimal feature extraction with ~10-11K param budget |
| `convKernel=5` | 5 | Captures ~50ms temporal context at 100Hz |
| `lstmHidden=24` | 24 | Sufficient temporal memory for HAR activities |
| `lstmLayers=1` | 1 | Single layer balances capacity vs efficiency |
| `bidirectional=True` | True | Full temporal context without 2× params (weight sharing) |
| `poolSize=2×2` | 4× total | Reduces sequence 128→32 for LSTM efficiency |

---

## Stage-by-Stage Architecture:

### Stage I: Convolutional Stem:

**Purpose**: Extract low-level temporal features from multi-channel sensor input.

```
Input:  x ∈ ℝ^(B × T × C)     [Batch, Time=128, Channels]
        |
        v
+-----------------------------------------------+
|  Transpose: (B, T, C) → (B, C, T)             |
|  Channels-first for Conv1D                    |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  Conv1D(in_channels=C, out_channels=16,       |
|         kernel_size=5, padding=2, bias=False) |
|  Extracts 16 feature maps from all sensors    |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  BatchNorm1D(16)                              |
|  Stabilizes training, normalizes activations  |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  ReLU activation                              |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  MaxPool1D(kernel_size=2, stride=2)           |
|  Temporal compression: T → T/2                |
+-----------------------------------------------+
        |
        v
Output: h1 ∈ ℝ^(B × 16 × T/2)   [Batch, Features=16, Time=64]
```

**Parameter Count** (for UCI-HAR with C=9):.
- Conv1D weights: 9 × 16 × 5 = 720.
- BatchNorm: 16 × 2 = 32.
- **Total Stage I**: ~752 parameters.

---

### Stage II: Second Convolutional Block:

**Purpose**: Further feature refinement and temporal compression.

```
Input:  h1 ∈ ℝ^(B × 16 × T/2)   [Batch, Features=16, Time=64]
        |
        v
+-----------------------------------------------+
|  Conv1D(in_channels=16, out_channels=16,      |
|         kernel_size=5, padding=2, bias=False) |
|  Refines feature representations              |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  BatchNorm1D(16)                              |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  ReLU activation                              |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  MaxPool1D(kernel_size=2, stride=2)           |
|  Temporal compression: T/2 → T/4              |
+-----------------------------------------------+
        |
        v
Output: h2 ∈ ℝ^(B × 16 × T/4)   [Batch, Features=16, Time=32]
```

**Parameter Count**:.
- Conv1D weights: 16 × 16 × 5 = 1,280.
- BatchNorm: 16 × 2 = 32.
- **Total Stage II**: ~1,312 parameters.

---

### Stage III: Bidirectional LSTM:

**Purpose**: Model long-range temporal dependencies with bidirectional context.

```
Input:  h2 ∈ ℝ^(B × 16 × T/4)   [Batch, Features=16, Time=32]
        |
        v
+-----------------------------------------------+
|  Transpose: (B, 16, T/4) → (B, T/4, 16)       |
|  Sequence-first format for LSTM               |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  LSTM(input_size=16, hidden_size=24,          |
|       num_layers=1, bidirectional=True,       |
|       batch_first=True)                       |
|                                               |
|  Forward LSTM:  h_fwd ∈ ℝ^(B × T/4 × 24)      |
|  Backward LSTM: h_bwd ∈ ℝ^(B × T/4 × 24)      |
|  Concatenate:   h_out ∈ ℝ^(B × T/4 × 48)      |
+-----------------------------------------------+
        |
        v
Output: h3 ∈ ℝ^(B × T/4 × 48)   [Batch, Time=32, Features=48]
```

**Parameter Count** (Bidirectional LSTM):.
- Input weights (forward):  4 × 24 × 16 = 1,536.
- Hidden weights (forward): 4 × 24 × 24 = 2,304.
- Biases (forward):         4 × 24 × 2 = 192.
- Forward total:            4,032.
- **Bidirectional total**:  4,032 × 2 = **8,064 parameters**.

---

### Stage IV: Temporal Aggregation:

**Purpose**: Collapse temporal dimension to fixed-size representation.

```
Input:  h3 ∈ ℝ^(B × T/4 × 48)   [Batch, Time=32, Features=48]
        |
        v
+-----------------------------------------------+
|  Option A: Last Timestep (default)            |
|  h_out = h3[:, -1, :]                         |
|  Uses final hidden state (captures full seq)  |
|                                               |
|  Option B: Mean Pooling (alternative)         |
|  h_out = mean(h3, dim=1)                      |
|  Aggregates all timestep representations      |
+-----------------------------------------------+
        |
        v
Output: h4 ∈ ℝ^(B × 48)   [Batch, Features=48]
```

**Note**: The bidirectional LSTM's last timestep contains:.
- Forward direction: information about entire sequence (processed left-to-right).
- Backward direction: information about entire sequence (processed right-to-left).

---

### Stage V: Classification Head:

**Purpose**: Map feature representation to class logits.

```
Input:  h4 ∈ ℝ^(B × 48)     [Batch, Features=48]
        |
        v
+-----------------------------------------------+
|  Dropout(p=dropout_rate)                      |
|  Regularization during training               |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  Linear(in_features=48, out_features=C)       |
|  C = number of activity classes               |
+-----------------------------------------------+
        |
        v
Output: logits ∈ ℝ^(B × num_classes)
```

**Parameter Count** (varies by dataset):.
- Weights: 48 × num_classes.
- Bias: num_classes.
- **Total Head**: 49 × num_classes.

| Dataset | Classes | Head Parameters |
|---------|---------|-----------------|
| UCI-HAR | 6 | 294 |
| MotionSense | 6 | 294 |
| WISDM | 6 | 294 |
| PAMAP2 | 12 | 588 |
| Opportunity | 5 | 245 |
| UniMiB | 9 | 441 |
| SKODA | 11 | 539 |
| Daphnet | 2 | 98 |

---

## Parameter Budget Analysis:

### Component Breakdown (UCI-HAR Example: C=9, Classes=6):

| Component | Formula | Parameters |
|-----------|---------|------------|
| Conv1D-1 (Stem) | C × 16 × 5 + 16 (bias) | 736 |
| BatchNorm-1 | 16 × 2 | 32 |
| Conv1D-2 | 16 × 16 × 5 + 16 (bias) | 1,296 |
| BatchNorm-2 | 16 × 2 | 32 |
| LSTM (BiDir) | 2 × 4 × 24 × (16 + 24 + 1) | 7,872 |
| Linear Head | 48 × 6 + 6 | 294 |
| **Total** | | **10,454** |

**Note**: Conv1D layers use `bias=True` to match baselines implementation.

### Total Parameter Count by Dataset:

| Dataset | Input Channels | Classes | **Total Params** |
|---------|----------------|---------|------------------|
| UCI-HAR | 9 | 6 | **10,454** |
| MotionSense | 6 | 6 | **10,214** |
| WISDM | 3 | 6 | **9,974** |
| PAMAP2 | 19 | 12 | **11,548** |
| Opportunity | 79 | 5 | **16,005** |
| UniMiB | 3 | 9 | **10,121** |
| SKODA | 30 | 11 | **12,379** |
| Daphnet | 9 | 2 | **10,258** |

**Note**: Parameter count varies primarily due to:.
- First Conv1D input channels (dataset-dependent).
- Classification head output size (class-dependent).

*Values verified using hook-based benchmark calculation (scripts/calculateBenchmarksVerified.py)*

---

## Computational Complexity Analysis:

### Verified MACs per Dataset:

| Dataset | Seq Len | **Total MACs** | **Total FLOPs** |
|---------|---------|----------------|------------------|
| UCI-HAR | 128 | **420.1K** | **840.2K** |
| MotionSense | 128 | **389.4K** | **778.8K** |
| WISDM | 128 | **358.7K** | **717.4K** |
| PAMAP2 | 128 | **523.0K** | **1.05M** |
| Opportunity | 128 | **1.14M** | **2.28M** |
| UniMiB | 128 | **358.9K** | **717.8K** |
| SKODA | 98 | **443.9K** | **887.8K** |
| Daphnet | 64 | **245.4K** | **490.8K** |

*Values measured using hook-based MAC counting (scripts/calculateBenchmarksVerified.py)*

### MAC Calculation Formulas:

**Conv1D MACs**:.
```
MACs = out_channels × out_length × in_channels × kernel_size
     = 16 × (T) × C × 5        # Stage I
     = 16 × (T/2) × 16 × 5     # Stage II
```

**LSTM MACs** (Bidirectional):.
```
MACs = 2 × seq_len × 4 × hidden × (input + hidden)
     = 2 × (T/4) × 4 × 24 × (16 + 24)
     = 2 × 32 × 4 × 24 × 40 = 245,760 (for T=128)
```

**Linear MACs**:.
```
MACs = in_features × out_features
     = 48 × num_classes
```

### FLOPs (2× MACs):

FLOPs values are calculated as 2× MACs. See the Verified MACs per Dataset table above for computed values.

---

## LSTM Mathematics:

### LSTM Cell Computation:

The bidirectional LSTM processes each timestep with the following gates:.

```
Forward pass (per timestep t):
  i_t = σ(W_ii·x_t + b_ii + W_hi·h_{t-1} + b_hi)   # Input gate
  f_t = σ(W_if·x_t + b_if + W_hf·h_{t-1} + b_hf)   # Forget gate
  g_t = tanh(W_ig·x_t + b_ig + W_hg·h_{t-1} + b_hg) # Cell gate
  o_t = σ(W_io·x_t + b_io + W_ho·h_{t-1} + b_ho)   # Output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                  # Cell state
  h_t = o_t ⊙ tanh(c_t)                            # Hidden state
```

Where:
- $\sigma$ is the sigmoid function.
- $\odot$ denotes element-wise multiplication.
- $W_{*i}$ are input-to-hidden weights (shape: hidden × input).
- $W_{*h}$ are hidden-to-hidden weights (shape: hidden × hidden).
- $b_{*}$ are biases (shape: hidden).

### Bidirectional Processing:

```
Forward LSTM:   h_fwd[t] = LSTM_fwd(x[0], x[1], ..., x[t])
Backward LSTM:  h_bwd[t] = LSTM_bwd(x[T], x[T-1], ..., x[t])
Output:         h[t] = concat(h_fwd[t], h_bwd[t])
```

---

## Design Rationale:

### Why 2 Conv Layers vs 4 (DeepConvLSTM)?:

| Aspect | 2 Layers | 4 Layers |
|--------|----------|----------|
| Feature Hierarchy | Basic → Refined | Basic → Mid → High → Abstract |
| Receptive Field | 9 timesteps | 17 timesteps |
| Parameters | ~2K | ~8K (conv only) |
| **Trade-off** | Sufficient for most HAR | Better for complex gestures |

For typical HAR activities (walking, running, sitting), 2 layers provide adequate feature extraction while keeping the model ultra-lightweight.

### Why MaxPool After Each Conv?:

| Benefit | Description |
|---------|-------------|
| **Temporal Compression** | 128 → 64 → 32 (4× reduction) |
| **LSTM Efficiency** | Shorter sequences = faster training |
| **Translation Invariance** | Reduces sensitivity to exact timing |
| **Parameter Reduction** | Smaller sequence = fewer LSTM computations |

### Why Single-Layer Bidirectional LSTM?:

| Design Choice | Rationale |
|---------------|-----------|
| **1 Layer** | Sufficient temporal modeling for ~3s windows |
| **Bidirectional** | Activities have symmetric temporal patterns |
| **Hidden=24** | Matches d_model of NanoHARMamba for fair comparison |
| **No Stacking** | Avoids overfitting on small HAR datasets |

### Why Last Timestep Aggregation?:

| Method | Pros | Cons |
|--------|------|------|
| **Last Timestep** | Contains full bidirectional context | May miss some patterns |
| Mean Pooling | Robust to timing variations | Dilutes important features |
| Attention | Learns importance weights | Adds parameters |

Last timestep is chosen for simplicity and efficiency—the bidirectional LSTM already captures full sequence context.

---

## Dataset Preprocessing:

Each dataset requires specific preprocessing to handle sensor noise, sampling rates, and class imbalance. The following table summarizes the preprocessing pipeline for each dataset.

### Preprocessing Summary:

| Dataset | Sampling Rate | Filter | Normalization | Windowing | Special Handling |
|---------|---------------|--------|---------------|-----------|------------------|
| **UCI-HAR** | 50 Hz | None | Z-Score | Pre-segmented (128) | Standard normalization |
| **MotionSense** | 50 Hz | None | Z-Score | 128, stride 64 | Subject-wise split |
| **WISDM** | 20 Hz | None | Z-Score | 128, stride 64 | User-wise split |
| **PAMAP2** | 100 Hz | **10 Hz Low-Pass** | **Robust Scaling** | 128, stride 64 | Signal Rescue pipeline |
| **Opportunity** | 30 Hz | None | Z-Score | 128, stride 64 | 79 body-worn channels |
| **UniMiB** | 50 Hz | None | Z-Score | 128 (fixed) | ADL only (9 classes) |
| **SKODA** | 98 Hz | **5 Hz Low-Pass** | Z-Score | 98, stride 24 | Signal Rescue pipeline |
| **Daphnet** | 64 Hz | **12 Hz Low-Pass** | Z-Score | 64, stride 32 | Signal Rescue + Class weights |

### Detailed Preprocessing per Dataset:

#### UCI-HAR (Standard Benchmark):
```
Preprocessing Pipeline:
├── Raw Data: 9 channels (body_acc_xyz, body_gyro_xyz, total_acc_xyz)
├── Sampling: 50 Hz
├── Window: Pre-segmented at 128 timesteps (2.56 sec)
├── Filter: None (data is already filtered by dataset authors)
├── Normalization: Channel-wise Z-Score (μ=0, σ=1)
└── Split: Subject-based (21 train / 9 test subjects)
```

#### MotionSense (In-the-Wild iPhone Data):
```
Preprocessing Pipeline:
├── Raw Data: 12 channels (acc_xyz, gyro_xyz, attitude_rpy, gravity_xyz)
├── Subset Used: 6 channels (accelerometer + gyroscope only)
├── Sampling: 50 Hz
├── Window: 128 samples, stride 64 (50% overlap)
├── Filter: None (natural phone movement noise preserved)
├── Normalization: Channel-wise Z-Score per split
└── Split: Subject-based (80% train / 20% test)
```

#### WISDM (Large-Scale Accelerometer):
```
Preprocessing Pipeline:
├── Raw Data: 3 channels (acc_x, acc_y, acc_z)
├── Sampling: 20 Hz (low sampling rate)
├── Window: 128 samples (~6.4 sec), stride 64
├── Filter: None
├── Normalization: Channel-wise Z-Score per split
└── Split: User-based (80% train / 20% test)
```

#### PAMAP2 (Multi-IMU Signal Rescue):
```
Signal Rescue Pipeline:
├── Raw Data: 52 channels → 19 channels used (3 IMUs: hand, chest, ankle)
├── Sampling: 100 Hz
├── STEP 1 - Low-Pass Filter:
│   ├── Type: 4th Order Butterworth
│   ├── Cutoff: 10 Hz
│   └── Purpose: Remove impact spikes from Running/Jumping activities
├── STEP 2 - Robust Scaling:
│   ├── Method: IQR-based normalization (instead of Z-Score)
│   ├── Formula: (x - median) / IQR
│   └── Purpose: Prevent outliers from compressing static activities
├── Window: 128 samples, stride 64
├── Class Weights: Enabled (inverse frequency)
└── Split: Subject-based leave-one-out or 80/20
```

**Why Signal Rescue for PAMAP2?**.
- Running and jumping create sharp acceleration spikes (high-frequency artifacts).
- These spikes dominate the signal and obscure patterns from static activities.
- 10 Hz low-pass converts impact spikes into smooth rhythmic curves.
- Robust Scaling prevents outliers from compressing Lying/Sitting into flat lines.

#### Opportunity (High-Dimensional Multi-Sensor):
```
Preprocessing Pipeline:
├── Raw Data: 113 channels → 79 body-worn IMU channels
├── Removed: Ambient sensors, object sensors
├── Sampling: 30 Hz
├── Window: 128 samples, stride 64
├── Filter: None
├── Normalization: Channel-wise Z-Score
├── Task: Locomotion (5 classes: Null, Stand, Walk, Sit, Lie)
└── Split: Subject-based (4 train / 1 test)
```

#### UniMiB-SHAR (Smartphone ADL + Falls):
```
Preprocessing Pipeline:
├── Raw Data: 3 channels (accelerometer x, y, z)
├── Sampling: 50 Hz
├── Window: 128 samples (fixed, pre-segmented)
├── Filter: None
├── Normalization: Channel-wise Z-Score
├── Task: ADL only (9 classes, excluding 8 fall types)
└── Split: Subject-based (24 train / 6 test subjects)
```

#### SKODA (Industrial Signal Rescue):
```
Signal Rescue Pipeline:
├── Raw Data: 30 channels (10 sensors × 3 axes, right arm)
├── Sampling: 98 Hz
├── STEP 1 - Low-Pass Filter:
│   ├── Type: 4th Order Butterworth
│   ├── Cutoff: 5 Hz
│   └── Purpose: Remove industrial machine vibration (50-60 Hz)
├── STEP 2 - Z-Score Normalization:
│   └── Applied AFTER filtering (critical ordering)
├── Window: 98 samples (~1 sec), stride 24 (75% overlap for train)
├── Label Smoothing: 0.1 (fuzzy gesture boundaries)
├── Batch Size: 512 (larger for stability)
└── Split: Time-based (first 80% train, last 20% test)
```

**Why Signal Rescue for SKODA?**.
- Industrial sensors pick up machine vibration at 50-60 Hz.
- Human arm manipulation movements are < 2-3 Hz.
- 5 Hz cutoff removes vibration while preserving gesture dynamics.
- Prevents aliasing when stride-4 tokenization is applied.

#### Daphnet (Parkinson's Freeze Detection):
```
Signal Rescue Pipeline:
├── Raw Data: 9 channels (3 IMUs: ankle, thigh, trunk × 3 axes)
├── Sampling: 64 Hz
├── STEP 1 - Low-Pass Filter:
│   ├── Type: 4th Order Butterworth
│   ├── Cutoff: 12 Hz
│   └── Purpose: Remove sensor jitter (20-30 Hz)
├── STEP 2 - Z-Score Normalization:
│   └── Applied AFTER filtering
├── Window: 64 samples (1 sec), stride 32
├── Class Weights: 15.0 for Freeze, 1.0 for Walk (extreme imbalance)
├── Batch Size: 512
└── Split: Subject-based leave-one-out
```

**Why Signal Rescue for Daphnet?**.
- The "Freeze" frequency band is 3-8 Hz.
- Sensor jitter at 20-30 Hz causes aliasing artifacts.
- 12 Hz cutoff preserves freeze patterns while removing noise.
- Extreme class weights address ~10% Freeze vs ~90% Walk imbalance.

### Normalization Methods Comparison:

| Method | Formula | Best For | Used In |
|--------|---------|----------|---------|
| **Z-Score** | $(x - \mu) / \sigma$ | Standard HAR | UCI-HAR, MotionSense, WISDM, Opportunity, UniMiB, SKODA, Daphnet |
| **Robust Scaling** | $(x - median) / IQR$ | Outlier-heavy data | PAMAP2 |
| **Min-Max** | $(x - min) / (max - min)$ | Bounded ranges | Not used |

### Butterworth Filter Parameters:

| Dataset | Cutoff (Hz) | Order | Purpose |
|---------|-------------|-------|---------|
| PAMAP2 | 10 Hz | 4 | Remove impact spikes from running/jumping |
| SKODA | 5 Hz | 4 | Remove industrial machine vibration (50-60 Hz) |
| Daphnet | 12 Hz | 4 | Remove sensor jitter (20-30 Hz) |

### Training Configuration per Dataset:

| Dataset | Batch Size | Epochs | Early Stop | Class Weights | Special |
|---------|------------|--------|------------|---------------|---------|
| UCI-HAR | 64 | 200 | 10 | No | - |
| MotionSense | 64 | 200 | 10 | No | - |
| WISDM | 64 | 200 | 10 | No | - |
| PAMAP2 | 64 | 200 | 10 | **Yes** | Robust Scaling |
| Opportunity | 64 | 200 | 10 | No | - |
| UniMiB | 64 | 200 | 10 | No | - |
| SKODA | **512** | 200 | 10 | Optional | Label Smoothing 0.1 |
| Daphnet | **512** | 200 | 10 | **Yes (15:1)** | Extreme imbalance |

---

## Baseline Comparison:

### Models Overview:

| Model | Architecture | Avg Params | Key Features |
|-------|--------------|------------|--------------|
| **MicroBiConvLSTM** | 2× Conv1d + BiLSTM | ~11.4K | Ultra-lightweight, O(N) complexity |
| TinierHAR | 4× DW-Sep Conv2d + BiGRU | ~33.5K | Depthwise separable, attention aggregation |
| TinyHAR | 4× Conv2d + Unidirectional LSTM | ~55.1K | Cross-channel attention, temporal attention |
| DeepConvLSTM | 4× Conv1d + 2-layer LSTM | ~135.6K | Classic baseline, high capacity |

*Values verified using hook-based benchmark calculation (scripts/calculateBenchmarksVerified.py)*

### Performance Comparison (F1 Score %) — Mean ± Std:

| Dataset | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|---------|-------------------|--------------|---------|-----------|
| UCI-HAR | 93.41 ± 0.35 | 93.53 ± 0.26 | **96.53 ± 0.41** | 96.37 ± 0.57 |
| MotionSense | 91.65 ± 0.43 | 92.90 ± 0.96 | 92.67 ± 0.67 | **91.99 ± 0.60** |
| WISDM | 73.17 ± 12.42 | 81.84 ± 1.46 | 77.09 ± 4.95 | **83.06 ± 3.24** |
| PAMAP2 | 60.75 ± 1.76 | 67.79 ± 1.50 | **73.22 ± 3.58** | 74.07 ± 1.16 |
| Opportunity | 87.58 ± 0.73 | 88.30 ± 0.72 | **88.69 ± 0.38** | 87.09 ± 0.90 |
| UniMiB | 79.43 ± 1.66 | **85.83 ± 1.22** | 77.61 ± 2.23 | 79.67 ± 4.45 |
| SKODA | 94.46 ± 1.31 | 94.63 ± 2.47 | **97.01 ± 0.53** | 96.99 ± 0.76 |
| Daphnet | 88.98 ± 1.64 | 88.95 ± 2.26 | 86.42 ± 3.64 | **89.84 ± 1.90** |
| **Average** | **83.68%** | **85.93%** | **86.16%** | **87.39%** |

### Accuracy Comparison (%) — Mean ± Std:

| Dataset | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|---------|-------------------|--------------|---------|-----------|
| UCI-HAR | 93.33 ± 0.35 | 93.44 ± 0.25 | **96.46 ± 0.43** | 96.30 ± 0.59 |
| MotionSense | 92.71 ± 0.46 | **94.12 ± 1.15** | 94.00 ± 0.97 | 93.28 ± 0.68 |
| WISDM | 81.73 ± 2.18 | 83.17 ± 2.02 | 83.83 ± 1.83 | **86.35 ± 0.94** |
| PAMAP2 | 65.79 ± 1.40 | 67.50 ± 1.78 | **74.98 ± 3.38** | 77.45 ± 1.52 |
| Opportunity | 86.62 ± 0.90 | 86.74 ± 0.74 | **87.45 ± 0.41** | 86.19 ± 0.78 |
| UniMiB | 91.13 ± 1.37 | **92.71 ± 0.95** | 90.49 ± 0.84 | 90.30 ± 0.79 |
| SKODA | 94.39 ± 1.35 | 94.88 ± 2.06 | **97.14 ± 0.49** | 96.88 ± 0.69 |
| Daphnet | 97.37 ± 0.35 | 97.37 ± 0.48 | 96.44 ± 1.60 | **97.41 ± 0.62** |
| **Average** | **87.88%** | **88.74%** | **90.10%** | **90.52%** |

---

## MACs and FLOPs Comparison:

### Per-Dataset MACs (Multiply-Accumulate Operations):

| Dataset | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|---------|-------------------|--------------|---------|-----------|
| UCI-HAR | **420.1K** | 16.62M | 6.05M | 1.08M |
| MotionSense | **389.4K** | 16.49M | 4.27M | 740.3K |
| WISDM | **358.7K** | 16.37M | 2.48M | 402.0K |
| PAMAP2 | **523.0K** | 17.03M | 12.01M | 2.21M |
| Opportunity | **1.14M** | 19.48M | 47.77M | 8.97M |
| UniMiB | **358.9K** | 16.37M | 2.48M | 402.1K |
| SKODA | **443.9K** | 13.29M | 12.36M | 1.91M |
| Daphnet | **245.4K** | 8.11M | 2.64M | 399.0K |
| **Average** | **485K** | **15.51M** | **9.29M** | **1.73M** |

*Values verified using hook-based benchmark calculation*

### Per-Dataset FLOPs (Floating-Point Operations = 2× MACs):

| Dataset | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|---------|-------------------|--------------|---------|-----------|
| UCI-HAR | **840.2K** | 33.24M | 12.11M | 2.16M |
| MotionSense | **778.8K** | 32.98M | 8.53M | 1.48M |
| WISDM | **717.4K** | 32.74M | 4.96M | 804.0K |
| PAMAP2 | **1.05M** | 34.05M | 24.03M | 4.41M |
| Opportunity | **2.28M** | 38.96M | 95.54M | 17.94M |
| UniMiB | **717.8K** | 32.74M | 4.96M | 804.2K |
| SKODA | **887.8K** | 26.57M | 24.72M | 3.82M |
| Daphnet | **490.8K** | 16.22M | 5.28M | 798.0K |
| **Average** | **970K** | **31.02M** | **18.57M** | **3.46M** |

*Values verified using hook-based benchmark calculation*

### Per-Dataset Parameters:

| Dataset | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|---------|-------------------|--------------|---------|-----------|
| UCI-HAR | **10,454** | 132,038 | 42,704 | 16,931 |
| MotionSense | **10,214** | 131,078 | 39,248 | 12,323 |
| WISDM | **9,974** | 130,118 | 35,792 | 7,715 |
| PAMAP2 | **11,548** | 135,628 | 54,518 | 32,489 |
| Opportunity | **16,005** | 154,373 | 123,295 | 124,418 |
| UniMiB | **10,121** | 130,313 | 35,939 | 7,814 |
| SKODA | **12,379** | 139,083 | 67,141 | 49,352 |
| Daphnet | **10,258** | 131,778 | 42,508 | 16,799 |
| **Average** | **11.4K** | **135.6K** | **55.1K** | **33.5K** |

*Values verified using hook-based benchmark calculation*

---

## Efficiency Analysis:

### Computational Efficiency Summary:

| Model | Avg Params | Avg MACs | Avg FLOPs | Avg F1 | Complexity |
|-------|------------|----------|-----------|--------|------------|
| **MicroBiConvLSTM** | **11.4K** | **485K** | **970K** | 83.68% | O(N) |
| TinierHAR | 33.5K | 1.73M | 3.46M | 87.39% | O(N²) |
| TinyHAR | 55.1K | 9.29M | 18.57M | 86.16% | O(N²) |
| DeepConvLSTM | 135.6K | 15.51M | 31.02M | 85.93% | O(N) |

*Values verified using hook-based benchmark calculation*

### Efficiency Metrics:

| Metric | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|--------|-------------------|--------------|---------|-----------|
| **F1 per K-Params** | **7.34** | 0.63 | 1.56 | 2.61 |
| **F1 per M-MACs** | **172.5** | 5.54 | 9.27 | 50.5 |
| **Params Reduction vs DeepConvLSTM** | **11.9×** | 1× | 2.5× | 4.0× |
| **MACs Reduction vs DeepConvLSTM** | **32.0×** | 1× | 1.7× | 9.0× |

*Efficiency metrics based on verified benchmark values*

### Model Size and Deployment:

| Model | Parameters | FP32 Size | INT8 Size | Flash Fit (64KB) |
|-------|------------|-----------|-----------|------------------|
| **MicroBiConvLSTM** | ~11.4K | ~46 KB | ~11 KB | ✓ Yes |
| TinierHAR | ~33.5K | ~134 KB | ~34 KB | ✓ Yes (INT8) |
| TinyHAR | ~55.1K | ~220 KB | ~55 KB | ✓ Yes (INT8) |
| DeepConvLSTM | ~135.6K | ~544 KB | ~136 KB | ✗ No |

### Latency Comparison (Inference, CPU):

| Dataset | MicroBiConvLSTM | DeepConvLSTM | TinyHAR | TinierHAR |
|---------|-------------------|--------------|---------|-----------|
| UCI-HAR | **0.49 ms** | 2.87 ms | 1.01 ms | 0.98 ms |
| MotionSense | **0.73 ms** | 2.65 ms | 1.66 ms | 0.92 ms |
| WISDM | **0.51 ms** | 2.52 ms | 1.03 ms | 1.28 ms |
| PAMAP2 | **0.49 ms** | 3.21 ms | 1.07 ms | 1.08 ms |
| Opportunity | **0.47 ms** | 4.35 ms | 2.47 ms | 1.35 ms |
| UniMiB | **1.08 ms** | 2.48 ms | 1.07 ms | 1.40 ms |

---

## MicroBiConvLSTM Strengths and Weaknesses:

### Strengths:

- ✓ **Ultra-low parameter count** (~11.4K average, verified).
- ✓ **Lowest MACs** (~485K average, 32× fewer than DeepConvLSTM).
- ✓ **Fastest inference** (0.47-1.08 ms on CPU).
- ✓ **No attention mechanism** (simpler implementation).
- ✓ **Bidirectional temporal context** (full sequence modeling).
- ✓ **Fits in 64KB Flash** (INT8 quantized: ~11 KB).
- ✓ **O(N) complexity** (linear with sequence length).
- ✓ **Works well on smartphone IMU data** (UCI-HAR, MotionSense).
- ✓ **Competitive on Daphnet** (97.37% accuracy, 88.98% F1).

### Weaknesses:

- ✗ **LSTM sequential nature** limits parallelization on GPU.
- ✗ **No cross-channel attention** (sensors processed jointly).
- ✗ **Fixed receptive field** from 2× Conv1D layers.
- ✗ **Less effective on multi-IMU setups** (PAMAP2: 60.75% F1).
- ✗ **Lower average F1** than baselines (-3.71% vs TinierHAR).
- ✗ **High variance on WISDM** (F1: 73.17 ± 12.42%).

### When to Use MicroBiConvLSTM:

| Scenario | Recommendation |
|----------|----------------|
| **Extreme resource constraints** (MCU, 64KB Flash) | ✓ **Best choice** |
| **Battery-limited wearables** | ✓ **Best choice** |
| **Real-time HAR on edge** | ✓ **Best choice** |
| **Smartphone IMU (3-9 channels)** | ✓ Good choice |
| **Multi-IMU body sensors (PAMAP2)** | ✗ Use TinyHAR/TinierHAR |
| **Maximum accuracy needed** | ✗ Use TinyHAR |
| **Best efficiency-accuracy tradeoff** | Use TinierHAR |

---

## Implementation Details:

### PyTorch Implementation:

```python
class MicroBiConvLSTM(nn.Module):
    def __init__(self, inChannels: int, numClasses: int, seqLen: int = 128,
                 convFilters: int = 16, lstmHidden: int = 24, dropout: float = 0.1):
        super().__init__()
        
        # Stage I: Conv Stem
        self.conv1 = nn.Conv1d(inChannels, convFilters, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(convFilters)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Stage II: Conv Block
        self.conv2 = nn.Conv1d(convFilters, convFilters, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(convFilters)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Stage III: BiDir LSTM
        self.lstm = nn.LSTM(
            input_size=convFilters,
            hidden_size=lstmHidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Stage IV & V: Classification
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstmHidden * 2, numClasses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]
        
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # [B, T/4, 16]
        x, _ = self.lstm(x)     # [B, T/4, 48]
        
        # Classification (last timestep)
        x = x[:, -1, :]         # [B, 48]
        x = self.dropout(x)
        x = self.classifier(x)  # [B, num_classes]
        
        return x
```

### Weight Initialization:

```python
def _initWeights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
```

---

## Training Commands:

```bash
# Train MicroBiConvLSTM on all datasets
python scripts/trainMicroBiConvLSTM.py --dataset all --seeds 5 --epochs 200

# Single dataset training
python scripts/trainMicroBiConvLSTM.py --dataset ucihar --seeds 5

# HPO for hyperparameter tuning
python scripts/hpoMicroBiConvLSTM.py --dataset ucihar --n-trials 50
```

---

## Citation:

```bibtex
@article{MicroBiConvLSTM_2026,
  title={MicroBiConvLSTM: An Ultra-Lightweight Convolutional LSTM 
         for Efficient Human Activity Recognition},
  author={...},
  journal={...},
  year={2026}
}
```

---

*Document Version: 1.0 | Last Updated: January 2026*
*Architecture: FROZEN | Training Hyperparameters Only: lr, weight_decay, dropout*
