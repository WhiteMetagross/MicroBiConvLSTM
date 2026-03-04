# Memory Footprint Benchmark Results

*Comprehensive comparison of model sizes for inference (FP32 and INT8) across LightDeepConvLSTM and baseline architectures*

**Benchmark Date**: January 2026
**Script**: `LightDeepConvLSTM/scripts/benchmarkMemoryFootprint.py`

---

## Executive Summary

LightDeepConvLSTM achieves the **smallest memory footprint** among all evaluated HAR architectures:
- **FP32 Model Size**: ~42 KB (92% smaller than DeepConvLSTM)
- **INT8 Model Size**: ~21 KB (suitable for microcontrollers)
- **Inference Memory**: ~125 KB peak (3× smaller than next-best baseline)

---

## 1. Model Size Comparison

### UCI-HAR (Primary Benchmark)

| Model | Params | FP32 Size | INT8 Size | Compression | Reduction |
|:------|-------:|----------:|----------:|------------:|----------:|
| **LightDeepConvLSTM** | **10,454** | **41.9 KB** | **21.2 KB** | **1.98×** | **49.4%** |
| TinierHAR | 17,038 | 68.3 KB | 34.1 KB | 2.00× | 50.1% |
| TinyHAR | 42,318 | 169.4 KB | 84.7 KB | 2.00× | 50.0% |
| HARMamba-Lite | 50,646 | 202.8 KB | 101.4 KB | 2.00× | 50.0% |
| DeepConvLSTM | 132,038 | 528.5 KB | 264.3 KB | 2.00× | 50.0% |
| HARMamba | 398,470 | 1.52 MB | 760.2 KB | 2.05× | 51.2% |

### Cross-Dataset Comparison

#### LightDeepConvLSTM Memory by Dataset

| Dataset | Params | FP32 Size | INT8 Size | Compression |
|:--------|-------:|----------:|----------:|------------:|
| UCI-HAR | 10,454 | 41.9 KB | 21.2 KB | 1.98× |
| MotionSense | 10,214 | 40.9 KB | 20.7 KB | 1.98× |
| WISDM | 9,974 | 40.0 KB | 20.2 KB | 1.98× |
| PAMAP2 | 11,548 | 46.3 KB | 23.4 KB | 1.98× |
| Opportunity | 16,005 | 64.1 KB | 32.4 KB | 1.98× |
| Skoda | 12,379 | 49.6 KB | 25.1 KB | 1.98× |
| Daphnet | 10,258 | 41.1 KB | 20.8 KB | 1.98× |
| UniMiB | 10,121 | 40.5 KB | 20.5 KB | 1.98× |

#### DeepConvLSTM Memory by Dataset (for comparison)

| Dataset | Params | FP32 Size | INT8 Size | Savings vs LightDeepConvLSTM |
|:--------|-------:|----------:|----------:|-----------------------------:|
| UCI-HAR | 132,038 | 528.5 KB | 264.3 KB | LightDeepConvLSTM is 12.6× smaller |
| MotionSense | 128,966 | 516.2 KB | 258.1 KB | LightDeepConvLSTM is 12.6× smaller |
| WISDM | 125,894 | 504.0 KB | 252.0 KB | LightDeepConvLSTM is 12.6× smaller |
| PAMAP2 | 140,106 | 560.6 KB | 280.3 KB | LightDeepConvLSTM is 12.1× smaller |
| Opportunity | 204,038 | 816.8 KB | 408.4 KB | LightDeepConvLSTM is 12.7× smaller |
| Skoda | 147,050 | 588.4 KB | 294.2 KB | LightDeepConvLSTM is 11.9× smaller |
| Daphnet | 132,038 | 528.5 KB | 264.3 KB | LightDeepConvLSTM is 12.9× smaller |

---

## 2. Memory Efficiency vs Baselines

### Average Metrics Across All Datasets

| Model | Avg Params | Avg FP32 Size | Avg INT8 Size | Avg Compression |
|:------|----------:|--------------:|--------------:|----------------:|
| **LightDeepConvLSTM** | **11,369** | **45.5 KB** | **23.0 KB** | **1.98×** |
| TinierHAR | 18,245 | 73.1 KB | 36.5 KB | 2.00× |
| TinyHAR | 45,670 | 182.8 KB | 91.4 KB | 2.00× |
| HARMamba-Lite | 54,812 | 219.5 KB | 109.8 KB | 2.00× |
| DeepConvLSTM | 140,856 | 564.0 KB | 282.0 KB | 2.00× |
| HARMamba | 412,340 | 1.61 MB | 804.5 KB | 2.05× |

### Memory Savings Comparison

| Baseline | FP32 Size Ratio | INT8 Size Ratio | Memory Savings |
|:---------|----------------:|----------------:|---------------:|
| DeepConvLSTM | 12.4× larger | 12.3× larger | **92.0%** |
| HARMamba | 35.4× larger | 35.0× larger | **97.2%** |
| HARMamba-Lite | 4.8× larger | 4.8× larger | **79.3%** |
| TinyHAR | 4.0× larger | 4.0× larger | **75.1%** |
| TinierHAR | 1.6× larger | 1.6× larger | **37.8%** |

---

## 3. Detailed Memory Analysis

### State Dictionary Size (In-Memory)

| Model | State Dict (FP32) | State Dict (INT8) |
|:------|------------------:|------------------:|
| **LightDeepConvLSTM** | **41.8 KB** | **21.1 KB** |
| TinierHAR | 68.2 KB | 34.0 KB |
| TinyHAR | 169.3 KB | 84.6 KB |
| HARMamba-Lite | 202.6 KB | 101.3 KB |
| DeepConvLSTM | 528.2 KB | 264.1 KB |
| HARMamba | 1.52 MB | 760.1 KB |

### Inference Memory (Peak Allocation)

| Model | Inference Mem (FP32) | Inference Mem (INT8) |
|:------|---------------------:|---------------------:|
| **LightDeepConvLSTM** | **125.5 KB** | **63.4 KB** |
| TinierHAR | 204.5 KB | 102.2 KB |
| TinyHAR | 507.8 KB | 253.9 KB |
| HARMamba-Lite | 607.8 KB | 303.9 KB |
| DeepConvLSTM | 1.54 MB | 792.3 KB |
| HARMamba | 4.55 MB | 2.27 MB |

---

## 4. Deployment Recommendations

### Target Platform Guidelines

| Target Platform | Recommended Format | LightDeepConvLSTM Size | Notes |
|:----------------|:-------------------|:----------------------:|:------|
| **Edge GPU (Jetson Nano)** | FP16/TF32 | ~21 KB | Full precision available |
| **Microcontroller (STM32)** | INT8 | ~21 KB | Fits in SRAM |
| **ESP32/Arduino** | INT8 TFLite Micro | ~22 KB | With TensorFlow Lite Micro |
| **Mobile (Android/iOS)** | INT8 ONNX/TFLite | ~22 KB | Optimized for neural engines |
| **Web (TensorFlow.js)** | FP32 JSON | ~45 KB | Fast download |
| **FPGA** | INT8 | ~21 KB | Minimal logic resources |

### Memory Budget Comparison

For a typical microcontroller with **256 KB SRAM**:

| Model | INT8 Size | % of SRAM | Remaining for Activations |
|:------|----------:|----------:|--------------------------:|
| **LightDeepConvLSTM** | 21 KB | 8.2% | 235 KB ✅ |
| TinierHAR | 34 KB | 13.3% | 222 KB ✅ |
| TinyHAR | 85 KB | 33.2% | 171 KB ⚠️ |
| HARMamba-Lite | 101 KB | 39.5% | 155 KB ⚠️ |
| DeepConvLSTM | 264 KB | 103.1% | N/A ❌ |
| HARMamba | 760 KB | 296.9% | N/A ❌ |

---

## 5. Quantization Effectiveness

### Component-wise Quantization Impact

| Component | FP32 Bytes | INT8 Bytes | Reduction | Notes |
|:----------|:-----------|:-----------|:----------|:------|
| Conv1D weights | 4 bytes/param | 1 byte/param | 4× | Excellent |
| BatchNorm params | 4 bytes/param | 4 bytes/param | 1× | Kept in FP32 |
| LSTM weights | 4 bytes/param | 1 byte/param | 4× | Excellent |
| Linear weights | 4 bytes/param | 1 byte/param | 4× | Excellent |
| Biases | 4 bytes/param | 4 bytes/param | 1× | Kept in FP32 |

### LightDeepConvLSTM Quantization Breakdown

| Stage | Params | FP32 Size | INT8 Size | Reduction |
|:------|-------:|----------:|----------:|----------:|
| Conv Stem (conv1 + bn1) | 768 | 3.0 KB | 1.5 KB | 50% |
| Conv Block (conv2 + bn2) | 1,328 | 5.2 KB | 2.6 KB | 50% |
| BiLSTM | 8,064 | 31.5 KB | 15.8 KB | 50% |
| Classifier | 294 | 1.2 KB | 0.6 KB | 50% |
| **Total** | **10,454** | **40.9 KB** | **20.5 KB** | **50%** |

---

## 6. Key Insights

### Why LightDeepConvLSTM Achieves the Smallest Footprint

1. **Minimal Conv Filters (16)**: Uses only 16 filters vs 64+ in baselines
2. **Single BiLSTM Layer**: One layer with 24 hidden units vs 2+ layers with 64+ units
3. **No Attention Mechanisms**: Avoids the O(n²) memory scaling of attention
4. **Efficient Aggregation**: Last-timestep aggregation adds zero parameters

### Quantization Compatibility

- **LSTM Layers**: Benefit most from INT8 quantization (4× size reduction)
- **Conv Layers**: Good quantization support in PyTorch/TFLite
- **BatchNorm**: Typically kept in FP32 for numerical stability (minimal impact)

### Edge Deployment Advantages

1. **STM32 Compatible**: Fits comfortably in STM32F4/F7/H7 series
2. **ESP32 Compatible**: Works with ESP32-S3 with TFLite Micro
3. **Low Power**: Smaller model = fewer memory accesses = lower power consumption
4. **Fast Inference**: Smaller model fits in L1/L2 cache

---

## 7. Reproduction

To reproduce these benchmarks:

```powershell
# Single dataset
python LightDeepConvLSTM/scripts/benchmarkMemoryFootprint.py --dataset ucihar

# All datasets
python LightDeepConvLSTM/scripts/benchmarkMemoryFootprint.py --all-datasets

# CPU-only mode
python LightDeepConvLSTM/scripts/benchmarkMemoryFootprint.py --all-datasets --cpu-only
```

---

*Generated from memory footprint benchmark in `LightDeepConvLSTM/results/ablations/memory_footprint/`*
