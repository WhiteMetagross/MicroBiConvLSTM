# LightDeepConvLSTM Ablation Studies (Auto-Generated)

> This file is written by `scripts/ablationStudiesLightDeepConvLSTM.py`.
> Run studies to populate metrics.

## Table I: Architectural Ablation Results

| Configuration | Params (K) | MACs (K) | UCI-HAR (F1) | SKODA (F1) | PAMAP2 (F1) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Proposed (Frozen)** | **10.5K** | **420.1K** | **92.84%** | **95.34%** | **66.24%** |
| A1: No MaxPool | 10.5K | 1,239.0K | 92.61% | 92.93% | 63.28% |
| A2: Unidirectional | 6.3K | 297.0K | 93.67% | 94.13% | 64.84% |
| A3: Single Conv | 9.1K | 584.0K | 92.85% | 93.42% | 63.93% |
| A4: Mean Pooling | 10.5K | 420.1K | 91.94% | 91.15% | 58.88% |

## Table II: Preprocessing & Hardware Proxy Summary

| Dataset | Normalization | Filter | Special Handling | INT8 Quant. Error |
| :--- | :--- | :--- | :--- | :--- |
| **PAMAP2** | Robust Scaling | 10Hz LPF | Impact Spike Removal | -0.06% ↑ |
| **SKODA** | Z-Score | 5Hz LPF | Machine Vib. Removal | 0.07% |
| **Daphnet** | Z-Score | 12Hz LPF | 15:1 Class Weighting | 1.09% |

## Table III: Memory Footprint Comparison (UCI-HAR)

| Model | Params | FP32 Size | INT8 Size | Memory Savings |
| :--- | ---: | ---: | ---: | ---: |
| **LightDeepConvLSTM** | **10.5K** | **41.9 KB** | **21.2 KB** | **Baseline** |
| TinierHAR | 17.0K | 68.3 KB | 34.1 KB | 38.7% larger |
| TinyHAR | 42.3K | 169.4 KB | 84.7 KB | 75.3% larger |
| HARMamba-Lite | 50.6K | 202.8 KB | 101.4 KB | 79.3% larger |
| DeepConvLSTM | 132.0K | 528.5 KB | 264.3 KB | 92.1% larger |
| HARMamba | 398.5K | 1.52 MB | 760.2 KB | 97.2% larger |

## Table IV: Deployment Suitability

| Model | STM32F4 (256KB) | ESP32 (320KB) | Jetson Nano | Mobile |
| :--- | :---: | :---: | :---: | :---: |
| **LightDeepConvLSTM** | ✅ (8% SRAM) | ✅ (7% SRAM) | ✅ | ✅ |
| TinierHAR | ✅ (13% SRAM) | ✅ (11% SRAM) | ✅ | ✅ |
| TinyHAR | ⚠️ (33% SRAM) | ⚠️ (26% SRAM) | ✅ | ✅ |
| HARMamba-Lite | ⚠️ (40% SRAM) | ⚠️ (32% SRAM) | ✅ | ✅ |
| DeepConvLSTM | ❌ (103% SRAM) | ❌ (82% SRAM) | ✅ | ✅ |
| HARMamba | ❌ (297% SRAM) | ❌ (238% SRAM) | ⚠️ | ⚠️ |

