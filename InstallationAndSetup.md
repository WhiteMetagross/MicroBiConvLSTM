# Installation and Setup Guide.

**Author:** Mridankan Mandal

**Paper:** [MicroBiConvLSTM on arXiv](https://arxiv.org/abs/2602.06523)

This document provides step-by-step instructions for setting up the MicroBiConvLSTM development environment, installing all dependencies, downloading benchmark datasets, and verifying the installation.

---

## Table of Contents.

1. [System Requirements](#system-requirements)
2. [Python Installation](#python-installation)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Installing Dependencies](#installing-dependencies)
5. [CUDA and GPU Setup](#cuda-and-gpu-setup)
6. [Dataset Download and Organization](#dataset-download-and-organization)
7. [Verifying the Installation](#verifying-the-installation)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements.

### Minimum Requirements.

- **Operating System:** Windows 10/11, Ubuntu 18.04+, or macOS 10.15+.
- **Python:** 3.8 or higher.
- **RAM:** 8 GB minimum.
- **Disk Space:** 2 GB for dependencies and datasets.

### Recommended Requirements.

- **Python:** 3.10 or 3.11 for best compatibility with PyTorch.
- **RAM:** 16 GB or higher.
- **GPU:** NVIDIA GPU with CUDA support (any modern NVIDIA GPU with 4 GB or more VRAM).
- **CUDA Toolkit:** 11.8 or 12.1 (matching the PyTorch build).

**Note:** MicroBiConvLSTM has approximately 10,500 parameters and can be trained entirely on CPU. GPU acceleration reduces training time but is not required.

---

## Python Installation.

### Windows.

1. Download Python from https://www.python.org/downloads/.
2. During installation, check the box labeled "Add Python to PATH".
3. Verify the installation by opening a terminal.

```powershell
python --version
```

### Linux.

Most Linux distributions include Python. If not, install it using the package manager.

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### macOS.

Install Python using Homebrew.

```bash
brew install python
```

---

## Virtual Environment Setup.

A virtual environment is strongly recommended to isolate this project from system-wide Python packages.

### Using venv (Standard Library).

```bash
# Create a virtual environment.
python -m venv microbiconv_env

# Activate the environment.
# On Windows:
microbiconv_env\Scripts\activate

# On Linux/macOS:
source microbiconv_env/bin/activate
```

### Using Conda.

```bash
# Create a conda environment.
conda create -n microbiconv python=3.10

# Activate the environment.
conda activate microbiconv
```

After activation, the terminal prompt will show the environment name. All subsequent commands should be run inside this activated environment.

---

## Installing Dependencies.

### Step 1: Install PyTorch.

Install PyTorch first, selecting the appropriate command for your system from https://pytorch.org/get-started/locally/.

For a system with an NVIDIA GPU and CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only systems:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install Remaining Dependencies.

```bash
pip install -r requirements.txt
```

This installs the following packages.

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| torch | 2.0.0 | Core deep learning framework. |
| numpy | 1.21.0 | Numerical computing. |
| pandas | 1.3.0 | Data loading and manipulation. |
| scikit-learn | 1.0.0 | Preprocessing, metrics, and train/test splitting. |
| scipy | 1.7.0 | Signal processing and statistical functions. |
| optuna | 3.0.0 | Hyperparameter optimization framework. |
| matplotlib | 3.4.0 | Plotting and figure generation. |
| seaborn | 0.11.0 | Statistical visualizations. |
| tqdm | 4.62.0 | Progress bars for training loops. |
| pyyaml | 6.0 | YAML configuration file parsing. |
| thop | 0.1.0 | (Optional) FLOPs and parameter counting. |

---

## CUDA and GPU Setup.

### Check GPU Availability.

After installing PyTorch, verify that CUDA is available.

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output for a system with an NVIDIA GPU:

```
CUDA available: True
Device: NVIDIA GeForce RTX 4060 Laptop GPU
```

### Installing CUDA Toolkit.

If CUDA is not available but you have an NVIDIA GPU:

1. Install the latest NVIDIA drivers from https://www.nvidia.com/Download/index.aspx.
2. Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads.
3. Reinstall PyTorch with the matching CUDA version.

**Note:** The CUDA version used by PyTorch must match or be compatible with the system CUDA Toolkit. PyTorch ships its own CUDA runtime, so often only the NVIDIA driver needs to be up to date.

---

## Dataset Download and Organization.

The datasets must be downloaded manually and placed in a `datasets/` folder at the repository root. Each dataset has its own subfolder.

### Required Directory Structure.

```
Micro-Bi-ConvLSTM/
|-- datasets/
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
    |   |-- data/
    |       |-- A_DeviceMotion_data/
    |-- WISDM_ar_v1.1/
    |   |-- WISDM_ar_v1.1_raw.txt
    |-- PAMAP2_Dataset/
    |   |-- Protocol/
    |       |-- subject101.dat
    |       |-- ...
    |-- Opportunity/
    |   |-- dataset/
    |       |-- S1-ADL1.dat
    |       |-- ...
    |-- UniMiB-SHAR/
    |   |-- data/
    |-- Skoda/
    |   |-- SkodaMiniCP_2015_08.csv (or similar)
    |-- Daphnet/
        |-- dataset/
            |-- S01R01.txt
            |-- ...
```

### Dataset Download Links.

| # | Dataset | Download URL |
|---|---------|-------------|
| 1 | UCI-HAR | https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones |
| 2 | MotionSense | https://github.com/mmalekzadeh/motion-sense |
| 3 | WISDM | https://www.cis.fordham.edu/wisdm/dataset.php |
| 4 | PAMAP2 | https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring |
| 5 | Opportunity | https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition |
| 6 | UniMiB-SHAR | http://www.sal.disco.unimib.it/technologies/unimib-shar/ |
| 7 | Skoda | https://sensor.informatik.uni-mannheim.de/ |
| 8 | Daphnet | https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait |

### Dataset Summary.

| Dataset | Channels | Sequence Length | Classes | Train Samples | Test Samples |
|---------|----------|----------------|---------|---------------|-------------|
| UCI-HAR | 9 | 128 | 6 | 7,352 | 2,947 |
| MotionSense | 6 | 128 | 6 | ~2,800 | ~1,200 |
| WISDM | 3 | 128 | 6 | ~4,500 | ~1,500 |
| PAMAP2 | 19 | 128 | 12 | ~12,000 | ~4,000 |
| Opportunity | 79 | 128 | 5 | ~8,000 | ~3,000 |
| UniMiB-SHAR | 3 | 128 | 9 | ~4,000 | ~1,700 |
| Skoda | 30 | 98 | 11 | ~6,500 | ~2,800 |
| Daphnet | 9 | 64 | 2 | ~7,000 | ~3,000 |

**Note:** Exact sample counts depend on the preprocessing configuration and windowing parameters.

---

## Verifying the Installation.

Run the following checks to confirm that the installation is correct.

### Step 1: Verify Python and Dependencies.

```bash
python -c "import torch; import numpy; import pandas; import sklearn; import optuna; print('All core dependencies imported successfully.')"
```

### Step 2: Verify Model Creation.

```bash
python -c "
from models import createMicroBiConvLstm
model = createMicroBiConvLstm('ucihar', dropout=0.15)
info = model.getModelInfo()
print(f'Model: {info[\"modelName\"]}')
print(f'Parameters: {info[\"totalParameters\"]:,}')
print('Model creation successful.')
"
```

Expected output:

```
Model: MicroBiConvLSTM
Parameters: 10,454
Model creation successful.
```

### Step 3: Verify Baseline Creation.

```bash
python -c "
from baselines.deepConvLstm import createDeepConvLstm
from baselines.tinyHar import createTinyHar
from baselines.tinierHar import createTinierHar
dcl = createDeepConvLstm('ucihar')
thar = createTinyHar('ucihar')
tinier = createTinierHar('ucihar')
print('All baseline models created successfully.')
"
```

### Step 4: Verify Data Loading (Requires Dataset).

```bash
python -c "
from data import loadUciHarData
X_train, y_train, X_test, y_test = loadUciHarData(rootPath='./datasets/UCI HAR Dataset')
print(f'Train: {X_train.shape}, Test: {X_test.shape}')
print('Data loading successful.')
"
```

This step requires that the UCI-HAR dataset has been downloaded and placed in the correct directory.

---

## Troubleshooting.

### PyTorch Not Detecting GPU.

- Ensure the NVIDIA driver is up to date.
- Verify the CUDA version with `nvidia-smi` and ensure the installed PyTorch build matches.
- Reinstall PyTorch with the correct CUDA version from https://pytorch.org/get-started/locally/.

### ModuleNotFoundError for Project Modules.

If Python cannot find `models`, `baselines`, or `data` modules, ensure you are running scripts from the repository root directory.

```bash
cd /path/to/Micro-Bi-ConvLSTM
python scripts/trainMicroBiConvLstm.py --dataset ucihar
```

Alternatively, add the repository root to the Python path.

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Micro-Bi-ConvLSTM"
```

On Windows (PowerShell):

```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\Micro-Bi-ConvLSTM"
```

### Out-of-Memory Errors.

MicroBiConvLSTM is extremely lightweight and should not cause memory issues. If memory errors occur with baseline models or large batch sizes:

- Reduce the batch size by modifying the DATASET_CONFIGS dictionary in the training script.
- Disable AMP by setting `use_amp = False` in the training function.
- Close other GPU-intensive applications.

### Optuna Database Errors.

If HPO crashes with database errors, ensure the output directory exists and is writable. Optuna stores study data in-memory by default, so this should not normally occur.

### Dataset Loading Failures.

- Verify that the dataset folder names match the expected directory structure shown above.
- Check that the files have been fully extracted (not still compressed).
- Ensure the root path passed to the data loader matches the actual location.
