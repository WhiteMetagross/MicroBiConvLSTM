"""
LightDeepConvLSTM Hyperparameter Optimization Script

This script performs HPO for the LightDeepConvLSTM architecture.
It uses Optuna with TPE sampler and FROZEN architecture hyperparameters.

FROZEN Architecture (NOT tuned):
    convFilters = 16    (number of conv filters)
    convKernel = 5      (convolution kernel size)
    lstmHidden = 24     (LSTM hidden dimension)
    lstmLayers = 1      (number of LSTM layers)
    bidirectional = True (bidirectional LSTM)

TUNED Hyperparameters (matches baselines for fairness):
    learning_rate:  [1e-4, 1e-2]   log-uniform
    weight_decay:   [1e-5, 0.05]   log-uniform
    dropout:        [0.0, 0.5]     uniform

HPO Configuration:
    sampler = TPE (Tree-structured Parzen Estimator)
    trials = 50
    epochs per trial = 50 (matches baselines for fairness)
    pruning = Median pruner with warmup
    patience = 5 (early stopping)
    optimization_target = F1 Score (Macro) - MAXIMIZED

Usage:
    python hpoLightDeepConvLSTM.py --dataset ucihar --n-trials 50
    python hpoLightDeepConvLSTM.py --dataset pamap2 --n-trials 100 --epochs 100
    python hpoLightDeepConvLSTM.py --dataset all --n-trials 50
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import LightDeepConvLSTM


# ============== Dataset Configurations ==============

DATASET_CONFIGS = {
    'ucihar': {
        'name': 'UCI-HAR',
        'inputChannels': 9,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64
    },
    'motionsense': {
        'name': 'MotionSense',
        'inputChannels': 6,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64
    },
    'wisdm': {
        'name': 'WISDM',
        'inputChannels': 3,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64
    },
    'pamap2': {
        'name': 'PAMAP2',
        'inputChannels': 19,
        'seqLen': 128,
        'numClasses': 12,
        'batchSize': 64
    },
    'opportunity': {
        'name': 'Opportunity',
        'inputChannels': 79,
        'seqLen': 128,
        'numClasses': 5,
        'batchSize': 64
    },
    'unimib': {
        'name': 'UniMiB-SHAR',
        'inputChannels': 3,
        'seqLen': 128,
        'numClasses': 9,
        'batchSize': 64
    },
    'skoda': {
        'name': 'Skoda',
        'inputChannels': 30,
        'seqLen': 98,
        'numClasses': 11,
        'batchSize': 512
    },
    'daphnet': {
        'name': 'Daphnet',
        'inputChannels': 9,
        'seqLen': 64,
        'numClasses': 2,
        'batchSize': 512
    }
}

# FROZEN architecture hyperparameters - DO NOT TUNE
FROZEN_ARCH = {
    'convFilters': 16,
    'convKernel': 5,
    'convPadding': 2,
    'lstmHidden': 24,
    'lstmLayers': 1,
    'bidirectional': True,
}


def setSeed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def getDevice():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def loadDataset(datasetName: str, batchSize: int):
    """
    Load dataset with train/val splits.
    
    Uses local data loaders from LightDeepConvLSTM/data with:
    - Signal Rescue filters for PAMAP2, Skoda, Daphnet
    - Class weights for imbalanced datasets
    - OLD stratified shuffle split for Skoda (paper reproducibility)
    """
    import sys
    from pathlib import Path
    
    # Add local data folder to path (self-contained for LightDeepConvLSTM paper)
    dataPath = Path(__file__).parent.parent / 'data'
    if str(dataPath) not in sys.path:
        sys.path.insert(0, str(dataPath))
    
    config = DATASET_CONFIGS[datasetName]
    classWeights = None
    
    # Convert dataset name to lowercase for loader imports
    dsName = datasetName.lower()
    
    if dsName == 'ucihar':
        from uciHar import getUciHarLoaders
        trainLoader, testLoader = getUciHarLoaders(
            root='./datasets/UCI HAR Dataset', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'motionsense':
        from motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(
            root='./datasets/motion-sense-master', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'wisdm':
        from wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(
            root='./datasets/WISDM_ar_v1.1', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'pamap2':
        from pamap2 import getPamap2Loaders
        trainLoader, testLoader, classWeights = getPamap2Loaders(
            root='./datasets/PAMAP2_Dataset', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'opportunity':
        from opportunity import getOpportunityLoaders
        trainLoader, testLoader, classWeights = getOpportunityLoaders(
            root='./datasets/Opportunity', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'unimib':
        from unimib import getUnimibLoaders
        trainLoader, testLoader = getUnimibLoaders(
            root='./datasets/UniMiB-SHAR', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'skoda':
        # Uses OLD stratified shuffle split for LightDeepConvLSTM paper reproducibility
        from skoda import getSkodaLoaders
        trainLoader, testLoader, classWeights = getSkodaLoaders(
            root='./datasets/Skoda', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'daphnet':
        from daphnet import getDaphnetLoaders
        trainLoader, testLoader, classWeights = getDaphnetLoaders(
            root='./datasets/Daphnet', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")
    
    return trainLoader, testLoader, classWeights


def trainEpoch(
    model: nn.Module,
    trainLoader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    useAmp: bool = True,
) -> tuple:
    """Train for one epoch."""
    model.train()
    totalLoss = 0.0
    allPreds = []
    allLabels = []
    
    for batch in trainLoader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if useAmp and device.type == 'cuda':
            with autocast(device_type='cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        totalLoss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        allPreds.extend(preds.cpu().numpy())
        allLabels.extend(y.cpu().numpy())
    
    avgLoss = totalLoss / len(trainLoader.dataset)
    f1 = f1_score(allLabels, allPreds, average='macro', zero_division=0)
    
    return avgLoss, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataLoader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on dataset."""
    model.eval()
    totalLoss = 0.0
    allPreds = []
    allLabels = []
    
    for batch in dataLoader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        totalLoss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        allPreds.extend(preds.cpu().numpy())
        allLabels.extend(y.cpu().numpy())
    
    avgLoss = totalLoss / len(dataLoader.dataset)
    f1 = f1_score(allLabels, allPreds, average='macro', zero_division=0)
    
    return avgLoss, f1


def objective(
    trial: optuna.Trial,
    datasetName: str,
    trainLoader: DataLoader,
    valLoader: DataLoader,
    classWeights: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
) -> float:
    """
    Optuna objective function for HPO.
    
    Tunes ONLY training hyperparameters:
    - learning_rate
    - weight_decay
    - dropout
    
    Returns F1 score (to be maximized).
    """
    # Sample hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weightDecay = trial.suggest_float('weight_decay', 1e-5, 0.05, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    config = DATASET_CONFIGS[datasetName]
    
    # Create model with FROZEN architecture
    model = LightDeepConvLSTM(
        numClasses=config['numClasses'],
        inChannels=config['inputChannels'],
        seqLen=config['seqLen'],
        dropout=dropout,
        aggregation='last',
    ).to(device)
    
    # Setup training
    if classWeights is not None:
        cw = classWeights.to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    useAmp = device.type == 'cuda'
    
    bestF1 = 0.0
    noImproveCount = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        trainLoss, trainF1 = trainEpoch(
            model, trainLoader, criterion, optimizer, device, scaler, useAmp
        )
        
        # Evaluate
        valLoss, valF1 = evaluate(model, valLoader, criterion, device)
        
        scheduler.step()
        
        # Track best
        if valF1 > bestF1:
            bestF1 = valF1
            noImproveCount = 0
        else:
            noImproveCount += 1
        
        # Report to Optuna for pruning
        trial.report(valF1, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if noImproveCount >= patience:
            break
    
    return bestF1


def runHPO(
    datasetName: str,
    nTrials: int = 50,
    epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
    saveDir: str = './hpo_results',
    verbose: bool = True,
) -> dict:
    """
    Run Hyperparameter Optimization for LightDeepConvLSTM on a specific dataset.
    
    Args:
        datasetName: Name of the dataset
        nTrials: Number of HPO trials
        epochs: Epochs per trial
        patience: Early stopping patience per trial
        seed: Random seed for reproducibility
        saveDir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing HPO results
    """
    setSeed(seed)
    device = getDevice()
    config = DATASET_CONFIGS[datasetName]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HPO for LightDeepConvLSTM on {config['name']}")
        print(f"Trials: {nTrials} | Epochs/trial: {epochs} | Device: {device}")
        print(f"{'='*60}")
    
    # Load dataset
    trainLoader, valLoader, classWeights = loadDataset(
        datasetName, config['batchSize']
    )
    
    # Create Optuna study
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize F1 score
        sampler=sampler,
        pruner=pruner,
        study_name=f'lightdeepconvlstm_{datasetName}',
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, datasetName, trainLoader, valLoader, 
            classWeights, device, epochs, patience
        ),
        n_trials=nTrials,
        show_progress_bar=verbose,
    )
    
    # Get best results
    bestTrial = study.best_trial
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HPO Results for {config['name']}")
        print(f"{'='*60}")
        print(f"Best Trial: {bestTrial.number}")
        print(f"Best F1 Score: {bestTrial.value*100:.2f}%")
        print(f"\nOptimal Hyperparameters:")
        for key, value in bestTrial.params.items():
            print(f"  {key}: {value:.6f}")
    
    # Save results
    os.makedirs(saveDir, exist_ok=True)
    resultsPath = os.path.join(saveDir, f'lightdeepconvlstm_{datasetName}_hpo.json')
    
    results = {
        'dataset': datasetName,
        'bestTrial': bestTrial.number,
        'bestF1': bestTrial.value,
        'bestParams': bestTrial.params,
        'nTrials': nTrials,
        'epochs': epochs,
        'allTrials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ],
    }
    
    with open(resultsPath, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {resultsPath}")
    
    return results


def main():
    """Main entry point for HPO script."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization for LightDeepConvLSTM'
    )
    parser.add_argument(
        '--dataset', type=str, default='ucihar',
        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 
                 'opportunity', 'unimib', 'skoda', 'daphnet', 'all'],
        help='Dataset to run HPO on (default: ucihar)'
    )
    parser.add_argument(
        '--n-trials', type=int, default=50,
        help='Number of HPO trials (default: 50)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Epochs per trial (default: 50)'
    )
    parser.add_argument(
        '--patience', type=int, default=5,
        help='Early stopping patience per trial (default: 5)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--save-dir', type=str, default='./hpo_results',
        help='Directory to save results (default: ./hpo_results)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]
    allResults = {}
    
    for dataset in datasets:
        results = runHPO(
            datasetName=dataset,
            nTrials=args.n_trials,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            saveDir=args.save_dir,
            verbose=not args.quiet,
        )
        allResults[dataset] = results
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"HPO Summary - LightDeepConvLSTM")
    print(f"{'='*60}")
    print(f"\n{'Dataset':<15} {'Best F1':<12} {'LR':<12} {'WD':<12} {'Dropout':<10}")
    print(f"{'-'*60}")
    
    for dataset, res in allResults.items():
        params = res['bestParams']
        print(f"{dataset:<15} {res['bestF1']*100:.2f}%      "
              f"{params['learning_rate']:.6f}   "
              f"{params['weight_decay']:.6f}   "
              f"{params['dropout']:.3f}")
    
    print(f"\n✓ HPO complete! Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
