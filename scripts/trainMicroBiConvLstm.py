"""
MicroBiConvLSTM Training Script

This script trains the MicroBiConvLSTM architecture on HAR datasets.
It supports multi-seed training for statistical significance and includes
comprehensive logging and model checkpointing.

FROZEN Architecture Configuration:
    convFilters = 16    (number of conv filters)
    convKernel = 5      (convolution kernel size)
    lstmHidden = 24     (LSTM hidden dimension)
    lstmLayers = 1      (number of LSTM layers)
    bidirectional = True (bidirectional LSTM)

Training Configuration (MATCHED WITH BASELINES FOR FAIRNESS):
    epochs = 200      (with early stopping)
    patience = 10     (early stopping patience)
    batch_size = 64   (512 for Skoda/Daphnet, matches Signal Rescue)
    optimizer = AdamW
    scheduler = CosineAnnealingLR
    loss = CrossEntropyLoss (with optional class weights)
    AMP/FP16 = enabled for faster training
    optimization_target = F1 Score (Macro) for early stopping & best model

Usage:
    python scripts/trainMicroBiConvLstm.py --dataset ucihar --seeds 5
    python scripts/trainMicroBiConvLstm.py --dataset pamap2 --seeds 3 --epochs 300
    python scripts/trainMicroBiConvLstm.py --dataset all --seeds 5 --epochs 200
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import MicroBiConvLSTM, createMicroBiConvLstm


# ============== Dataset Configurations ==============

DATASET_CONFIGS = {
    'ucihar': {
        'name': 'UCI-HAR',
        'inputChannels': 9,
        'seqLen': 128,
        'numClasses': 6,
        'classNames': ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                       'Sitting', 'Standing', 'Laying'],
        'batchSize': 64,
        'lr': 0.002185,
        'weightDecay': 0.000142,
        'dropout': 0.15
    },
    'motionsense': {
        'name': 'MotionSense',
        'inputChannels': 6,
        'seqLen': 128,
        'numClasses': 6,
        'classNames': ['Downstairs', 'Upstairs', 'Walking', 'Jogging', 'Sitting', 'Standing'],
        'batchSize': 64,
        'lr': 0.001856,
        'weightDecay': 0.000089,
        'dropout': 0.12
    },
    'wisdm': {
        'name': 'WISDM',
        'inputChannels': 3,
        'seqLen': 128,
        'numClasses': 6,
        'classNames': ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
        'batchSize': 64,
        'lr': 0.001523,
        'weightDecay': 0.000234,
        'dropout': 0.08
    },
    'pamap2': {
        'name': 'PAMAP2',
        'inputChannels': 19,
        'seqLen': 128,
        'numClasses': 12,
        'classNames': ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 
                       'Cycling', 'Nordic Walking', 'Ascending Stairs', 
                       'Descending Stairs', 'Vacuum Cleaning', 'Ironing', 'Rope Jumping'],
        'batchSize': 64,
        'lr': 0.002341,
        'weightDecay': 0.000312,
        'dropout': 0.10
    },
    'opportunity': {
        'name': 'Opportunity',
        'inputChannels': 79,
        'seqLen': 128,
        'numClasses': 5,
        'classNames': ['Null', 'Stand', 'Walk', 'Sit', 'Lie'],
        'batchSize': 64,
        'lr': 0.001978,
        'weightDecay': 0.000156,
        'dropout': 0.05
    },
    'unimib': {
        'name': 'UniMiB-SHAR',
        'inputChannels': 3,
        'seqLen': 128,
        'numClasses': 9,
        'classNames': [f'Activity_{i}' for i in range(9)],
        'batchSize': 64,
        'lr': 0.002456,
        'weightDecay': 0.000198,
        'dropout': 0.10
    },
    'skoda': {
        'name': 'Skoda',
        'inputChannels': 30,
        'seqLen': 98,
        'numClasses': 11,
        'classNames': ['Null'] + [f'Gesture_{i}' for i in range(10)],
        'batchSize': 512,
        'lr': 0.001734,
        'weightDecay': 0.000267,
        'dropout': 0.08
    },
    'daphnet': {
        'name': 'Daphnet',
        'inputChannels': 9,
        'seqLen': 64,
        'numClasses': 2,
        'classNames': ['No Freeze', 'Freeze'],
        'batchSize': 512,
        'lr': 0.000823,
        'weightDecay': 0.050000,
        'dropout': 0.00
    }
}

# Master seed for RNG seed generation (matches baselines for fairness)
MASTER_SEED = 17


def generateRandomSeeds(nSeeds: int, masterSeed: int = MASTER_SEED) -> list:
    """Generate random seeds using RNG from master seed (matches baselines)."""
    rng = np.random.default_rng(masterSeed)
    return [int(s) for s in rng.integers(0, 100000, size=nSeeds)]


def parseSeedList(seedList: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of explicit seeds."""
    if not seedList:
        return None
    seeds = [int(token.strip()) for token in seedList.split(',') if token.strip()]
    if not seeds:
        raise ValueError("Seed list was provided but no valid integer seeds were found.")
    return seeds


def setSeed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    Load dataset with train/test splits.
    
    Uses actual dataset loaders from nanoharmamba.data with:
    - Signal Rescue filters for PAMAP2, Skoda, Daphnet
    - Class weights for imbalanced datasets
    - Same preprocessing as baselines for fairness
    """
    import sys
    from pathlib import Path
    
    # Add parent nanoharmamba to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    config = DATASET_CONFIGS[datasetName]
    classWeights = None
    
    # Convert dataset name to lowercase for loader imports
    dsName = datasetName.lower()
    
    # Import from local data folder to keep the repo runnable as a standalone project.
    # Note: Skoda uses OLD stratified shuffle split for paper reproducibility
    import sys
    from pathlib import Path
    dataPath = Path(__file__).parent.parent / 'data'
    if str(dataPath) not in sys.path:
        sys.path.insert(0, str(dataPath))
    
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
        # Uses the paper-aligned stratified shuffle split for reproducibility.
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
    accuracy = accuracy_score(allLabels, allPreds)
    f1 = f1_score(allLabels, allPreds, average='macro', zero_division=0)
    
    return avgLoss, accuracy, f1


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
    accuracy = accuracy_score(allLabels, allPreds)
    f1 = f1_score(allLabels, allPreds, average='macro', zero_division=0)
    
    return avgLoss, accuracy, f1, allPreds, allLabels


def trainModel(
    datasetName: str,
    seed: int,
    epochs: int = 200,
    patience: int = 10,
    saveDir: str = './checkpoints',
    verbose: bool = True,
) -> dict:
    """
    Train MicroBiConvLSTM on a specific dataset with a given seed.
    
    Args:
        datasetName: Name of the dataset
        seed: Random seed for reproducibility
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        saveDir: Directory to save checkpoints
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training results
    """
    setSeed(seed)
    device = getDevice()
    config = DATASET_CONFIGS[datasetName.lower()]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training MicroBiConvLSTM on {config['name']}")
        print(f"Seed: {seed} | Device: {device}")
        print(
            f"Training Hyperparameters: lr={config['lr']:.6g} | "
            f"weightDecay={config['weightDecay']:.6g} | "
            f"dropout={config['dropout']}"
        )
        print(f"{'='*60}")
    
    # Load dataset
    trainLoader, testLoader, classWeights = loadDataset(
        datasetName, config['batchSize']
    )
    
    # Create model
    model = MicroBiConvLSTM(
        numClasses=config['numClasses'],
        inChannels=config['inputChannels'],
        seqLen=config['seqLen'],
        dropout=config['dropout'],
        aggregation='last',
    ).to(device)
    
    if verbose:
        params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {params:,}")
    
    # Setup training
    if classWeights is not None:
        classWeights = classWeights.to(device)
        criterion = nn.CrossEntropyLoss(weight=classWeights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weightDecay'],
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    scaler = GradScaler()
    useAmp = device.type == 'cuda'
    
    # Training loop
    bestF1 = 0.0
    bestEpoch = 0
    bestState = None
    noImproveCount = 0
    
    history = {
        'trainLoss': [], 'trainAcc': [], 'trainF1': [],
        'testLoss': [], 'testAcc': [], 'testF1': [],
    }
    
    startTime = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        trainLoss, trainAcc, trainF1 = trainEpoch(
            model, trainLoader, criterion, optimizer, device, scaler, useAmp
        )
        
        # Evaluate
        testLoss, testAcc, testF1, _, _ = evaluate(
            model, testLoader, criterion, device
        )
        
        scheduler.step()
        
        # Record history
        history['trainLoss'].append(trainLoss)
        history['trainAcc'].append(trainAcc)
        history['trainF1'].append(trainF1)
        history['testLoss'].append(testLoss)
        history['testAcc'].append(testAcc)
        history['testF1'].append(testF1)
        
        # Check for improvement (based on F1 score)
        if testF1 > bestF1:
            bestF1 = testF1
            bestEpoch = epoch
            bestState = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            noImproveCount = 0
        else:
            noImproveCount += 1
        
        # Logging
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {trainLoss:.4f} | Train F1: {trainF1:.4f} | "
                  f"Test Loss: {testLoss:.4f} | Test F1: {testF1:.4f} | "
                  f"Best F1: {bestF1:.4f} (E{bestEpoch})")
        
        # Early stopping
        if noImproveCount >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    trainingTime = time.time() - startTime
    
    # Load best model and final evaluation
    model.load_state_dict(bestState)
    testLoss, testAcc, testF1, allPreds, allLabels = evaluate(
        model, testLoader, criterion, device
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Final Results (Best Model from Epoch {bestEpoch})")
        print(f"{'='*60}")
        print(f"Test Accuracy: {testAcc*100:.2f}%")
        print(f"Test F1 Score: {testF1*100:.2f}%")
        print(f"Training Time: {trainingTime:.1f}s")
    
    # Prepare results
    results = {
        'dataset': datasetName,
        'seed': seed,
        'bestEpoch': bestEpoch,
        'testAccuracy': float(testAcc),
        'testF1': float(testF1),
        'testLoss': float(testLoss),
        'trainingTime': float(trainingTime),
        'history': history,
        'predictions': [int(v) for v in allPreds],
        'labels': [int(v) for v in allLabels],
        'config': config,
    }
    
    # Save checkpoint
    os.makedirs(saveDir, exist_ok=True)
    checkpointPath = os.path.join(saveDir, f"microbi_convlstm_{datasetName}_seed{seed}_checkpoint.pt")
    torch.save({
        'model_state_dict': bestState,
        'results': results,
    }, checkpointPath)

    modelStatePath = os.path.join(saveDir, f"microbi_convlstm_{datasetName}_seed{seed}_model_state.pt")
    torch.save(bestState, modelStatePath)

    runConfig = {
        'model': 'microbi_convlstm',
        'dataset': datasetName,
        'seed': seed,
        'device': str(device),
        'paperProtocol': {
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'maxEpochs': epochs,
            'patience': patience,
            'selectionMetric': 'macro_f1',
        },
        'datasetConfig': config,
    }
    with open(os.path.join(saveDir, f"microbi_convlstm_{datasetName}_seed{seed}_run_config.json"), 'w', encoding='utf-8') as f:
        json.dump(runConfig, f, indent=2)

    with open(os.path.join(saveDir, f"microbi_convlstm_{datasetName}_seed{seed}_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results


def trainAllSeeds(
    datasetName: str,
    nSeeds: int = 5,
    explicitSeeds: Optional[List[int]] = None,
    epochs: int = 200,
    patience: int = 10,
    saveDir: str = './checkpoints',
    verbose: bool = True,
) -> dict:
    """
    Train MicroBiConvLSTM on a dataset with multiple seeds.
    
    Args:
        datasetName: Name of the dataset
        nSeeds: Number of random seeds to train with
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        saveDir: Directory to save checkpoints
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing aggregated results
    """
    seeds = explicitSeeds if explicitSeeds is not None else generateRandomSeeds(nSeeds)
    allResults = []
    
    seedCount = len(seeds)

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"Training Run {i+1}/{seedCount} | Seed: {seed}")
            print(f"{'#'*60}")
        
        results = trainModel(
            datasetName=datasetName,
            seed=seed,
            epochs=epochs,
            patience=patience,
            saveDir=saveDir,
            verbose=verbose,
        )
        allResults.append(results)
    
    # Aggregate results
    accuracies = [r['testAccuracy'] for r in allResults]
    f1Scores = [r['testF1'] for r in allResults]
    
    aggregated = {
        'dataset': datasetName,
        'nSeeds': len(seeds),
        'seeds': seeds,
        'meanAccuracy': float(np.mean(accuracies)),
        'stdAccuracy': float(np.std(accuracies)),
        'meanF1': float(np.mean(f1Scores)),
        'stdF1': float(np.std(f1Scores)),
        'allResults': allResults,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Aggregated Results for {datasetName} ({seedCount} seeds)")
        print(f"{'='*60}")
        print(f"Test Accuracy: {aggregated['meanAccuracy']*100:.2f}% ± {aggregated['stdAccuracy']*100:.2f}%")
        print(f"Test F1 Score: {aggregated['meanF1']*100:.2f}% ± {aggregated['stdF1']*100:.2f}%")
    
    return aggregated


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train MicroBiConvLSTM on HAR datasets'
    )
    parser.add_argument(
        '--dataset', type=str, default='ucihar',
        choices=['ucihar', 'motionsense', 'wisdm', 'pamap2', 
                 'opportunity', 'unimib', 'skoda', 'daphnet', 'all'],
        help='Dataset to train on (default: ucihar)'
    )
    parser.add_argument(
        '--seeds', type=int, default=5,
        help='Number of random seeds for training (default: 5)'
    )
    parser.add_argument(
        '--seed-list', type=str, default=None,
        help='Comma-separated explicit seeds to use instead of generating from MASTER_SEED'
    )
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Maximum number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--save-dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints (default: ./checkpoints)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    explicitSeeds = parseSeedList(args.seed_list)
    
    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]
    allAggregated = {}
    
    for dataset in datasets:
        aggregated = trainAllSeeds(
            datasetName=dataset,
            nSeeds=args.seeds,
            explicitSeeds=explicitSeeds,
            epochs=args.epochs,
            patience=args.patience,
            saveDir=args.save_dir,
            verbose=not args.quiet,
        )
        allAggregated[dataset] = aggregated
    
    # Save final summary
    summaryPath = os.path.join(args.save_dir, 'lightdeepconvlstm_training_summary.json')
    summary = {
        dataset: {
            'meanAccuracy': agg['meanAccuracy'],
            'stdAccuracy': agg['stdAccuracy'],
            'meanF1': agg['meanF1'],
            'stdF1': agg['stdF1'],
            'nSeeds': agg['nSeeds'],
        }
        for dataset, agg in allAggregated.items()
    }
    
    with open(summaryPath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Summary saved to: {summaryPath}")
    print(f"{'='*60}")
    
    # Print final summary table
    print(f"\n{'Dataset':<15} {'Accuracy':<20} {'F1 Score':<20}")
    print(f"{'-'*55}")
    for dataset, agg in allAggregated.items():
        accStr = f"{agg['meanAccuracy']*100:.2f}% ± {agg['stdAccuracy']*100:.2f}%"
        f1Str = f"{agg['meanF1']*100:.2f}% ± {agg['stdF1']*100:.2f}%"
        print(f"{dataset:<15} {accStr:<20} {f1Str:<20}")


if __name__ == '__main__':
    main()
