"""
Baseline Training Script for MicroBiConvLSTM Research Paper

Trains baseline models (TinyHAR, TinierHAR, DeepConvLSTM) on HAR datasets.
These baselines do NOT include Mamba-based models.

Usage:
    python trainBaselines.py --dataset ucihar --model all --seeds 5
    python trainBaselines.py --dataset all --model tinyhar --seeds 3
    python trainBaselines.py --dataset skoda --model deepconvlstm --seeds 5
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add parent directory to path for imports
_THIS_FILE = Path(__file__).resolve()
_LIGHTDEEPCONVLSTM_DIR = _THIS_FILE.parents[1]
_REPO_ROOT = _THIS_FILE.parents[2]

sys.path.insert(0, str(_LIGHTDEEPCONVLSTM_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from baselines import TinyHAR, TinierHAR, DeepConvLSTM


# ============== Constants ==============

MASTER_SEED = 17

MODELS = ['tinyhar', 'tinierhar', 'deepconvlstm']

DATASETS = ['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet']

DATASET_CONFIGS = {
    'ucihar': {
        'name': 'UCI-HAR',
        'inputChannels': 9,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64,
        'lr': 0.002,
        'weightDecay': 0.0001,
        'root': './datasets/UCI HAR Dataset'
    },
    'motionsense': {
        'name': 'MotionSense',
        'inputChannels': 6,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64,
        'lr': 0.002,
        'weightDecay': 0.0001,
        'root': './datasets/motion-sense-master'
    },
    'wisdm': {
        'name': 'WISDM',
        'inputChannels': 3,
        'seqLen': 128,
        'numClasses': 6,
        'batchSize': 64,
        'lr': 0.002,
        'weightDecay': 0.0001,
        'root': './datasets/WISDM_ar_v1.1'
    },
    'pamap2': {
        'name': 'PAMAP2',
        'inputChannels': 19,
        'seqLen': 128,
        'numClasses': 12,
        'batchSize': 64,
        'lr': 0.002,
        'weightDecay': 0.0001,
        'root': './datasets/PAMAP2_Dataset'
    },
    'opportunity': {
        'name': 'Opportunity',
        'inputChannels': 79,
        'seqLen': 128,
        'numClasses': 5,
        'batchSize': 64,
        'lr': 0.002,
        'weightDecay': 0.0001,
        'root': './datasets/Opportunity'
    },
    'unimib': {
        'name': 'UniMiB-SHAR',
        'inputChannels': 3,
        'seqLen': 128,
        'numClasses': 9,
        'batchSize': 64,
        'lr': 0.002,
        'weightDecay': 0.0001,
        'root': './datasets/UniMiB-SHAR'
    },
    'skoda': {
        'name': 'Skoda',
        'inputChannels': 30,
        'seqLen': 98,
        'numClasses': 11,
        'batchSize': 512,
        'lr': 0.002,
        'weightDecay': 0.01,
        'root': './datasets/Skoda'
    },
    'daphnet': {
        'name': 'Daphnet',
        'inputChannels': 9,
        'seqLen': 64,
        'numClasses': 2,
        'batchSize': 512,
        'lr': 0.002,
        'weightDecay': 0.01,
        'root': './datasets/Daphnet'
    },
}


# ============== Utilities ==============

def setSeed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generateRandomSeeds(masterSeed: int, count: int) -> List[int]:
    """Generate reproducible random seeds."""
    rng = np.random.default_rng(masterSeed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(count)]


def getDevice() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def loadDataset(datasetName: str, batchSize: int) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """Load train and test data loaders for the specified dataset."""
    dsName = datasetName.lower()
    classWeights = None
    
    if dsName == 'ucihar':
        from data.uciHar import getUciHarLoaders
        trainLoader, testLoader = getUciHarLoaders(
            root='./datasets/UCI HAR Dataset', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'motionsense':
        from data.motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(
            root='./datasets/motion-sense-master', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'wisdm':
        from data.wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(
            root='./datasets/WISDM_ar_v1.1', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'pamap2':
        from data.pamap2 import getPamap2Loaders
        trainLoader, testLoader, classWeights = getPamap2Loaders(
            root='./datasets/PAMAP2_Dataset', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'opportunity':
        from data.opportunity import getOpportunityLoaders
        trainLoader, testLoader, classWeights = getOpportunityLoaders(
            root='./datasets/Opportunity', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'unimib':
        from data.unimib import getUnimibLoaders
        trainLoader, testLoader = getUnimibLoaders(
            root='./datasets/UniMiB-SHAR', batchSize=batchSize, numWorkers=0
        )
    elif dsName == 'skoda':
        # Use LOCAL Skoda loader with OLD stratified split + shuffle
        from data.skoda import getSkodaLoaders
        trainLoader, testLoader, classWeights = getSkodaLoaders(
            root='./datasets/Skoda', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    elif dsName == 'daphnet':
        from data.daphnet import getDaphnetLoaders
        trainLoader, testLoader, classWeights = getDaphnetLoaders(
            root='./datasets/Daphnet', batchSize=batchSize, numWorkers=0, returnWeights=True
        )
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")
    
    return trainLoader, testLoader, classWeights


def createModel(modelName: str, config: dict) -> nn.Module:
    """Create a model instance based on name and config."""
    modelName = modelName.lower()
    
    if modelName == 'tinyhar':
        return TinyHAR(
            numClasses=config['numClasses'],
            inChannels=config['inputChannels'],
            seqLen=config['seqLen'],
        )
    elif modelName == 'tinierhar':
        return TinierHAR(
            numClasses=config['numClasses'],
            inChannels=config['inputChannels'],
            seqLen=config['seqLen'],
        )
    elif modelName == 'deepconvlstm':
        return DeepConvLSTM(
            numClasses=config['numClasses'],
            inChannels=config['inputChannels'],
            seqLen=config['seqLen'],
        )
    else:
        raise ValueError(f"Unknown model: {modelName}")


# ============== Training ==============

def trainEpoch(
    model: nn.Module,
    trainLoader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    useAmp: bool = True,
) -> Tuple[float, float, float]:
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
) -> Tuple[float, float, float, list, list]:
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
    modelName: str,
    datasetName: str,
    seed: int,
    epochs: int = 200,
    patience: int = 10,
    saveDir: str = './results/baselines',
    verbose: bool = True,
) -> dict:
    """Train a baseline model on a specific dataset with a given seed."""
    setSeed(seed)
    device = getDevice()
    config = DATASET_CONFIGS[datasetName.lower()]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {modelName.upper()} on {config['name']}")
        print(f"Seed: {seed} | Device: {device}")
        print(f"{'='*60}")
    
    # Load dataset
    trainLoader, testLoader, classWeights = loadDataset(
        datasetName, config['batchSize']
    )
    
    # Create model
    model = createModel(modelName, config).to(device)
    
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
    noBetterCount = 0
    history = {'train_loss': [], 'train_f1': [], 'test_loss': [], 'test_f1': []}
    
    startTime = time.time()
    
    for epoch in range(1, epochs + 1):
        trainLoss, trainAcc, trainF1 = trainEpoch(
            model, trainLoader, criterion, optimizer, device, scaler, useAmp
        )
        
        testLoss, testAcc, testF1, _, _ = evaluate(
            model, testLoader, criterion, device
        )
        
        scheduler.step()
        
        history['train_loss'].append(trainLoss)
        history['train_f1'].append(trainF1)
        history['test_loss'].append(testLoss)
        history['test_f1'].append(testF1)
        
        if testF1 > bestF1:
            bestF1 = testF1
            bestEpoch = epoch
            noBetterCount = 0
            bestState = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            noBetterCount += 1
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}: Train F1={trainF1:.4f} | Test F1={testF1:.4f} | Best={bestF1:.4f}")
        
        if noBetterCount >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    totalTime = time.time() - startTime
    
    # Final evaluation
    model.load_state_dict(bestState)
    _, _, finalF1, preds, labels = evaluate(model, testLoader, criterion, device)
    finalAcc = accuracy_score(labels, preds)
    
    if verbose:
        print(f"\nFinal Results:")
        print(f"  Test Accuracy: {finalAcc*100:.2f}%")
        print(f"  Test F1 (Macro): {finalF1*100:.2f}%")
        print(f"  Training Time: {totalTime:.1f}s")
    
    # Save results
    result = {
        'model': modelName,
        'dataset': datasetName,
        'seed': seed,
        'best_epoch': bestEpoch,
        'test_accuracy': float(finalAcc),
        'test_f1': float(finalF1),
        'training_time': totalTime,
        'parameters': sum(p.numel() for p in model.parameters()),
        'history': history,
    }
    
    # Save checkpoint
    saveDir = Path(saveDir) / modelName / datasetName
    saveDir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': bestState,
        'config': config,
        'result': result,
    }, saveDir / f"seed{seed}_checkpoint.pt")
    
    with open(saveDir / f"seed{seed}_results.json", 'w') as f:
        # Convert numpy arrays in history to lists for JSON serialization
        resultForJson = {k: v for k, v in result.items() if k != 'history'}
        json.dump(resultForJson, f, indent=2)
    
    return result


def runMultiSeed(
    modelName: str,
    datasetName: str,
    numSeeds: int = 5,
    epochs: int = 200,
    patience: int = 10,
    saveDir: str = './results/baselines',
) -> dict:
    """Run training with multiple seeds and aggregate results."""
    seeds = generateRandomSeeds(MASTER_SEED, numSeeds)
    results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"# Run {i+1}/{numSeeds} - Seed: {seed}")
        print(f"{'#'*60}")
        
        result = trainModel(
            modelName=modelName,
            datasetName=datasetName,
            seed=seed,
            epochs=epochs,
            patience=patience,
            saveDir=saveDir,
        )
        results.append(result)
    
    # Aggregate
    accuracies = [r['test_accuracy'] for r in results]
    f1s = [r['test_f1'] for r in results]
    
    summary = {
        'model': modelName,
        'dataset': datasetName,
        'num_seeds': numSeeds,
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'individual_results': results,
    }
    
    # Save summary
    saveDir = Path(saveDir) / modelName / datasetName
    saveDir.mkdir(parents=True, exist_ok=True)
    
    with open(saveDir / "summary.json", 'w') as f:
        summaryForJson = {k: v for k, v in summary.items() if k != 'individual_results'}
        json.dump(summaryForJson, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {modelName.upper()} on {datasetName.upper()}")
    print(f"  Accuracy: {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}%")
    print(f"  F1 Score: {summary['f1_mean']*100:.2f}% ± {summary['f1_std']*100:.2f}%")
    print(f"{'='*60}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Training for Baseline Models")
    parser.add_argument('--model', type=str, default='all',
                        choices=MODELS + ['all'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, default='ucihar',
                        choices=DATASETS + ['all'],
                        help='Dataset to use')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--outDir', type=str, default='./results/baselines',
                        help='Output directory')
    
    args = parser.parse_args()
    
    models = MODELS if args.model == 'all' else [args.model]
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    
    allSummaries = []
    
    for model in models:
        for dataset in datasets:
            try:
                summary = runMultiSeed(
                    modelName=model,
                    datasetName=dataset,
                    numSeeds=args.seeds,
                    epochs=args.epochs,
                    patience=args.patience,
                    saveDir=args.outDir,
                )
                allSummaries.append(summary)
            except Exception as e:
                print(f"ERROR training {model} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<15} {'Dataset':<15} {'Accuracy':<20} {'F1 Score':<20}")
    print("-"*80)
    for s in allSummaries:
        acc = f"{s['accuracy_mean']*100:.2f}% ± {s['accuracy_std']*100:.2f}%"
        f1 = f"{s['f1_mean']*100:.2f}% ± {s['f1_std']*100:.2f}%"
        print(f"{s['model']:<15} {s['dataset']:<15} {acc:<20} {f1:<20}")
    print("="*80)


if __name__ == '__main__':
    main()
