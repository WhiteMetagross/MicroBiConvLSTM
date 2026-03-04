"""Baseline Models Hyperparameter Optimization Script.

Tunes training hyperparameters (lr, weight_decay, dropout) for TinyHAR,
TinierHAR, and DeepConvLSTM using Optuna TPE sampler.

Usage:
    python hpoBaselines.py --model tinyhar --dataset ucihar --n-trials 50
    python hpoBaselines.py --model all --dataset all --n-trials 50
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

from baselines import TinyHAR, TinierHAR, DeepConvLSTM


MODEL_REGISTRY = {
    'tinyhar': {
        'name': 'TinyHAR',
        'class': TinyHAR,
        'params': '~42k',
    },
    'tinierhar': {
        'name': 'TinierHAR',
        'class': TinierHAR,
        'params': '~17k',
    },
    'deepconvlstm': {
        'name': 'DeepConvLSTM',
        'class': DeepConvLSTM,
        'params': '~132k',
    },
}


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
        'inputChannels': 60,
        'seqLen': 98,
        'numClasses': 11,
        'batchSize': 64
    },
    'daphnet': {
        'name': 'Daphnet',
        'inputChannels': 9,
        'seqLen': 128,
        'numClasses': 3,
        'batchSize': 64
    },
}


def setSeed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def getDevice() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def loadDataset(datasetName: str, batchSize: int):
    """Load train/test data loaders and optional class weights."""
    dataPath = Path(__file__).parent.parent / 'data'
    if str(dataPath) not in sys.path:
        sys.path.insert(0, str(dataPath))
    
    classWeights = None
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


def createModel(
    modelName: str,
    numClasses: int,
    inChannels: int,
    seqLen: int,
    dropout: float = 0.3,
) -> nn.Module:
    """Create a baseline model instance."""
    modelName = modelName.lower()
    
    if modelName == 'tinyhar':
        return TinyHAR(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
        )
    elif modelName == 'tinierhar':
        return TinierHAR(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
        )
    elif modelName == 'deepconvlstm':
        return DeepConvLSTM(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {modelName}. Choose from: {list(MODEL_REGISTRY.keys())}")


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
        
        optimizer.zero_grad()
        
        if useAmp:
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
    modelName: str,
    datasetName: str,
    trainLoader: DataLoader,
    valLoader: DataLoader,
    classWeights: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
) -> float:
    """Optuna objective: tunes lr, weight_decay, dropout. Returns best F1."""
    # Sample hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weightDecay = trial.suggest_float('weight_decay', 1e-5, 0.05, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    config = DATASET_CONFIGS[datasetName]
    
    # Create model
    model = createModel(
        modelName=modelName,
        numClasses=config['numClasses'],
        inChannels=config['inputChannels'],
        seqLen=config['seqLen'],
        dropout=dropout,
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
    modelName: str,
    datasetName: str,
    nTrials: int = 50,
    epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
    saveDir: str = './hpo_results',
    verbose: bool = True,
) -> dict:
    """Run HPO for a baseline model on a specific dataset. Returns results dict."""
    setSeed(seed)
    device = getDevice()
    
    modelInfo = MODEL_REGISTRY[modelName.lower()]
    config = DATASET_CONFIGS[datasetName]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HPO for {modelInfo['name']} ({modelInfo['params']}) on {config['name']}")
        print(f"Trials: {nTrials} | Epochs/trial: {epochs} | Device: {device}")
        print(f"{'='*60}")
    
    # Load dataset
    trainLoader, valLoader, classWeights = loadDataset(
        datasetName, config['batchSize']
    )
    
    # Create Optuna study
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    studyName = f'{modelName.lower()}_{datasetName}'
    study = optuna.create_study(
        direction='maximize',  # Maximize F1 score
        sampler=sampler,
        pruner=pruner,
        study_name=studyName,
    )
    
    # Run optimization
    startTime = time.time()
    
    study.optimize(
        lambda trial: objective(
            trial, modelName, datasetName, trainLoader, valLoader, 
            classWeights, device, epochs, patience
        ),
        n_trials=nTrials,
        show_progress_bar=verbose,
    )
    
    elapsedTime = time.time() - startTime
    
    # Get best results
    bestTrial = study.best_trial
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HPO Results for {modelInfo['name']} on {config['name']}")
        print(f"{'='*60}")
        print(f"Best F1 Score: {bestTrial.value:.4f}")
        print(f"Best Hyperparameters:")
        for key, value in bestTrial.params.items():
            if 'rate' in key or 'decay' in key:
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value:.4f}")
        print(f"Total time: {elapsedTime/60:.1f} minutes")
        print(f"Completed trials: {len(study.trials)}")
        print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Save results
    os.makedirs(saveDir, exist_ok=True)
    results = {
        'model': modelName,
        'modelName': modelInfo['name'],
        'modelParams': modelInfo['params'],
        'dataset': datasetName,
        'datasetName': config['name'],
        'bestF1': bestTrial.value,
        'bestParams': bestTrial.params,
        'nTrials': nTrials,
        'completedTrials': len(study.trials),
        'prunedTrials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'elapsedTimeMinutes': elapsedTime / 60,
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
    }
    
    # Save JSON
    jsonPath = os.path.join(saveDir, f'{studyName}_hpo.json')
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {jsonPath}")
    
    return results


def runAllHPO(
    modelName: str = 'all',
    datasetName: str = 'all',
    nTrials: int = 50,
    epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
    saveDir: str = './hpo_results',
    verbose: bool = True,
) -> list:
    """Run HPO for multiple models/datasets. Returns list of results."""
    models = list(MODEL_REGISTRY.keys()) if modelName.lower() == 'all' else [modelName.lower()]
    datasets = list(DATASET_CONFIGS.keys()) if datasetName.lower() == 'all' else [datasetName.lower()]
    
    allResults = []
    total = len(models) * len(datasets)
    current = 0
    
    for model in models:
        for dataset in datasets:
            current += 1
            if verbose:
                print(f"\n\n{'#'*60}")
                print(f"# Progress: {current}/{total}")
                print(f"# Model: {model} | Dataset: {dataset}")
                print(f"{'#'*60}")
            
            try:
                result = runHPO(
                    modelName=model,
                    datasetName=dataset,
                    nTrials=nTrials,
                    epochs=epochs,
                    patience=patience,
                    seed=seed,
                    saveDir=saveDir,
                    verbose=verbose,
                )
                allResults.append(result)
            except Exception as e:
                print(f"Error running HPO for {model} on {dataset}: {e}")
                allResults.append({
                    'model': model,
                    'dataset': dataset,
                    'error': str(e),
                })
    
    # Save summary
    if verbose and len(allResults) > 1:
        print(f"\n\n{'='*60}")
        print("HPO SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Dataset':<15} {'Best F1':>10}")
        print('-' * 45)
        for r in allResults:
            if 'error' not in r:
                print(f"{r['model']:<15} {r['dataset']:<15} {r['bestF1']:>10.4f}")
            else:
                print(f"{r['model']:<15} {r['dataset']:<15} {'ERROR':>10}")
    
    # Save combined results
    summaryPath = os.path.join(saveDir, 'baselines_hpo_summary.json')
    with open(summaryPath, 'w') as f:
        json.dump(allResults, f, indent=2)
    
    if verbose:
        print(f"\nSummary saved to: {summaryPath}")
    
    return allResults


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization for Baseline Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run HPO for TinyHAR on UCI-HAR
    python hpoBaselines.py --model tinyhar --dataset ucihar
    
    # Run HPO for all models on WISDM with 100 trials
    python hpoBaselines.py --model all --dataset wisdm --n-trials 100
    
    # Run HPO for DeepConvLSTM on all datasets
    python hpoBaselines.py --model deepconvlstm --dataset all
    
    # Run HPO for all models on all datasets
    python hpoBaselines.py --model all --dataset all
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='tinyhar',
        choices=['tinyhar', 'tinierhar', 'deepconvlstm', 'all'],
        help='Model to optimize (default: tinyhar)'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='ucihar',
        choices=list(DATASET_CONFIGS.keys()) + ['all'],
        help='Dataset to use (default: ucihar)'
    )
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='Number of HPO trials (default: 50)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Epochs per trial (default: 50)'
    )
    parser.add_argument(
        '--patience', '-p',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./hpo_results/baselines',
        help='Directory to save results (default: ./hpo_results/baselines)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    runAllHPO(
        modelName=args.model,
        datasetName=args.dataset,
        nTrials=args.n_trials,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        saveDir=args.save_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
