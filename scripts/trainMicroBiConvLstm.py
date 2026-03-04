"""
MicroBiConvLSTM Training Script.

Trains the MicroBiConvLSTM architecture on HAR benchmark datasets with
multi-seed evaluation, early stopping, and AMP/FP16 acceleration.

Usage:
    python scripts/trainMicroBiConvLstm.py --dataset ucihar --seeds 5
    python scripts/trainMicroBiConvLstm.py --dataset all --seeds 5

Author: Mridankan Mandal
Paper: https://arxiv.org/abs/2602.06523
"""

import os
import sys
import argparse
import random
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import MicroBiConvLSTM


# HPO-tuned training hyperparameters per dataset. Architecture is frozen.
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
        'dropout': 0.15,
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
        'dropout': 0.12,
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
        'dropout': 0.08,
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
        'dropout': 0.10,
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
        'dropout': 0.05,
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
        'dropout': 0.10,
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
        'dropout': 0.08,
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
        'dropout': 0.00,
    },
}

MASTER_SEED = 17


def generateRandomSeeds(nSeeds: int, masterSeed: int = MASTER_SEED) -> list:
    """Generate reproducible random seeds from a master seed."""
    rng = np.random.default_rng(masterSeed)
    return [int(s) for s in rng.integers(0, 100000, size=nSeeds)]


def setSeed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getDevice():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def loadDataset(datasetName: str, batchSize: int):
    """Load train/test data loaders and optional class weights for a dataset."""
    dataPath = Path(__file__).parent.parent / 'data'
    if str(dataPath) not in sys.path:
        sys.path.insert(0, str(dataPath))

    classWeights = None
    dsName = datasetName.lower()

    if dsName == 'ucihar':
        from uciHar import getUciHarLoaders
        trainLoader, testLoader = getUciHarLoaders(root='./datasets/UCI HAR Dataset', batchSize=batchSize, numWorkers=0)
    elif dsName == 'motionsense':
        from motionSense import getMotionSenseLoaders
        trainLoader, testLoader = getMotionSenseLoaders(root='./datasets/motion-sense-master', batchSize=batchSize, numWorkers=0)
    elif dsName == 'wisdm':
        from wisdm import getWisdmLoaders
        trainLoader, testLoader = getWisdmLoaders(root='./datasets/WISDM_ar_v1.1', batchSize=batchSize, numWorkers=0)
    elif dsName == 'pamap2':
        from pamap2 import getPamap2Loaders
        trainLoader, testLoader, classWeights = getPamap2Loaders(root='./datasets/PAMAP2_Dataset', batchSize=batchSize, numWorkers=0, returnWeights=True)
    elif dsName == 'opportunity':
        from opportunity import getOpportunityLoaders
        trainLoader, testLoader, classWeights = getOpportunityLoaders(root='./datasets/Opportunity', batchSize=batchSize, numWorkers=0, returnWeights=True)
    elif dsName == 'unimib':
        from unimib import getUnimibLoaders
        trainLoader, testLoader = getUnimibLoaders(root='./datasets/UniMiB-SHAR', batchSize=batchSize, numWorkers=0)
    elif dsName == 'skoda':
        from skoda import getSkodaLoaders
        trainLoader, testLoader, classWeights = getSkodaLoaders(root='./datasets/Skoda', batchSize=batchSize, numWorkers=0, returnWeights=True)
    elif dsName == 'daphnet':
        from daphnet import getDaphnetLoaders
        trainLoader, testLoader, classWeights = getDaphnetLoaders(root='./datasets/Daphnet', batchSize=batchSize, numWorkers=0, returnWeights=True)
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")

    return trainLoader, testLoader, classWeights


def trainEpoch(model, trainLoader, criterion, optimizer, device, scaler, useAmp=True):
    """Train for one epoch. Returns (loss, accuracy, f1)."""
    model.train()
    totalLoss = 0.0
    allPreds = []
    allLabels = []

    for x, y in trainLoader:
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
        allPreds.extend(logits.argmax(dim=-1).cpu().numpy())
        allLabels.extend(y.cpu().numpy())

    avgLoss = totalLoss / len(trainLoader.dataset)
    accuracy = accuracy_score(allLabels, allPreds)
    f1 = f1_score(allLabels, allPreds, average='macro', zero_division=0)
    return avgLoss, accuracy, f1


@torch.no_grad()
def evaluate(model, dataLoader, criterion, device):
    """Evaluate model. Returns (loss, accuracy, f1, predictions, labels)."""
    model.eval()
    totalLoss = 0.0
    allPreds = []
    allLabels = []

    for x, y in dataLoader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        totalLoss += loss.item() * x.size(0)
        allPreds.extend(logits.argmax(dim=-1).cpu().numpy())
        allLabels.extend(y.cpu().numpy())

    avgLoss = totalLoss / len(dataLoader.dataset)
    accuracy = accuracy_score(allLabels, allPreds)
    f1 = f1_score(allLabels, allPreds, average='macro', zero_division=0)
    return avgLoss, accuracy, f1, allPreds, allLabels


def trainModel(datasetName, seed, epochs=200, patience=10, saveDir='./checkpoints', verbose=True):
    """Train MicroBiConvLSTM on a dataset with a given seed. Returns results dict."""
    setSeed(seed)
    device = getDevice()
    config = DATASET_CONFIGS[datasetName.lower()]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training MicroBiConvLSTM on {config['name']}")
        print(f"Seed: {seed} | Device: {device}")
        print(f"{'=' * 60}")

    trainLoader, testLoader, classWeights = loadDataset(datasetName, config['batchSize'])

    model = MicroBiConvLSTM(
        numClasses=config['numClasses'],
        inChannels=config['inputChannels'],
        seqLen=config['seqLen'],
        dropout=config['dropout'],
        aggregation='last',
    ).to(device)

    if verbose:
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if classWeights is not None:
        criterion = nn.CrossEntropyLoss(weight=classWeights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    useAmp = device.type == 'cuda'

    bestF1 = 0.0
    bestEpoch = 0
    bestState = None
    noImproveCount = 0
    history = {'trainLoss': [], 'trainAcc': [], 'trainF1': [], 'testLoss': [], 'testAcc': [], 'testF1': []}

    startTime = time.time()

    for epoch in range(1, epochs + 1):
        trainLoss, trainAcc, trainF1 = trainEpoch(model, trainLoader, criterion, optimizer, device, scaler, useAmp)
        testLoss, testAcc, testF1, _, _ = evaluate(model, testLoader, criterion, device)
        scheduler.step()

        history['trainLoss'].append(trainLoss)
        history['trainAcc'].append(trainAcc)
        history['trainF1'].append(trainF1)
        history['testLoss'].append(testLoss)
        history['testAcc'].append(testAcc)
        history['testF1'].append(testF1)

        if testF1 > bestF1:
            bestF1 = testF1
            bestEpoch = epoch
            bestState = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            noImproveCount = 0
        else:
            noImproveCount += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {trainLoss:.4f} | Train F1: {trainF1:.4f} | "
                  f"Test Loss: {testLoss:.4f} | Test F1: {testF1:.4f} | "
                  f"Best F1: {bestF1:.4f} (E{bestEpoch})")

        if noImproveCount >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    trainingTime = time.time() - startTime

    model.load_state_dict(bestState)
    testLoss, testAcc, testF1, allPreds, allLabels = evaluate(model, testLoader, criterion, device)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Final Results (Best Model from Epoch {bestEpoch})")
        print(f"{'=' * 60}")
        print(f"Test Accuracy: {testAcc * 100:.2f}%")
        print(f"Test F1 Score: {testF1 * 100:.2f}%")
        print(f"Training Time: {trainingTime:.1f}s")

    results = {
        'dataset': datasetName,
        'seed': seed,
        'bestEpoch': bestEpoch,
        'testAccuracy': testAcc,
        'testF1': testF1,
        'testLoss': testLoss,
        'trainingTime': trainingTime,
        'history': history,
        'predictions': allPreds,
        'labels': allLabels,
        'config': config,
    }

    os.makedirs(saveDir, exist_ok=True)
    checkpointPath = os.path.join(saveDir, f"microBiConvLstm_{datasetName}_seed{seed}.pt")
    torch.save({'model_state_dict': bestState, 'results': results}, checkpointPath)

    return results


def trainAllSeeds(datasetName, nSeeds=5, epochs=200, patience=10, saveDir='./checkpoints', verbose=True):
    """Train MicroBiConvLSTM with multiple seeds and aggregate results."""
    seeds = generateRandomSeeds(nSeeds)
    allResults = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#' * 60}")
            print(f"Training Run {i + 1}/{nSeeds} | Seed: {seed}")
            print(f"{'#' * 60}")

        results = trainModel(datasetName, seed, epochs, patience, saveDir, verbose)
        allResults.append(results)

    accuracies = [r['testAccuracy'] for r in allResults]
    f1Scores = [r['testF1'] for r in allResults]

    aggregated = {
        'dataset': datasetName,
        'nSeeds': nSeeds,
        'seeds': seeds,
        'meanAccuracy': np.mean(accuracies),
        'stdAccuracy': np.std(accuracies),
        'meanF1': np.mean(f1Scores),
        'stdF1': np.std(f1Scores),
        'allResults': allResults,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Aggregated Results for {datasetName} ({nSeeds} seeds)")
        print(f"{'=' * 60}")
        print(f"Test Accuracy: {aggregated['meanAccuracy'] * 100:.2f}% +/- {aggregated['stdAccuracy'] * 100:.2f}%")
        print(f"Test F1 Score: {aggregated['meanF1'] * 100:.2f}% +/- {aggregated['stdF1'] * 100:.2f}%")

    return aggregated


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train MicroBiConvLSTM on HAR datasets.')
    parser.add_argument('--dataset', type=str, default='ucihar',
                        choices=list(DATASET_CONFIGS.keys()) + ['all'],
                        help='Dataset to train on (default: ucihar).')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds (default: 5).')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum training epochs (default: 200).')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10).')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint directory.')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity.')
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]
    allAggregated = {}

    for dataset in datasets:
        aggregated = trainAllSeeds(dataset, args.seeds, args.epochs, args.patience, args.save_dir, not args.quiet)
        allAggregated[dataset] = aggregated

    summaryPath = os.path.join(args.save_dir, 'microBiConvLstm_training_summary.json')
    summary = {
        dataset: {
            'meanAccuracy': agg['meanAccuracy'], 'stdAccuracy': agg['stdAccuracy'],
            'meanF1': agg['meanF1'], 'stdF1': agg['stdF1'], 'nSeeds': agg['nSeeds'],
        }
        for dataset, agg in allAggregated.items()
    }

    with open(summaryPath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training Complete. Summary saved to: {summaryPath}")
    print(f"{'=' * 60}")
    print(f"\n{'Dataset':<15} {'Accuracy':<20} {'F1 Score':<20}")
    print(f"{'-' * 55}")
    for dataset, agg in allAggregated.items():
        accStr = f"{agg['meanAccuracy'] * 100:.2f}% +/- {agg['stdAccuracy'] * 100:.2f}%"
        f1Str = f"{agg['meanF1'] * 100:.2f}% +/- {agg['stdF1'] * 100:.2f}%"
        print(f"{dataset:<15} {accStr:<20} {f1Str:<20}")


if __name__ == '__main__':
    main()
