"""Ablation studies runner for MicroBiConvLSTM.

Implements architectural ablations (A0-A4), efficiency proxies (Pareto, complexity,
INT8 PTQ), interpretability (1D Grad-CAM), and robustness evaluations.

Usage:
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study arch --seeds 3
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study pareto --seeds 1
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study complexity
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study quant
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset ucihar --study gradcam --gradcam-class "Walking"
    python scripts/ablationStudiesMicroBiConvLstm.py --dataset pamap2 --study sensitivity
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from sklearn.metrics import f1_score, accuracy_score

# Ensure local imports work whether run from repo root or module folder
_THIS_FILE = Path(__file__).resolve()
_REPO_DIR = _THIS_FILE.parents[1]
_REPO_ROOT = _THIS_FILE.parents[2]

import sys

sys.path.insert(0, str(_REPO_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from models.microBiConvLstm import MicroBiConvLSTM
from models.microBiConvLstmVariants import createVariantModel, makeVariantSpec

from scripts.trainMicroBiConvLstm import (
    DATASET_CONFIGS,
    MASTER_SEED,
    generateRandomSeeds,
    setSeed,
    getDevice,
    trainEpoch,
)


def _safe_torch_load(path: Path):
    """Load a torch checkpoint safely across torch versions.

    Newer PyTorch versions support weights_only=True; older ones don't.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_latest_checkpoint(dataset: str, *, variant_prefix: str = "A0") -> Path:
    """Find the newest checkpoint for a specific variant prefix (default A0 baseline)."""
    arch_dir = dataset_dir(dataset) / "arch"
    if not arch_dir.exists():
        raise RuntimeError(
            f"No arch directory found under {arch_dir}. Run --study arch first to create checkpoints."
        )

    # Prefer checkpoints whose parent folder name starts with the requested prefix.
    candidates = [p for p in arch_dir.glob(f"{variant_prefix}_*/checkpoint.pt")]
    if not candidates:
        # Fallback to any nested checkpoint (keeps behavior but makes it explicit).
        candidates = list(arch_dir.glob("**/checkpoint.pt"))

    if not candidates:
        raise RuntimeError(
            f"No checkpoint.pt found under {arch_dir}. Run --study arch first to create checkpoints."
        )

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


# --------------------------
# Output + paths
# --------------------------

def out_dir() -> Path:
    p = _REPO_DIR / "results" / "ablations"
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_dir(dataset: str) -> Path:
    p = out_dir() / dataset.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------
# Dataset loading (absolute paths + preprocessing bypass)
# --------------------------

def _datasets_root() -> Path:
    return _REPO_ROOT / "datasets"


def load_dataset(
    dataset_name: str,
    batch_size: int,
    *,
    bypass_preprocessing: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.Tensor]]:
    """Load train/test loaders with optional preprocessing bypass."""
    dataPath = Path(__file__).parent.parent / 'data'
    if str(dataPath) not in sys.path:
        sys.path.insert(0, str(dataPath))

    ds = dataset_name.lower()

    if ds == "ucihar":
        from uciHar import getUciHarLoaders

        return (*getUciHarLoaders(root=str(_datasets_root() / "UCI HAR Dataset"), batchSize=batch_size, numWorkers=0), None)

    if ds == "motionsense":
        from motionSense import getMotionSenseLoaders

        return (*getMotionSenseLoaders(root=str(_datasets_root() / "motion-sense-master"), batchSize=batch_size, numWorkers=0), None)

    if ds == "wisdm":
        from wisdm import getWisdmLoaders

        return (*getWisdmLoaders(root=str(_datasets_root() / "WISDM_ar_v1.1"), batchSize=batch_size, numWorkers=0), None)

    if ds == "opportunity":
        from opportunity import getOpportunityLoaders

        train_loader, test_loader, class_weights = getOpportunityLoaders(
            root=str(_datasets_root() / "Opportunity"),
            batchSize=batch_size,
            numWorkers=0,
            returnWeights=True,
        )
        return train_loader, test_loader, class_weights

    if ds == "unimib":
        from unimib import getUnimibLoaders

        return (*getUnimibLoaders(root=str(_datasets_root() / "UniMiB-SHAR"), batchSize=batch_size, numWorkers=0), None)

    if ds == "pamap2":
        # Need direct dataset construction to control applyFilter.
        # Import from main nanoharmamba for direct class access (not affected by Skoda split change)
        from nanoharmamba.data.pamap2 import Pamap2Dataset, computeClassWeights, ACTIVITY_IDS
        from torch.utils.data import DataLoader

        train_ds = Pamap2Dataset(
            root=str(_datasets_root() / "PAMAP2_Dataset"),
            split="train",
            windowSize=128,
            stride=64,
            channels="compact",
            normalize=True,
            applyFilter=not bypass_preprocessing,
            filterCutoff=10.0,
            useRobustScaling=True,
        )

        test_ds = Pamap2Dataset(
            root=str(_datasets_root() / "PAMAP2_Dataset"),
            split="test",
            windowSize=128,
            stride=32,
            channels="compact",
            normalize=False,
            applyFilter=not bypass_preprocessing,
            filterCutoff=10.0,
            useRobustScaling=True,
        )

        # Apply train normalization to test
        if len(train_ds.windows) > 0 and len(test_ds.windows) > 0:
            if train_ds.useRobustScaling and train_ds.median is not None:
                test_ds.median = train_ds.median
                test_ds.iqr = train_ds.iqr
                test_ds.windows = (test_ds.windows - train_ds.median) / train_ds.iqr
                test_ds.normalize = True
                test_ds.useRobustScaling = True
            elif train_ds.mean is not None:
                test_ds.mean = train_ds.mean
                test_ds.std = train_ds.std
                test_ds.windows = (test_ds.windows - train_ds.mean) / train_ds.std
                test_ds.normalize = True

        class_weights = None
        if len(train_ds.labels) > 0:
            class_weights = computeClassWeights(train_ds.labels, numClasses=len(ACTIVITY_IDS))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return train_loader, test_loader, class_weights

    if ds == "skoda":
        from skoda import SkodaDataset, computeSkodaClassWeights, NUM_CLASSES
        from torch.utils.data import DataLoader

        train_ds = SkodaDataset(
            root=str(_datasets_root() / "Skoda"),
            split="train",
            windowSize=98,
            stride=24,
            normalize=True,
            applyFilter=not bypass_preprocessing,
            filterCutoff=5.0,
        )
        test_ds = SkodaDataset(
            root=str(_datasets_root() / "Skoda"),
            split="test",
            windowSize=98,
            stride=98,
            normalize=False,
            applyFilter=not bypass_preprocessing,
            filterCutoff=5.0,
        )

        if len(train_ds.windows) > 0 and len(test_ds.windows) > 0:
            test_ds.mean = train_ds.mean
            test_ds.std = train_ds.std
            test_ds.windows = (test_ds.windows - test_ds.mean) / test_ds.std
            test_ds.normalize = True

        class_weights = None
        if len(train_ds.labels) > 0:
            class_weights = computeSkodaClassWeights(train_ds.labels, numClasses=NUM_CLASSES)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return train_loader, test_loader, class_weights

    if ds == "daphnet":
        from daphnet import getDaphnetLoaders

        train_loader, test_loader, class_weights = getDaphnetLoaders(
            root=str(_datasets_root() / "Daphnet"),
            batchSize=batch_size,
            numWorkers=0,
            returnWeights=True,
            applyFilter=not bypass_preprocessing,
            filterCutoff=12.0,
            aggressiveWeights=True,
        )
        return train_loader, test_loader, class_weights

    raise ValueError(f"Unknown dataset: {dataset_name}")


# --------------------------
# Metrics + proxies
# --------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(
    *,
    seq_len: int,
    in_channels: int,
    num_classes: int,
    conv_filters: int,
    kernel: int,
    use_conv2: bool,
    use_pool1: bool,
    use_pool2: bool,
    lstm_hidden: int,
    bidirectional: bool,
) -> int:
    """Analytical MAC estimate per *sequence* (not per batch).

    Conventions:
    - Conv1d stride=1, same padding => output length equals input length.
    - MaxPool1d(kernel=2,stride=2) => length halves.
    - LSTM MACs use standard gate computations: 4*h*(in+h) per timestep.

    Returns:
        Estimated MACs as integer.
    """

    length = int(seq_len)

    # Conv1 MACs: L * Cout * (Cin * K)
    macs = length * conv_filters * (in_channels * kernel)

    # Pool1 affects length
    if use_pool1:
        length = length // 2

    if use_conv2:
        macs += length * conv_filters * (conv_filters * kernel)

    if use_pool2:
        length = length // 2

    # LSTM over length timesteps
    directions = 2 if bidirectional else 1
    # per timestep per direction
    macs_per_timestep = 4 * lstm_hidden * (conv_filters + lstm_hidden)
    macs += length * directions * macs_per_timestep

    # Classifier
    lstm_out = lstm_hidden * directions
    macs += lstm_out * num_classes

    return int(macs)


@torch.no_grad()
def evaluate_simple(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "accuracy": float(acc), "f1": float(f1)}


def train_variant(
    *,
    dataset: str,
    seed: int,
    variant_id: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    bypass_preprocessing: bool = False,
    conv_filters: Optional[int] = None,
    lstm_hidden: Optional[int] = None,
) -> Dict:
    setSeed(seed)
    device = getDevice()

    config = DATASET_CONFIGS[dataset]
    train_loader, test_loader, class_weights = load_dataset(
        dataset, config["batchSize"], bypass_preprocessing=bypass_preprocessing
    )

    spec = makeVariantSpec(
        variantId=variant_id,
        numClasses=config["numClasses"],
        inChannels=config["inputChannels"],
        seqLen=config["seqLen"],
        dropout=dropout,
        convFilters=conv_filters,
        lstmHidden=lstm_hidden,
    )

    model = createVariantModel(spec).to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = GradScaler()
    use_amp = device.type == "cuda"

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    history = {
        "trainLoss": [],
        "trainF1": [],
        "testLoss": [],
        "testF1": [],
    }

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, _, train_f1 = trainEpoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        metrics = evaluate_simple(model, test_loader, criterion, device)

        scheduler.step()

        history["trainLoss"].append(float(train_loss))
        history["trainF1"].append(float(train_f1))
        history["testLoss"].append(float(metrics["loss"]))
        history["testF1"].append(float(metrics["f1"]))

        if metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    train_time = time.time() - start

    # Final eval best model
    assert best_state is not None
    model.load_state_dict(best_state)
    final_metrics = evaluate_simple(model, test_loader, criterion, device)

    result = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant_id,
        "spec": asdict(spec),
        "fixed_protocol": {"optimizer": "AdamW", "lr": lr, "weightDecay": weight_decay},
        "bestEpoch": best_epoch,
        "final": final_metrics,
        "trainingTimeSec": float(train_time),
        "history": history,
        "params": int(count_params(model)),
        "macs": int(
            estimate_macs(
                seq_len=config["seqLen"],
                in_channels=config["inputChannels"],
                num_classes=config["numClasses"],
                conv_filters=spec.convFilters,
                kernel=spec.convKernel,
                use_conv2=spec.useConv2,
                use_pool1=spec.usePool1,
                use_pool2=spec.usePool2,
                lstm_hidden=spec.lstmHidden,
                bidirectional=spec.bidirectional,
            )
        ),
        "bypassPreprocessing": bool(bypass_preprocessing),
    }

    return result, best_state


def save_run_artifacts(
    *,
    dataset: str,
    study: str,
    run_name: str,
    result: Dict,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Path:
    base = dataset_dir(dataset) / study
    base.mkdir(parents=True, exist_ok=True)

    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if state_dict is not None:
        torch.save({"model_state_dict": state_dict, "result": result}, run_dir / "checkpoint.pt")

    return run_dir


# --------------------------
# Studies
# --------------------------

def study_arch(args) -> None:
    dataset = args.dataset
    ds_cfg = DATASET_CONFIGS[dataset]

    # Baseline HPO hyperparams used for controlled experiments (except LR fixed to 1e-3)
    weight_decay = float(ds_cfg["weightDecay"])
    dropout = float(ds_cfg["dropout"])
    lr = float(args.lr)

    seeds = generateRandomSeeds(args.seeds, masterSeed=MASTER_SEED)

    # A0 must be the frozen base model; train once and reuse for A4.
    a0_results = []
    a0_state = None

    for seed in seeds:
        res, state = train_variant(
            dataset=dataset,
            seed=seed,
            variant_id="A0",
            epochs=args.epochs,
            patience=args.patience,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        run_dir = save_run_artifacts(
            dataset=dataset,
            study="arch",
            run_name=f"A0_seed{seed}_{now_tag()}",
            result=res,
            state_dict=state,
        )
        a0_results.append({"runDir": str(run_dir), **res})
        a0_state = state  # keep last seed for A4 eval-only

    # A4 eval-only: load A0 weights but change aggregation to mean.
    if a0_state is not None:
        # Use the variant model spec for A4 but with the same parameters.
        seed = seeds[0]
        setSeed(seed)
        device = getDevice()

        train_loader, test_loader, class_weights = load_dataset(dataset, ds_cfg["batchSize"], bypass_preprocessing=False)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        spec_a4 = makeVariantSpec(
            variantId="A4",
            numClasses=ds_cfg["numClasses"],
            inChannels=ds_cfg["inputChannels"],
            seqLen=ds_cfg["seqLen"],
            dropout=dropout,
        )
        model_a4 = createVariantModel(spec_a4).to(device)
        model_a4.load_state_dict(a0_state, strict=False)
        metrics_a4 = evaluate_simple(model_a4, test_loader, criterion, device)

        res_a4 = {
            "dataset": dataset,
            "seed": seed,
            "variant": "A4",
            "spec": asdict(spec_a4),
            "final": metrics_a4,
            "params": int(count_params(model_a4)),
            "macs": int(
                estimate_macs(
                    seq_len=ds_cfg["seqLen"],
                    in_channels=ds_cfg["inputChannels"],
                    num_classes=ds_cfg["numClasses"],
                    conv_filters=spec_a4.convFilters,
                    kernel=spec_a4.convKernel,
                    use_conv2=spec_a4.useConv2,
                    use_pool1=spec_a4.usePool1,
                    use_pool2=spec_a4.usePool2,
                    lstm_hidden=spec_a4.lstmHidden,
                    bidirectional=spec_a4.bidirectional,
                )
            ),
            "note": "Eval-only: loaded A0 weights, changed aggregation to mean",
        }
        save_run_artifacts(
            dataset=dataset,
            study="arch",
            run_name=f"A4_evalonly_fromA0_{now_tag()}",
            result=res_a4,
            state_dict=None,
        )

    # Retrain-required variants
    for variant_id in ["A1", "A2", "A3"]:
        for seed in seeds:
            res, state = train_variant(
                dataset=dataset,
                seed=seed,
                variant_id=variant_id,
                epochs=args.epochs,
                patience=args.patience,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
            )
            save_run_artifacts(
                dataset=dataset,
                study="arch",
                run_name=f"{variant_id}_seed{seed}_{now_tag()}",
                result=res,
                state_dict=state,
            )


def study_pareto(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = args.dataset
    ds_cfg = DATASET_CONFIGS[dataset]

    weight_decay = float(ds_cfg["weightDecay"])
    dropout = float(ds_cfg["dropout"])
    lr = float(args.lr)

    seeds = generateRandomSeeds(args.seeds, masterSeed=MASTER_SEED)

    conv_grid = [int(x) for x in args.conv_grid.split(",")]
    lstm_grid = [int(x) for x in args.lstm_grid.split(",")]

    # Collect results in-memory so we can plot a Pareto curve without cross-run aggregation.
    collected = []

    for conv_filters in conv_grid:
        for lstm_hidden in lstm_grid:
            for seed in seeds:
                res, state = train_variant(
                    dataset=dataset,
                    seed=seed,
                    variant_id="A0",
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=lr,
                    weight_decay=weight_decay,
                    dropout=dropout,
                    conv_filters=conv_filters,
                    lstm_hidden=lstm_hidden,
                )
                res["pareto"] = {"convFilters": conv_filters, "lstmHidden": lstm_hidden}
                save_run_artifacts(
                    dataset=dataset,
                    study="pareto",
                    run_name=f"conv{conv_filters}_lstm{lstm_hidden}_seed{seed}_{now_tag()}",
                    result=res,
                    state_dict=state,
                )
                collected.append(res)

    # Build a per-(conv,lstm) aggregate (mean over seeds)
    by_cfg: Dict[Tuple[int, int], List[Dict]] = {}
    for r in collected:
        key = (int(r["pareto"]["convFilters"]), int(r["pareto"]["lstmHidden"]))
        by_cfg.setdefault(key, []).append(r)

    points = []
    for (cf, lh), runs in by_cfg.items():
        f1s = [float(rr["final"]["f1"]) for rr in runs]
        params = int(runs[0]["params"])
        macs = int(runs[0]["macs"])
        points.append({"convFilters": cf, "lstmHidden": lh, "params": params, "macs": macs, "f1_mean": float(np.mean(f1s))})

    # Pareto frontier (maximize f1, minimize params)
    points_sorted = sorted(points, key=lambda d: (d["params"], -d["f1_mean"]))
    frontier = []
    best_f1 = -1.0
    for p in points_sorted:
        if p["f1_mean"] > best_f1:
            frontier.append(p)
            best_f1 = p["f1_mean"]

    out = dataset_dir(dataset) / "pareto"
    out.mkdir(parents=True, exist_ok=True)

    # Plot
    xs = [p["params"] / 1000.0 for p in points]
    ys = [p["f1_mean"] * 100.0 for p in points]

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xs, ys, s=35)
    for p in points:
        ax.annotate(f"c{p['convFilters']},h{p['lstmHidden']}", (p["params"] / 1000.0, p["f1_mean"] * 100.0), fontsize=7, alpha=0.8)

    fx = [p["params"] / 1000.0 for p in frontier]
    fy = [p["f1_mean"] * 100.0 for p in frontier]
    ax.plot(fx, fy, color="crimson", linewidth=2, label="Pareto frontier")

    ax.set_title(f"Pareto: Params vs F1 ({dataset.upper()})")
    ax.set_xlabel("Parameters (K)")
    ax.set_ylabel("Macro F1 (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = out / f"pareto_{now_tag()}.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    (out / f"pareto_{now_tag()}.json").write_text(
        json.dumps({"dataset": dataset, "points": points, "frontier": frontier, "plot": str(plot_path)}, indent=2),
        encoding="utf-8",
    )


def study_complexity(args) -> None:
    dataset = args.dataset
    ds_cfg = DATASET_CONFIGS[dataset]

    # Use frozen reference specs
    in_channels = int(ds_cfg["inputChannels"])
    num_classes = int(ds_cfg["numClasses"])

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    rows = []
    for seq_len in seq_lens:
        macs = estimate_macs(
            seq_len=seq_len,
            in_channels=in_channels,
            num_classes=num_classes,
            conv_filters=16,
            kernel=5,
            use_conv2=True,
            use_pool1=True,
            use_pool2=True,
            lstm_hidden=24,
            bidirectional=True,
        )
        rows.append({"seqLen": seq_len, "macs": int(macs)})

    out = dataset_dir(dataset) / "complexity"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"complexity_{now_tag()}.json").write_text(json.dumps({"dataset": dataset, "rows": rows}, indent=2))


def study_quant(args) -> None:
    dataset = args.dataset
    ds_cfg = DATASET_CONFIGS[dataset]

    ckpt_path = _find_latest_checkpoint(dataset, variant_prefix="A0")
    payload = _safe_torch_load(ckpt_path)
    state = payload["model_state_dict"]

    # Build baseline A0 model and evaluate FP32 on CPU for comparability with quant.
    dropout = float(ds_cfg["dropout"])

    spec = makeVariantSpec(
        variantId="A0",
        numClasses=ds_cfg["numClasses"],
        inChannels=ds_cfg["inputChannels"],
        seqLen=ds_cfg["seqLen"],
        dropout=dropout,
    )
    model_fp32 = createVariantModel(spec)
    model_fp32.load_state_dict(state, strict=False)
    model_fp32.eval()

    train_loader, test_loader, class_weights = load_dataset(dataset, ds_cfg["batchSize"], bypass_preprocessing=False)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    fp32 = evaluate_simple(model_fp32, test_loader, criterion, device=torch.device("cpu"))

    # PTQ (dynamic quantization) - weights to INT8 for LSTM+Linear
    try:
        from torch.ao.quantization import quantize_dynamic

        model_int8 = quantize_dynamic(model_fp32, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
        int8 = evaluate_simple(model_int8, test_loader, criterion, device=torch.device("cpu"))
        quant_error = float(fp32["f1"] - int8["f1"])
    except Exception as e:
        int8 = None
        quant_error = None
        err = str(e)
    else:
        err = None

    res = {
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "fp32": fp32,
        "int8": int8,
        "quantizationErrorF1": quant_error,
        "note": "PTQ simulation via torch.ao.quantization.quantize_dynamic (LSTM/Linear) on CPU.",
        "error": err,
    }

    save_run_artifacts(dataset=dataset, study="quant", run_name=f"ptq_{now_tag()}", result=res)


# --------------------------
# Grad-CAM (1D)
# --------------------------


def compute_gradcam_1d(
    *,
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    conv2_module_name: str = "conv2",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D Grad-CAM for conv2.

    Returns:
        cam_upsampled: [T] numpy array in [0,1]
        logits: [num_classes] numpy array
    """

    model.eval()

    feats = {}
    grads = {}

    def fwd_hook(_module, _inp, out):
        feats["value"] = out
        out.retain_grad()

    # Locate module
    target_module = dict(model.named_modules()).get(conv2_module_name)
    if target_module is None:
        raise ValueError(f"Could not find module '{conv2_module_name}' in model")

    h = target_module.register_forward_hook(fwd_hook)

    x = x.requires_grad_(True)
    logits = model(x)
    score = logits[:, target_class].sum()

    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    h.remove()

    fmap: torch.Tensor = feats["value"]  # [B, F, T']
    grad: torch.Tensor = fmap.grad  # [B, F, T']

    # Global-average-pool gradients across time
    weights = grad.mean(dim=2, keepdim=True)  # [B, F, 1]
    cam = torch.relu((weights * fmap).sum(dim=1))  # [B, T']

    cam = cam[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Upsample to original T (input time)
    T = x.shape[1]
    cam_up = torch.nn.functional.interpolate(cam.view(1, 1, -1), size=T, mode="linear", align_corners=False)
    cam_up = cam_up.view(-1)

    return cam_up.detach().cpu().numpy(), logits[0].detach().cpu().numpy()


def study_gradcam(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = args.dataset
    ds_cfg = DATASET_CONFIGS[dataset]

    ckpt_path = _find_latest_checkpoint(dataset, variant_prefix="A0")
    payload = _safe_torch_load(ckpt_path)
    state = payload["model_state_dict"]

    model = MicroBiConvLSTM(
        numClasses=ds_cfg["numClasses"],
        inChannels=ds_cfg["inputChannels"],
        seqLen=ds_cfg["seqLen"],
        dropout=float(ds_cfg["dropout"]),
        aggregation="last",
    )
    model.load_state_dict(state, strict=False)

    train_loader, test_loader, _ = load_dataset(dataset, ds_cfg["batchSize"], bypass_preprocessing=False)

    class_names = ds_cfg.get("classNames")
    if class_names is None:
        raise RuntimeError("Dataset config missing classNames; required for gradcam class lookup")

    target_idx = None
    for i, n in enumerate(class_names):
        if n.lower() == args.gradcam_class.lower():
            target_idx = i
            break
    if target_idx is None:
        raise ValueError(f"Unknown class '{args.gradcam_class}'. Available: {class_names}")

    # Pick a sample of that class from test loader
    chosen_x = None
    chosen_y = None
    for xb, yb in test_loader:
        mask = (yb == target_idx)
        if mask.any():
            chosen_x = xb[mask][0:1]  # [1,T,C]
            chosen_y = int(target_idx)
            break

    if chosen_x is None:
        raise RuntimeError(f"Could not find class '{args.gradcam_class}' in test set")

    cam, logits = compute_gradcam_1d(model=model, x=chosen_x, target_class=chosen_y, conv2_module_name="conv2")

    # Visualize: overlay cam on signal magnitude (simple summary)
    x_np = chosen_x[0].detach().numpy()  # [T,C]
    signal_mag = np.linalg.norm(x_np, axis=1)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(signal_mag, label="|signal| (L2 across channels)", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(cam, color="crimson", alpha=0.7, label="Grad-CAM (conv2)", linewidth=1.0)
    ax.set_title(f"1D Grad-CAM on conv2 | dataset={dataset} | class={args.gradcam_class}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal magnitude")
    ax2.set_ylabel("Grad-CAM")

    out = dataset_dir(dataset) / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    plot_path = out / f"gradcam_{args.gradcam_class}_{now_tag()}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    res = {
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "class": args.gradcam_class,
        "classIndex": int(target_idx),
        "plot": str(plot_path),
    }
    save_run_artifacts(dataset=dataset, study="gradcam", run_name=f"{args.gradcam_class}_{now_tag()}", result=res)


# --------------------------
# Robustness / sensitivity
# --------------------------

def _apply_channel_dropout(x: torch.Tensor, channel_indices: Sequence[int]) -> torch.Tensor:
    x = x.clone()
    x[:, :, list(channel_indices)] = 0.0
    return x


def _apply_sampling_jitter_hold(x: torch.Tensor, every_k: int) -> torch.Tensor:
    """Simulate sensor lag by dropping every k-th sample and holding last value."""
    if every_k <= 1:
        return x
    x = x.clone()
    # replace every k-th timestep with previous timestep (zero-order hold)
    idx = torch.arange(x.shape[1], device=x.device)
    drop = (idx % every_k == 0)
    # avoid t=0 which has no previous
    drop[0] = False
    x[:, drop, :] = x[:, idx[drop] - 1, :]
    return x


@torch.no_grad()
def evaluate_with_corruptions(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    channel_dropout: Optional[Sequence[int]] = None,
    sampling_every_k: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        if channel_dropout is not None:
            x = _apply_channel_dropout(x, channel_dropout)
        if sampling_every_k is not None:
            x = _apply_sampling_jitter_hold(x, sampling_every_k)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "accuracy": float(acc), "f1": float(f1)}


def study_sensitivity(args) -> None:
    dataset = args.dataset
    ds_cfg = DATASET_CONFIGS[dataset]

    ckpt_path = _find_latest_checkpoint(dataset, variant_prefix="A0")
    payload = _safe_torch_load(ckpt_path)
    state = payload["model_state_dict"]

    device = torch.device("cpu")

    model = MicroBiConvLSTM(
        numClasses=ds_cfg["numClasses"],
        inChannels=ds_cfg["inputChannels"],
        seqLen=ds_cfg["seqLen"],
        dropout=float(ds_cfg["dropout"]),
        aggregation="last",
    ).to(device)
    model.load_state_dict(state, strict=False)

    _, test_loader, class_weights = load_dataset(dataset, ds_cfg["batchSize"], bypass_preprocessing=False)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    base = evaluate_simple(model, test_loader, criterion, device)

    def channel_groups(ds_name: str, n_channels: int) -> Dict[str, List[int]]:
        """Dataset-aware modality groupings for channel dropout."""
        ds_name = ds_name.lower()

        # UCI-HAR: 9 channels typically ordered as:
        # body_acc(0-2), body_gyro(3-5), total_acc(6-8)
        if ds_name == "ucihar" and n_channels == 9:
            return {
                "gyro": [3, 4, 5],
                "body_acc": [0, 1, 2],
                "total_acc": [6, 7, 8],
            }

        # MotionSense: 6 channels commonly acc(0-2), gyro(3-5)
        if ds_name == "motionsense" and n_channels == 6:
            return {
                "gyro": [3, 4, 5],
                "acc": [0, 1, 2],
            }

        # Fallback: expose a coarse grouping (last third)
        return {
            "drop_last_third": list(range(n_channels * 2 // 3, n_channels)),
        }

    channel_sets = channel_groups(dataset, int(ds_cfg["inputChannels"]))

    jitter_every_k = int(args.jitter_every_k)

    results = {
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "baseline": base,
        "channelDropout": {},
        "samplingJitter": {},
        "preprocessingBypass": {},
    }

    for name, ch_idx in channel_sets.items():
        m = evaluate_with_corruptions(model, test_loader, criterion, channel_dropout=ch_idx)
        results["channelDropout"][name] = {"channels": ch_idx, "metrics": m, "deltaF1": float(base["f1"] - m["f1"]) }

    if jitter_every_k > 1:
        m = evaluate_with_corruptions(model, test_loader, criterion, sampling_every_k=jitter_every_k)
        results["samplingJitter"][f"every_{jitter_every_k}"] = {"metrics": m, "deltaF1": float(base["f1"] - m["f1"]) }

    # Preprocessing bypass evaluation (no retraining) is intentionally NOT done,
    # because the plan requests disabling filters as a preprocessing ablation.
    # We run an inference-only evaluation with bypassed preprocessing to quantify mismatch.
    if dataset in {"pamap2", "skoda", "daphnet"}:
        _, test_bypass, class_weights_bypass = load_dataset(dataset, ds_cfg["batchSize"], bypass_preprocessing=True)
        if class_weights_bypass is not None:
            crit_bypass = nn.CrossEntropyLoss(weight=class_weights_bypass)
        else:
            crit_bypass = nn.CrossEntropyLoss()

        m = evaluate_simple(model, test_bypass, crit_bypass, device)
        results["preprocessingBypass"]["inference_only"] = {"metrics": m, "deltaF1": float(base["f1"] - m["f1"]) }

    save_run_artifacts(dataset=dataset, study="sensitivity", run_name=f"robust_{now_tag()}", result=results)


# --------------------------
# Publication tables
# --------------------------

def write_publication_tables_stub() -> None:
    # Create a main stub; actual values will be filled by your runs.
    p = out_dir() / "PublicationTables.md"
    if p.exists():
        return

    content = """# MicroBiConvLSTM Ablation Studies (Auto-Generated)

> This file is written by `scripts/ablationStudiesMicroBiConvLstm.py`.
> Run studies to populate metrics.

## Table I: Architectural Ablation Results

| Configuration | Params (K) | MACs (K) | UCI-HAR (F1) | SKODA (F1) | PAMAP2 (F1) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Proposed (Frozen)** | **10.5K** | **420.1K** | **93.41%** | **94.46%** | **60.75%** |
| A1: No MaxPool | 10.5K | 1,217.2K | - | - | - |
| A2: Unidirectional | 6.5K | 299.0K | - | - | - |
| A3: Single Conv | 9.2K | 369.6K | - | - | - |
| A4: Mean Pooling | 10.5K | 420.1K | - | - | - |

## Table II: Preprocessing & Hardware Proxy Summary

| Dataset | Normalization | Filter | Special Handling | INT8 Quant. Error |
| :--- | :--- | :--- | :--- | :--- |
| **PAMAP2** | Robust Scaling | 10Hz LPF | Impact Spike Removal | TBD |
| **SKODA** | Z-Score | 5Hz LPF | Machine Vib. Removal | TBD |
| **Daphnet** | Z-Score | 12Hz LPF | 15:1 Class Weighting | TBD |
"""
    p.write_text(content, encoding="utf-8")


# --------------------------
# CLI
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        choices=["arch", "pareto", "complexity", "quant", "gradcam", "sensitivity"],
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)

    # Fixed protocol requested: AdamW + lr=1e-3
    parser.add_argument("--lr", type=float, default=1e-3)

    # Pareto
    parser.add_argument("--conv-grid", type=str, default="8,16,24")
    parser.add_argument("--lstm-grid", type=str, default="16,24,32")

    # Complexity
    parser.add_argument("--seq-lens", type=str, default="64,96,128,160,192,224,256")

    # Grad-CAM
    parser.add_argument("--gradcam-class", type=str, default="Walking")

    # Sensitivity
    parser.add_argument("--jitter-every-k", type=int, default=5)

    args = parser.parse_args()

    write_publication_tables_stub()

    if args.study == "arch":
        study_arch(args)
    elif args.study == "pareto":
        study_pareto(args)
    elif args.study == "complexity":
        study_complexity(args)
    elif args.study == "quant":
        study_quant(args)
    elif args.study == "gradcam":
        study_gradcam(args)
    elif args.study == "sensitivity":
        study_sensitivity(args)
    else:
        raise ValueError(args.study)


if __name__ == "__main__":
    main()
