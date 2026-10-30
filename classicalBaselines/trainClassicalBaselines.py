"""Train reproducible classical HAR baselines on the paper dataset splits.

The held-out loader is used once for final evaluation. All learned preprocessing,
including scaling and PCA, is fitted exclusively on the training split.

Examples:
    python classicalBaselines/trainClassicalBaselines.py --models all --datasets ucihar --seed-list 29
    python classicalBaselines/trainClassicalBaselines.py --models all --datasets all --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.trainBaselines import DATASET_CONFIGS, DATASETS, loadDataset


MASTER_SEED = 17
MODELS = ("logistic_regression", "rbf_svm", "xgboost")
FEATURE_VERSION = 1
TEMPORAL_FEATURES = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "q25",
    "q75",
    "rms",
    "mean_abs",
    "skewness",
    "excess_kurtosis",
    "diff_mean",
    "diff_std",
    "diff_mean_abs",
    "zero_crossing_rate",
)
SPECTRAL_FEATURES = (
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
    "low_band_power",
    "mid_band_power",
    "high_band_power",
)


@contextmanager
def working_directory(path: Path):
    """Temporarily satisfy the legacy dataset loaders' relative path contract."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_csv(value: str, allowed: Sequence[str], label: str) -> List[str]:
    """Parse an `all` value or a comma-separated ordered subset."""
    if value.strip().lower() == "all":
        return list(allowed)
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(parsed) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)}")
    if not parsed:
        raise ValueError(f"At least one {label} must be selected.")
    return list(dict.fromkeys(parsed))


def parse_seed_list(value: str | None) -> List[int] | None:
    if value is None:
        return None
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seed-list did not contain an integer seed.")
    return seeds


def generate_random_seeds(master_seed: int, count: int) -> List[int]:
    """Match the deterministic seed generation used by the DL baseline runner."""
    if count < 1:
        raise ValueError("--seeds must be at least one.")
    rng = np.random.default_rng(master_seed)
    return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=count)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_names(input_channels: int) -> List[str]:
    descriptors = TEMPORAL_FEATURES + SPECTRAL_FEATURES
    return [
        f"channel_{channel:02d}_{descriptor}"
        for channel in range(input_channels)
        for descriptor in descriptors
    ]


def _as_time_channel(batch: torch.Tensor, seq_len: int, channels: int) -> np.ndarray:
    values = batch.detach().cpu().numpy()
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError(f"Expected a 3-D sensor batch, received shape {values.shape}.")
    if values.shape[1:] == (seq_len, channels):
        return values.astype(np.float32, copy=False)
    if values.shape[1:] == (channels, seq_len):
        return values.transpose(0, 2, 1).astype(np.float32, copy=False)
    raise ValueError(
        f"Cannot map sensor batch {values.shape} to [batch, {seq_len}, {channels}]."
    )


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Compute compact per-channel temporal and frequency-domain descriptors."""
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [samples, time, channels], received {x.shape}.")

    eps = np.float32(1e-8)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    centered = x - mean[:, None, :]
    safe_std = np.maximum(std, eps)
    z_score = centered / safe_std[:, None, :]
    difference = np.diff(x, axis=1)

    temporal = [
        mean,
        std,
        x.min(axis=1),
        x.max(axis=1),
        np.median(x, axis=1),
        np.quantile(x, 0.25, axis=1),
        np.quantile(x, 0.75, axis=1),
        np.sqrt(np.mean(np.square(x), axis=1)),
        np.mean(np.abs(x), axis=1),
        np.mean(np.power(z_score, 3), axis=1),
        np.mean(np.power(z_score, 4), axis=1) - 3.0,
        difference.mean(axis=1),
        difference.std(axis=1),
        np.mean(np.abs(difference), axis=1),
        np.mean(x[:, 1:, :] * x[:, :-1, :] < 0.0, axis=1),
    ]

    spectrum = np.fft.rfft(x, axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)
    power[:, 0, :] = 0.0
    total_power = np.maximum(power.sum(axis=1), eps)
    probability = power / total_power[:, None, :]
    frequency = np.fft.rfftfreq(x.shape[1]).astype(np.float32)
    normalized_frequency = frequency / np.float32(0.5)
    positive_bins = max(1, power.shape[1] - 1)

    dominant_frequency = normalized_frequency[np.argmax(power, axis=1)]
    spectral_centroid = np.sum(
        probability * normalized_frequency[None, :, None], axis=1
    )
    spectral_entropy = -np.sum(
        probability * np.log(np.maximum(probability, eps)), axis=1
    ) / np.log(np.float32(positive_bins + 1))
    low_power = probability[:, normalized_frequency <= 0.10, :].sum(axis=1)
    mid_power = probability[
        :, (normalized_frequency > 0.10) & (normalized_frequency <= 0.25), :
    ].sum(axis=1)
    high_power = probability[:, normalized_frequency > 0.25, :].sum(axis=1)

    # Stack by channel so columns align with feature_names().
    features = np.stack(
        temporal
        + [
            dominant_frequency,
            spectral_centroid,
            spectral_entropy,
            low_power,
            mid_power,
            high_power,
        ],
        axis=2,
    ).transpose(0, 2, 1)
    features = features.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def loader_to_features(
    loader: DataLoader,
    config: Dict[str, Any],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract every sample from a loader dataset, including a dropped final batch."""
    sequential_loader = DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    feature_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    for sensor_batch, label_batch in sequential_loader:
        windows = _as_time_channel(
            sensor_batch, config["seqLen"], config["inputChannels"]
        )
        feature_batches.append(extract_features(windows))
        label_batches.append(np.asarray(label_batch).reshape(-1).astype(np.int64))
    if not feature_batches:
        raise RuntimeError("The selected dataset split contains no samples.")
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def load_feature_data(
    dataset: str,
    cache_dir: Path,
    feature_batch_size: int,
    refresh_cache: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    config = DATASET_CONFIGS[dataset]
    cache_path = cache_dir / f"{dataset}_features_v{FEATURE_VERSION}.npz"
    if cache_path.exists() and not refresh_cache:
        cached = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(cached["metadata_json"].item()))
        return (
            cached["x_train"],
            cached["y_train"],
            cached["x_test"],
            cached["y_test"],
            list(cached["feature_names"].astype(str)),
            metadata,
        )

    print(f"Loading and featurizing {dataset}...")
    started = time.perf_counter()
    with working_directory(_WORKSPACE_ROOT):
        train_loader, test_loader, _ = loadDataset(dataset, config["batchSize"])
    x_train, y_train = loader_to_features(train_loader, config, feature_batch_size)
    x_test, y_test = loader_to_features(test_loader, config, feature_batch_size)
    names = feature_names(config["inputChannels"])
    if x_train.shape[1] != len(names):
        raise RuntimeError(
            f"Feature schema has {len(names)} names but data has {x_train.shape[1]} columns."
        )

    metadata = {
        "dataset": dataset,
        "featureVersion": FEATURE_VERSION,
        "featureCount": int(x_train.shape[1]),
        "trainSamples": int(len(y_train)),
        "testSamples": int(len(y_test)),
        "trainClassCounts": dict(sorted(Counter(map(int, y_train)).items())),
        "testClassCounts": dict(sorted(Counter(map(int, y_test)).items())),
        "extractionSeconds": time.perf_counter() - started,
        "splitProtocol": "existing paper dataset train/test loaders",
        "preprocessingProtocol": "feature extraction is fixed; learned transforms are train-fitted",
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=np.asarray(names),
        metadata_json=json.dumps(metadata),
    )
    return x_train, y_train, x_test, y_test, names, metadata


def stratified_cap(
    x: np.ndarray,
    y: np.ndarray,
    maximum: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Optionally cap costly exact-SVM fitting while retaining class proportions."""
    if maximum <= 0 or len(y) <= maximum:
        return x, y, False
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    classes, counts = np.unique(y, return_counts=True)
    allocations = np.maximum(1, np.floor(maximum * counts / len(y)).astype(int))
    while allocations.sum() > maximum:
        allocations[np.argmax(allocations)] -= 1
    while allocations.sum() < maximum:
        room = counts - allocations
        allocations[np.argmax(room)] += 1
    for class_id, count in zip(classes, allocations):
        indices = np.flatnonzero(y == class_id)
        selected.append(rng.choice(indices, size=min(int(count), len(indices)), replace=False))
    chosen = np.concatenate(selected)
    rng.shuffle(chosen)
    return x[chosen], y[chosen], True


def create_estimator(
    model: str,
    seed: int,
    pca_components: int,
    jobs: int,
    num_classes: int,
) -> Any:
    if model == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        tol=1e-4,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model == "rbf_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=pca_components,
                        svd_solver="randomized",
                        random_state=seed,
                    ),
                ),
                (
                    "classifier",
                    SVC(
                        C=10.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        cache_size=2048,
                    ),
                ),
            ]
        )
    if model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("XGBoost is required: pip install xgboost") from error
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic" if num_classes == 2 else "multi:softprob",
            eval_metric="logloss" if num_classes == 2 else "mlogloss",
            tree_method="hist",
            n_jobs=jobs,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model}")


def fit_and_evaluate(
    model: str,
    dataset: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    names: Sequence[str],
    output_dir: Path,
    pca_components: int,
    jobs: int,
    svm_max_train_samples: int,
) -> Dict[str, Any]:
    set_seed(seed)
    fit_x, fit_y = x_train, y_train
    capped = False
    if model == "rbf_svm":
        fit_x, fit_y, capped = stratified_cap(
            x_train, y_train, svm_max_train_samples, seed
        )
    actual_components = min(pca_components, fit_x.shape[1], len(fit_y) - 1)
    estimator = create_estimator(
        model,
        seed,
        actual_components,
        jobs,
        DATASET_CONFIGS[dataset]["numClasses"],
    )

    print(
        f"Fitting {model} on {dataset}, seed {seed} "
        f"({len(fit_y)} train, {len(y_test)} test, {x_train.shape[1]} features)..."
    )
    fit_started = time.perf_counter()
    fit_kwargs: Dict[str, Any] = {}
    if model == "xgboost":
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", fit_y)
    estimator.fit(fit_x, fit_y, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    predictions = estimator.predict(x_test).astype(np.int64)
    inference_seconds = time.perf_counter() - inference_started
    labels = np.arange(DATASET_CONFIGS[dataset]["numClasses"])
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macroF1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weightedF1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "macroPrecision": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macroRecall": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "fitSeconds": fit_seconds,
        "inferenceSeconds": inference_seconds,
        "inferenceMillisecondsPerWindow": 1000.0 * inference_seconds / len(y_test),
        "trainSamplesAvailable": int(len(y_train)),
        "trainSamplesUsed": int(len(fit_y)),
        "testSamples": int(len(y_test)),
        "featureCount": int(x_train.shape[1]),
        "pcaComponents": actual_components if model == "rbf_svm" else None,
        "svmTrainingCapped": capped,
        "confusionMatrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classificationReport": report,
        "protocol": {
            "split": "existing paper dataset train/test loaders",
            "modelSelection": "fixed hyperparameters; no held-out test feedback",
            "learnedPreprocessing": "fitted on training data only",
            "classImbalance": "balanced class/sample weighting",
        },
    }

    run_dir = output_dir / model / dataset / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"{model}_{dataset}_seed{seed}.joblib"
    joblib.dump(estimator, model_path, compress=3)
    np.savez_compressed(run_dir / "predictions.npz", labels=y_test, predictions=predictions)
    metrics["modelPath"] = str(model_path.resolve())

    if model == "xgboost":
        importances = np.asarray(estimator.feature_importances_)
        order = np.argsort(importances)[::-1][:50]
        metrics["topFeatureImportances"] = [
            {"feature": names[index], "importance": float(importances[index])}
            for index in order
        ]
    with open(run_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        f"Completed {model}/{dataset}/seed{seed}: "
        f"accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macroF1']:.4f}, "
        f"fit={fit_seconds:.1f}s"
    )
    return metrics


def aggregate_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for result in results:
        groups.setdefault((result["model"], result["dataset"]), []).append(result)
    rows = []
    for (model, dataset), records in sorted(groups.items()):
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "seeds": [record["seed"] for record in records],
                "runs": len(records),
                "accuracyMean": float(np.mean([record["accuracy"] for record in records])),
                "accuracyStd": float(np.std([record["accuracy"] for record in records])),
                "macroF1Mean": float(np.mean([record["macroF1"] for record in records])),
                "macroF1Std": float(np.std([record["macroF1"] for record in records])),
                "balancedAccuracyMean": float(
                    np.mean([record["balancedAccuracy"] for record in records])
                ),
                "fitSecondsMean": float(
                    np.mean([record["fitSeconds"] for record in records])
                ),
            }
        )
    return {"groups": rows, "completedRuns": sum(row["runs"] for row in rows)}


def write_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    models = parse_csv(args.models, MODELS, "model")
    datasets = parse_csv(args.datasets, DATASETS, "dataset")
    explicit_seeds = parse_seed_list(args.seed_list)
    seeds = explicit_seeds or generate_random_seeds(args.master_seed, args.seeds)
    run_name = args.run_name or datetime.now().strftime("classical_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    cache_dir = Path(args.feature_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = {
        "models": models,
        "datasets": datasets,
        "seeds": seeds,
        "masterSeed": args.master_seed,
        "featureVersion": FEATURE_VERSION,
        "pcaComponents": args.pca_components,
        "svmMaxTrainSamples": args.svm_max_train_samples,
        "jobs": args.jobs,
        "createdAt": datetime.now().isoformat(),
        "protocol": {
            "testUsage": "single final evaluation after fixed training",
            "learnedPreprocessing": "training split only",
            "primaryMetric": "macro-F1",
        },
    }
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(experiment, handle, indent=2)
    if args.dry_run:
        print(json.dumps(experiment, indent=2))
        return output_dir

    status: Dict[str, Any] = {"experiment": experiment, "runs": []}
    completed: List[Dict[str, Any]] = []
    for dataset in datasets:
        feature_data = load_feature_data(
            dataset,
            cache_dir,
            args.feature_batch_size,
            args.refresh_feature_cache,
        )
        x_train, y_train, x_test, y_test, names, metadata = feature_data
        dataset_meta_dir = output_dir / "dataset_metadata"
        dataset_meta_dir.mkdir(parents=True, exist_ok=True)
        with open(dataset_meta_dir / f"{dataset}.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        for model in models:
            for seed in seeds:
                result_path = output_dir / model / dataset / f"seed{seed}" / "results.json"
                if args.resume and result_path.exists():
                    with open(result_path, "r", encoding="utf-8") as handle:
                        result = json.load(handle)
                    completed.append(result)
                    status["runs"].append(
                        {"model": model, "dataset": dataset, "seed": seed, "state": "reused"}
                    )
                    continue
                try:
                    result = fit_and_evaluate(
                        model,
                        dataset,
                        seed,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        names,
                        output_dir,
                        args.pca_components,
                        args.jobs,
                        args.svm_max_train_samples,
                    )
                    completed.append(result)
                    status["runs"].append(
                        {"model": model, "dataset": dataset, "seed": seed, "state": "complete"}
                    )
                except Exception as error:
                    status["runs"].append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "state": "failed",
                            "error": repr(error),
                        }
                    )
                    write_status(output_dir / "status.json", status)
                    if not args.continue_on_error:
                        raise
                    print(f"ERROR: {model}/{dataset}/seed{seed}: {error}")
                write_status(output_dir / "status.json", status)

    summary = aggregate_results(completed)
    summary["experiment"] = experiment
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Results saved to {output_dir.resolve()}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="all or a comma-separated model list")
    parser.add_argument("--datasets", default="all", help="all or a comma-separated dataset list")
    parser.add_argument("--seeds", type=int, default=5, help="number of master-seed-generated runs")
    parser.add_argument("--seed-list", default=None, help="explicit comma-separated seeds")
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument(
        "--svm-max-train-samples",
        type=int,
        default=0,
        help="optional stratified exact-SVM cap; zero uses the complete training split",
    )
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--feature-cache-dir", default="./results/classical_feature_cache")
    parser.add_argument("--output-dir", default="./results/classical_baselines")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
