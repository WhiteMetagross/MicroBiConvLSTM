"""Benchmark cross-paradigm storage and CPU-cycle efficiency.

The benchmark compares complete raw-window inference paths on one pinned WSL CPU:
MicroBiConvLSTM forward inference and classical feature extraction plus prediction.
Cycle counts are invariant-TSC reference cycles, not architecture-independent MACs.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import platform
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.microBiConvLstm import MicroBiConvLSTM
from classicalBaselines.trainClassicalBaselines import (
    DATASET_CONFIGS,
    _as_time_channel,
    extract_features,
    loadDataset,
)


DATASETS = (
    "ucihar",
    "motionsense",
    "wisdm",
    "pamap2",
    "opportunity",
    "unimib",
    "skoda",
    "daphnet",
)
CLASSICAL_MODELS = ("logistic_regression", "rbf_svm", "xgboost")


class TscReader:
    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library))
        self.library.tsc_begin.restype = ctypes.c_uint64
        self.library.tsc_end.restype = ctypes.c_uint64

    def begin(self) -> int:
        return int(self.library.tsc_begin())

    def end(self) -> int:
        return int(self.library.tsc_end())


def pin_single_cpu() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    if not available:
        return None
    cpu = available[0]
    os.sched_setaffinity(0, {cpu})
    return cpu


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def benchmark_cycles(
    reader: TscReader,
    inference_functions: Sequence[Callable[[], Any]],
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    if not inference_functions:
        raise ValueError("At least one inference function is required.")
    with torch.inference_mode():
        for index in range(warmup):
            inference_functions[index % len(inference_functions)]()

    measurements: List[int] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.inference_mode():
            for _ in range(repeats):
                for inference in inference_functions:
                    start = reader.begin()
                    inference()
                    stop = reader.end()
                    measurements.append(stop - start)
    finally:
        if gc_enabled:
            gc.enable()

    return {
        "measurementCount": len(measurements),
        "cyclesMedian": float(np.median(measurements)),
        "cyclesMean": mean(measurements),
        "cyclesStd": pstdev(measurements),
        "cyclesP25": percentile(measurements, 0.25),
        "cyclesP75": percentile(measurements, 0.75),
        "cyclesMin": float(min(measurements)),
        "cyclesMax": float(max(measurements)),
    }


def compact_classical_state_bytes(
    model_name: str,
    estimator: Any,
    complexity_record: Dict[str, Any],
) -> int:
    """Return compact FP32-equivalent inference-state bytes."""
    if model_name == "logistic_regression":
        scaler = estimator.named_steps["scaler"]
        classifier = estimator.named_steps["classifier"]
        scalars = (
            scaler.mean_.size
            + scaler.scale_.size
            + classifier.coef_.size
            + classifier.intercept_.size
        )
        return int(scalars * 4)
    if model_name == "rbf_svm":
        scaler = estimator.named_steps["scaler"]
        pca = estimator.named_steps["pca"]
        classifier = estimator.named_steps["classifier"]
        scalars = (
            scaler.mean_.size
            + scaler.scale_.size
            + pca.mean_.size
            + pca.components_.size
            + classifier.support_vectors_.size
            + classifier.dual_coef_.size
            + classifier.intercept_.size
            + 1  # gamma
        )
        return int(scalars * 4)
    if model_name == "xgboost":
        # Compact structure-of-arrays estimate: feature ID (u16), threshold
        # (f32), children (2*i32), missing direction (u8), and f32 leaves.
        split_nodes = int(complexity_record["splitNodes"])
        leaf_nodes = int(complexity_record["leafNodes"])
        trees = int(complexity_record["trees"])
        return split_nodes * 15 + leaf_nodes * 4 + trees * 4
    raise ValueError(model_name)


def load_raw_test_windows(dataset: str, count: int) -> List[np.ndarray]:
    config = DATASET_CONFIGS[dataset]
    previous = Path.cwd()
    os.chdir(_WORKSPACE_ROOT)
    try:
        _, test_loader, _ = loadDataset(dataset, config["batchSize"])
    finally:
        os.chdir(previous)
    test_dataset = test_loader.dataset
    selected = np.linspace(0, len(test_dataset) - 1, min(count, len(test_dataset)), dtype=int)
    windows = []
    for index in selected:
        sensor, _ = test_dataset[int(index)]
        batch = sensor.unsqueeze(0) if isinstance(sensor, torch.Tensor) else torch.as_tensor(sensor)[None]
        windows.append(
            np.ascontiguousarray(
                _as_time_channel(batch, config["seqLen"], config["inputChannels"])
            )
        )
    return windows


def micro_paths(micro_root: Path, dataset: str) -> Tuple[Path, Path]:
    folder = micro_root / dataset
    state = folder / f"microbi_convlstm_{dataset}_seed29_model_state.pt"
    result = folder / f"microbi_convlstm_{dataset}_seed29_results.json"
    if not state.exists() or not result.exists():
        raise FileNotFoundError(f"Missing MicroBiConvLSTM artifacts for {dataset}: {folder}")
    return state, result


def load_micro_model(dataset: str, state_path: Path) -> MicroBiConvLSTM:
    config = DATASET_CONFIGS[dataset]
    model = MicroBiConvLSTM(
        numClasses=config["numClasses"],
        inChannels=config["inputChannels"],
        seqLen=config["seqLen"],
        dropout=config.get("dropout", 0.0),
        aggregation="last",
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def micro_state_bytes(model: torch.nn.Module) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()))


def aggregate_seed_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "model": records[0]["model"],
        "dataset": records[0]["dataset"],
        "seedCount": len(records),
    }
    for field in ("macroF1", "stateBytes", "cyclesMedian"):
        values = [float(record[field]) for record in records]
        result[f"{field}Mean"] = mean(values)
        result[f"{field}Std"] = pstdev(values) if len(values) > 1 else 0.0
    result["stateKiB"] = result["stateBytesMean"] / 1024.0
    result["millionCyclesPerWindow"] = result["cyclesMedianMean"] / 1_000_000.0
    f1_points = 100.0 * result["macroF1Mean"]
    result["f1PointsPerKiB"] = f1_points / result["stateKiB"]
    result["f1PointsPerMillionCycles"] = f1_points / result["millionCyclesPerWindow"]
    return result


def summarize_models(aggregates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for model in ("microbi_convlstm",) + CLASSICAL_MODELS:
        rows = [row for row in aggregates if row["model"] == model]
        f1_points = 100.0 * mean(row["macroF1Mean"] for row in rows)
        state_kib = mean(row["stateKiB"] for row in rows)
        million_cycles = mean(row["millionCyclesPerWindow"] for row in rows)
        summaries.append(
            {
                "model": model,
                "datasetCount": len(rows),
                "macroF1Percent": f1_points,
                "stateKiB": state_kib,
                "millionCyclesPerWindow": million_cycles,
                "f1PointsPerKiB": f1_points / state_kib,
                "f1PointsPerMillionCycles": f1_points / million_cycles,
            }
        )
    return summaries


def write_markdown(
    path: Path,
    aggregates: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    lines = [
        "# Cross-Paradigm Efficiency Results:",
        "",
        "All cycle measurements cover one complete raw-window inference. MicroBiConvLSTM includes its full neural forward pass. Classical pipelines include handcrafted feature extraction, scaling, PCA where applicable, and final prediction. Measurements are invariant-TSC reference cycles on one pinned WSL CPU thread.",
        "",
        "Model-state size is a compact FP32-equivalent representation, not framework container size. XGBoost uses compact split and leaf arrays.",
        "",
        f"- CPU affinity: `{metadata['cpuAffinity']}`.",
        f"- Samples per estimator: `{metadata['samples']}`.",
        f"- Repeats per sample: `{metadata['repeats']}`.",
        f"- Measurements per estimator: `{metadata['samples'] * metadata['repeats']}`.",
        "",
        "## Cross-Dataset Summary:",
        "",
        "| Model | Mean Macro-F1 (%) | Mean State (KiB) | Mean Mcycles/window | F1 points/KiB | F1 points/Mcycle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['macroF1Percent']:.2f} | "
            f"{row['stateKiB']:.2f} | {row['millionCyclesPerWindow']:.3f} | "
            f"{row['f1PointsPerKiB']:.3f} | "
            f"{row['f1PointsPerMillionCycles']:.3f} |"
        )
    lines.extend(
        [
        "",
        "## Per-Dataset Results:",
        "",
        "| Model | Dataset | Macro-F1 (%) | State (KiB) | Mcycles/window | F1 points/KiB | F1 points/Mcycle |",
        "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{100.0 * row['macroF1Mean']:.2f} | {row['stateKiB']:.2f} | "
            f"{row['millionCyclesPerWindow']:.3f} | {row['f1PointsPerKiB']:.3f} | "
            f"{row['f1PointsPerMillionCycles']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation:",
            "",
            "- State size includes every learned or fitted numeric array required for inference.",
            "- Cycle counts are host-implementation measurements and should not be presented as Pico or ESP32 cycles.",
            "- Cross-device claims require rerunning every complete pipeline on the same target hardware and compiler configuration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical-run",
        default="./results/classical_baselines/classical_master17_5seeds",
    )
    parser.add_argument(
        "--micro-root",
        default=(
            "./results/paper_retraining/paper_retrain_seed29_20261028_194010/"
            "microbi_convlstm"
        ),
    )
    parser.add_argument("--tsc-library", default="./results/cross_paradigm/librdtsc.so")
    parser.add_argument("--output-dir", default="./results/cross_paradigm")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cpu = pin_single_cpu()
    reader = TscReader(Path(args.tsc_library).resolve())
    classical_root = Path(args.classical_run)
    micro_root = Path(args.micro_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    complexity_payload = json.loads(
        (classical_root / "classical_complexity.json").read_text(encoding="utf-8")
    )
    complexity_by_run = {
        (record["model"], record["dataset"], int(record["seed"])): record
        for record in complexity_payload["records"]
    }
    raw_records: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        print(f"Loading raw held-out windows for {dataset}...")
        windows = load_raw_test_windows(dataset, args.samples)

        state_path, result_path = micro_paths(micro_root, dataset)
        micro = load_micro_model(dataset, state_path)
        micro_result = json.loads(result_path.read_text(encoding="utf-8"))
        torch_windows = [torch.from_numpy(window) for window in windows]
        micro_functions = [lambda sample=sample: micro(sample) for sample in torch_windows]
        cycle_stats = benchmark_cycles(reader, micro_functions, args.warmup, args.repeats)
        micro_record = {
            "model": "microbi_convlstm",
            "dataset": dataset,
            "seed": 29,
            "macroF1": float(micro_result["testF1"]),
            "stateBytes": micro_state_bytes(micro),
            **cycle_stats,
        }
        raw_records.append(micro_record)
        print(
            f"  microbi_convlstm: {micro_record['stateBytes'] / 1024:.2f} KiB, "
            f"{micro_record['cyclesMedian'] / 1e6:.3f} Mcycles"
        )

        for model_name in CLASSICAL_MODELS:
            for model_path in sorted((classical_root / model_name / dataset).glob("seed*/*.joblib")):
                seed = int(model_path.parent.name.removeprefix("seed"))
                estimator = joblib.load(model_path)
                if model_name == "xgboost":
                    estimator.set_params(n_jobs=1)
                functions = [
                    lambda window=window, fitted=estimator: fitted.predict(extract_features(window))
                    for window in windows
                ]
                cycle_stats = benchmark_cycles(reader, functions, args.warmup, args.repeats)
                result_payload = json.loads(
                    (model_path.parent / "results.json").read_text(encoding="utf-8")
                )
                complexity = complexity_by_run[(model_name, dataset, seed)]
                record = {
                    "model": model_name,
                    "dataset": dataset,
                    "seed": seed,
                    "macroF1": float(result_payload["macroF1"]),
                    "stateBytes": compact_classical_state_bytes(
                        model_name, estimator, complexity
                    ),
                    **cycle_stats,
                }
                raw_records.append(record)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in raw_records:
        grouped.setdefault((record["model"], record["dataset"]), []).append(record)
    model_order = ("microbi_convlstm",) + CLASSICAL_MODELS
    aggregates = [
        aggregate_seed_records(grouped[(model, dataset)])
        for model in model_order
        for dataset in DATASETS
    ]
    summaries = summarize_models(aggregates)
    metadata = {
        "host": platform.platform(),
        "processor": platform.processor(),
        "cpuAffinity": cpu,
        "torchThreads": torch.get_num_threads(),
        "samples": args.samples,
        "requestedSamples": args.samples,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cycleUnit": "invariant TSC reference cycles",
        "statePrecision": "compact FP32-equivalent inference state",
    }
    payload = {
        "metadata": metadata,
        "records": raw_records,
        "aggregates": aggregates,
        "modelSummaries": summaries,
    }
    (output_dir / "cross_paradigm_efficiency.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "cross_paradigm_efficiency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(aggregates[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)
    with (output_dir / "cross_paradigm_efficiency_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    write_markdown(
        output_dir / "cross_paradigm_efficiency.md",
        aggregates,
        summaries,
        metadata,
    )
    print(f"Saved {len(aggregates)} aggregate rows to {output_dir.resolve()}.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
