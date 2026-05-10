from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


THIS_FILE = Path(__file__).resolve()
MICROBI_DIR = THIS_FILE.parents[1]
RESULTS_DIR = MICROBI_DIR / "results" / "paperRetraining"
EDGE_EXPORTS_DIR = MICROBI_DIR / "edgeExports"

ALL_MODELS = ("microbi", "deepconvlstm", "tinyhar", "tinierhar")
ALL_DATASETS = ("ucihar", "motionsense", "wisdm", "pamap2", "opportunity", "unimib", "skoda", "daphnet")


def parse_csv_list(raw: str, allowed: Sequence[str], label: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(allowed)

    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)}")
    return requested


def find_latest_successful_run() -> Path:
    run_dirs = sorted(
        [path for path in RESULTS_DIR.iterdir() if path.is_dir() and path.name.startswith("paperRetrainSeed")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if status and all(int(entry.get("returncode", 1)) == 0 for entry in status):
            return run_dir
    raise FileNotFoundError(f"Could not find a successful paper retraining run under {RESULTS_DIR}")


def checkpoint_for(run_dir: Path, model: str, dataset: str) -> Path:
    if model == "microbi":
        checkpoint = run_dir / "microBiConvLstm" / dataset / f"microBiConvLstm_{dataset}_seed29_checkpoint.pt"
    else:
        checkpoint = run_dir / "baselines" / model / dataset / "seed29_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model}/{dataset}: {checkpoint}")
    return checkpoint


def build_command(run_dir: Path, model: str, dataset: str, representative_samples: int) -> Dict[str, object]:
    checkpoint = checkpoint_for(run_dir, model, dataset)
    output_dir = EDGE_EXPORTS_DIR / run_dir.name / model / dataset
    exporter = MICROBI_DIR / "scripts" / "exportEdgeModels.py"
    command = [
        sys.executable,
        str(exporter),
        "--model",
        model,
        "--dataset",
        dataset,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--representative-samples",
        str(representative_samples),
        "--seed",
        "29",
    ]
    return {
        "model": model,
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "log_path": str(output_dir / "conversion.log"),
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert completed paper retraining runs to ONNX/TFLite/TFLM bundles.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Paper retraining run directory. Defaults to latest successful run.")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset list or 'all'.")
    parser.add_argument("--representative-samples", type=int, default=128, help="Calibration samples for full-int quantization.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without running them.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir is not None else find_latest_successful_run()
    models = parse_csv_list(args.models, ALL_MODELS, "models")
    datasets = parse_csv_list(args.datasets, ALL_DATASETS, "datasets")

    bundle_root = EDGE_EXPORTS_DIR / run_dir.name
    bundle_root.mkdir(parents=True, exist_ok=True)

    jobs = [build_command(run_dir, model, dataset, args.representative_samples) for model in models for dataset in datasets]
    manifest = {
        "run_dir": str(run_dir),
        "bundle_root": str(bundle_root),
        "job_count": len(jobs),
        "jobs": [
            {
                "model": job["model"],
                "dataset": job["dataset"],
                "checkpoint": job["checkpoint"],
                "output_dir": job["output_dir"],
                "log_path": job["log_path"],
                "command": " ".join(f'"{token}"' if " " in token else token for token in job["command"]),
            }
            for job in jobs
        ],
    }
    (bundle_root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    status: List[Dict[str, object]] = []
    for index, job in enumerate(jobs, start=1):
        printable = " ".join(f'"{token}"' if " " in token else token for token in job["command"])
        output_dir = Path(str(job["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(str(job["log_path"]))
        print(f"[{index}/{len(jobs)}] Converting {job['model']} on {job['dataset']}")
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(printable + "\n\n")
            completed = subprocess.run(
                job["command"],
                cwd=str(MICROBI_DIR),
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        summary = None
        report_path = output_dir / "parity_report.json"
        if report_path.exists():
            try:
                summary = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = None
        record = {
            "index": index,
            "model": job["model"],
            "dataset": job["dataset"],
            "checkpoint": job["checkpoint"],
            "output_dir": job["output_dir"],
            "log_path": job["log_path"],
            "returncode": completed.returncode,
            "command": printable,
            "summary": summary,
        }
        status.append(record)
        (bundle_root / "conversion_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        if summary is not None:
            int8_size = int(summary.get("sizes", {}).get("tflite_int8_bytes", 0))
            int8_ram = int(summary.get("ram_estimates", {}).get("tflite_int8_tensor_bytes", 0))
            parity = float(summary.get("parity", {}).get("tflite_int8", {}).get("parity_percent", 0.0))
            print(f"  size={int8_size}B ram={int8_ram}B int8_parity={parity:.3f}%")
        print(f"  log={log_path}")

    print()
    print(f"Completed {len(jobs)} conversions into {bundle_root}")


if __name__ == "__main__":
    main()
