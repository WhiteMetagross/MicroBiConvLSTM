"""
Sequential paper retraining launcher for MicroBiConvLSTM and baselines.

This script runs the final paper training protocol across:
- MicroBiConvLSTM (implemented in trainMicroBiConvLstm.py)
- DeepConvLSTM
- TinyHAR
- TinierHAR

It is designed for explicit seed-based retraining, for example seed 29 only.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


ALL_MODELS = ["microbiconvlstm", "deepconvlstm", "tinyhar", "tinierhar"]
ALL_DATASETS = ["ucihar", "motionsense", "wisdm", "pamap2", "opportunity", "unimib", "skoda", "daphnet"]


def parse_csv_arg(value: str, allowed: Iterable[str], arg_name: str) -> List[str]:
    """Parse a comma-separated CLI argument with validation."""
    if value == "all":
        return list(allowed)

    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{arg_name} must contain at least one value.")

    invalid = [token for token in tokens if token not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {arg_name}: {', '.join(invalid)}")
    return tokens


def parse_seed_list(seed_list: str) -> List[int]:
    """Parse explicit seed list."""
    seeds = [int(token.strip()) for token in seed_list.split(",") if token.strip()]
    if not seeds:
        raise ValueError("At least one explicit seed is required.")
    return seeds


def build_command(
    project_root: Path,
    python_exe: str,
    model_name: str,
    dataset_name: str,
    seed_list: str,
    epochs: int,
    patience: int,
    out_dir: Path,
) -> List[str]:
    """Build the concrete subprocess command for one model/dataset pair."""
    scripts_dir = project_root / "scripts"

    if model_name == "microbiconvlstm":
        save_dir = out_dir / "microBiConvLstm" / dataset_name
        return [
            python_exe,
            str(scripts_dir / "trainMicroBiConvLstm.py"),
            "--dataset",
            dataset_name,
            "--epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--seed-list",
            seed_list,
            "--save-dir",
            str(save_dir),
        ]

    return [
        python_exe,
        str(scripts_dir / "trainBaselines.py"),
        "--model",
        model_name,
        "--dataset",
        dataset_name,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--seed-list",
        seed_list,
        "--outDir",
        str(out_dir / "baselines"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final paper retraining sequentially.")
    parser.add_argument("--models", default="all", help="Comma-separated models or 'all'")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets or 'all'")
    parser.add_argument("--seed-list", default="29", help="Comma-separated explicit seeds, default: 29")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--outDir", default="./results/paperRetraining", help="Output root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if one run fails")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    models = parse_csv_arg(args.models, ALL_MODELS, "models")
    datasets = parse_csv_arg(args.datasets, ALL_DATASETS, "datasets")
    seeds = parse_seed_list(args.seed_list)
    out_dir = Path(args.outDir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"paperRetrainSeed{'-'.join(str(seed) for seed in seeds)}_{timestamp}"
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for model_name in models:
        for dataset_name in datasets:
            commands.append(
                build_command(
                    project_root=project_root,
                    python_exe=sys.executable,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    seed_list=args.seed_list,
                    epochs=args.epochs,
                    patience=args.patience,
                    out_dir=run_dir,
                )
            )

    command_lines = [shlex.join(command) for command in commands]
    commands_path = run_dir / "commands.sh"
    commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    manifest = {
        "projectRoot": str(project_root),
        "workspaceRoot": str(workspace_root),
        "pythonExecutable": sys.executable,
        "models": models,
        "datasets": datasets,
        "seeds": seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "runDirectory": str(run_dir),
        "paperProtocol": {
            "trainingMaxEpochs": 200,
            "trainingPatience": 10,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "selectionMetric": "macro_f1",
            "hpoReference": "results/hpo/*.json and docs/2602.06523v1 paper sources",
        },
        "commands": command_lines,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print(f"Saved command list: {commands_path}")

    if args.dry_run:
        print("\nDry run command plan:\n")
        for idx, line in enumerate(command_lines, start=1):
            print(f"{idx:02d}. {line}")
        return 0

    statuses = []
    for idx, command in enumerate(commands, start=1):
        display = shlex.join(command)
        print(f"\n[{idx}/{len(commands)}] {display}")
        completed = subprocess.run(command, cwd=workspace_root)
        status = {
            "index": idx,
            "command": display,
            "returncode": completed.returncode,
        }
        statuses.append(status)
        if completed.returncode != 0 and not args.continue_on_error:
            (run_dir / "status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
            return completed.returncode

    (run_dir / "status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
