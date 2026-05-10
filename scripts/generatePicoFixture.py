"""
Generate a reproducible Pico-side inference fixture for an exported HAR model.

The fixture contains:
- one test sample
- the expected label and class names
- PyTorch reference logits
- TFLite FP32 and INT8 reference logits

This lets the microcontroller firmware validate that the on-device output matches
the desktop export path while also timing repeated inference.
"""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

from exportEdgeModels import (
    DATASET_CONFIGS,
    build_model,
    extract_state_dict,
    load_checkpoint_payload,
    load_dataset_loaders,
    run_tflite_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Pico inference fixture header.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the exported model bundle and parity_report.json.",
    )
    parser.add_argument(
        "--model-name",
        default="microbi",
        choices=["microbi", "deepconvlstm", "tinyhar", "tinierhar"],
        help="Logical model family used by exportEdgeModels.py.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key used by exportEdgeModels.py, for example motionsense.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based index into the test split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Header file to create.",
    )
    return parser.parse_args()


def iter_test_samples(loader) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for batch_x, batch_y in loader:
        batch_x_np = batch_x.detach().cpu().numpy().astype(np.float32)
        batch_y_np = batch_y.detach().cpu().numpy().astype(np.int64)
        for sample, label in zip(batch_x_np, batch_y_np):
            yield sample[np.newaxis, ...], label


def cpp_float(value: float) -> str:
    scalar = float(value)
    formatted = f"{scalar:.9g}"
    if "e" not in formatted.lower() and "." not in formatted:
        formatted += ".0"
    return f"{formatted}f"


def emit_float_array(values: Sequence[float], line_width: int = 8) -> str:
    parts: List[str] = []
    for idx in range(0, len(values), line_width):
        row = values[idx : idx + line_width]
        parts.append("  " + ", ".join(cpp_float(v) for v in row) + ",")
    return "\n".join(parts)


def emit_string_array(values: Sequence[str]) -> str:
    rows = [f'  "{value}",' for value in values]
    return "\n".join(rows)


def resolve_class_names(dataset_name: str, test_loader, num_classes: int) -> List[str]:
    dataset = getattr(test_loader, "dataset", None)
    class_names_attr = getattr(dataset, "classNames", None)
    if callable(class_names_attr):
        names = list(class_names_attr())
        if len(names) == num_classes:
            return [str(name) for name in names]
    if isinstance(class_names_attr, (list, tuple)):
        names = list(class_names_attr)
        if len(names) == num_classes:
            return [str(name) for name in names]

    module = importlib.import_module(DATASET_CONFIGS[dataset_name]["loader_module"])
    constant_candidates = [
        "ACTIVITIES",
        "PAMAP2_ACTIVITIES",
        "SKODA_ACTIVITIES",
        "DAPHNET_ACTIVITIES",
        "ACTIVITY_LABELS",
        "ADL_LABELS",
        "LOCOMOTION_LABELS",
    ]
    for constant_name in constant_candidates:
        value = getattr(module, constant_name, None)
        if isinstance(value, dict):
            ordered = [str(value[idx]) for idx in sorted(value.keys())]
            if len(ordered) == num_classes:
                return ordered
        if isinstance(value, (list, tuple)) and len(value) == num_classes:
            return [str(item) for item in value]

    metadata_candidates = [
        "metadata",
        "config",
    ]
    for attr_name in metadata_candidates:
        metadata = getattr(dataset, attr_name, None)
        if isinstance(metadata, dict):
            labels = metadata.get("labels") or metadata.get("activities")
            if isinstance(labels, dict):
                ordered = [str(labels[idx]) for idx in sorted(labels.keys())]
                if len(ordered) == num_classes:
                    return ordered
            if isinstance(labels, (list, tuple)) and len(labels) == num_classes:
                return [str(item) for item in labels]

    return [f"class_{idx}" for idx in range(num_classes)]


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    parity_report_path = model_dir / "parity_report.json"
    if not parity_report_path.exists():
        raise FileNotFoundError(f"Missing parity report: {parity_report_path}")

    parity_payload = json.loads(parity_report_path.read_text(encoding="utf-8"))
    checkpoint_path = model_dir / (
        "microbi_convlstm.pth"
        if args.model_name == "microbi"
        else f"{args.model_name}.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing canonical checkpoint: {checkpoint_path}")

    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_state_dict(payload)
    model = build_model(args.model_name, args.dataset)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _train_loader, test_loader = load_dataset_loaders(args.dataset)
    test_iter = iter_test_samples(test_loader)

    sample_np = None
    label = None
    for idx, (maybe_sample, maybe_label) in enumerate(test_iter):
        if idx == args.sample_index:
            sample_np = maybe_sample
            label = int(maybe_label)
            break
    if sample_np is None or label is None:
        raise IndexError(f"Test sample index {args.sample_index} is out of range.")

    sample_tensor = torch.from_numpy(sample_np)
    with torch.no_grad():
        torch_output = model(sample_tensor).detach().cpu().numpy().astype(np.float32).reshape(-1)
    class_names = resolve_class_names(args.dataset, test_loader, int(torch_output.shape[0]))

    stem = "microbi_convlstm" if args.model_name == "microbi" else args.model_name
    fp32_tflite = model_dir / f"{stem}.tflite"
    int8_tflite = model_dir / f"{stem}_quant.tflite"
    fp32_output, _fp32_in, _fp32_out = run_tflite_inference(fp32_tflite, sample_np, expected_output_shape=(1, len(class_names)))
    int8_output, _int8_in, _int8_out = run_tflite_inference(int8_tflite, sample_np, expected_output_shape=(1, len(class_names)))
    fp32_output = fp32_output.astype(np.float32).reshape(-1)
    int8_output = int8_output.astype(np.float32).reshape(-1)

    flat_input = sample_np.reshape(-1).astype(np.float32)
    input_timesteps = int(sample_np.shape[1])
    input_channels = int(sample_np.shape[2])
    num_classes = len(class_names)

    header = f"""#pragma once

#include <cstddef>

static const char kFixtureModelName[] = "{args.model_name}";
static const char kFixtureDatasetName[] = "{args.dataset}";
constexpr int kFixtureSampleIndex = {args.sample_index};
constexpr size_t kFixtureInputTimesteps = {input_timesteps};
constexpr size_t kFixtureInputChannels = {input_channels};
constexpr size_t kFixtureInputElementCount = {flat_input.size};
constexpr size_t kFixtureNumClasses = {num_classes};
constexpr int kFixtureExpectedLabel = {label};
constexpr int kFixtureExpectedPyTorchTop1 = {int(np.argmax(torch_output))};
constexpr int kFixtureExpectedTfliteInt8Top1 = {int(np.argmax(int8_output))};
constexpr float kFixtureDesktopInt8ParityPercent = {cpp_float(float(parity_payload["parity"]["tflite_int8"]["parity_percent"]))};

static const char* const kFixtureClassNames[kFixtureNumClasses] = {{
{emit_string_array(class_names)}
}};

alignas(16) static const float kFixtureInput[kFixtureInputElementCount] = {{
{emit_float_array(flat_input.tolist())}
}};

static const float kFixtureExpectedPyTorchLogits[kFixtureNumClasses] = {{
{emit_float_array(torch_output.tolist(), line_width=num_classes)}
}};

static const float kFixtureExpectedTfliteFp32Logits[kFixtureNumClasses] = {{
{emit_float_array(fp32_output.tolist(), line_width=num_classes)}
}};

static const float kFixtureExpectedTfliteInt8Logits[kFixtureNumClasses] = {{
{emit_float_array(int8_output.tolist(), line_width=num_classes)}
}};
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header, encoding="ascii")
    print(f"Wrote Pico fixture to {args.output}")


if __name__ == "__main__":
    main()
