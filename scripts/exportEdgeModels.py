"""
Export MicroBi-ConvLSTM and compatible baselines for edge deployment.

Current default flow:
1. Discover the best A0 MicroBi-ConvLSTM checkpoint for a dataset.
2. Normalize it into a stable `.pth` artifact.
3. Export ONNX (opset 11).
4. Convert ONNX -> SavedModel -> TFLite (float32 + full integer).
5. Emit `microbi_model.h/.cpp` for TFLite Micro embedding.
6. Run one-sample parity checks against PyTorch, ONNX Runtime, and TFLite.

The script is written so baseline checkpoints can be passed in explicitly
later using the same pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

# Keep TensorFlow quieter and a bit more deterministic for parity work.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from onnx2tf import convert as onnx_to_tf
except ImportError:
    onnx_to_tf = None


THIS_FILE = Path(__file__).resolve()
MICROBI_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[1]

if str(MICROBI_DIR) not in sys.path:
    sys.path.insert(0, str(MICROBI_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import DeepConvLSTM, TinyHAR, TinierHAR
from models import MicroBiConvLSTM


def require_module(module, package_name: str):
    if module is None:
        raise ImportError(
            f"Missing required dependency '{package_name}'. Install it in the active environment to use this path."
        )
    return module


DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "ucihar": {
        "pretty_name": "UCI-HAR",
        "in_channels": 9,
        "seq_len": 128,
        "num_classes": 6,
        "loader_module": "data.uciHar",
        "loader_name": "getUciHarLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "UCI HAR Dataset",
        "batch_size": 64,
    },
    "motionsense": {
        "pretty_name": "MotionSense",
        "in_channels": 6,
        "seq_len": 128,
        "num_classes": 6,
        "loader_module": "data.motionSense",
        "loader_name": "getMotionSenseLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "motion-sense-master",
        "batch_size": 64,
    },
    "wisdm": {
        "pretty_name": "WISDM",
        "in_channels": 3,
        "seq_len": 128,
        "num_classes": 6,
        "loader_module": "data.wisdm",
        "loader_name": "getWisdmLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "WISDM_ar_v1.1",
        "batch_size": 64,
    },
    "pamap2": {
        "pretty_name": "PAMAP2",
        "in_channels": 19,
        "seq_len": 128,
        "num_classes": 12,
        "loader_module": "data.pamap2",
        "loader_name": "getPamap2Loaders",
        "dataset_root": REPO_ROOT / "datasets" / "PAMAP2_Dataset",
        "batch_size": 64,
    },
    "opportunity": {
        "pretty_name": "Opportunity",
        "in_channels": 79,
        "seq_len": 128,
        "num_classes": 5,
        "loader_module": "data.opportunity",
        "loader_name": "getOpportunityLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "Opportunity",
        "batch_size": 64,
    },
    "unimib": {
        "pretty_name": "UniMiB-SHAR",
        "in_channels": 3,
        "seq_len": 128,
        "num_classes": 9,
        "loader_module": "data.unimib",
        "loader_name": "getUnimibLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "UniMiB-SHAR",
        "batch_size": 64,
    },
    "skoda": {
        "pretty_name": "Skoda",
        "in_channels": 30,
        "seq_len": 98,
        "num_classes": 11,
        "loader_module": "data.skoda",
        "loader_name": "getSkodaLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "Skoda",
        "batch_size": 512,
    },
    "daphnet": {
        "pretty_name": "Daphnet",
        "in_channels": 9,
        "seq_len": 64,
        "num_classes": 2,
        "loader_module": "data.daphnet",
        "loader_name": "getDaphnetLoaders",
        "dataset_root": REPO_ROOT / "datasets" / "Daphnet",
        "batch_size": 512,
    },
}


MODEL_FACTORIES: Dict[str, Callable[[str], nn.Module]] = {
    "microbi": lambda dataset: MicroBiConvLSTM(
        numClasses=int(DATASET_CONFIGS[dataset]["num_classes"]),
        inChannels=int(DATASET_CONFIGS[dataset]["in_channels"]),
        seqLen=int(DATASET_CONFIGS[dataset]["seq_len"]),
        dropout=0.0,
        aggregation="last",
    ),
    "deepconvlstm": lambda dataset: DeepConvLSTM(
        numClasses=int(DATASET_CONFIGS[dataset]["num_classes"]),
        inChannels=int(DATASET_CONFIGS[dataset]["in_channels"]),
        seqLen=int(DATASET_CONFIGS[dataset]["seq_len"]),
    ),
    "tinyhar": lambda dataset: TinyHAR(
        numClasses=int(DATASET_CONFIGS[dataset]["num_classes"]),
        inChannels=int(DATASET_CONFIGS[dataset]["in_channels"]),
        seqLen=int(DATASET_CONFIGS[dataset]["seq_len"]),
    ),
    "tinierhar": lambda dataset: TinierHAR(
        numClasses=int(DATASET_CONFIGS[dataset]["num_classes"]),
        inChannels=int(DATASET_CONFIGS[dataset]["in_channels"]),
        seqLen=int(DATASET_CONFIGS[dataset]["seq_len"]),
    ),
}


@dataclass
class ParityReport:
    label: str
    top1_match: bool
    top1_ref: int
    top1_test: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    cosine_similarity: float
    parity_percent: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "top1_match": self.top1_match,
            "top1_ref": self.top1_ref,
            "top1_test": self.top1_test,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "cosine_similarity": self.cosine_similarity,
            "parity_percent": self.parity_percent,
        }


@dataclass
class CheckpointMatch:
    model_name: str
    dataset_guess: Optional[str]
    checkpoint_path: Path
    parameter_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export edge-ready model artifacts.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORIES.keys()),
        default="microbi",
        help="Model family to export.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        default="motionsense",
        help="Dataset-specific configuration to use.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. For MicroBi, the best A0 checkpoint is auto-discovered when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write exported artifacts.",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=128,
        help="Number of samples to use for full-integer calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=74096,
        help="Preferred seed when multiple MicroBi A0 checkpoints are available with close scores.",
    )
    return parser.parse_args()


def artifact_stem(model_name: str) -> str:
    return "microbi_convlstm" if model_name == "microbi" else model_name


def infer_checkpoint_kind_and_signature(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[str, Optional[int], Optional[int], int]:
    keys = list(state_dict.keys())
    key_set = set(keys)
    param_count = int(sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)))

    if any(k.startswith("convLayers.") for k in key_set) and any(k.startswith("channelAttention.") for k in key_set):
        in_channels = None
        num_classes = None
        first_weight = state_dict.get("channelFusion.weight")
        head_weight = state_dict.get("classifier.weight")
        conv_weight = state_dict.get("convLayers.0.0.weight")
        if conv_weight is not None and head_weight is not None and first_weight is not None:
            filter_num = int(conv_weight.shape[0])
            in_channels = int(first_weight.shape[1] // filter_num)
            num_classes = int(head_weight.shape[0])
        return "tinyhar", in_channels, num_classes, param_count

    if any(k.startswith("convBlocks.") for k in key_set) and any(k.startswith("gru.") for k in key_set) and any(k.startswith("attention.") for k in key_set):
        in_channels = None
        num_classes = None
        depthwise = state_dict.get("convBlocks.0.main.0.depthwise.weight")
        classifier = state_dict.get("classifier.weight")
        gru_weight = state_dict.get("gru.weight_ih_l0")
        if depthwise is not None and classifier is not None and gru_weight is not None:
            nb_filters = int(state_dict["convBlocks.1.main.0.pointwise.weight"].shape[0] // 2)
            gru_input_dim = int(gru_weight.shape[1])
            in_channels = int(gru_input_dim // (2 * nb_filters))
            num_classes = int(classifier.shape[0])
        return "tinierhar", in_channels, num_classes, param_count

    if any(k.startswith("convLayers.") for k in key_set) and any(k.startswith("lstm.") for k in key_set) and any(k.startswith("classifier.1.") for k in key_set):
        in_channels = None
        num_classes = None
        conv_weight = state_dict.get("convLayers.0.weight")
        classifier_weight = state_dict.get("classifier.1.weight")
        if conv_weight is not None and classifier_weight is not None:
            in_channels = int(conv_weight.shape[1])
            num_classes = int(classifier_weight.shape[0])
        return "deepconvlstm", in_channels, num_classes, param_count

    if "conv1.weight" in key_set and "conv2.weight" in key_set and any(k.startswith("lstm.") for k in key_set) and "classifier.weight" in key_set:
        in_channels = int(state_dict["conv1.weight"].shape[1])
        num_classes = int(state_dict["classifier.weight"].shape[0])
        return "microbi", in_channels, num_classes, param_count

    return "unknown", None, None, param_count


def dataset_from_signature(in_channels: Optional[int], num_classes: Optional[int]) -> Optional[str]:
    if in_channels is None or num_classes is None:
        return None
    matches = [
        name
        for name, cfg in DATASET_CONFIGS.items()
        if int(cfg["in_channels"]) == in_channels and int(cfg["num_classes"]) == num_classes
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def discover_checkpoint_candidates(model_name: str, dataset: str) -> List[CheckpointMatch]:
    search_roots = [REPO_ROOT / "results", MICROBI_DIR / "results"]
    candidates: List[CheckpointMatch] = []
    for root in search_roots:
        if not root.exists():
            continue
        for checkpoint_path in root.rglob("*"):
            if not checkpoint_path.is_file() or checkpoint_path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
                continue
            try:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            state_dict = payload.get("model_state_dict") or payload.get("state_dict")
            if state_dict is None and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
                state_dict = payload
            if state_dict is None:
                continue
            kind, in_channels, num_classes, param_count = infer_checkpoint_kind_and_signature(state_dict)
            if kind != model_name:
                continue
            dataset_guess = dataset_from_signature(in_channels, num_classes)
            candidates.append(
                CheckpointMatch(
                    model_name=kind,
                    dataset_guess=dataset_guess,
                    checkpoint_path=checkpoint_path.resolve(),
                    parameter_count=param_count,
                )
            )

    exact = [candidate for candidate in candidates if candidate.dataset_guess == dataset]
    return exact if exact else candidates


def find_best_microbi_checkpoint(dataset: str, preferred_seed: int) -> Path:
    arch_dir = MICROBI_DIR / "results" / "ablations" / dataset / "arch"
    candidates = sorted(arch_dir.glob("A0_*/checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(f"No A0 checkpoint found under {arch_dir}")

    scored: List[Tuple[float, int, Path]] = []
    for checkpoint_path in candidates:
        payload = torch.load(checkpoint_path, map_location="cpu")
        result = payload.get("result", {})
        final = result.get("final", {})
        f1 = float(final.get("f1", -math.inf))
        seed = int(result.get("seed", -1))
        scored.append((f1, seed, checkpoint_path))

    # Highest F1 first, then exact preferred seed, then lexicographic path.
    scored.sort(
        key=lambda item: (
            item[0],
            1 if item[1] == preferred_seed else 0,
            str(item[2]),
        ),
        reverse=True,
    )
    return scored[0][2]


def load_checkpoint_payload(checkpoint_path: Path) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} did not contain a dictionary payload.")
    return payload


def extract_state_dict(payload: Dict[str, object]) -> Dict[str, torch.Tensor]:
    state_dict = payload.get("model_state_dict") or payload.get("state_dict")
    if state_dict is None:
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            state_dict = payload
        else:
            raise KeyError("Could not locate a model state dict inside the checkpoint payload.")
    return state_dict


def build_model(model_name: str, dataset: str) -> nn.Module:
    return MODEL_FACTORIES[model_name](dataset)


def save_canonical_checkpoint(
    model_name: str,
    dataset: str,
    source_checkpoint: Path,
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    metadata: Dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / f"{artifact_stem(model_name)}.pth"
    payload = {
        "model_name": model_name,
        "dataset": dataset,
        "source_checkpoint": str(source_checkpoint),
        "model_state_dict": state_dict,
        "metadata": metadata,
    }
    torch.save(payload, canonical_path)
    return canonical_path


def load_dataset_loaders(dataset: str):
    config = DATASET_CONFIGS[dataset]
    data_dir = MICROBI_DIR / "data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    module_name = str(config["loader_module"]).split(".")[-1]
    module = importlib.import_module(module_name)
    loader_fn = getattr(module, str(config["loader_name"]))
    root = str(config["dataset_root"])
    batch_size = int(config["batch_size"])
    result = loader_fn(root=root, batchSize=batch_size, numWorkers=0)
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    raise RuntimeError(f"Unexpected loader return signature for dataset {dataset}.")


def take_samples(loader, limit: int) -> List[np.ndarray]:
    samples: List[np.ndarray] = []
    for batch_x, _batch_y in loader:
        batch_np = batch_x.detach().cpu().numpy().astype(np.float32)
        for sample in batch_np:
            samples.append(sample[np.newaxis, ...])
            if len(samples) >= limit:
                return samples
    return samples


def export_onnx(model: nn.Module, sample_input: torch.Tensor, onnx_path: Path) -> None:
    require_module(onnx, "onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if sidecar_path.exists():
        sidecar_path.unlink()
    torch.onnx.export(
        model,
        sample_input,
        str(onnx_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
        dynamo=False,
        external_data=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def convert_onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    require_module(onnx_to_tf, "onnx2tf")
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    onnx_to_tf(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        keep_shape_absolutely_input_names=["input"],
        keep_nwc_or_nhwc_or_ndhwc_input_names=["input"],
        enable_rnn_unroll=True,
        output_integer_quantized_tflite=False,
        output_dynamic_range_quantized_tflite=False,
        disable_model_save=False,
        non_verbose=True,
        verbosity="error",
    )


def make_representative_dataset(samples: Sequence[np.ndarray]):
    def generator() -> Iterator[List[np.ndarray]]:
        for sample in samples:
            yield [sample.astype(np.float32)]

    return generator


def export_tflite_models(
    saved_model_dir: Path,
    float_tflite_path: Path,
    int_tflite_path: Path,
    mixed_int16act_tflite_path: Path,
    representative_samples: Sequence[np.ndarray],
) -> None:
    require_module(tf, "tensorflow")
    float_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    float_converter.experimental_enable_resource_variables = True
    float_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    float_model = float_converter.convert()
    float_tflite_path.write_bytes(float_model)

    int_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    int_converter.experimental_enable_resource_variables = True
    int_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int_converter.representative_dataset = make_representative_dataset(representative_samples)
    int_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int_converter.inference_input_type = tf.int8
    int_converter.inference_output_type = tf.int8
    int_model = int_converter.convert()
    int_tflite_path.write_bytes(int_model)

    mixed_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    mixed_converter.experimental_enable_resource_variables = True
    mixed_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    mixed_converter.representative_dataset = make_representative_dataset(representative_samples)
    mixed_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    mixed_converter.inference_input_type = tf.float32
    mixed_converter.inference_output_type = tf.float32
    mixed_model = mixed_converter.convert()
    mixed_int16act_tflite_path.write_bytes(mixed_model)


def should_promote_mixed_quantized(
    int_report: ParityReport,
    mixed_report: ParityReport,
) -> bool:
    if mixed_report.parity_percent < int_report.parity_percent + 5.0:
        return False
    if mixed_report.parity_percent < 90.0 and mixed_report.top1_match is False:
        return False
    if int_report.parity_percent >= 90.0 and mixed_report.parity_percent < int_report.parity_percent:
        return False
    return True


def run_onnx_inference(onnx_path: Path, sample_input: np.ndarray) -> np.ndarray:
    require_module(ort, "onnxruntime")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": sample_input.astype(np.float32)})
    return outputs[0]


def _select_output_detail(
    output_details: Sequence[Dict[str, object]],
    expected_shape: Optional[Sequence[int]],
) -> Dict[str, object]:
    if not output_details:
        raise RuntimeError("TFLite interpreter did not expose any outputs.")
    if expected_shape is None:
        return output_details[0]

    expected_tuple = tuple(int(dim) for dim in expected_shape)
    expected_numel = int(np.prod(np.array(expected_tuple, dtype=np.int64), dtype=np.int64))

    exact_matches: List[Dict[str, object]] = []
    numel_matches: List[Dict[str, object]] = []
    for detail in output_details:
        shape = tuple(int(dim) for dim in np.array(detail.get("shape", []), dtype=np.int64).tolist())
        if shape == expected_tuple:
            exact_matches.append(detail)
        if shape and int(np.prod(np.array(shape, dtype=np.int64), dtype=np.int64)) == expected_numel:
            numel_matches.append(detail)

    if exact_matches:
        return exact_matches[-1]
    if numel_matches:
        return numel_matches[-1]
    return output_details[-1]


def align_output_to_expected_shape(
    output: np.ndarray,
    expected_shape: Optional[Sequence[int]],
) -> np.ndarray:
    if expected_shape is None:
        return output

    expected_tuple = tuple(int(dim) for dim in expected_shape)
    if tuple(output.shape) == expected_tuple:
        return output

    if (
        output.ndim == len(expected_tuple)
        and expected_tuple
        and expected_tuple[0] == 1
        and tuple(output.shape[1:]) == tuple(expected_tuple[1:])
        and output.shape[0] > 1
    ):
        return output[-1:, ...]

    expected_numel = int(np.prod(np.array(expected_tuple, dtype=np.int64), dtype=np.int64))
    if output.size == expected_numel:
        return output.reshape(expected_tuple)

    return output


def run_tflite_inference(
    model_path: Path,
    sample_input: np.ndarray,
    expected_output_shape: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    require_module(tf, "tensorflow")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = _select_output_detail(interpreter.get_output_details(), expected_output_shape)

    if np.dtype(input_details["dtype"]) == np.int8:
        scale, zero_point = input_details["quantization"]
        quantized = np.round(sample_input / scale + zero_point).astype(np.int32)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized)
    else:
        interpreter.set_tensor(input_details["index"], sample_input.astype(np.float32))

    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details["index"])

    if np.dtype(output_details["dtype"]) == np.int8:
        out_scale, out_zero_point = output_details["quantization"]
        output = (raw_output.astype(np.float32) - out_zero_point) * out_scale
    else:
        output = raw_output.astype(np.float32)

    output = align_output_to_expected_shape(output, expected_output_shape)
    return output, input_details, output_details


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0.0:
        return 1.0
    return float(np.dot(a_flat, b_flat) / denom)


def parity_percent(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    rmse = math.sqrt(float(np.mean((ref - cand) ** 2)))
    scale = float(np.max(np.abs(ref))) + 1e-8
    parity = max(0.0, 1.0 - (rmse / scale))
    return float(parity * 100.0)


def build_parity_report(label: str, reference: np.ndarray, candidate: np.ndarray) -> ParityReport:
    diff = np.abs(reference - candidate)
    return ParityReport(
        label=label,
        top1_match=int(reference.argmax()) == int(candidate.argmax()),
        top1_ref=int(reference.argmax()),
        top1_test=int(candidate.argmax()),
        max_abs_error=float(diff.max()),
        mean_abs_error=float(diff.mean()),
        rmse=float(math.sqrt(float(np.mean((reference - candidate) ** 2)))),
        cosine_similarity=cosine_similarity(reference, candidate),
        parity_percent=parity_percent(reference, candidate),
    )


def bytes_to_human(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0 or unit == "GB":
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} GB"


def estimate_tensor_bytes(details: Iterable[Dict[str, object]]) -> int:
    total = 0
    for detail in details:
        shape = np.array(detail.get("shape", []), dtype=np.int64)
        if shape.size == 0:
            continue
        if np.any(shape < 0):
            continue
        count = int(np.prod(shape, dtype=np.int64))
        if count <= 0:
            continue
        dtype_size = np.dtype(detail["dtype"]).itemsize
        total += count * dtype_size
    return total


def inspect_tflite_model(model_path: Path) -> Dict[str, object]:
    require_module(tf, "tensorflow")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    try:
        op_details = interpreter._get_ops_details()  # type: ignore[attr-defined]
        op_names = [detail.get("op_name", "UNKNOWN") for detail in op_details]
    except Exception:
        op_names = []

    return {
        "tensor_bytes_estimate": estimate_tensor_bytes(tensor_details),
        "input_dtype": np.dtype(input_details["dtype"]).name,
        "output_dtype": np.dtype(output_details["dtype"]).name,
        "input_quantization": tuple(float(x) for x in input_details.get("quantization", (0.0, 0.0))),
        "output_quantization": tuple(float(x) for x in output_details.get("quantization", (0.0, 0.0))),
        "op_names": op_names,
    }


def emit_cpp_array(
    binary_path: Path,
    header_path: Path,
    source_path: Path,
    symbol_name: str,
    quant_metadata: Dict[str, object],
) -> None:
    data = binary_path.read_bytes()
    hex_lines: List[str] = []
    chunk_size = 12
    for start in range(0, len(data), chunk_size):
        chunk = data[start : start + chunk_size]
        hex_lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    hex_blob = "\n".join(hex_lines)

    input_scale, input_zero = quant_metadata.get("input_quantization", (0.0, 0.0))
    output_scale, output_zero = quant_metadata.get("output_quantization", (0.0, 0.0))

    header_path.write_text(
        f"""#pragma once

#include <cstddef>
#include <cstdint>

// ESP32 keeps model bytes in flash via PROGMEM. RP2040/Pico stores `const`
// data in XIP flash by default, so the empty fallback is the desired behavior.
#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
#include <pgmspace.h>
#define MICROBI_MODEL_STORAGE PROGMEM
#else
#define MICROBI_MODEL_STORAGE
#endif

#if defined(__GNUC__)
#define MICROBI_MODEL_ALIGN __attribute__((aligned(16)))
#else
#define MICROBI_MODEL_ALIGN
#endif

extern MICROBI_MODEL_ALIGN const unsigned char {symbol_name}[] MICROBI_MODEL_STORAGE;
extern const unsigned int {symbol_name}_len;

extern const float {symbol_name}_input_scale;
extern const int {symbol_name}_input_zero_point;
extern const float {symbol_name}_output_scale;
extern const int {symbol_name}_output_zero_point;
""",
        encoding="ascii",
    )

    source_path.write_text(
        f"""#include "{header_path.name}"

MICROBI_MODEL_ALIGN const unsigned char {symbol_name}[] MICROBI_MODEL_STORAGE = {{
{hex_blob}
}};

const unsigned int {symbol_name}_len = {len(data)};
const float {symbol_name}_input_scale = {float(input_scale):.10g}f;
const int {symbol_name}_input_zero_point = {int(input_zero)};
const float {symbol_name}_output_scale = {float(output_scale):.10g}f;
const int {symbol_name}_output_zero_point = {int(output_zero)};
""",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.lower()
    model_name = args.model.lower()
    stem = artifact_stem(model_name)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (MICROBI_DIR / "edge_exports" / f"{dataset}_{stem}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if model_name == "microbi":
            checkpoint_path = find_best_microbi_checkpoint(dataset, preferred_seed=args.seed)
        else:
            discovered = discover_checkpoint_candidates(model_name=model_name, dataset=dataset)
            if not discovered:
                raise FileNotFoundError(
                    f"No trained checkpoint for baseline model '{model_name}' and dataset '{dataset}' "
                    "was found anywhere under the workspace search roots."
                )
            checkpoint_path = discovered[0].checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_state_dict(payload)
    model = build_model(model_name, dataset)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    metadata = {
        "dataset_config": {
            "pretty_name": DATASET_CONFIGS[dataset]["pretty_name"],
            "in_channels": DATASET_CONFIGS[dataset]["in_channels"],
            "seq_len": DATASET_CONFIGS[dataset]["seq_len"],
            "num_classes": DATASET_CONFIGS[dataset]["num_classes"],
        },
        "checkpoint_result": payload.get("result"),
    }
    canonical_checkpoint_path = save_canonical_checkpoint(
        model_name=model_name,
        dataset=dataset,
        source_checkpoint=checkpoint_path,
        state_dict=state_dict,
        output_dir=output_dir,
        metadata=metadata,
    )

    train_loader, test_loader = load_dataset_loaders(dataset)
    representative_samples = take_samples(train_loader, limit=args.representative_samples)
    if not representative_samples:
        raise RuntimeError("Could not extract representative samples from the training loader.")

    test_samples = take_samples(test_loader, limit=1)
    if not test_samples:
        raise RuntimeError("Could not extract a test sample for parity validation.")

    sample_np = test_samples[0]
    sample_tensor = torch.from_numpy(sample_np)

    with torch.no_grad():
        torch_output = model(sample_tensor).detach().cpu().numpy()

    onnx_path = output_dir / f"{stem}.onnx"
    export_onnx(model, sample_tensor, onnx_path)

    onnx_output = run_onnx_inference(onnx_path, sample_np)

    saved_model_dir = output_dir / "saved_model"
    convert_onnx_to_saved_model(onnx_path, saved_model_dir)

    float_tflite_path = output_dir / f"{stem}.tflite"
    int_tflite_path = output_dir / f"{stem}_quant.tflite"
    full_int8_tflite_path = output_dir / f"{stem}_quant_fullint8.tflite"
    mixed_int16act_tflite_path = output_dir / f"{stem}_quant_int16act.tflite"
    export_tflite_models(
        saved_model_dir=saved_model_dir,
        float_tflite_path=float_tflite_path,
        int_tflite_path=full_int8_tflite_path,
        mixed_int16act_tflite_path=mixed_int16act_tflite_path,
        representative_samples=representative_samples,
    )

    float_output, _float_in, _float_out = run_tflite_inference(
        float_tflite_path,
        sample_np,
        expected_output_shape=torch_output.shape,
    )
    int_output, full_int8_input_details, full_int8_output_details = run_tflite_inference(
        full_int8_tflite_path,
        sample_np,
        expected_output_shape=torch_output.shape,
    )
    mixed_output, mixed_input_details, mixed_output_details = run_tflite_inference(
        mixed_int16act_tflite_path,
        sample_np,
        expected_output_shape=torch_output.shape,
    )

    onnx_report = build_parity_report("PyTorch vs ONNX", torch_output, onnx_output)
    float_report = build_parity_report("PyTorch vs TFLite FP32", torch_output, float_output)
    full_int8_report = build_parity_report("PyTorch vs TFLite INT8", torch_output, int_output)
    mixed_int16act_report = build_parity_report(
        "PyTorch vs TFLite INT16Act-INT8W",
        torch_output,
        mixed_output,
    )

    use_mixed_quantized = should_promote_mixed_quantized(full_int8_report, mixed_int16act_report)
    if use_mixed_quantized:
        shutil.copyfile(mixed_int16act_tflite_path, int_tflite_path)
        quant_input_details = mixed_input_details
        quant_output_details = mixed_output_details
        quant_report = mixed_int16act_report
        quant_recipe = "int16_activations_int8_weights"
    else:
        shutil.copyfile(full_int8_tflite_path, int_tflite_path)
        quant_input_details = full_int8_input_details
        quant_output_details = full_int8_output_details
        quant_report = full_int8_report
        quant_recipe = "full_int8"

    float_info = inspect_tflite_model(float_tflite_path)
    full_int8_info = inspect_tflite_model(full_int8_tflite_path)
    mixed_int16act_info = inspect_tflite_model(mixed_int16act_tflite_path)
    int_info = inspect_tflite_model(int_tflite_path)

    header_path = output_dir / f"{stem}_model.h"
    source_path = output_dir / f"{stem}_model.cpp"
    emit_cpp_array(
        binary_path=int_tflite_path,
        header_path=header_path,
        source_path=source_path,
        symbol_name=f"{stem}_model",
        quant_metadata={
            "input_quantization": quant_input_details.get("quantization", (0.0, 0.0)),
            "output_quantization": quant_output_details.get("quantization", (0.0, 0.0)),
        },
    )

    param_count = int(sum(param.numel() for param in model.parameters()))
    report = {
        "model": model_name,
        "dataset": dataset,
        "source_checkpoint": str(checkpoint_path),
        "canonical_checkpoint": str(canonical_checkpoint_path),
        "parameter_count": param_count,
        "artifacts": {
            "onnx": str(onnx_path),
            "saved_model": str(saved_model_dir),
            "tflite_fp32": str(float_tflite_path),
            "tflite_int8": str(int_tflite_path),
            "tflite_int8_full": str(full_int8_tflite_path),
            "tflite_int16act": str(mixed_int16act_tflite_path),
            "c_header": str(header_path),
            "c_source": str(source_path),
        },
        "sizes": {
            "onnx_bytes": onnx_path.stat().st_size,
            "tflite_fp32_bytes": float_tflite_path.stat().st_size,
            "tflite_int8_bytes": int_tflite_path.stat().st_size,
            "tflite_int8_full_bytes": full_int8_tflite_path.stat().st_size,
            "tflite_int16act_bytes": mixed_int16act_tflite_path.stat().st_size,
        },
        "ram_estimates": {
            "tflite_fp32_tensor_bytes": int(float_info["tensor_bytes_estimate"]),
            "tflite_int8_tensor_bytes": int(int_info["tensor_bytes_estimate"]),
            "tflite_int8_full_tensor_bytes": int(full_int8_info["tensor_bytes_estimate"]),
            "tflite_int16act_tensor_bytes": int(mixed_int16act_info["tensor_bytes_estimate"]),
        },
        "tflite_fp32": float_info,
        "tflite_int8_full": full_int8_info,
        "tflite_int16act": mixed_int16act_info,
        "tflite_int8": int_info,
        "quantized_selection": {
            "recipe": quant_recipe,
            "promoted_mixed_quantized": use_mixed_quantized,
        },
        "parity": {
            "onnx": onnx_report.to_dict(),
            "tflite_fp32": float_report.to_dict(),
            "tflite_int8": quant_report.to_dict(),
            "tflite_int8_full": full_int8_report.to_dict(),
            "tflite_int16act": mixed_int16act_report.to_dict(),
        },
    }

    report_path = output_dir / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print()
    print(
        "SUMMARY | "
        f"Model Size: {bytes_to_human(report['sizes']['tflite_int8_bytes'])} | "
        f"RAM Req: {bytes_to_human(report['ram_estimates']['tflite_int8_tensor_bytes'])} | "
        f"Parity: {quant_report.parity_percent:.3f}% | "
        f"Paths: {int_tflite_path} ; {header_path} ; {source_path}"
    )


if __name__ == "__main__":
    main()
