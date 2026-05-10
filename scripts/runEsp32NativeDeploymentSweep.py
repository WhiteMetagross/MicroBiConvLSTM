from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
MICROBI_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
PROJECT_DIR = MICROBI_DIR / "embedded" / "esp32NativeDeployment"
EXPORT_ROOT = MICROBI_DIR / "edgeExports"
MICROBI_PICO_METRICS = MICROBI_DIR / "picoDeployments" / "microbiPico2Metrics.json"
BASELINE_VALID_RUNS = MICROBI_DIR / "picoDeployments" / "baselineValidRunsSummary.json"
ESP_IDF_EXPORT_BAT = os.environ.get("ESP_IDF_EXPORT_BAT", "export.bat")
IDF_PYTHON_ENV_PATH = os.environ.get("IDF_PYTHON_ENV_PATH", "")
WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu")
WSL_MAMBAHAR_PYTHON = os.environ.get("WSL_MAMBAHAR_PYTHON", "python")
DEFAULT_PORT = "COM9"
MODEL_CHOICES = ("microbi", "deepconvlstm", "tinyhar", "tinierhar")
DEFAULT_DATASETS = (
    "ucihar",
    "motionsense",
    "wisdm",
    "pamap2",
    "opportunity",
    "unimib",
    "skoda",
    "daphnet",
)
ARENA_STEP_BYTES = 16 * 1024
MIN_ARENA_BYTES = 96 * 1024
MAX_ARENA_BYTES = 256 * 1024
SUMMARY_FLOAT_RE = re.compile(r"^([a-z0-9_]+)=(-?\d+(?:\.\d+)?)$")
SUMMARY_CLASS_RE = re.compile(r"^([a-z_]+)=(\d+) \((.+)\)$")
RUN_LATENCY_RE = re.compile(r"latency_us=(\d+)")
BUILD_SIZE_RE = re.compile(
    r"microbi_deployment\.bin binary size 0x([0-9a-fA-F]+) bytes\. Smallest app partition is 0x([0-9a-fA-F]+) bytes\."
)
BOOT_APP_RE = re.compile(r"factory app\s+00 00\s+00010000\s+([0-9A-Fa-f]{8})")

OP_NAME_TO_RESOLVER_CALL: Dict[str, str] = {
    "ADD": "AddAdd",
    "BATCH_MATMUL": "AddBatchMatMul",
    "CONCATENATION": "AddConcatenation",
    "CONV_2D": "AddConv2D",
    "DEEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "DEQUANTIZE": "AddDequantize",
    "FULLY_CONNECTED": "AddFullyConnected",
    "GATHER": "AddGather",
    "LOGISTIC": "AddLogistic",
    "MAX_POOL_2D": "AddMaxPool2D",
    "MUL": "AddMul",
    "PACK": "AddPack",
    "PAD": "AddPad",
    "QUANTIZE": "AddQuantize",
    "RESHAPE": "AddReshape",
    "REVERSE_V2": "AddReverseV2",
    "SOFTMAX": "AddSoftmax",
    "SPLIT": "AddSplit",
    "SUB": "AddSub",
    "SUM": "AddSum",
    "TANH": "AddTanh",
    "TRANSPOSE": "AddTranspose",
    "UNPACK": "AddUnpack",
}


@dataclass
class Bundle:
    model: str
    dataset: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native ESP-IDF deployment sweeps for exported HAR models on classic ESP32 boards."
    )
    parser.add_argument("--models", default="all", help="Comma-separated models or 'all'.")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets or 'all'.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="ESP32 serial port.")
    parser.add_argument("--sample-index", type=int, default=0, help="Test-set sample index for fixture generation.")
    parser.add_argument("--timeout-seconds", type=int, default=35, help="Serial capture timeout per attempt.")
    parser.add_argument("--max-arena-kb", type=int, default=MAX_ARENA_BYTES // 1024, help="Maximum tensor arena size in KB.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered bundles and arena candidates.")
    return parser.parse_args()


def parse_csv_or_all(value: str, choices: Sequence[str]) -> List[str]:
    if value.strip().lower() == "all":
        return list(choices)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in choices]
    if invalid:
        raise ValueError(f"Invalid values: {invalid}. Choices: {choices}")
    return selected


def win_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved).replace("\\", "/")[2:]
    return f"/mnt/{drive}{tail}"


def run(cmd: Sequence[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
    )


def discover_bundles(export_root: Path, models: Sequence[str], datasets: Sequence[str]) -> List[Bundle]:
    bundles: List[Bundle] = []
    for model in models:
        for dataset in datasets:
            candidate = export_root / model / dataset
            if (candidate / "parity_report.json").exists():
                bundles.append(Bundle(model=model, dataset=dataset, path=candidate))
    return bundles


def load_desktop_metrics(bundle: Bundle) -> Dict[str, object]:
    payload = json.loads((bundle.path / "parity_report.json").read_text(encoding="utf-8"))
    return {
        "desktop_int8_parity_pct": payload["parity"]["tflite_int8"]["parity_percent"],
        "desktop_int8_tensor_bytes": payload["ram_estimates"]["tflite_int8_tensor_bytes"],
        "model_bytes": payload["sizes"]["tflite_int8_bytes"],
        "parameter_count": payload["parameter_count"],
    }


def load_required_ops(bundle: Bundle) -> List[str]:
    payload = json.loads((bundle.path / "parity_report.json").read_text(encoding="utf-8"))
    op_names = payload.get("tflite_int8", {}).get("op_names") or payload.get("tflite_fp32", {}).get("op_names")
    if not op_names:
        raise KeyError(f"Missing op_names in parity report for {bundle.model}/{bundle.dataset}")
    unique_ops = [
        op_name
        for op_name in dict.fromkeys(str(op_name) for op_name in op_names)
        if op_name not in {"DELEGATE"}
    ]
    unsupported = [op_name for op_name in unique_ops if op_name not in OP_NAME_TO_RESOLVER_CALL]
    if unsupported:
        raise KeyError(
            f"Unsupported ops for resolver generation in {bundle.model}/{bundle.dataset}: {unsupported}"
        )
    return unique_ops


def load_reference_arena_map() -> Dict[Tuple[str, str], int]:
    arena_map: Dict[Tuple[str, str], int] = {}
    if MICROBI_PICO_METRICS.exists():
        payload = json.loads(MICROBI_PICO_METRICS.read_text(encoding="utf-8"))
        for row in payload:
            value = row.get("tensor_arena_used_bytes")
            if value:
                arena_map[(row["model"], row["dataset"])] = int(value)
    if BASELINE_VALID_RUNS.exists():
        payload = json.loads(BASELINE_VALID_RUNS.read_text(encoding="utf-8"))
        for model, results in payload.get("results", {}).items():
            for dataset, row in results.items():
                value = row.get("arena_used_bytes")
                if value:
                    arena_map[(model, dataset)] = int(value)
    return arena_map


def round_up(value: int, multiple: int = ARENA_STEP_BYTES) -> int:
    return int(math.ceil(value / multiple) * multiple)


def build_arena_candidates(
    bundle: Bundle,
    desktop_metrics: Dict[str, object],
    arena_reference_map: Dict[Tuple[str, str], int],
    max_arena_bytes: int,
) -> List[int]:
    reference = arena_reference_map.get((bundle.model, bundle.dataset))
    desktop_tensor = int(desktop_metrics["desktop_int8_tensor_bytes"])
    seeds: List[int] = []
    if reference:
        seeds.extend(
            [
                round_up(reference + 8 * 1024),
                round_up(reference + 24 * 1024),
            ]
        )
    heuristic = max(MIN_ARENA_BYTES, round_up(int(desktop_tensor * 1.35)))
    seeds.append(heuristic)
    if bundle.model == "microbi":
        seeds.append(128 * 1024)
        seeds.append(160 * 1024)
    elif bundle.model == "tinyhar":
        seeds.append(160 * 1024)
        seeds.append(192 * 1024)
    elif bundle.model == "tinierhar":
        seeds.append(192 * 1024)
        seeds.append(224 * 1024)
    else:
        seeds.append(224 * 1024)
        seeds.append(256 * 1024)

    candidates = sorted(
        {
            min(max_arena_bytes, max(MIN_ARENA_BYTES, candidate))
            for candidate in seeds
        }
    )
    expanded: List[int] = []
    for candidate in candidates:
        if not expanded or candidate > expanded[-1]:
            expanded.append(candidate)
        next_candidate = candidate + ARENA_STEP_BYTES
        if next_candidate <= max_arena_bytes and next_candidate not in expanded:
            expanded.append(next_candidate)
    if max_arena_bytes not in expanded:
        expanded.append(max_arena_bytes)
    return sorted({candidate for candidate in expanded if candidate <= max_arena_bytes})


def prepare_fixture(bundle: Bundle, project_dir: Path, sample_index: int) -> None:
    output_header = project_dir / "main" / "edge_fixture.h"
    command = " ".join(
        [
            shlex.quote(WSL_MAMBAHAR_PYTHON),
            shlex.quote(win_to_wsl(MICROBI_DIR / "scripts" / "generatePicoFixture.py")),
            "--model-dir",
            shlex.quote(win_to_wsl(bundle.path)),
            "--model-name",
            shlex.quote(bundle.model),
            "--dataset",
            shlex.quote(bundle.dataset),
            "--sample-index",
            shlex.quote(str(sample_index)),
            "--output",
            shlex.quote(win_to_wsl(output_header)),
        ]
    )
    completed = run(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", command],
        cwd=REPO_ROOT,
        check=True,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)


def emit_model_array(bundle: Bundle, project_dir: Path) -> None:
    stem = "microbi_convlstm" if bundle.model == "microbi" else bundle.model
    model_path = bundle.path / f"{stem}_quant.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing INT8 TFLite model: {model_path}")

    model_bytes = model_path.read_bytes()
    header = (
        "#pragma once\n\n"
        "#include <cstddef>\n\n"
        "extern const unsigned char edge_model[];\n"
        "extern const unsigned int edge_model_len;\n"
    )

    rows: List[str] = []
    for offset in range(0, len(model_bytes), 12):
        chunk = model_bytes[offset : offset + 12]
        rows.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    source = (
        '#include "edge_model_data.h"\n\n'
        "const unsigned char edge_model[] = {\n"
        + "\n".join(rows)
        + "\n};\n\n"
        f"const unsigned int edge_model_len = {len(model_bytes)};\n"
    )

    (project_dir / "main" / "edge_model_data.h").write_text(header, encoding="ascii")
    (project_dir / "main" / "edge_model_data.cc").write_text(source, encoding="ascii")


def emit_resolver_config(bundle: Bundle, project_dir: Path) -> None:
    ops = load_required_ops(bundle)
    lines = [
        "#pragma once",
        "",
        '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"',
        "",
        f"constexpr unsigned int kEdgeResolverOpCount = {len(ops)};",
        "",
        "inline void RegisterEdgeOps(",
        "    tflite::MicroMutableOpResolver<kEdgeResolverOpCount>& resolver) {",
    ]
    for op_name in ops:
        lines.append(f"  resolver.{OP_NAME_TO_RESOLVER_CALL[op_name]}();")
    lines.extend(["}", ""])
    (project_dir / "main" / "edge_ops_config.h").write_text("\n".join(lines), encoding="ascii")


def cleanup_port_holders(port: str) -> None:
    script = f"""
$targets = Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -match '{port}' -or
  $_.CommandLine -match 'idf_monitor.py' -or
  $_.CommandLine -match 'esp_idf_monitor' -or
  $_.CommandLine -match 'idf.py'
}}
foreach ($p in $targets) {{
  try {{ Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }} catch {{}}
}}
"""
    run(["powershell", "-NoProfile", "-Command", script], check=False)


def run_idf(project_dir: Path, idf_args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    arg_text = " ".join(idf_args)
    env_prefix = ""
    if IDF_PYTHON_ENV_PATH:
        env_prefix = (
            f"set IDF_PYTHON_ENV_PATH={IDF_PYTHON_ENV_PATH}&& "
            f"set PATH={IDF_PYTHON_ENV_PATH}\\Scripts;%PATH%&& "
        )
    cmd = f"{env_prefix}call {ESP_IDF_EXPORT_BAT} && cd /d {project_dir} && idf.py {arg_text}"
    return run(["cmd", "/c", cmd], cwd=REPO_ROOT, check=check)


def compile_project(
    project_dir: Path,
    arena_bytes: int,
    *,
    clean: bool,
    use_static_arena: bool = False,
) -> Tuple[Dict[str, int], str]:
    if clean:
        build_dir = project_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
    args: List[str] = [
        f"-DEDGE_TENSOR_ARENA_SIZE={arena_bytes}",
        f"-DEDGE_USE_STATIC_ARENA={1 if use_static_arena else 0}",
        "build",
    ]
    completed = run_idf(project_dir, args, check=False)
    combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    metrics: Dict[str, int] = {}
    size_match = BUILD_SIZE_RE.search(combined)
    if size_match:
        metrics["firmware_flash_bytes"] = int(size_match.group(1), 16)
        metrics["app_partition_bytes"] = int(size_match.group(2), 16)
    return metrics, combined


def flash_project(project_dir: Path, port: str) -> str:
    last_completed: Optional[subprocess.CompletedProcess[str]] = None
    logs: List[str] = []
    for attempt in range(1, 4):
        cleanup_port_holders(port)
        completed = run_idf(project_dir, ["-p", port, "flash"], check=False)
        last_completed = completed
        combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        if combined:
            logs.append(f"=== FLASH ATTEMPT {attempt} ===\n{combined}")
        if completed.returncode == 0:
            return "\n\n".join(logs)

        retryable = any(
            marker in combined
            for marker in (
                "Wrong boot mode detected",
                "No serial data received",
                "Failed to connect",
                "Invalid head of packet",
            )
        )
        if not retryable or attempt == 3:
            break
        time.sleep(2.0)

    assert last_completed is not None
    raise subprocess.CalledProcessError(
        last_completed.returncode,
        last_completed.args,
        output="\n\n".join(logs),
        stderr=last_completed.stderr,
    )


def capture_serial(port: str, timeout_seconds: int) -> str:
    script = f"""
$portName = '{port}'
$deadline = (Get-Date).AddSeconds(20)
$serial = $null
$lastError = $null
Write-Output 'HOST_STAGE=wait_for_port_begin'
while ((Get-Date) -lt $deadline -and $null -eq $serial) {{
  try {{
    $serial = New-Object System.IO.Ports.SerialPort $portName,115200,None,8,one
    $serial.ReadTimeout = 1000
    $serial.DtrEnable = $false
    $serial.RtsEnable = $false
    $serial.Open()
  }} catch {{
    $lastError = $_.Exception.Message
    $serial = $null
    Start-Sleep -Milliseconds 500
  }}
}}
if ($null -eq $serial) {{
  Write-Output 'HOST_STAGE=wait_for_port_failed'
  if ($lastError) {{ Write-Output $lastError }}
  exit 2
}}
Write-Output 'HOST_STAGE=serial_open_done'
try {{
  $endTime = (Get-Date).AddSeconds({timeout_seconds})
  while ((Get-Date) -lt $endTime) {{
    try {{
      $line = $serial.ReadLine()
      if ($line) {{
        $trimmed = $line.TrimEnd("`r", "`n")
        Write-Output $trimmed
        if ($trimmed -eq '=== DONE ===' -or $trimmed -eq '=== FAILED ===') {{ break }}
      }}
    }} catch [System.TimeoutException] {{
    }}
  }}
}} finally {{
  Write-Output 'HOST_STAGE=serial_close'
  try {{ $serial.Close() }} catch {{}}
}}
"""
    completed = run(["powershell", "-NoProfile", "-Command", script], check=False)
    combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return combined


def parse_failure_reason(log_text: str) -> str:
    preferred = (
        "Failed to resize buffer.",
        "Failed to allocate tail memory.",
        "Failed to allocate temp memory.",
        "Failed to allocate memory for memory planning",
        "Failed to allocate tensor arena from internal SRAM.",
        "Arena allocation failed.",
        "Didn't find op for builtin opcode",
        "Failed to get registration from op code",
        "Schema mismatch:",
        "Node ",
        "Guru Meditation Error:",
        "abort() was called",
        "region `dram0_0_seg' overflowed",
        "partition-table",
        "AllocateTensors failed.",
    )
    for line in log_text.splitlines():
        clean = line.strip()
        if clean.startswith(preferred):
            return clean
    for line in log_text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("HOST_STAGE="):
            continue
        if "\x00" in clean:
            continue
        if clean.startswith("I (") or clean.startswith("W (") or clean.startswith("E ("):
            continue
        if clean.startswith("Mode:") or clean.startswith("Fixture ") or clean.startswith("Model bytes:"):
            continue
        if clean.startswith("Tensor arena bytes") or clean.startswith("Heap total internal bytes before setup:"):
            continue
        if clean.startswith("Heap free internal bytes before setup:") or clean.startswith("Largest internal 8-bit block before setup:"):
            continue
        if clean in {"AllocateTensors failed.", "Invoke failed.", "=== FAILED ==="}:
            continue
        return clean
    return "unknown failure"


def parse_serial_metrics(log_text: str) -> Dict[str, object]:
    result: Dict[str, object] = {"success": "=== SUMMARY ===" in log_text, "raw_log": log_text}
    run_latencies: List[int] = []
    for line in log_text.splitlines():
        run_match = RUN_LATENCY_RE.search(line)
        if run_match:
            run_latencies.append(int(run_match.group(1)))
        float_match = SUMMARY_FLOAT_RE.match(line.strip())
        if float_match:
            key, value = float_match.groups()
            result[key] = float(value) if "." in value else int(value)
            continue
        class_match = SUMMARY_CLASS_RE.match(line.strip())
        if class_match:
            key, idx, label = class_match.groups()
            result[key] = {"index": int(idx), "label": label}
            continue
        if line.startswith("Heap total internal bytes before setup:"):
            result["heap_total_internal_bytes_before_setup"] = int(line.rsplit(":", 1)[1].strip())
        elif line.startswith("Heap free internal bytes before setup:"):
            result["heap_free_internal_bytes_before_setup"] = int(line.rsplit(":", 1)[1].strip())
        elif line.startswith("Largest internal 8-bit block before setup:"):
            result["largest_internal_block_before_setup_bytes"] = int(line.rsplit(":", 1)[1].strip())
        elif line.startswith("Tensor arena bytes (static internal SRAM):") or line.startswith(
            "Tensor arena bytes (dynamic internal SRAM):"
        ):
            result["tensor_arena_alloc_bytes"] = int(line.rsplit(":", 1)[1].strip())
        else:
            boot_match = BOOT_APP_RE.search(line)
            if boot_match:
                result["boot_app_partition_bytes"] = int(boot_match.group(1), 16)
    if run_latencies:
        mean = sum(run_latencies) / len(run_latencies)
        result["latency_min_us"] = min(run_latencies)
        result["latency_max_us"] = max(run_latencies)
        result["latency_std_us"] = (
            sum((value - mean) ** 2 for value in run_latencies) / len(run_latencies)
        ) ** 0.5
    if not result["success"]:
        result["failure_reason"] = parse_failure_reason(log_text)
    return result


def should_retry_with_larger_arena(serial_metrics: Dict[str, object]) -> bool:
    reason = str(serial_metrics.get("failure_reason", ""))
    raw_log = str(serial_metrics.get("raw_log", ""))
    if "AllocateTensors failed." in raw_log:
        return True
    return any(
        marker in reason
        for marker in (
            "Failed to resize buffer.",
            "Failed to allocate tail memory.",
            "Failed to allocate temp memory.",
            "Failed to allocate memory for memory planning",
            "Failed to allocate tensor arena from internal SRAM.",
            "Arena allocation failed.",
            "failed to prepare with status",
        )
    )


def compute_status_row(
    bundle: Bundle,
    desktop_metrics: Dict[str, object],
    compile_metrics: Dict[str, int],
    serial_metrics: Dict[str, object],
    arena_bytes: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model": bundle.model,
        "dataset": bundle.dataset,
        **desktop_metrics,
        **compile_metrics,
        **serial_metrics,
        "requested_arena_bytes": arena_bytes,
    }
    if row.get("success"):
        desktop_parity = float(row["desktop_int8_parity_pct"])
        esp_parity = float(row.get("avg_parity_vs_pytorch_pct", 0.0))
        row["parity_drop_pct_points"] = desktop_parity - esp_parity
        if row.get("predicted_class") and row.get("expected_class"):
            row["top1_match_on_esp32"] = (
                row["predicted_class"]["index"] == row["expected_class"]["index"]
            )
    return row


def write_reports(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "esp32_native_deployment_status.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    lines = [
        "# ESP32 Native Deployment Sweep",
        "",
        "| Model | Dataset | Status | Arena KB | Flash B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Heap Used B | Arena Used B | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        status = "RUNS" if row.get("success") else "FAILS"
        note = (
            f"top1={row.get('top1_match_on_esp32', '')}"
            if row.get("success")
            else str(row.get("failure_reason", "unknown"))
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["dataset"],
                    status,
                    str(int(row.get("requested_arena_bytes", 0)) // 1024),
                    str(row.get("firmware_flash_bytes", "")),
                    f"{float(row['avg_latency_us']) / 1000.0:.3f}" if row.get("avg_latency_us") is not None else "",
                    f"{float(row['avg_parity_vs_pytorch_pct']):.3f}" if row.get("avg_parity_vs_pytorch_pct") is not None else "",
                    f"{float(row['desktop_int8_parity_pct']):.3f}",
                    str(row.get("heap_used_internal_bytes", "")),
                    str(row.get("tensor_arena_used_bytes", "")),
                    note.replace("|", "/"),
                ]
            )
            + " |"
        )
    (output_dir / "esp32_native_deployment_status.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def update_paper_results(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    paper_results = MICROBI_DIR / "ESP32paper_results.md"
    successful = [row for row in rows if row.get("success")]
    failed = [row for row in rows if not row.get("success")]
    lines = [
        "# ESP32 Deployment Results",
        "",
        "## Native Sweep Summary",
        "",
        f"- Bundles tested: `{len(rows)}`",
        f"- Successful on-device runs: `{len(successful)}`",
        f"- Failed deployments: `{len(failed)}`",
        f"- Sweep report: [esp32_native_deployment_status.md]({output_dir / 'esp32_native_deployment_status.md'})",
        "",
        "## Successful Runs",
        "",
        "| Model | Dataset | Arena KB | Flash B | Heap Used B | Arena Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Degradation pp | Top-1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in successful:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["dataset"],
                    str(int(row.get("requested_arena_bytes", 0)) // 1024),
                    str(row.get("firmware_flash_bytes", "")),
                    str(row.get("heap_used_internal_bytes", "")),
                    str(row.get("tensor_arena_used_bytes", "")),
                    f"{float(row['avg_latency_us']) / 1000.0:.3f}",
                    f"{float(row['avg_parity_vs_pytorch_pct']):.3f}",
                    f"{float(row['desktop_int8_parity_pct']):.3f}",
                    f"{float(row['parity_drop_pct_points']):.3f}",
                    (
                        f"{row['predicted_class']['label']} / {row['expected_class']['label']}"
                        if row.get("predicted_class") and row.get("expected_class")
                        else ""
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failed Runs",
            "",
            "| Model | Dataset | Arena KB | Failure Reason | Flash B | Model B | Desktop Tensor B |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in failed:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["dataset"],
                    str(int(row.get("requested_arena_bytes", 0)) // 1024),
                    str(row.get("failure_reason", "unknown")).replace("|", "/"),
                    str(row.get("firmware_flash_bytes", "")),
                    str(row.get("model_bytes", "")),
                    str(row.get("desktop_int8_tensor_bytes", "")),
                ]
            )
            + " |"
        )
    paper_results.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    models = parse_csv_or_all(args.models, MODEL_CHOICES)
    datasets = parse_csv_or_all(args.datasets, DEFAULT_DATASETS)
    bundles = discover_bundles(EXPORT_ROOT, models, datasets)
    if not bundles:
        raise SystemExit("No matching export bundles were found.")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = MICROBI_DIR / "esp32Deployments" / f"nativeEsp32Sweep_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_map = load_reference_arena_map()
    max_arena_bytes = args.max_arena_kb * 1024

    if args.dry_run:
        for bundle in bundles:
            desktop_metrics = load_desktop_metrics(bundle)
            print(
                f"{bundle.model}/{bundle.dataset} -> "
                f"{build_arena_candidates(bundle, desktop_metrics, reference_map, max_arena_bytes)}"
            )
        return

    rows: List[Dict[str, object]] = []
    clean_build = True
    for bundle in bundles:
        desktop_metrics = load_desktop_metrics(bundle)
        candidates = build_arena_candidates(bundle, desktop_metrics, reference_map, max_arena_bytes)
        bundle_log_dir = output_dir / bundle.model / bundle.dataset
        bundle_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Deploying {bundle.model}/{bundle.dataset} ===")
        print(f"Arena candidates: {[value // 1024 for value in candidates]} KB")

        prepare_fixture(bundle, PROJECT_DIR, args.sample_index)
        emit_model_array(bundle, PROJECT_DIR)
        emit_resolver_config(bundle, PROJECT_DIR)

        final_row: Optional[Dict[str, object]] = None
        for attempt_index, arena_bytes in enumerate(candidates, start=1):
            attempt_dir = bundle_log_dir / f"attempt_{attempt_index:02d}_{arena_bytes // 1024}kb"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            print(f"Attempt {attempt_index}: arena={arena_bytes // 1024} KB")
            try:
                compile_metrics, compile_log = compile_project(
                    PROJECT_DIR,
                    arena_bytes,
                    clean=clean_build,
                )
                clean_build = False
                (attempt_dir / "compile.log").write_text(compile_log + "\n", encoding="utf-8")

                flash_log = flash_project(PROJECT_DIR, args.port)
                (attempt_dir / "flash.log").write_text(flash_log + "\n", encoding="utf-8")

                serial_log = capture_serial(args.port, args.timeout_seconds)
                (attempt_dir / "serial.log").write_text(serial_log + "\n", encoding="utf-8")
                serial_metrics = parse_serial_metrics(serial_log)
                row = compute_status_row(bundle, desktop_metrics, compile_metrics, serial_metrics, arena_bytes)
                final_row = row
                if row.get("success"):
                    break
                if not should_retry_with_larger_arena(serial_metrics):
                    break
            except subprocess.CalledProcessError as exc:
                error_text = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
                (attempt_dir / "error.log").write_text(error_text + "\n", encoding="utf-8")
                final_row = {
                    "model": bundle.model,
                    "dataset": bundle.dataset,
                    **desktop_metrics,
                    "requested_arena_bytes": arena_bytes,
                    "success": False,
                    "failure_reason": f"host_command_failed: {error_text.splitlines()[-1] if error_text else exc}",
                }
                if "overflowed by" in error_text or "No space left" in error_text:
                    break
            finally:
                if final_row is not None:
                    (attempt_dir / "result.json").write_text(
                        json.dumps({k: v for k, v in final_row.items() if k != "raw_log"}, indent=2),
                        encoding="utf-8",
                    )
        if final_row is None:
            final_row = {
                "model": bundle.model,
                "dataset": bundle.dataset,
                **desktop_metrics,
                "success": False,
                "failure_reason": "no_attempt_completed",
            }

        rows.append(final_row)
        write_reports(output_dir, rows)
        update_paper_results(output_dir, rows)
        print(json.dumps({k: v for k, v in final_row.items() if k != "raw_log"}, indent=2))

    write_reports(output_dir, rows)
    update_paper_results(output_dir, rows)
    print(f"Wrote native ESP32 sweep outputs to {output_dir}")


if __name__ == "__main__":
    main()
