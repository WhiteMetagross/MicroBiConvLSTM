from __future__ import annotations

import argparse
import json
import os
import re
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

ARDUINO_CLI = Path(os.environ.get("ARDUINO_CLI", "arduino-cli"))
PICOTOOL = Path(os.environ.get("PICOTOOL", "picotool"))
ARDUINO_PYTHON = Path(os.environ.get("PICO_TOOL_PYTHON", sys.executable))
UF2CONV = Path(os.environ.get("UF2CONV", "uf2conv.py"))
PYSERIAL_DIR = Path(os.environ.get("PYSERIAL_DIR", "."))
WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu")
WSL_DEPLOY_PYTHON = os.environ.get("WSL_DEPLOY_PYTHON", "python")
PROJECT_DIR = MICROBI_DIR / "embedded" / "pico2EdgeRuntime"
EXPORT_ROOT = MICROBI_DIR / "edgeExports"

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

COMPILE_FLASH_RE = re.compile(r"Sketch uses (\d+) bytes")
COMPILE_RAM_RE = re.compile(
    r"Global variables use (\d+) bytes .* leaving (\d+) bytes"
)
SUMMARY_FLOAT_RE = re.compile(r"^([a-z0-9_]+)=(-?\d+(?:\.\d+)?)$")
SUMMARY_CLASS_RE = re.compile(r"^([a-z_]+)=(\d+) \((.+)\)$")
RUN_LATENCY_RE = re.compile(r"latency_us=(\d+)")


@dataclass
class Bundle:
    model: str
    dataset: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy converted edge models to Raspberry Pi Pico 2 and capture runtime status."
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=EXPORT_ROOT,
        help="Root folder containing converted edge model bundles.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR,
        help="Reusable Arduino sketch folder for deployment.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model families or 'all'.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated datasets or 'all'.",
    )
    parser.add_argument(
        "--port",
        default="COM8",
        help="Serial port for the Pico 2 runtime device.",
    )
    parser.add_argument(
        "--arena-bytes",
        type=int,
        default=425_984,
        help="Tensor arena size passed to the generic sketch.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Test-set sample index used to build the verification fixture.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=45,
        help="Serial capture timeout after flashing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a timestamped folder under MicroBiConvLSTM/picoDeployments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered bundles without compiling or flashing.",
    )
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


def run_powershell(script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        ["powershell", "-NoProfile", "-Command", script],
        check=check,
    )


def run_python(
    args: Sequence[str], *, cwd: Optional[Path] = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return run([str(ARDUINO_PYTHON), *args], cwd=cwd, check=check)


def find_bootsel_drive() -> Optional[str]:
    script = (
        "$vol = Get-CimInstance Win32_Volume | "
        "Where-Object { $_.Label -match 'RP2350|RPI|PICO' -and $_.DriveLetter } | "
        "Select-Object -First 1 -ExpandProperty DriveLetter; "
        "if ($vol) { Write-Output $vol }"
    )
    completed = run_powershell(script, check=False)
    drive = (completed.stdout or "").strip()
    return drive or None


def run_powershell_with_timeout(
    script: str, *, timeout_seconds: int, cwd: Optional[Path] = None
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                ["cmd", "/c", "taskkill", "/PID", str(process.pid), "/F", "/T"],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            stdout, stderr = process.communicate()
        raise subprocess.CalledProcessError(
            returncode=-9,
            cmd=process.args,
            output=stdout,
            stderr=(stderr or "") + "\nflash_and_capture_timeout",
        )
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def discover_bundles(export_root: Path, models: Sequence[str], datasets: Sequence[str]) -> List[Bundle]:
    bundles: List[Bundle] = []
    for model in models:
        for dataset in datasets:
            candidate = export_root / model / dataset
            if (candidate / "parity_report.json").exists():
                bundles.append(Bundle(model=model, dataset=dataset, path=candidate))
    return bundles


def find_model_artifacts(bundle: Bundle) -> Tuple[Path, Path, str]:
    header_candidates = sorted(bundle.path.glob("*_model.h"))
    source_candidates = sorted(bundle.path.glob("*_model.cpp"))
    if len(header_candidates) != 1 or len(source_candidates) != 1:
        raise FileNotFoundError(f"Expected one model header/source in {bundle.path}")
    header = header_candidates[0]
    source = source_candidates[0]
    stem = header.name[: -len("_model.h")]
    return header, source, stem


def prepare_fixture(bundle: Bundle, project_dir: Path, sample_index: int) -> None:
    output_header = project_dir / "edge_fixture.h"
    command = (
        f"cd '{win_to_wsl(MICROBI_DIR)}' && "
        f"{WSL_DEPLOY_PYTHON} scripts/generatePicoFixture.py "
        f"--model-dir '{win_to_wsl(bundle.path)}' "
        f"--model-name {bundle.model} "
        f"--dataset {bundle.dataset} "
        f"--sample-index {sample_index} "
        f"--output '{win_to_wsl(output_header)}'"
    )
    completed = run(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-ic", command],
        check=True,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)


def rewrite_model_files(bundle: Bundle, project_dir: Path) -> None:
    header_src, source_src, stem = find_model_artifacts(bundle)
    header_text = header_src.read_text(encoding="utf-8")
    source_text = source_src.read_text(encoding="utf-8")

    replacements = {
        f'{stem}_model.h': "edge_model.h",
        f"{stem}_model_len": "edge_model_len",
        f"{stem}_model_input_scale": "edge_model_input_scale",
        f"{stem}_model_input_zero_point": "edge_model_input_zero_point",
        f"{stem}_model_output_scale": "edge_model_output_scale",
        f"{stem}_model_output_zero_point": "edge_model_output_zero_point",
        f"{stem}_model": "edge_model",
    }
    for old, new in replacements.items():
        header_text = header_text.replace(old, new)
        source_text = source_text.replace(old, new)

    (project_dir / "edge_model.h").write_text(header_text, encoding="utf-8")
    (project_dir / "edge_model.cpp").write_text(source_text, encoding="utf-8")


def should_retry_compile_with_clean(log_text: str) -> bool:
    retry_markers = (
        "arm-none-eabi-ar:",
        "No such file or directory",
        "undefined reference to `setup'",
        "undefined reference to `loop'",
    )
    return any(marker in log_text for marker in retry_markers)


def compile_project(project_dir: Path, arena_bytes: int, *, clean: bool) -> Tuple[Dict[str, int], str]:
    src_dir = project_dir / "src"
    def run_compile(clean_build: bool) -> subprocess.CompletedProcess[str]:
        compile_cmd = [
            str(ARDUINO_CLI),
            "compile",
            "--export-binaries",
            "--fqbn",
            "rp2040:rp2040:rpipico2:flash=4194304_0,arch=arm,uploadmethod=picotool,freq=150,opt=Optimize2,rtti=Disabled,exceptions=Disabled,dbgport=Disabled,dbglvl=None,usbstack=picosdk,ipbtstack=ipv4only",
            "--build-property",
            f"compiler.cpp.extra_flags=-I{src_dir} -DEDGE_TENSOR_ARENA_SIZE={arena_bytes}",
            "--build-property",
            f"compiler.c.extra_flags=-I{src_dir} -DEDGE_TENSOR_ARENA_SIZE={arena_bytes}",
            str(project_dir),
        ]
        if clean_build:
            compile_cmd.insert(2, "--clean")
        return run(compile_cmd, cwd=REPO_ROOT, check=False)

    completed = run_compile(clean)
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0 and not clean and should_retry_compile_with_clean(combined):
        completed = run_compile(True)
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    metrics: Dict[str, int] = {}
    flash_match = COMPILE_FLASH_RE.search(combined)
    ram_match = COMPILE_RAM_RE.search(combined)
    if flash_match:
        metrics["firmware_flash_bytes"] = int(flash_match.group(1))
    if ram_match:
        metrics["global_ram_bytes"] = int(ram_match.group(1))
        metrics["global_ram_free_bytes"] = int(ram_match.group(2))
    return metrics, combined


def flash_and_capture(project_dir: Path, port: str, timeout_seconds: int) -> str:
    uf2 = project_dir / "build" / "rp2040.rp2040.rpipico2" / "pico2EdgeRuntime.ino.uf2"
    deploy_logs: List[str] = []

    bootsel_drive = find_bootsel_drive()
    if bootsel_drive:
        picotool_cmd = [str(PICOTOOL), "load", str(uf2), "-x"]
        completed = run(picotool_cmd, cwd=REPO_ROOT, check=False)
        combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        deploy_logs.append(f"HOST_STAGE=picotool_bootsel\n{combined}")
        if completed.returncode != 0:
            target = f"{bootsel_drive}\\NEW.UF2"
            copy_script = (
                f"Copy-Item -LiteralPath '{uf2}' -Destination '{target}' -Force; "
                f"Write-Output 'Flashed {target}'"
            )
            completed = run_powershell(copy_script, check=False)
            combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            deploy_logs.append(f"HOST_STAGE=copy_drive\n{combined}")
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    ["powershell", "-NoProfile", "-Command", copy_script],
                    output="\n\n".join(deploy_logs),
                    stderr="direct_copy_failed",
                )
    else:
        deploy_attempts: List[Tuple[str, Sequence[str]]] = [
        (
            "picotool_force",
            [str(PICOTOOL), "load", str(uf2), "-f", "-x"],
        ),
        (
            "direct_drive",
            [str(UF2CONV), "--family", "RP2040", "--deploy", str(uf2)],
        ),
        (
            "serial_reset",
            [str(UF2CONV), "--serial", port, "--family", "RP2040", "--deploy", str(uf2)],
        ),
        ]
        flash_ok = False
        for label, deploy_cmd in deploy_attempts:
            runner = run if Path(deploy_cmd[0]).suffix.lower() == ".exe" else run_python
            completed = runner(deploy_cmd, cwd=REPO_ROOT, check=False)
            combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            deploy_logs.append(f"HOST_STAGE={label}\n{combined}")
            if completed.returncode == 0:
                flash_ok = True
                break
        if not flash_ok:
            raise subprocess.CalledProcessError(
                1,
                [str(ARDUINO_PYTHON), str(UF2CONV)],
                output="\n\n".join(deploy_logs),
                stderr="uf2_upload_failed",
            )

    capture_script = f"""
import sys
import time
sys.path.insert(0, r"{PYSERIAL_DIR}")
import serial
import serial.tools.list_ports

port = sys.argv[1]
deadline = time.time() + 25.0
ser = None
last_error = None
print("HOST_STAGE=wait_for_port_begin")
while time.time() < deadline and ser is None:
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if port in ports:
        try:
            ser = serial.Serial(port, 115200, timeout=1.0)
            ser.dtr = True
            ser.rts = True
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
    else:
        time.sleep(0.5)

if ser is None:
    print("HOST_STAGE=wait_for_port_failed")
    if last_error:
        print(last_error)
    sys.exit(2)

print("HOST_STAGE=serial_open_done")
end_time = time.time() + {timeout_seconds}
try:
    while time.time() < end_time:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("utf-8", errors="replace").rstrip("\\r\\n")
        print(text)
        if text == "=== DONE ===":
            break
finally:
    print("HOST_STAGE=serial_close")
    try:
        ser.close()
    except Exception:
        pass
"""
    completed = run_python(
        ["-c", capture_script, port],
        cwd=REPO_ROOT,
        check=False,
    )
    combined_capture = (completed.stdout or "").strip()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            [str(ARDUINO_PYTHON), "-c", "<capture_script>", port],
            output="\n\n".join(deploy_logs) + ("\n\n" + combined_capture if combined_capture else ""),
            stderr=completed.stderr or "serial_capture_failed",
        )
    return "\n\n".join(deploy_logs + [combined_capture]).strip()


def parse_failure_reason(log_text: str) -> str:
    preferred_prefixes = (
        "Didn't find op for builtin opcode",
        "Failed to get registration from op code",
        "Failed to resize buffer.",
        "Failed to allocate tail memory.",
        "Failed to allocate temp memory.",
        "Failed to allocate memory for memory planning",
        "Too many buffers",
        "Arena allocation failed.",
        "Schema mismatch:",
    )
    for line in log_text.splitlines():
        clean = line.strip()
        if clean.startswith(preferred_prefixes):
            return clean

    for line in log_text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean == '"':
            continue
        if clean.startswith("touch_failed="):
            continue
        if clean.startswith("touch_failed_all_attempts="):
            continue
        if clean.startswith("serial_open_retry="):
            continue
        if clean.startswith("HOST_STAGE="):
            continue
        if clean.startswith("Family id ") or clean.startswith("Loading into Flash:"):
            continue
        if re.match(r"^\d{8}->\d{8}$", clean):
            continue
        if clean == "The device was rebooted to start the application.":
            continue
        if clean.startswith("=== Pico 2") or clean.startswith("Fixture model:") or clean.startswith("Fixture dataset:"):
            continue
        if clean.startswith("Model bytes:") or clean.startswith("Tensor arena size:") or clean.startswith("Input scale/zp:"):
            continue
        if clean.startswith("Output scale/zp:") or clean.startswith("Expected label:") or clean.startswith("Desktop INT8 parity"):
            continue
        if clean.startswith("Heap total bytes:"):
            continue
        if clean in {"AllocateTensors failed.", "Model setup failed.", "Invoke failed."}:
            continue
        return clean
    return "unknown failure"


def parse_serial_metrics(log_text: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "success": "=== SUMMARY ===" in log_text,
        "raw_log": log_text,
    }
    run_latencies: List[int] = []
    for line in log_text.splitlines():
        run_match = RUN_LATENCY_RE.search(line)
        if run_match:
            run_latencies.append(int(run_match.group(1)))
        float_match = SUMMARY_FLOAT_RE.match(line.strip())
        if float_match:
            key, value = float_match.groups()
            if "." in value:
                result[key] = float(value)
            else:
                result[key] = int(value)
            continue
        class_match = SUMMARY_CLASS_RE.match(line.strip())
        if class_match:
            key, idx, label = class_match.groups()
            result[key] = {"index": int(idx), "label": label}
    if run_latencies:
        result["latency_min_us"] = min(run_latencies)
        result["latency_max_us"] = max(run_latencies)
        result["latency_std_us"] = (
            sum((value - (sum(run_latencies) / len(run_latencies))) ** 2 for value in run_latencies)
            / len(run_latencies)
        ) ** 0.5
    if not result["success"]:
        result["failure_reason"] = parse_failure_reason(log_text)
    return result


def load_desktop_metrics(bundle: Bundle) -> Dict[str, object]:
    payload = json.loads((bundle.path / "parity_report.json").read_text(encoding="utf-8"))
    return {
        "desktop_int8_parity_pct": payload["parity"]["tflite_int8"]["parity_percent"],
        "desktop_int8_tensor_bytes": payload["ram_estimates"]["tflite_int8_tensor_bytes"],
        "model_bytes": payload["sizes"]["tflite_int8_bytes"],
        "parameter_count": payload["parameter_count"],
    }


def compute_status_row(bundle: Bundle, compile_metrics: Dict[str, int], serial_metrics: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model": bundle.model,
        "dataset": bundle.dataset,
        **load_desktop_metrics(bundle),
        **compile_metrics,
    }
    row.update(serial_metrics)
    if row.get("success"):
        desktop_parity = float(row["desktop_int8_parity_pct"])
        pico_parity = float(row.get("avg_parity_vs_pytorch_pct", 0.0))
        row["parity_drop_pct_points"] = desktop_parity - pico_parity
        row["top1_match_on_pico"] = (
            row.get("predicted_class", {}).get("index")
            == row.get("expected_class", {}).get("index")
        )
    return row


def write_reports(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pico2_deployment_status.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Pico 2 Deployment Sweep",
        "",
        "| Model | Dataset | Status | Failure/Notes | Flash B | Global RAM B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Top-1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        status = "RUNS" if row.get("success") else "FAILS"
        note = ""
        if row.get("success"):
            note = (
                f"arena={row.get('tensor_arena_alloc_bytes', '')}, "
                f"drop_pp={float(row.get('parity_drop_pct_points', 0.0)):.3f}"
            )
        else:
            note = str(row.get("failure_reason", "unknown"))
        avg_latency_ms = ""
        if row.get("avg_latency_us") is not None:
            avg_latency_ms = f"{float(row['avg_latency_us']) / 1000.0:.3f}"
        top1 = ""
        if row.get("predicted_class") and row.get("expected_class"):
            top1 = (
                f"{row['predicted_class']['label']} / {row['expected_class']['label']}"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["dataset"],
                    status,
                    note.replace("|", "/"),
                    str(row.get("firmware_flash_bytes", "")),
                    str(row.get("global_ram_bytes", "")),
                    str(row.get("tensor_arena_used_bytes", "")),
                    str(row.get("heap_used_bytes", "")),
                    avg_latency_ms,
                    f"{float(row['avg_parity_vs_pytorch_pct']):.3f}" if row.get("avg_parity_vs_pytorch_pct") is not None else "",
                    f"{float(row['desktop_int8_parity_pct']):.3f}",
                    top1,
                ]
            )
            + " |"
        )
    (output_dir / "pico2_deployment_status.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def update_paper_results(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    paper_results = MICROBI_DIR / "paper_results.md"
    successful = [row for row in rows if row.get("success")]
    failed = [row for row in rows if not row.get("success")]

    lines = [
        "# Pico 2 Deployment Results",
        "",
        "## Sweep Summary",
        "",
        f"- Bundles tested: `{len(rows)}`",
        f"- Successful on-device runs: `{len(successful)}`",
        f"- Failed deployments: `{len(failed)}`",
        f"- Sweep report: [pico2_deployment_status.md]({output_dir / 'pico2_deployment_status.md'})",
        "",
        "## Successful Runs",
        "",
        "| Model | Dataset | Flash B | Arena Used B | Heap Used B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Degradation pp | Top-1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in successful:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["dataset"],
                    str(row.get("firmware_flash_bytes", "")),
                    str(row.get("tensor_arena_used_bytes", "")),
                    str(row.get("heap_used_bytes", "")),
                    f"{float(row['avg_latency_us']) / 1000.0:.3f}",
                    f"{float(row['avg_parity_vs_pytorch_pct']):.3f}",
                    f"{float(row['desktop_int8_parity_pct']):.3f}",
                    f"{float(row['parity_drop_pct_points']):.3f}",
                    f"{row['predicted_class']['label']} / {row['expected_class']['label']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failed Runs",
            "",
            "| Model | Dataset | Failure Reason | Desktop Tensor B | Model B |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in failed:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["dataset"],
                    str(row.get("failure_reason", "unknown")).replace("|", "/"),
                    str(row.get("desktop_int8_tensor_bytes", "")),
                    str(row.get("model_bytes", "")),
                ]
            )
            + " |"
        )
    paper_results.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    models = parse_csv_or_all(args.models, MODEL_CHOICES)
    datasets = parse_csv_or_all(args.datasets, DEFAULT_DATASETS)
    bundles = discover_bundles(args.export_root, models, datasets)
    if not bundles:
        raise SystemExit("No matching export bundles were found.")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = MICROBI_DIR / "picoDeployments" / f"pico2Sweep_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for bundle in bundles:
            print(f"{bundle.model}/{bundle.dataset} -> {bundle.path}")
        return

    rows: List[Dict[str, object]] = []
    clean_compile = True
    for bundle in bundles:
        print(f"=== Deploying {bundle.model}/{bundle.dataset} ===")
        bundle_log_dir = output_dir / bundle.model / bundle.dataset
        bundle_log_dir.mkdir(parents=True, exist_ok=True)
        row: Dict[str, object]
        try:
            prepare_fixture(bundle, args.project_dir, args.sample_index)
            rewrite_model_files(bundle, args.project_dir)
            compile_metrics, compile_output = compile_project(
                args.project_dir, args.arena_bytes, clean=clean_compile
            )
            clean_compile = False
            (bundle_log_dir / "compile.log").write_text(
                compile_output, encoding="utf-8"
            )
            serial_log = flash_and_capture(
                args.project_dir, args.port, args.timeout_seconds
            )
            (bundle_log_dir / "serial.log").write_text(serial_log + "\n", encoding="utf-8")
            serial_metrics = parse_serial_metrics(serial_log)
            row = compute_status_row(bundle, compile_metrics, serial_metrics)
        except subprocess.CalledProcessError as exc:
            error_text = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
            (bundle_log_dir / "error.log").write_text(error_text + "\n", encoding="utf-8")
            row = {
                "model": bundle.model,
                "dataset": bundle.dataset,
                **load_desktop_metrics(bundle),
                "success": False,
                "failure_reason": f"host_command_failed: {error_text.splitlines()[-1] if error_text else exc}",
            }
        except Exception as exc:
            (bundle_log_dir / "error.log").write_text(str(exc) + "\n", encoding="utf-8")
            row = {
                "model": bundle.model,
                "dataset": bundle.dataset,
                **load_desktop_metrics(bundle),
                "success": False,
                "failure_reason": str(exc),
            }
        rows.append(row)
        (bundle_log_dir / "result.json").write_text(
            json.dumps({k: v for k, v in row.items() if k != "raw_log"}, indent=2),
            encoding="utf-8",
        )
        write_reports(output_dir, rows)
        update_paper_results(output_dir, rows)
        print(json.dumps({k: v for k, v in row.items() if k != "raw_log"}, indent=2))

    write_reports(output_dir, rows)
    update_paper_results(output_dir, rows)
    print(f"Wrote sweep outputs to {output_dir}")


if __name__ == "__main__":
    main()
