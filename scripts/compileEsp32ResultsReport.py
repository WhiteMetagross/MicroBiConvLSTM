from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from runEsp32NativeDeploymentSweep import BUILD_SIZE_RE, parse_serial_metrics


THIS_FILE = Path(__file__).resolve()
MICROBI_DIR = THIS_FILE.parents[1]
DEPLOY_ROOT = MICROBI_DIR / "esp32Deployments"
REPORT_JSON = MICROBI_DIR / "esp32PaperResults.json"
REPORT_MD = MICROBI_DIR / "ESP32paperResults.md"

SOURCE_FILES: List[Path] = []

MODEL_ORDER = ["microbi", "tinyhar", "tinierhar", "deepconvlstm"]
DATASET_ORDER = [
    "ucihar",
    "motionsense",
    "wisdm",
    "pamap2",
    "opportunity",
    "unimib",
    "skoda",
    "daphnet",
]
MODEL_LABELS = {
    "microbi": "MicroBi-ConvLSTM",
    "tinyhar": "TinyHAR",
    "tinierhar": "TinierHAR",
    "deepconvlstm": "DeepConvLSTM",
}


def load_rows() -> Dict[Tuple[str, str], Dict[str, object]]:
    rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    for source in SOURCE_FILES:
        if not source.exists():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                rows[(row["model"], row["dataset"])] = row
    source_dirs = [path for path in DEPLOY_ROOT.iterdir() if path.is_dir()] if DEPLOY_ROOT.exists() else []
    for directory in source_dirs:
        if not directory.exists():
            continue
        for result_path in sorted(directory.rglob("result.json")):
            row = json.loads(result_path.read_text(encoding="utf-8"))
            serial_path = result_path.with_name("serial.log")
            compile_path = result_path.with_name("compile.log")
            if serial_path.exists():
                parsed = parse_serial_metrics(serial_path.read_text(encoding="utf-8", errors="ignore"))
                if not row.get("success") and parsed.get("failure_reason"):
                    row["failure_reason"] = parsed["failure_reason"]
            if compile_path.exists() and not row.get("firmware_flash_bytes"):
                match = BUILD_SIZE_RE.search(compile_path.read_text(encoding="utf-8", errors="ignore"))
                if match:
                    row["firmware_flash_bytes"] = int(match.group(1), 16)
                    row["app_partition_bytes"] = int(match.group(2), 16)
            row["artifact_dir"] = str(result_path.parent)
            rows[(row["model"], row["dataset"])] = row
    return rows


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def fmt_int(value: object) -> str:
    if value is None:
        return ""
    return str(int(value))


def summarize(rows: Dict[Tuple[str, str], Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for model in MODEL_ORDER:
        model_rows = [rows[(model, dataset)] for dataset in DATASET_ORDER if (model, dataset) in rows]
        success_rows = [row for row in model_rows if row.get("success")]
        summary[model] = {
            "total": len(model_rows),
            "success": len(success_rows),
            "avg_latency_ms": (
                sum(float(row["avg_latency_us"]) for row in success_rows) / len(success_rows) / 1000.0
                if success_rows
                else None
            ),
            "avg_parity_pct": (
                sum(float(row["avg_parity_vs_pytorch_pct"]) for row in success_rows) / len(success_rows)
                if success_rows
                else None
            ),
            "max_arena_used_bytes": (
                max(int(row.get("tensor_arena_used_bytes", 0)) for row in success_rows)
                if success_rows
                else None
            ),
        }
    return summary


def write_json(rows: Dict[Tuple[str, str], Dict[str, object]], summary: Dict[str, Dict[str, object]]) -> None:
    payload = {
        "summary": summary,
        "results": [
            rows[(model, dataset)]
            for model in MODEL_ORDER
            for dataset in DATASET_ORDER
            if (model, dataset) in rows
        ],
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(rows: Dict[Tuple[str, str], Dict[str, object]], summary: Dict[str, Dict[str, object]]) -> None:
    lines: List[str] = [
        "# ESP32 Deployment Results",
        "",
        "Native ESP-IDF results on the connected classic `ESP32-D0WD-V3` board (`4 MB` flash, no PSRAM).",
        "",
        "## Comparison",
        "",
        "- `MicroBi-ConvLSTM` is the only family that completed a full `8/8` dataset sweep on this board.",
        "- `TinyHAR` runs on `5/8` datasets, but parity is highly variable and the heaviest datasets hit on-device memory limits.",
        "- `TinierHAR` now runs on `6/8` datasets after enabling single-core IRAM-as-8-bit heap support and a `176 KB` native arena; only `pamap2` and `opportunity` still exceed the classic ESP32 arena ceiling.",
        "- `DeepConvLSTM` remains blocked by the classic ESP32 contiguous internal SRAM ceiling; every dataset fails tensor-arena allocation before inference begins.",
        "",
        "## Family Summary",
        "",
        "| Model | Success / Total | Avg Latency ms | Avg PyTorch Parity % | Max Arena Used B |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_ORDER:
        item = summary[model]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[model],
                    f"{item['success']} / {item['total']}",
                    fmt_float(item["avg_latency_ms"]),
                    fmt_float(item["avg_parity_pct"]),
                    fmt_int(item["max_arena_used_bytes"]),
                ]
            )
            + " |"
        )

    for model in MODEL_ORDER:
        lines.extend(
            [
                "",
                f"## {MODEL_LABELS[model]}",
                "",
                "| Dataset | Status | Arena KB | Flash B | Avg Latency ms | PyTorch Parity % | Desktop INT8 Parity % | Heap Used B | Arena Used B | Top-1 / Failure |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for dataset in DATASET_ORDER:
            row = rows.get((model, dataset))
            if not row:
                continue
            if row.get("success"):
                tail = (
                    f"{row['predicted_class']['label']} / {row['expected_class']['label']}"
                    if row.get("predicted_class") and row.get("expected_class")
                    else ""
                )
                status = "RUNS"
            else:
                tail = str(row.get("failure_reason", ""))
                status = "FAILS"
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        status,
                        fmt_int(int(row.get("requested_arena_bytes", 0)) // 1024),
                        fmt_int(row.get("firmware_flash_bytes")),
                        fmt_float((float(row["avg_latency_us"]) / 1000.0) if row.get("avg_latency_us") is not None else None),
                        fmt_float(row.get("avg_parity_vs_pytorch_pct")),
                        fmt_float(row.get("desktop_int8_parity_pct")),
                        fmt_int(row.get("heap_used_internal_bytes")),
                        fmt_int(row.get("tensor_arena_used_bytes")),
                        tail.replace("|", "/"),
                    ]
                )
                + " |"
            )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_json(rows, summary)
    write_markdown(rows, summary)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
