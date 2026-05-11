from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Pico2Models" / "Results"
INPUT_JSON = RESULTS_DIR / "pico2Fp32Int8Results.json"
OUTPUT_MD = RESULTS_DIR / "pico2Fp32Int8Results.md"
MODELS = ("microbi", "tinyhar", "tinierhar", "deepconvlstm")


def load_payload() -> Dict[str, Any]:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def md_cell(value: Optional[Any]) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("\n", " ").replace("|", "/")


def latency_ms(record: Dict[str, Any]) -> str:
    avg_latency_us = record.get("avg_latency_us")
    if avg_latency_us is None:
        return "-"
    return md_cell(round(float(avg_latency_us) / 1000.0, 3))


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Pico 2 FP32 and INT8 Deployment Matrix")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at_utc']}")
    lines.append("")
    lines.append("## Coverage:")
    lines.append("")
    lines.append("| Model | INT8 Runs | INT8 Fails | INT8 Missing | FP32 Runs | FP32 Fails | FP32 Missing |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        int8 = payload["summary"]["int8"][model]
        fp32 = payload["summary"]["fp32"][model]
        lines.append(
            f"| {model} | {int8['runs']} | {int8['fails']} | {int8['missing']} | "
            f"{fp32['runs']} | {fp32['fails']} | {fp32['missing']} |"
        )

    for model in MODELS:
        lines.append("")
        lines.append(f"## {model}:")
        lines.append("")
        lines.append(
            "| Dataset | Variant | Success | Latency (ms) | PyTorch Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
        datasets = payload["models"][model]
        for dataset in payload["datasets"]:
            for variant in ("int8", "fp32"):
                record = datasets[variant][dataset]
                source_name = Path(record.get("source", "")).name
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            md_cell(dataset),
                            md_cell(variant),
                            md_cell(record.get("success")),
                            latency_ms(record),
                            md_cell(record.get("avg_parity_vs_pytorch_pct")),
                            md_cell(record.get("tensor_arena_used_bytes")),
                            md_cell(record.get("firmware_flash_bytes")),
                            md_cell(record.get("top1_match_on_pico")),
                            md_cell(record.get("failure_reason")),
                            md_cell(source_name),
                        ]
                    )
                    + " |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = load_payload()
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
