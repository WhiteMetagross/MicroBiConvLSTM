from __future__ import annotations

import argparse
import json
from pathlib import Path

from runEsp32NativeDeploymentSweep import (
    Bundle,
    PROJECT_DIR,
    EXPORT_ROOT,
    compile_project,
    compute_status_row,
    emit_model_array,
    emit_resolver_config,
    flash_project,
    load_desktop_metrics,
    parse_serial_metrics,
    prepare_fixture,
    capture_serial,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ESP32 native deployment bundle.")
    parser.add_argument("--model", required=True, choices=["microbi", "deepconvlstm", "tinyhar", "tinierhar"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arena-kb", required=True, type=int)
    parser.add_argument("--port", default="COM9")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--static-arena", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = Bundle(
        model=args.model,
        dataset=args.dataset,
        path=EXPORT_ROOT / args.model / args.dataset,
    )
    if not bundle.path.exists():
        raise FileNotFoundError(f"Missing export bundle: {bundle.path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_fixture(bundle, PROJECT_DIR, args.sample_index)
    emit_model_array(bundle, PROJECT_DIR)
    emit_resolver_config(bundle, PROJECT_DIR)

    desktop_metrics = load_desktop_metrics(bundle)
    arena_bytes = args.arena_kb * 1024

    compile_metrics, compile_log = compile_project(
        PROJECT_DIR,
        arena_bytes,
        clean=args.clean,
        use_static_arena=args.static_arena,
    )
    (output_dir / "compile.log").write_text(compile_log + "\n", encoding="utf-8")

    flash_log = flash_project(PROJECT_DIR, args.port)
    (output_dir / "flash.log").write_text(flash_log + "\n", encoding="utf-8")

    serial_log = capture_serial(args.port, args.timeout_seconds)
    (output_dir / "serial.log").write_text(serial_log + "\n", encoding="utf-8")

    serial_metrics = parse_serial_metrics(serial_log)
    row = compute_status_row(bundle, desktop_metrics, compile_metrics, serial_metrics, arena_bytes)
    (output_dir / "result.json").write_text(
        json.dumps({k: v for k, v in row.items() if k != "raw_log"}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in row.items() if k != "raw_log"}, indent=2))


if __name__ == "__main__":
    main()
