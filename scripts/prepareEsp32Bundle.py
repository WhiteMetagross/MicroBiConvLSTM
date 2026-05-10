from __future__ import annotations

import argparse
from pathlib import Path

from runEsp32NativeDeploymentSweep import (
    Bundle,
    EXPORT_ROOT,
    PROJECT_DIR,
    emit_model_array,
    emit_resolver_config,
    prepare_fixture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ESP32 fixture/model/resolver for one bundle.")
    parser.add_argument("--model", required=True, choices=["microbi", "deepconvlstm", "tinyhar", "tinierhar"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
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
    prepare_fixture(bundle, PROJECT_DIR, args.sample_index)
    emit_model_array(bundle, PROJECT_DIR)
    emit_resolver_config(bundle, PROJECT_DIR)
    print(f"Prepared {args.model}/{args.dataset}")


if __name__ == "__main__":
    main()
