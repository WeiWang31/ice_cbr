from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from icecbr.run import build_config
from icecbr.pipelines import run_target_month_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper's main target-month export pipeline.")
    parser.add_argument("--output-tag", required=True, help="Output tag created under outputs/ or ICECBR_OUTPUT_DIR.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model ids to export.")
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--test-end-year", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sic-eof-dim", type=int, default=None)
    parser.add_argument("--retrieval-rho", type=float, default=None)
    parser.add_argument("--target-eval-start-year", type=int, default=None)
    parser.add_argument("--target-eval-end-year", type=int, default=None)
    parser.add_argument("--enable-revise", action="store_true")
    parser.add_argument("--revise-variance-quantile", type=float, default=None)
    parser.add_argument("--revise-downweight-strength", type=float, default=None)
    parser.add_argument("--enable-retain", action="store_true")
    parser.add_argument("--disable-retain-online-growth", action="store_true")
    parser.add_argument("--disable-retain-persist", action="store_true")
    parser.add_argument("--retain-reload", action="store_true")
    parser.add_argument("--enable-selective-retain", action="store_true")
    parser.add_argument("--enable-bounded-retain-memory", action="store_true")
    parser.add_argument("--enable-retained-case-boost", action="store_true")
    parser.add_argument("--enable-representative-retained-memory", action="store_true")
    parser.add_argument("--retain-admission-mode", choices=["none", "permissive", "strict"], default=None)
    parser.add_argument("--retained-case-cap", type=int, default=None)
    parser.add_argument("--retained-case-cap-per-lead", type=int, default=None)
    parser.add_argument("--retained-case-cap-per-month", type=int, default=None)
    parser.add_argument("--retain-representative-cap-per-lead", type=int, default=None)
    parser.add_argument("--retained-case-boost-strength", type=float, default=None)
    parser.add_argument("--retained-case-boost-max", type=float, default=None)
    parser.add_argument("--run-mode", choices=["fresh", "resume"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    result = run_target_month_export(config)
    if hasattr(result, "to_json"):
        print(result.to_json(orient="records"))
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
