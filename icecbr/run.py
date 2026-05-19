from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .pipelines import (
    run_agr_region_merge,
    run_e1,
    run_e2,
    run_e3,
    run_e4,
    run_e5,
    run_e6,
    run_e7,
    run_finalist_compare,
    run_forecast_compare,
    run_pilot_compare,
    run_pilot_full,
    run_smoke_test,
    run_target_month_compare,
    run_target_month_export,
    run_v1_anchor_ablation,
)
from .seas5 import (
    download_seas5,
    download_seas5_hindcast,
    fit_seas5_bias_correction,
    process_seas5,
    process_seas5_hindcast,
    run_seas5_baseline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run IceCBR paper-code experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",
            "e6",
            "e7",
            "pilot_full",
            "pilot_compare",
            "forecast_compare",
            "v1_anchor_ablation",
            "agr_region_merge",
            "finalist_compare",
            "target_month_compare",
            "target_month_export",
            "download_seas5",
            "download_seas5_hindcast",
            "process_seas5",
            "process_seas5_hindcast",
            "fit_seas5_bias_correction",
            "seas5_baseline",
            "smoke",
        ],
    )
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--test-end-year", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sic-eof-dim", type=int, default=None)
    parser.add_argument("--retrieval-rho", type=float, default=None)
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
    parser.add_argument("--target-eval-start-year", type=int, default=None)
    parser.add_argument("--target-eval-end-year", type=int, default=None)
    parser.add_argument("--run-mode", choices=["fresh", "resume"], default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    config = ExperimentConfig()
    if args.test_start_year is not None:
        config.test_start_year = args.test_start_year
    if args.test_end_year is not None:
        config.test_end_year = args.test_end_year
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.sic_eof_dim is not None:
        config.sic_eof_dim = args.sic_eof_dim
    if args.retrieval_rho is not None:
        config.retrieval_rho = args.retrieval_rho
    if args.enable_revise:
        config.enable_revise = True
    if args.revise_variance_quantile is not None:
        config.revise_variance_quantile = args.revise_variance_quantile
    if args.revise_downweight_strength is not None:
        config.revise_downweight_strength = args.revise_downweight_strength
    if args.enable_retain:
        config.enable_retain = True
    if args.disable_retain_online_growth:
        config.retain_online_growth = False
    if args.disable_retain_persist:
        config.retain_persist_jsonl = False
    if args.retain_reload:
        config.retain_reload = True
    if args.enable_selective_retain:
        config.enable_selective_retain = True
    if args.enable_bounded_retain_memory:
        config.enable_bounded_retain_memory = True
    if args.enable_retained_case_boost:
        config.enable_retained_case_boost = True
    if args.enable_representative_retained_memory:
        config.enable_representative_retained_memory = True
    if args.retain_admission_mode is not None:
        config.retain_admission_mode = args.retain_admission_mode
    if args.retained_case_cap is not None:
        config.retained_case_cap = args.retained_case_cap
    if args.retained_case_cap_per_lead is not None:
        config.retained_case_cap_per_lead = args.retained_case_cap_per_lead
    if args.retained_case_cap_per_month is not None:
        config.retained_case_cap_per_month = args.retained_case_cap_per_month
    if args.retain_representative_cap_per_lead is not None:
        config.retain_representative_cap_per_lead = args.retain_representative_cap_per_lead
    if args.retained_case_boost_strength is not None:
        config.retained_case_boost_strength = args.retained_case_boost_strength
    if args.retained_case_boost_max is not None:
        config.retained_case_boost_max = args.retained_case_boost_max
    if args.target_eval_start_year is not None:
        config.target_eval_start_year = args.target_eval_start_year
    if args.target_eval_end_year is not None:
        config.target_eval_end_year = args.target_eval_end_year
    if args.run_mode is not None:
        config.run_mode = args.run_mode
    if args.output_tag is not None:
        config.output_tag = args.output_tag
    if args.models is not None:
        config.selected_models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    return config


def main():
    args = parse_args()
    config = build_config(args)
    runners = {
        "e1": run_e1,
        "e2": run_e2,
        "e3": run_e3,
        "e4": run_e4,
        "e5": run_e5,
        "e6": run_e6,
        "e7": run_e7,
        "pilot_full": run_pilot_full,
        "pilot_compare": run_pilot_compare,
        "forecast_compare": run_forecast_compare,
        "v1_anchor_ablation": run_v1_anchor_ablation,
        "agr_region_merge": run_agr_region_merge,
        "finalist_compare": run_finalist_compare,
        "target_month_compare": run_target_month_compare,
        "target_month_export": run_target_month_export,
        "download_seas5": download_seas5,
        "download_seas5_hindcast": download_seas5_hindcast,
        "process_seas5": process_seas5,
        "process_seas5_hindcast": process_seas5_hindcast,
        "fit_seas5_bias_correction": fit_seas5_bias_correction,
        "seas5_baseline": run_seas5_baseline,
        "smoke": run_smoke_test,
    }
    result = runners[args.experiment](config)
    if hasattr(result, "to_json"):
        print(result.to_json(orient="records"))
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
