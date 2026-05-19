from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from icecbr.config import ExperimentConfig
from icecbr.data import load_bundle
from scripts.generate_case_study import _summarize_method_adaptation
from icecbr.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Build a multi-case support table for the manuscript.")
    parser.add_argument("--baseline-tag", type=str, required=True)
    parser.add_argument("--method-tag", type=str, required=True)
    parser.add_argument("--baseline-model", type=str, default="a0")
    parser.add_argument("--method-model", type=str, default="a0_mrlinear")
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def _rank_candidates(config: ExperimentConfig, baseline_npz: Path, method_npz: Path, top_n: int):
    base = np.load(baseline_npz)
    method = np.load(method_npz)
    threshold = config.target_threshold
    area = config.cell_area_km2 / 1e6
    bp = base["preds"].astype(np.float32)
    mp = method["preds"].astype(np.float32)
    truth = base["truths"].astype(np.float32)

    base_rmse = np.sqrt(np.mean((bp - truth) ** 2, axis=(1, 2)))
    method_rmse = np.sqrt(np.mean((mp - truth) ** 2, axis=(1, 2)))
    base_iiee = (
        np.logical_and(bp >= threshold, truth < threshold).sum(axis=(1, 2))
        + np.logical_and(bp < threshold, truth >= threshold).sum(axis=(1, 2))
    ) * area
    method_iiee = (
        np.logical_and(mp >= threshold, truth < threshold).sum(axis=(1, 2))
        + np.logical_and(mp < threshold, truth >= threshold).sum(axis=(1, 2))
    ) * area

    rows = []
    for sample_idx in range(bp.shape[0]):
        for lead_pos, lead in enumerate(base["lead_months"].tolist()):
            rows.append(
                {
                    "issue_ym": str(base["issue_ym"][sample_idx, lead_pos]),
                    "target_ym": str(base["target_ym"][sample_idx]),
                    "lead": int(lead),
                    "sample_index": int(sample_idx),
                    "lead_position": int(lead_pos),
                    "delta_iiee": float(base_iiee[sample_idx, lead_pos] - method_iiee[sample_idx, lead_pos]),
                    "delta_rmse": float(base_rmse[sample_idx, lead_pos] - method_rmse[sample_idx, lead_pos]),
                }
            )
    rows.sort(key=lambda row: (row["delta_iiee"], row["delta_rmse"]), reverse=True)
    return rows[:top_n]


def main():
    args = parse_args()
    config = ExperimentConfig()
    bundle = load_bundle(config)

    baseline_npz = (
        config.results_dir / args.baseline_tag / "target_month_export" / args.baseline_model / "predictions.npz"
    )
    method_npz = (
        config.results_dir / args.method_tag / "target_month_export" / args.method_model / "predictions.npz"
    )

    candidates = _rank_candidates(config, baseline_npz, method_npz, args.top_n)
    table_rows = []
    cache: dict[int, object] = {}
    for row in candidates:
        issue_index = next(i for i, ts in enumerate(bundle.time_index) if ts.strftime("%Y-%m") == row["issue_ym"])
        if issue_index not in cache:
            model = build_model(args.method_model)
            model.fit(np.arange(issue_index), bundle, config)
            cache[issue_index] = model
        model = cache[issue_index]
        adaptation_summary = _summarize_method_adaptation(
            method_model=model,
            issue_index=issue_index,
            lead=row["lead"],
            ocean_mask=bundle.land_mask,
        )
        dominant = max(adaptation_summary, key=lambda item: abs(item["extent_delta"]))
        table_rows.append(
            {
                "query": f"{row['issue_ym']}→{row['target_ym']}",
                "lead": row["lead"],
                "delta_iiee": row["delta_iiee"],
                "delta_rmse": row["delta_rmse"],
                "dominant_precedent": f"{dominant['case_issue_ym']}→{dominant['case_target_ym']}",
                "adaptation_delta_extent": dominant["extent_delta"],
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(table_rows, indent=2))

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query",
                    "lead",
                    "delta_iiee",
                    "delta_rmse",
                    "dominant_precedent",
                    "adaptation_delta_extent",
                ],
            )
            writer.writeheader()
            writer.writerows(table_rows)

    print(json.dumps(table_rows, indent=2))
    print(output_json)


if __name__ == "__main__":
    main()
