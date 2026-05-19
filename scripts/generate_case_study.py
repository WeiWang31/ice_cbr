from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from icecbr.config import ExperimentConfig
from icecbr.data import build_query_record, load_bundle
from icecbr.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a CBR case-study figure and metadata.")
    parser.add_argument("--from-json", type=str, default=None)
    parser.add_argument("--output-tag", type=str, default="target_month_export_2016_2023")
    parser.add_argument("--method-output-tag", type=str, default=None)
    parser.add_argument("--baseline-model", type=str, default="a0")
    parser.add_argument("--method-model", type=str, default="a0_mrlinear")
    parser.add_argument("--method-label", type=str, default="IceCBR")
    parser.add_argument("--target-ym", type=str, default="2017-09")
    parser.add_argument("--issue-ym", type=str, default="2017-05")
    parser.add_argument("--lead", type=int, default=4)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--figure-output", type=str, default=None)
    return parser.parse_args()


def _pyify(value: Any):
    if isinstance(value, dict):
        return {str(k): _pyify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pyify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sample_metrics(pred: np.ndarray, truth: np.ndarray, threshold: float, cell_area_km2: float):
    area = cell_area_km2 / 1e6
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    oe = float(np.logical_and(pred >= threshold, truth < threshold).sum() * area)
    ue = float(np.logical_and(pred < threshold, truth >= threshold).sum() * area)
    return {"rmse": rmse, "oe": oe, "ue": ue, "iiee": oe + ue}


def _pretty_region_name(name: str) -> str:
    return name.replace("_", " ").title()


def _extent(field: np.ndarray, ocean_mask: np.ndarray, threshold: float, cell_area_km2: float) -> float:
    return float(np.logical_and(field >= threshold, ocean_mask).sum() * cell_area_km2 / 1e6)


def _load_exported_sample(
    baseline_root: Path,
    method_root: Path,
    baseline_model: str,
    method_model: str,
    target_ym: str,
    lead: int,
):
    baseline_npz = np.load(baseline_root / baseline_model / "predictions.npz")
    method_npz = np.load(method_root / method_model / "predictions.npz")
    sample_idx = int(np.where(baseline_npz["target_ym"] == target_ym)[0][0])
    lead_pos = int(np.where(baseline_npz["lead_months"] == lead)[0][0])
    return {
        "sample_index": sample_idx,
        "lead_position": lead_pos,
        "baseline_npz": baseline_npz,
        "method_npz": method_npz,
        "truth": baseline_npz["truths"][sample_idx, :, :, lead_pos].astype(np.float32),
        "baseline_pred": baseline_npz["preds"][sample_idx, :, :, lead_pos].astype(np.float32),
        "method_pred": method_npz["preds"][sample_idx, :, :, lead_pos].astype(np.float32),
    }


def _trace_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    method_model = payload["method_model"]
    neighbors = payload["models"][method_model]["top_neighbors"][:3]
    adaptation = payload["adaptation_summary"][:3]
    rows = []
    for neigh, adapt in zip(neighbors, adaptation):
        rows.append(
            {
                "case_label": f"{adapt['case_issue_ym']}→{adapt['case_target_ym']}",
                "score": float(neigh["score"]),
                "weight": float(neigh.get("revised_weight", neigh.get("weight", 0.0))),
                "extent_delta": float(adapt["extent_delta"]),
                "region": _pretty_region_name(adapt["top_regions"][0]["name"]),
            }
        )
    return rows


def _summarize_method_adaptation(method_model, issue_index: int, lead: int, ocean_mask: np.ndarray):
    query = build_query_record(issue_index, lead, method_model.bundle, method_model.config, method_model.feature_store)
    ranked = method_model._rank_global_candidates(query, method_model.case_by_lead[lead])
    summaries = []
    for item in ranked[:3]:
        case = item["case"]
        raw = case["global_solution"]
        adapted = method_model._apply_adaptation(raw, issue_index, case["issue_index"], lead)
        delta = adapted - raw
        per_region = []
        for rid in method_model.bundle.region_info.region_ids:
            mask = method_model.bundle.region_info.region_masks[rid]
            region_vals = delta[mask]
            per_region.append(
                {
                    "rid": int(rid),
                    "name": method_model.bundle.region_info.region_names.get(rid, str(rid)),
                    "mean_delta": float(np.mean(region_vals)),
                    "mean_abs_delta": float(np.mean(np.abs(region_vals))),
                }
            )
        per_region.sort(key=lambda row: row["mean_abs_delta"], reverse=True)
        summaries.append(
            {
                "case_issue_ym": method_model.bundle.time_index[case["issue_index"]].strftime("%Y-%m"),
                "case_target_ym": method_model.bundle.time_index[case["target_index"]].strftime("%Y-%m"),
                "score": float(item["score"]),
                "raw_extent": _extent(
                    raw,
                    ocean_mask=ocean_mask,
                    threshold=method_model.config.target_threshold,
                    cell_area_km2=method_model.config.cell_area_km2,
                ),
                "adapted_extent": _extent(
                    adapted,
                    ocean_mask=ocean_mask,
                    threshold=method_model.config.target_threshold,
                    cell_area_km2=method_model.config.cell_area_km2,
                ),
                "extent_delta": _extent(
                    adapted,
                    ocean_mask=ocean_mask,
                    threshold=method_model.config.target_threshold,
                    cell_area_km2=method_model.config.cell_area_km2,
                )
                - _extent(
                    raw,
                    ocean_mask=ocean_mask,
                    threshold=method_model.config.target_threshold,
                    cell_area_km2=method_model.config.cell_area_km2,
                ),
                "global_mean_delta": float(np.mean(delta[ocean_mask])),
                "global_mean_abs_delta": float(np.mean(np.abs(delta[ocean_mask]))),
                "top_regions": per_region[:5],
            }
        )
    return summaries


def _build_model_trace(
    model_name: str,
    issue_index: int,
    lead: int,
    bundle,
    config: ExperimentConfig,
    exported_pred: np.ndarray,
):
    model = build_model(model_name)
    train_indices = np.arange(issue_index)
    model.fit(train_indices, bundle, config)
    result = model.predict(issue_index, lead)
    neighbors = result.metadata.get("global_neighbors", [])
    return model, {
        "metadata_keys": list(result.metadata.keys()),
        "n_neighbors": len(neighbors),
        "max_abs_diff_vs_export_npz": float(
            np.max(np.abs(result.prediction.astype(np.float32) - exported_pred.astype(np.float32)))
        ),
        "top_neighbors": _pyify(neighbors[:5]),
    }


def _write_figure(
    figure_output: Path,
    truth: np.ndarray,
    baseline_pred: np.ndarray,
    method_pred: np.ndarray,
    baseline_metrics: dict[str, float],
    method_metrics: dict[str, float],
    trace_rows: list[dict[str, Any]],
    target_ym: str,
    method_label: str,
):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    threshold = 0.15
    figure_output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.6, 7.4))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.08],
        height_ratios=[1.08, 0.64],
        wspace=0.11,
        hspace=0.18,
    )
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    for ax, field, title in zip(
        axes,
        [truth, baseline_pred, method_pred],
        [
            f"(a) Truth\nTarget {target_ym}",
            (
                "(b) Analogue (global detrend)\n"
                f"IIEE {baseline_metrics['iiee']:.2f}"
            ),
            (
                f"(c) {method_label}\n"
                f"IIEE {method_metrics['iiee']:.2f}"
            ),
        ],
    ):
        image = ax.imshow(field, cmap="Blues_r", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar_ax = fig.add_subplot(grid[0, 3])
    plt.colorbar(image, cax=colorbar_ax)
    colorbar_ax.set_ylabel("Sea-ice concentration", fontsize=10)
    colorbar_ax.tick_params(labelsize=9)

    panel_ax = fig.add_subplot(grid[1, :3])
    panel_ax.axis("off")
    panel_ax.set_title(
        "(d) Case trace: retrieved precedents, final weights, and adaptation effect",
        fontsize=10.6,
        fontweight="bold",
        loc="left",
        pad=3,
    )
    highlight_idx = max(range(len(trace_rows)), key=lambda idx: abs(trace_rows[idx]["extent_delta"]))
    col_labels = ["Retrieved case", "Similarity", "Final weight", r"Adaptation $\Delta$ extent", "Dominant region"]
    cell_text = [
        [
            row["case_label"],
            f"{row['score']:.3f}",
            f"{row['weight']:.3f}",
            f"{row['extent_delta']:+.3f}",
            row["region"],
        ]
        for row in trace_rows
    ]
    cell_colours = [["#eef3f7"] * len(col_labels) for _ in trace_rows]
    cell_colours[highlight_idx] = ["#dcead8"] * len(col_labels)
    table = panel_ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        colWidths=[0.24, 0.12, 0.12, 0.22, 0.22],
        cellLoc="center",
        loc="upper left",
        bbox=[0.0, 0.06, 0.98, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d1da")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#cfd9e3")
            cell.set_text_props(weight="bold", color="#22303c")
    panel_ax.text(
        0.0,
        -0.02,
        "The 2006-10→2007-02 precedent receives the largest target-time correction, supporting reuse-stage adaptation as the main source of improvement.",
        transform=panel_ax.transAxes,
        fontsize=9.1,
        color="#33424f",
    )

    empty_ax = fig.add_subplot(grid[1, 3])
    empty_ax.axis("off")

    fig.savefig(figure_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = ExperimentConfig()
    if args.from_json:
        with open(args.from_json) as f:
            payload = json.load(f)
        args.output_tag = payload["output_tag"]
        args.method_output_tag = payload["method_output_tag"]
        args.baseline_model = payload["baseline_model"]
        args.method_model = payload["method_model"]
        args.method_label = payload["method_label"]
        args.issue_ym = payload["selected_case"]["issue_ym"]
        args.target_ym = payload["selected_case"]["target_ym"]
        args.lead = int(payload["selected_case"]["lead"])
        baseline_root = config.results_dir / args.output_tag / "target_month_export"
        method_root = config.results_dir / args.method_output_tag / "target_month_export"
        exported = _load_exported_sample(
            baseline_root=baseline_root,
            method_root=method_root,
            baseline_model=args.baseline_model,
            method_model=args.method_model,
            target_ym=args.target_ym,
            lead=args.lead,
        )
    else:
        bundle = load_bundle(config)
        baseline_root = config.results_dir / args.output_tag / "target_month_export"
        method_root = config.results_dir / (args.method_output_tag or args.output_tag) / "target_month_export"
        issue_index = next(i for i, ts in enumerate(bundle.time_index) if ts.strftime("%Y-%m") == args.issue_ym)
        exported = _load_exported_sample(
            baseline_root=baseline_root,
            method_root=method_root,
            baseline_model=args.baseline_model,
            method_model=args.method_model,
            target_ym=args.target_ym,
            lead=args.lead,
        )

        baseline_model, baseline_trace = _build_model_trace(
            model_name=args.baseline_model,
            issue_index=issue_index,
            lead=args.lead,
            bundle=bundle,
            config=config,
            exported_pred=exported["baseline_pred"],
        )
        method_model, method_trace = _build_model_trace(
            model_name=args.method_model,
            issue_index=issue_index,
            lead=args.lead,
            bundle=bundle,
            config=config,
            exported_pred=np.load(method_root / args.method_model / "predictions.npz")["preds"][
                exported["sample_index"], :, :, exported["lead_position"]
            ].astype(np.float32),
        )
        adaptation_summary = _summarize_method_adaptation(
            method_model=method_model,
            issue_index=issue_index,
            lead=args.lead,
            ocean_mask=bundle.land_mask,
        )

        baseline_metrics = _sample_metrics(
            exported["baseline_pred"],
            exported["truth"],
            threshold=config.target_threshold,
            cell_area_km2=config.cell_area_km2,
        )
        method_metrics = _sample_metrics(
            exported["method_pred"],
            exported["truth"],
            threshold=config.target_threshold,
            cell_area_km2=config.cell_area_km2,
        )

        payload = {
            "output_tag": args.output_tag,
            "method_output_tag": args.method_output_tag or args.output_tag,
            "selected_case": {
                "issue_ym": args.issue_ym,
                "target_ym": args.target_ym,
                "lead": int(args.lead),
                "issue_index": int(issue_index),
                "sample_index": int(exported["sample_index"]),
                "lead_position": int(exported["lead_position"]),
            },
            "baseline_model": args.baseline_model,
            "method_model": args.method_model,
            "method_label": args.method_label,
            "baseline_metrics": baseline_metrics,
            "method_metrics": method_metrics,
            "models": {
                args.baseline_model: baseline_trace,
                args.method_model: method_trace,
            },
            "adaptation_summary": adaptation_summary,
        }

    outputs_dir = method_root / "case_study"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    json_output = (
        Path(args.json_output)
        if args.json_output
        else outputs_dir / f"{args.issue_ym.replace('-', '_')}_lead{args.lead}_{args.method_model}.json"
    )
    figure_output = (
        Path(args.figure_output)
        if args.figure_output
        else config.root_dir / "ICCBR_2026_LNCS_Template" / "img" / f"case_study_{args.issue_ym.replace('-', '_')}_lead{args.lead}.png"
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w") as f:
        json.dump(_pyify(payload), f, indent=2)
    _write_figure(
        figure_output=figure_output,
        truth=exported["truth"],
        baseline_pred=exported["baseline_pred"],
        method_pred=exported["method_pred"],
        baseline_metrics=payload["baseline_metrics"],
        method_metrics=payload["method_metrics"],
        trace_rows=_trace_rows_from_payload(payload),
        target_ym=payload["selected_case"]["target_ym"],
        method_label=payload["method_label"],
    )
    print(json_output)
    print(figure_output)


if __name__ == "__main__":
    main()
