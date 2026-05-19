from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from icecbr.config import ExperimentConfig
from icecbr.evaluate_exported_predictions import _evaluate_model_npz


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

VARIANT_SPECS = [
    ("Traditional Analogue (SIC-only)", "tmp_analogue_ablation_2016_2023", "analog_global_sic"),
    ("Enhanced Analogue (SIC+BG+Hist)", "tmp_analogue_ablation_2016_2023", "analog_global_sic_bg_hist"),
    ("Regional Retrieval (No Reuse)", "tmp_standard_analog_knn_2016_2023", "standard_analog_knn"),
    ("CBR--Reuse (global detrend)", "target_month_export_2016_2023_a0_icemamba_inputs_v1", "a0"),
    ("CBR--Reuse (CSA-Adapt)", "target_month_export_2016_2023_a0_mrlinear_icemamba_inputs_v1", "a0_mrlinear"),
    ("CBR--Reuse+Revise", "target_month_export_2016_2023_revise_icemamba_inputs_v1", "a0_mrlinear"),
    ("IceCBR", "target_month_export_2016_2023_revise_retain_final_icemamba_inputs_v1", "a0_mrlinear"),
]

METHOD_COLORS = {
    "Traditional Analogue (SIC-only)": "#8C8C8C",
    "Enhanced Analogue (SIC+BG+Hist)": "#B08B57",
    "Regional Retrieval (No Reuse)": "#6E7783",
    "CBR--Reuse (global detrend)": "#D17C00",
    "CBR--Reuse (CSA-Adapt)": "#C44E52",
    "CBR--Reuse+Revise": "#0072B2",
    "IceCBR": "#009E73",
}

METHOD_MARKERS = {
    "Traditional Analogue (SIC-only)": "o",
    "Enhanced Analogue (SIC+BG+Hist)": "s",
    "Regional Retrieval (No Reuse)": "^",
    "CBR--Reuse (global detrend)": "D",
    "CBR--Reuse (CSA-Adapt)": "P",
    "CBR--Reuse+Revise": "X",
    "IceCBR": "*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lead-time and target-month comparison for selected CBR variants.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root directory containing the exported result tags. Defaults to ICECBR_OUTPUT_DIR or ../results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _resolve_variant_dirs(results_root: Path) -> list[tuple[str, Path]]:
    return [
        (label, results_root / tag / "target_month_export" / model_name)
        for label, tag, model_name in VARIANT_SPECS
    ]


def _load_metric_arrays(variant_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mse_mae_path = variant_dir / "mse_mae_all.npz"
    iiee_path = variant_dir / "IIEE_all.npz"
    if mse_mae_path.exists() and iiee_path.exists():
        mse_mae = np.load(mse_mae_path)
        iiee = np.load(iiee_path)
        return np.sqrt(mse_mae["MSE"]), iiee["IIEE"]

    config = ExperimentConfig()
    land_mask = np.load(config.land_mask_path).astype(bool)
    results = _evaluate_model_npz(variant_dir / "predictions.npz", land_mask, start_date="2016-01")
    return np.sqrt(results["mse_all"]), results["iiee_all"]


def load_variant_arrays(variant_name: str, variant_dir: Path) -> pd.DataFrame:
    rmse, iiee_arr = _load_metric_arrays(variant_dir)

    if rmse.shape != (6, 12) or iiee_arr.shape != (6, 12):
        raise ValueError(f"Unexpected metric shape for {variant_name}: rmse={rmse.shape}, iiee={iiee_arr.shape}")

    rows = []
    for lead_idx in range(6):
        for month_idx in range(12):
            rows.append(
                {
                    "model": variant_name,
                    "lead": lead_idx + 1,
                    "target_month": month_idx + 1,
                    "target_month_label": MONTH_LABELS[month_idx],
                    "rmse": float(rmse[lead_idx, month_idx]),
                    "iiee": float(iiee_arr[lead_idx, month_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_plot_frame() -> pd.DataFrame:
    frames = []
    results_root = Path(
        __import__("os").environ.get("ICECBR_OUTPUT_DIR", Path(__file__).resolve().parents[2] / "results")
    ).expanduser()
    for variant_name, variant_dir in _resolve_variant_dirs(results_root):
        frames.append(load_variant_arrays(variant_name, variant_dir))
    return pd.concat(frames, ignore_index=True)


def plot_combined_curves(df: pd.DataFrame, output_dir: Path) -> None:
    lead_summary = df.groupby(["model", "lead"], as_index=False)[["rmse", "iiee"]].mean()
    month_summary = df.groupby(["model", "target_month"], as_index=False)[["rmse", "iiee"]].mean()

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), dpi=220)
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.08, right=0.98, hspace=0.34, wspace=0.24)
    panel_specs = [
        (axes[0, 0], lead_summary, "lead", "Lead time (months)", ("rmse", "RMSE"), "RMSE by lead time"),
        (axes[0, 1], lead_summary, "lead", "Lead time (months)", ("iiee", "IIEE"), "IIEE by lead time"),
        (axes[1, 0], month_summary, "target_month", "Target month", ("rmse", "RMSE"), "RMSE by target month"),
        (axes[1, 1], month_summary, "target_month", "Target month", ("iiee", "IIEE"), "IIEE by target month"),
    ]

    for ax, source_df, x_key, x_label, (metric_key, metric_label), title in panel_specs:
        for model, _, _ in VARIANT_SPECS:
            subset = source_df[source_df["model"] == model].sort_values(x_key)
            ax.plot(
                subset[x_key],
                subset[metric_key],
                marker=METHOD_MARKERS[model],
                linewidth=2.2,
                markersize=4.8,
                color=METHOD_COLORS[model],
                label=model,
            )
        if x_key == "lead":
            ax.set_xticks(range(1, 7))
        else:
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(output_dir / "lead_target_month_curves_rmse_iiee.png", bbox_inches="tight")
    fig.savefig(output_dir / "lead_target_month_curves_rmse_iiee.pdf", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    results_root = args.results_root or config.root_dir.parent / "results"
    output_dir = args.output_dir or (config.results_dir / "cbr_variant_compare_2016_2023" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for variant_name, variant_dir in _resolve_variant_dirs(results_root):
        frames.append(load_variant_arrays(variant_name, variant_dir))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_dir / "cbr_variant_month_lead_metrics.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_combined_curves(df, output_dir)

    print(output_dir / "lead_target_month_curves_rmse_iiee.png")


if __name__ == "__main__":
    main()
