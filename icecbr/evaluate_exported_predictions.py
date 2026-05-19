from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .metrics import calculate_acc, month_lead_iiee, month_lead_mae_mse


DEFAULT_START_DATE = "2016-01"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate exported prediction npz files.")
    parser.add_argument("--output-tag", type=str, default="target_month_export_2016_2023")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    return parser.parse_args()


def _write_json(path: Path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _evaluate_model_npz(npz_path: Path, land_mask: np.ndarray, start_date: str) -> dict:
    payload = np.load(npz_path)
    preds = payload["preds"].astype(np.float32)
    truths = payload["truths"].astype(np.float32)

    mae_all, mse_all = month_lead_mae_mse(preds, truths, land_mask, start_date=start_date)
    iiee_all, oe_all, ue_all = month_lead_iiee(
        preds,
        truths,
        start_date=start_date,
        threshold=0.15,
        cell_area_km2=625.0,
    )
    acc = calculate_acc(preds, truths, land_mask)

    summary = {
        "mae_mean": float(np.mean(mae_all)),
        "rmse_mean": float(np.sqrt(np.mean(mse_all))),
        "iiee_mean": float(np.mean(iiee_all)),
        "oe_mean": float(np.mean(oe_all)),
        "ue_mean": float(np.mean(ue_all)),
        "acc_mean": float(np.nanmean(acc)),
    }
    return {
        "preds": preds,
        "truths": truths,
        "mae_all": mae_all,
        "mse_all": mse_all,
        "iiee_all": iiee_all,
        "oe_all": oe_all,
        "ue_all": ue_all,
        "acc": acc,
        "summary": summary,
    }


def main():
    args = parse_args()
    config = ExperimentConfig()
    root = config.results_dir / args.output_tag / "target_month_export"
    land_mask = np.load(config.land_mask_path).astype(bool)

    rows = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        npz_path = model_dir / "predictions.npz"
        if not npz_path.exists():
            continue
        results = _evaluate_model_npz(npz_path, land_mask, start_date=args.start_date)

        np.savez(model_dir / "mse_mae_all.npz", MAE=results["mae_all"], MSE=results["mse_all"])
        np.savez(
            model_dir / "IIEE_all.npz",
            IIEE=results["iiee_all"],
            OE=results["oe_all"],
            UE=results["ue_all"],
        )
        _write_json(
            model_dir / "acc_metrics.json",
            {
                "acc_per_lead": results["acc"].tolist(),
                "acc_mean": results["summary"]["acc_mean"],
            },
        )
        _write_json(model_dir / "metric_summary.json", results["summary"])

        row = {"model": model_dir.name}
        row.update(results["summary"])
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("model")
    summary_df.to_csv(root / "baseline_metric_summary.csv", index=False)
    print(summary_df.to_json(orient="records"))


if __name__ == "__main__":
    main()
