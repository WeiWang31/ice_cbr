from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from icecbr.metrics import calculate_acc, month_lead_iiee, month_lead_mae_mse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LAND_MASK_PATH = PROJECT_ROOT.parent / "data_preprocess" / "g2202_land.npy"
START_DATE = "2016-01"


MAIN_RESULTS = [
    (
        "CBR--Reuse (global detrend)",
        RESULTS_ROOT
        / "target_month_export_2016_2023_a0_icemamba_inputs_v1"
        / "target_month_export"
        / "a0"
        / "predictions.npz",
    ),
    (
        "CBR--Reuse (CSA-Adapt)",
        RESULTS_ROOT
        / "target_month_export_2016_2023_a0_mrlinear_icemamba_inputs_v1"
        / "target_month_export"
        / "a0_mrlinear"
        / "predictions.npz",
    ),
    (
        "CBR--Reuse+Revise",
        RESULTS_ROOT
        / "target_month_export_2016_2023_revise_icemamba_inputs_v1"
        / "target_month_export"
        / "a0_mrlinear"
        / "predictions.npz",
    ),
    (
        "IceCBR",
        RESULTS_ROOT
        / "target_month_export_2016_2023_revise_retain_final_icemamba_inputs_v1"
        / "target_month_export"
        / "a0_mrlinear"
        / "predictions.npz",
    ),
]


def _format_value(value: float) -> str:
    return f"{value:.4f}"


def _load_land_mask() -> np.ndarray:
    if not LAND_MASK_PATH.exists():
        raise FileNotFoundError(f"Missing land mask file: {LAND_MASK_PATH}")
    return np.load(LAND_MASK_PATH).astype(bool)


def _evaluate_model_npz(npz_path: Path, land_mask: np.ndarray) -> dict[str, float]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {npz_path}")
    payload = np.load(npz_path)
    preds = payload["preds"].astype(np.float32)
    truths = payload["truths"].astype(np.float32)

    mae_all, mse_all = month_lead_mae_mse(preds, truths, land_mask, start_date=START_DATE)
    iiee_all, _, _ = month_lead_iiee(
        preds,
        truths,
        start_date=START_DATE,
        threshold=0.15,
        cell_area_km2=625.0,
    )
    acc = calculate_acc(preds, truths, land_mask)

    return {
        "MAE": float(np.mean(mae_all)),
        "RMSE": float(np.sqrt(np.mean(mse_all))),
        "IIEE": float(np.mean(iiee_all)),
        "ACC": float(np.nanmean(acc)),
    }


def _build_dataframe() -> pd.DataFrame:
    land_mask = _load_land_mask()
    rows = []
    for method, npz_path in MAIN_RESULTS:
        metrics = _evaluate_model_npz(npz_path, land_mask)
        rows.append({"Method": method, **metrics})
    return pd.DataFrame(rows, columns=["Method", "MAE", "RMSE", "IIEE", "ACC"])


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    formatted = df.copy()
    for col in ["MAE", "RMSE", "IIEE", "ACC"]:
        formatted[col] = formatted[col].map(_format_value)
    formatted.to_csv(path, index=False)


def _write_tex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & MAE $\downarrow$ & RMSE $\downarrow$ & IIEE $\downarrow$ & ACC $\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Method']} & "
            f"{_format_value(row['MAE'])} & "
            f"{_format_value(row['RMSE'])} & "
            f"{_format_value(row['IIEE'])} & "
            f"{_format_value(row['ACC'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe()
    csv_path = OUTPUT_DIR / "main_results_table.csv"
    tex_path = OUTPUT_DIR / "main_results_table.tex"
    _write_csv(df, csv_path)
    _write_tex(df, tex_path)
    print(df.to_json(orient="records"))
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
