from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_bundle
from .metrics import (
    calculate_iiee_single,
    brier_skill_score,
    calculate_acc,
    month_lead_iiee,
    month_lead_mae_mse,
    regional_rmse,
    sie_absolute_error,
)
from .models import _retain_store_path, build_export_model, build_model, summarize_retain_diagnostics
from .seas5 import (
    download_seas5,
    download_seas5_hindcast,
    evaluate_seas5_for_year,
    fit_seas5_bias_correction,
    load_processed_seas5,
    load_seas5_bias_correction,
    process_seas5,
    process_seas5_hindcast,
)
from .stats import block_bootstrap_ci, diebold_mariano, paired_significance


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _experiment_dir(config: ExperimentConfig, experiment_name: str, smoke: bool = False) -> Path:
    base = config.smoke_results_dir if smoke else config.results_dir
    if config.output_tag and not smoke:
        base = base / config.output_tag
    out_dir = base / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_json(path: Path, payload):
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _append_jsonl(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row)))
            f.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _persist_retained_case(config: ExperimentConfig, model_name: str, retained_case: dict):
    if not config.enable_retain or not config.retain_persist_jsonl or retained_case is None:
        return
    path = _retain_store_path(config, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(path, [retained_case])
    meta_path = path.with_suffix(".meta.json")
    count = 0
    if path.exists():
        with open(path, "r") as f:
            count = sum(1 for line in f if line.strip())
    _write_json(
        meta_path,
        {
            "model": model_name,
            "retain_namespace": config.retain_namespace,
            "enable_retain": config.enable_retain,
            "retain_online_growth": config.retain_online_growth,
            "retain_reload": config.retain_reload,
            "retained_case_count": count,
        },
    )


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {
            "status": "pending",
            "current_variant": None,
            "completed_years": [],
            "started_at": None,
            "updated_at": None,
        }
    with open(path, "r") as f:
        return json.load(f)


def _save_progress(path: Path, payload: dict):
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(path, payload)


def _prepare_variant_dir(variant_dir: Path, run_mode: str):
    if run_mode == "fresh" and variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "predictions").mkdir(parents=True, exist_ok=True)


def _single_field_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    ocean_mask: np.ndarray,
    region_masks: dict[int, np.ndarray],
    threshold: float,
    cell_area_km2: float,
    climatology_prob: np.ndarray,
):
    diff = pred[ocean_mask] - truth[ocean_mask]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    pred_bin = (pred > threshold).astype(np.int32)
    truth_bin = (truth > threshold).astype(np.int32)
    oe, ue, iiee = calculate_iiee_single(truth_bin, pred_bin, cell_area_km2=cell_area_km2)
    pred_sie = int((pred[ocean_mask] > threshold).sum())
    truth_sie = int((truth[ocean_mask] > threshold).sum())
    sie_abs_error = float(abs(pred_sie - truth_sie))
    y = (truth > threshold).astype(np.float32)
    p = np.clip(pred, 0.0, 1.0)
    bs = float(np.mean((p[ocean_mask] - y[ocean_mask]) ** 2))
    bs_ref = float(np.mean((climatology_prob[ocean_mask] - y[ocean_mask]) ** 2))
    bss = float(1.0 - bs / bs_ref) if bs_ref != 0 else np.nan
    regional = {}
    for rid, mask in region_masks.items():
        region_diff = pred[mask] - truth[mask]
        regional[rid] = float(np.sqrt(np.mean(region_diff**2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "iiee": float(iiee),
        "oe": float(oe),
        "ue": float(ue),
        "sie_abs_error": sie_abs_error,
        "bss": bss,
        "regional": regional,
    }


def _target_index_range(config: ExperimentConfig, bundle) -> range:
    start_idx = (config.target_eval_start_year - config.start_year) * 12
    end_idx = (config.target_eval_end_year - config.start_year + 1) * 12
    return range(start_idx, end_idx)


def _evaluate_internal_model_target_months(
    model_name: str,
    config: ExperimentConfig,
    bundle,
):
    original_namespace = config.retain_namespace
    config.retain_namespace = "internal_target_month_eval"
    target_indices = list(_target_index_range(config, bundle))
    unique_issue_indices = sorted({target_idx - lead for target_idx in target_indices for lead in config.lead_times})
    unique_issue_indices = [idx for idx in unique_issue_indices if idx >= 0]

    pred_cache: dict[tuple[int, int], np.ndarray] = {}
    metadata_cache: dict[tuple[int, int], dict] = {}
    retained_cases: list[dict] = []
    for issue_index in unique_issue_indices:
        train_indices = np.arange(issue_index)
        model = build_model(model_name)
        model.fit(train_indices, bundle, config)
        if config.enable_retain and config.retain_online_growth:
            for retained_case in retained_cases:
                model.append_retained_case(retained_case)
        for lead in config.lead_times:
            target_index = issue_index + lead
            if target_index >= bundle.sic.shape[0]:
                continue
            result = model.predict(issue_index, lead)
            pred_cache[(issue_index, lead)] = result.prediction
            metadata_cache[(issue_index, lead)] = result.metadata
            if config.enable_retain:
                retained_case = model.retain_case(issue_index, lead, result, bundle.sic[target_index])
                if retained_case is not None:
                    retained_cases.append(retained_case)
                    _persist_retained_case(config, model_name, retained_case)
                    if config.retain_online_growth:
                        model.append_retained_case(retained_case)

    rows = []
    region_rows = []
    pred_by_lead = {lead: [] for lead in config.lead_times}
    truth_by_lead = {lead: [] for lead in config.lead_times}
    clim_prob_cache: dict[int, np.ndarray] = {}
    for target_index in target_indices:
        target_date = bundle.time_index[target_index]
        truth = bundle.sic[target_index]
        for lead in config.lead_times:
            issue_index = target_index - lead
            if issue_index < 0 or (issue_index, lead) not in pred_cache:
                continue
            if issue_index not in clim_prob_cache:
                clim_prob_cache[issue_index] = np.mean(
                    (bundle.sic[:issue_index] > config.target_threshold).astype(np.float32),
                    axis=0,
                )
            pred = pred_cache[(issue_index, lead)]
            metrics = _single_field_metrics(
                pred,
                truth,
                bundle.land_mask,
                bundle.region_info.region_masks,
                config.target_threshold,
                config.cell_area_km2,
                clim_prob_cache[issue_index],
            )
            issue_date = bundle.time_index[issue_index]
            row = {
                "model": model_name,
                "target_year": int(target_date.year),
                "target_month": int(target_date.month),
                "target_ym": target_date.strftime("%Y-%m"),
                "lead": int(lead),
                "issue_year": int(issue_date.year),
                "issue_month": int(issue_date.month),
                "issue_ym": issue_date.strftime("%Y-%m"),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "iiee": metrics["iiee"],
                "oe": metrics["oe"],
                "ue": metrics["ue"],
                "sie_abs_error": metrics["sie_abs_error"],
                "bss": metrics["bss"],
            }
            rows.append(row)
            for rid, value in metrics["regional"].items():
                region_rows.append(
                    {
                        "model": model_name,
                        "target_year": int(target_date.year),
                        "target_month": int(target_date.month),
                        "target_ym": target_date.strftime("%Y-%m"),
                        "lead": int(lead),
                        "region_id": rid,
                        "region_name": bundle.region_info.region_names.get(rid, str(rid)),
                        "rmse": value,
                    }
                )
            pred_by_lead[lead].append(pred)
            truth_by_lead[lead].append(truth)

    acc_by_lead = {}
    for lead in config.lead_times:
        preds = pred_by_lead[lead]
        truths = truth_by_lead[lead]
        if not preds:
            acc_by_lead[lead] = np.nan
            continue
        stacked_preds = np.stack(preds, axis=0)[:, :, :, None]
        stacked_truths = np.stack(truths, axis=0)[:, :, :, None]
        acc_by_lead[lead] = float(calculate_acc(stacked_preds, stacked_truths, bundle.land_mask)[0])
    for row in rows:
        row["acc"] = acc_by_lead.get(row["lead"], np.nan)
    config.retain_namespace = original_namespace
    return pd.DataFrame(rows), pd.DataFrame(region_rows)


def _export_internal_model_target_months(
    model_name: str,
    config: ExperimentConfig,
    bundle,
    progress_hook=None,
):
    original_namespace = config.retain_namespace
    config.retain_namespace = "target_month_export"
    target_indices = list(_target_index_range(config, bundle))
    target_dates = bundle.time_index[target_indices]
    leads = list(config.lead_times)
    height, width = bundle.sic.shape[1:]
    n_targets = len(target_indices)
    n_leads = len(leads)

    unique_issue_indices = sorted({target_idx - lead for target_idx in target_indices for lead in leads})
    unique_issue_indices = [idx for idx in unique_issue_indices if idx >= 0]

    pred_cache: dict[tuple[int, int], np.ndarray] = {}
    model = build_export_model(model_name)
    retained_cases: list[dict] = []
    retain_events: list[dict] = []
    size_trace: list[dict] = []
    total_issues = len(unique_issue_indices)
    for issue_pos, issue_index in enumerate(unique_issue_indices, start=1):
        if progress_hook is not None:
            progress_hook(issue_index, issue_pos - 1, total_issues, "fitting")
        train_indices = np.arange(issue_index)
        model.fit(train_indices, bundle, config)
        if config.enable_retain and config.retain_online_growth:
            for retained_case in retained_cases:
                model.append_retained_case(retained_case)
        for lead in leads:
            target_index = issue_index + lead
            if target_index >= bundle.sic.shape[0]:
                continue
            result = model.predict(issue_index, lead)
            pred_cache[(issue_index, lead)] = result.prediction.astype(np.float32)
            if config.enable_retain:
                retained_case = model.retain_case(issue_index, lead, result, bundle.sic[target_index])
                if retained_case is not None:
                    retained_cases.append(retained_case)
                    _persist_retained_case(config, model_name, retained_case)
                    if config.retain_online_growth:
                        model.append_retained_case(retained_case)
                    retain_metadata = retained_case.get("retain_metadata") or {}
                    admission = retain_metadata.get("admission") or {}
                    selection = retain_metadata.get("selection") or {}
                    usage = result.metadata.get("retained_case_usage") or {}
                    snapshot = model.retain_diagnostics_snapshot() if hasattr(model, "retain_diagnostics_snapshot") else {}
                    retain_events.append(
                        {
                            "issue_index": int(issue_index),
                            "issue_ym": bundle.time_index[issue_index].strftime("%Y-%m"),
                            "lead": int(lead),
                            "target_index": int(target_index),
                            "target_ym": bundle.time_index[target_index].strftime("%Y-%m"),
                            "admitted": bool(admission.get("admitted", True)),
                            "admission_score": float(admission.get("score", 0.0)),
                            "admission_mode": admission.get("mode", config.retain_admission_mode),
                            "representative_decision": selection.get("representative_decision", "disabled"),
                            "active_retained_cases": int(snapshot.get("active_retained", 0)),
                            "evicted_count": int(snapshot.get("evicted", 0)),
                            "retained_retrieval_count": int(usage.get("retained_count", 0)),
                            "retained_contribution": float(usage.get("retained_contribution", 0.0)),
                        }
                    )
        if progress_hook is not None:
            progress_hook(issue_index, issue_pos, total_issues, "completed")
        snapshot = model.retain_diagnostics_snapshot() if hasattr(model, "retain_diagnostics_snapshot") else {}
        size_trace.append(
            {
                "issue_index": int(issue_index),
                "issue_ym": bundle.time_index[issue_index].strftime("%Y-%m"),
                "active_retained_cases": int(snapshot.get("active_retained", 0)),
                "generated": int(snapshot.get("generated", 0)),
                "admitted": int(snapshot.get("admitted", 0)),
                "rejected": int(snapshot.get("rejected", 0)),
                "evicted": int(snapshot.get("evicted", 0)),
            }
        )

    preds = np.full((n_targets, height, width, n_leads), np.nan, dtype=np.float32)
    truths = np.full_like(preds, np.nan)
    issue_ym = np.empty((n_targets, n_leads), dtype="<U7")
    issue_ym[:] = ""

    for target_pos, target_index in enumerate(target_indices):
        truth = bundle.sic[target_index].astype(np.float32)
        for lead_pos, lead in enumerate(leads):
            issue_index = target_index - lead
            if issue_index < 0:
                continue
            issue_date = bundle.time_index[issue_index]
            issue_ym[target_pos, lead_pos] = issue_date.strftime("%Y-%m")
            truths[target_pos, :, :, lead_pos] = truth
            pred = pred_cache.get((issue_index, lead))
            if pred is not None:
                preds[target_pos, :, :, lead_pos] = pred

    target_ym = np.array([ts.strftime("%Y-%m") for ts in target_dates], dtype="<U7")
    payload = {
        "preds": preds,
        "truths": truths,
        "target_ym": target_ym,
        "issue_ym": issue_ym,
        "lead_months": np.asarray(leads, dtype=np.int32),
        "retain_diagnostics": {**summarize_retain_diagnostics(retain_events), "size_trace": size_trace},
        "retain_events": retain_events,
    }
    config.retain_namespace = original_namespace
    return payload


def _evaluate_seas5_target_months(config: ExperimentConfig, bundle):
    if not config.seas5_processed_path.exists():
        if not (config.seas5_download_manifest_path.exists() and config.seas5_raw_blocks_dir.exists()):
            download_seas5(config)
        process_seas5(config)
    seas5_preds, init_times, lead_months = load_processed_seas5(config)
    init_lookup = {(ts.year, ts.month): idx for idx, ts in enumerate(init_times)}
    lead_lookup = {int(lead): idx for idx, lead in enumerate(lead_months.tolist())}

    target_indices = list(_target_index_range(config, bundle))
    rows = []
    region_rows = []
    pred_by_lead = {lead: [] for lead in config.lead_times}
    truth_by_lead = {lead: [] for lead in config.lead_times}
    clim_prob_cache: dict[int, np.ndarray] = {}
    for target_index in target_indices:
        target_date = bundle.time_index[target_index]
        truth = bundle.sic[target_index]
        for lead in config.lead_times:
            issue_index = target_index - lead
            if issue_index < 0:
                continue
            issue_date = bundle.time_index[issue_index]
            init_idx = init_lookup.get((int(issue_date.year), int(issue_date.month)))
            lead_idx = lead_lookup.get(int(lead))
            if init_idx is None or lead_idx is None:
                continue
            if issue_index not in clim_prob_cache:
                clim_prob_cache[issue_index] = np.mean(
                    (bundle.sic[:issue_index] > config.target_threshold).astype(np.float32),
                    axis=0,
                )
            pred = seas5_preds[init_idx, :, :, lead_idx]
            metrics = _single_field_metrics(
                pred,
                truth,
                bundle.land_mask,
                bundle.region_info.region_masks,
                config.target_threshold,
                config.cell_area_km2,
                clim_prob_cache[issue_index],
            )
            rows.append(
                {
                    "model": "seas5",
                    "target_year": int(target_date.year),
                    "target_month": int(target_date.month),
                    "target_ym": target_date.strftime("%Y-%m"),
                    "lead": int(lead),
                    "issue_year": int(issue_date.year),
                    "issue_month": int(issue_date.month),
                    "issue_ym": issue_date.strftime("%Y-%m"),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "iiee": metrics["iiee"],
                    "oe": metrics["oe"],
                    "ue": metrics["ue"],
                    "sie_abs_error": metrics["sie_abs_error"],
                    "bss": metrics["bss"],
                }
            )
            for rid, value in metrics["regional"].items():
                region_rows.append(
                    {
                        "model": "seas5",
                        "target_year": int(target_date.year),
                        "target_month": int(target_date.month),
                        "target_ym": target_date.strftime("%Y-%m"),
                        "lead": int(lead),
                        "region_id": rid,
                        "region_name": bundle.region_info.region_names.get(rid, str(rid)),
                        "rmse": value,
                    }
                )
            pred_by_lead[lead].append(pred)
            truth_by_lead[lead].append(truth)

    acc_by_lead = {}
    for lead in config.lead_times:
        preds = pred_by_lead[lead]
        truths = truth_by_lead[lead]
        if not preds:
            acc_by_lead[lead] = np.nan
            continue
        stacked_preds = np.stack(preds, axis=0)[:, :, :, None]
        stacked_truths = np.stack(truths, axis=0)[:, :, :, None]
        acc_by_lead[lead] = float(calculate_acc(stacked_preds, stacked_truths, bundle.land_mask)[0])
    for row in rows:
        row["acc"] = acc_by_lead.get(row["lead"], np.nan)
    return pd.DataFrame(rows), pd.DataFrame(region_rows)


def run_target_month_compare(config: ExperimentConfig):
    if config.output_tag is None:
        config.output_tag = f"target_month_compare_{config.target_eval_start_year}_{config.target_eval_end_year}"
    out_dir = _experiment_dir(config, "target_month_compare")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(config)
    models = list(config.target_compare_models)
    all_rows = []
    all_region_rows = []
    original_scheme = config.region_scheme
    for model_name in models:
        config.region_scheme = "nsidc18"
        if model_name == "seas5":
            rows_df, region_df = _evaluate_seas5_target_months(config, bundle)
        else:
            rows_df, region_df = _evaluate_internal_model_target_months(model_name, config, bundle)
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        rows_df.to_csv(model_dir / "target_lead_metrics.csv", index=False)
        region_df.to_csv(model_dir / "region_metrics.csv", index=False)
        all_rows.append(rows_df)
        all_region_rows.append(region_df)
    config.region_scheme = original_scheme

    metrics_df = pd.concat(all_rows, ignore_index=True)
    region_df = pd.concat(all_region_rows, ignore_index=True)
    metrics_df.to_csv(out_dir / "target_lead_metrics.csv", index=False)
    region_df.to_csv(out_dir / "region_metrics.csv", index=False)

    summary_df = (
        metrics_df.groupby("model", as_index=False)[["rmse", "mae", "bss", "sie_abs_error", "iiee", "acc"]].mean()
        .rename(
            columns={
                "rmse": "rmse_mean",
                "mae": "mae_mean",
                "bss": "bss_mean",
                "sie_abs_error": "sie_abs_error_mean",
                "iiee": "iiee_mean",
                "acc": "acc_mean",
            }
        )
    )
    sample_counts_df = (
        metrics_df.groupby("model", as_index=False)
        .size()
        .rename(columns={"size": "n_target_lead_samples"})
    )
    summary_df = summary_df.merge(sample_counts_df, on="model", how="left")
    lead_df = (
        metrics_df.groupby(["model", "lead"], as_index=False)[["rmse", "mae", "bss", "sie_abs_error", "iiee", "acc"]].mean()
    )
    lead_counts_df = (
        metrics_df.groupby(["model", "lead"], as_index=False)
        .size()
        .rename(columns={"size": "n_target_samples"})
    )
    lead_df = lead_df.merge(lead_counts_df, on=["model", "lead"], how="left")
    target_month_df = (
        metrics_df.groupby(["model", "target_year", "target_month", "target_ym"], as_index=False)[
            ["rmse", "mae", "bss", "sie_abs_error", "iiee"]
        ].mean()
    )
    target_month_counts_df = (
        metrics_df.groupby(["model", "target_year", "target_month", "target_ym"], as_index=False)
        .size()
        .rename(columns={"size": "n_leads_present"})
    )
    target_month_df = target_month_df.merge(
        target_month_counts_df,
        on=["model", "target_year", "target_month", "target_ym"],
        how="left",
    )
    target_year_df = (
        metrics_df.groupby(["model", "target_year"], as_index=False)[["rmse", "mae", "bss", "sie_abs_error", "iiee", "acc"]].mean()
    )
    target_year_counts_df = (
        metrics_df.groupby(["model", "target_year"], as_index=False)
        .size()
        .rename(columns={"size": "n_target_lead_samples"})
    )
    target_year_df = target_year_df.merge(target_year_counts_df, on=["model", "target_year"], how="left")
    region_summary_df = (
        region_df.groupby(["model", "region_id", "region_name"], as_index=False)[["rmse"]].mean()
    )
    metadata = {
        "protocol": "target_month_compare",
        "target_eval_start_year": config.target_eval_start_year,
        "target_eval_end_year": config.target_eval_end_year,
        "lead_times": list(config.lead_times),
        "models": models,
        "region_scheme": "nsidc18",
        "notes": "Records are organized by target month and lead; issue month is target month minus lead months.",
    }

    summary_df.to_csv(out_dir / "summary.csv", index=False)
    lead_df.to_csv(out_dir / "lead_summary.csv", index=False)
    target_month_df.to_csv(out_dir / "target_month_summary.csv", index=False)
    target_year_df.to_csv(out_dir / "target_year_summary.csv", index=False)
    region_summary_df.to_csv(out_dir / "region_summary.csv", index=False)
    _write_json(out_dir / "metadata.json", metadata)
    return summary_df


def run_target_month_export(config: ExperimentConfig):
    if config.output_tag is None:
        config.output_tag = f"target_month_export_{config.target_eval_start_year}_{config.target_eval_end_year}"
    out_dir = _experiment_dir(config, "target_month_export")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(config)
    models = list(config.selected_models or config.target_export_models)
    progress_path = out_dir / "progress.json"
    progress = {
        "status": "running",
        "current_model": None,
        "completed_models": [],
        "target_eval_start_year": config.target_eval_start_year,
        "target_eval_end_year": config.target_eval_end_year,
        "lead_months": list(config.lead_times),
    }
    _save_progress(progress_path, progress)

    original_scheme = config.region_scheme
    for model_name in models:
        if config.run_mode == "resume" and (out_dir / model_name / "predictions.npz").exists():
            progress["completed_models"].append(model_name)
            continue
        progress["current_model"] = model_name
        _save_progress(progress_path, progress)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] START target_month_export model={model_name}",
            flush=True,
        )
        config.region_scheme = "nsidc18"
        model_dir = out_dir / model_name
        if config.run_mode == "fresh" and model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        def _issue_progress(issue_index: int, completed_issues: int, total_issues: int, phase: str):
            issue_date = bundle.time_index[issue_index]
            _save_progress(
                progress_path,
                {
                    "status": "running",
                    "current_model": model_name,
                    "completed_models": list(progress["completed_models"]),
                    "target_eval_start_year": config.target_eval_start_year,
                    "target_eval_end_year": config.target_eval_end_year,
                    "lead_months": list(config.lead_times),
                    "current_issue_index": int(issue_index),
                    "current_issue_ym": issue_date.strftime("%Y-%m"),
                    "current_issue_phase": phase,
                    "completed_issue_count": int(completed_issues),
                    "total_issue_count": int(total_issues),
                },
            )

        payload = _export_internal_model_target_months(
            model_name,
            config,
            bundle,
            progress_hook=_issue_progress,
        )
        np.savez_compressed(
            model_dir / "predictions.npz",
            preds=payload["preds"],
            truths=payload["truths"],
            target_ym=payload["target_ym"],
            issue_ym=payload["issue_ym"],
            lead_months=payload["lead_months"],
        )
        model_metadata = {
            "model": model_name,
            "target_eval_start_year": config.target_eval_start_year,
            "target_eval_end_year": config.target_eval_end_year,
            "lead_months": list(config.lead_times),
            "prediction_shape": list(payload["preds"].shape),
            "region_scheme": "nsidc18",
        }
        if payload.get("retain_diagnostics") is not None:
            model_metadata["retain_diagnostics"] = payload["retain_diagnostics"]
        _write_json(model_dir / "metadata.json", model_metadata)
        if payload.get("retain_diagnostics") is not None:
            _write_json(model_dir / "retain_diagnostics.json", payload["retain_diagnostics"])
            if payload.get("retain_events"):
                pd.DataFrame(payload["retain_events"]).to_csv(model_dir / "retain_diagnostics.csv", index=False)
            size_trace = payload["retain_diagnostics"].get("size_trace", [])
            if size_trace:
                pd.DataFrame(size_trace).to_csv(model_dir / "active_case_base_size.csv", index=False)
        progress["completed_models"].append(model_name)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] DONE target_month_export model={model_name} shape={payload['preds'].shape}",
            flush=True,
        )

    config.region_scheme = original_scheme
    progress["status"] = "completed"
    progress["current_model"] = None
    progress["models"] = models
    progress["prediction_shape"] = [len(list(_target_index_range(config, bundle))), *bundle.sic.shape[1:], len(config.lead_times)]
    _save_progress(progress_path, progress)
    metadata = {
        "protocol": "target_month_export",
        "target_eval_start_year": config.target_eval_start_year,
        "target_eval_end_year": config.target_eval_end_year,
        "lead_months": list(config.lead_times),
        "models": models,
        "region_scheme": "nsidc18",
        "prediction_shape": progress["prediction_shape"],
    }
    _write_json(out_dir / "metadata.json", metadata)
    return metadata


def _flatten_case_ids(metadata: dict) -> list[tuple[int, int]]:
    case_ids: list[tuple[int, int]] = []
    if "global_neighbors" in metadata:
        for item in metadata["global_neighbors"]:
            case_ids.append(tuple(item["case_id"]))
    for region_items in metadata.get("regional", {}).values():
        for item in region_items:
            case_ids.append(tuple(item["case_id"]))
    return case_ids


def evaluate_model_for_year(
    model_name: str,
    config: ExperimentConfig,
    bundle,
    held_out_year: int,
):
    original_namespace = config.retain_namespace
    config.retain_namespace = "yearly_eval"
    train_end = (held_out_year - config.start_year) * 12
    train_indices = np.arange(train_end)
    model = build_model(model_name)
    model.fit(train_indices, bundle, config)
    months = list(config.issue_months)
    leads = list(config.lead_times)
    year_preds = np.full((len(months),) + bundle.sic.shape[1:] + (len(leads),), np.nan, dtype=np.float32)
    year_truths = np.full_like(year_preds, np.nan)
    year_meta = []

    for month_idx, month in enumerate(months):
        issue_index = (held_out_year - config.start_year) * 12 + (month - 1)
        for lead_idx, lead in enumerate(leads):
            target_index = issue_index + lead
            if target_index >= bundle.sic.shape[0]:
                continue
            result = model.predict(issue_index, lead)
            year_preds[month_idx, :, :, lead_idx] = result.prediction
            year_truths[month_idx, :, :, lead_idx] = bundle.sic[target_index]
            if config.enable_retain:
                retained_case = model.retain_case(issue_index, lead, result, bundle.sic[target_index])
                if retained_case is not None:
                    _persist_retained_case(config, model_name, retained_case)
                    if config.retain_online_growth:
                        model.append_retained_case(retained_case)
            year_meta.append(
                {
                    "held_out_year": held_out_year,
                    "issue_month": month,
                    "issue_index": issue_index,
                    "target_index": target_index,
                    "lead": lead,
                    "metadata": result.metadata,
                }
            )

    if not np.isfinite(year_preds).any():
        config.retain_namespace = original_namespace
        return None

    start_date = f"{held_out_year}-01"
    mae_by_ml, mse_by_ml = month_lead_mae_mse(year_preds, year_truths, bundle.land_mask, start_date=start_date)
    iiee_by_ml, oe_by_ml, ue_by_ml = month_lead_iiee(
        year_preds,
        year_truths,
        start_date=start_date,
        threshold=config.target_threshold,
        cell_area_km2=config.cell_area_km2,
    )
    acc = calculate_acc(year_preds, year_truths, bundle.land_mask)
    regional = regional_rmse(year_preds, year_truths, bundle.region_info.region_masks)
    sie_err = sie_absolute_error(year_preds, year_truths, bundle.land_mask, threshold=config.target_threshold)
    clim_prob = np.mean((bundle.sic[train_indices] > config.target_threshold).astype(np.float32), axis=0)
    bss = brier_skill_score(year_preds, year_truths, clim_prob, bundle.land_mask, threshold=config.target_threshold)

    valid_month_lead = int(np.isfinite(mae_by_ml).sum())
    year_row = {
        "model": model_name,
        "held_out_year": held_out_year,
        "rmse": float(np.sqrt(np.nanmean(mse_by_ml))),
        "mae": float(np.nanmean(mae_by_ml)),
        "iiee": float(np.nanmean(iiee_by_ml)),
        "acc": float(np.nanmean(acc)),
        "sie_abs_error": float(np.nanmean(sie_err)),
        "bss": float(bss),
        "valid_month_lead_count": valid_month_lead,
    }
    month_lead_rows = []
    region_rows = []
    for rid, value in regional.items():
        year_row[f"region_{rid}_rmse"] = value
        region_rows.append(
            {
                "model": model_name,
                "held_out_year": held_out_year,
                "region_id": rid,
                "region_name": bundle.region_info.region_names.get(rid, str(rid)),
                "rmse": float(value),
            }
        )
    for lead_idx, lead in enumerate(config.lead_times):
        for month_idx, month in enumerate(range(1, 13)):
            issue_index = (held_out_year - config.start_year) * 12 + (month - 1)
            target_index = issue_index + lead
            target_date = bundle.time_index[target_index] if target_index < bundle.sic.shape[0] else None
            month_lead_rows.append(
                {
                    "model": model_name,
                    "held_out_year": held_out_year,
                    "issue_year": held_out_year,
                    "issue_month": month,
                    "issue_ym": f"{held_out_year}-{month:02d}",
                    "lead": lead,
                    "target_month": month,
                    "target_year_actual": int(target_date.year) if target_date is not None else np.nan,
                    "target_month_actual": int(target_date.month) if target_date is not None else np.nan,
                    "target_ym_actual": target_date.strftime("%Y-%m") if target_date is not None else None,
                    "mae": float(mae_by_ml[lead_idx, month_idx]),
                    "rmse": float(np.sqrt(mse_by_ml[lead_idx, month_idx])),
                    "iiee": float(iiee_by_ml[lead_idx, month_idx]),
                    "oe": float(oe_by_ml[lead_idx, month_idx]),
                    "ue": float(ue_by_ml[lead_idx, month_idx]),
                    "acc": float(acc[lead_idx]),
                    "is_valid": bool(np.isfinite(mae_by_ml[lead_idx, month_idx])),
                }
            )
    result = {
        "year_row": year_row,
        "month_lead_rows": month_lead_rows,
        "region_rows": region_rows,
        "metadata": year_meta,
        "preds": year_preds,
        "truths": year_truths,
    }
    config.retain_namespace = original_namespace
    return result


def evaluate_model_over_years(
    model_name: str,
    config: ExperimentConfig,
    years: list[int] | None = None,
    output_dir: Path | None = None,
):
    config.ensure_dirs()
    bundle = load_bundle(config)
    years = years or list(range(config.test_start_year, config.test_end_year + 1))
    output_dir = output_dir or config.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    yearly_results = []
    rows = []
    for held_out_year in years:
        result = evaluate_model_for_year(model_name, config, bundle, held_out_year)
        if result is None:
            continue
        yearly_results.append(result)
        rows.append(result["year_row"])

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{model_name}_metrics.csv", index=False)
    month_lead_rows = []
    region_rows = []
    meta_store = []
    pred_store = []
    truth_store = []
    for result in yearly_results:
        month_lead_rows.extend(result["month_lead_rows"])
        region_rows.extend(result["region_rows"])
        meta_store.extend(result["metadata"])
        pred_store.append(result["preds"])
        truth_store.append(result["truths"])
    pd.DataFrame(month_lead_rows).to_csv(output_dir / f"{model_name}_month_lead_metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(output_dir / f"{model_name}_regional_metrics.csv", index=False)
    if pred_store:
        np.save(output_dir / f"{model_name}_preds.npy", np.concatenate(pred_store, axis=0))
        np.save(output_dir / f"{model_name}_truths.npy", np.concatenate(truth_store, axis=0))
    _write_json(output_dir / f"{model_name}_metadata.json", meta_store)
    return df


def run_e1(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E1")
    models = [
        "climatology",
        "damped_persistence",
        "linear_trend_persistence",
        "analog_regional_sic_bg_hist",
        "analog_knn_global_detrend",
        "icecbr_lite",
        "icecbr_full",
        "ridge",
        "random_forest",
    ]
    all_results = [evaluate_model_over_years(name, config, output_dir=out_dir / name) for name in models]
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_dir / "E1_end_to_end.csv", index=False)
    return combined


def _aggregate_variant_outputs(variant_dir: Path):
    yearly_path = variant_dir / "yearly_metrics.csv"
    month_lead_path = variant_dir / "month_lead_metrics.csv"
    regional_path = variant_dir / "regional_metrics.csv"
    metadata_path = variant_dir / "metadata.jsonl"

    yearly_df = pd.read_csv(yearly_path) if yearly_path.exists() else pd.DataFrame()
    month_lead_df = pd.read_csv(month_lead_path) if month_lead_path.exists() else pd.DataFrame()
    regional_df = pd.read_csv(regional_path) if regional_path.exists() else pd.DataFrame()
    metadata = _read_jsonl(metadata_path)
    return yearly_df, month_lead_df, regional_df, metadata


def _summarize_model_outputs(
    model_name: str,
    years: list[int],
    yearly_df: pd.DataFrame,
    month_lead_df: pd.DataFrame,
    regional_df: pd.DataFrame,
):
    summary = {
        "model": model_name,
        "held_out_years": years,
        "n_years_completed": int(len(yearly_df)),
        "rmse_mean": float(yearly_df["rmse"].mean()) if not yearly_df.empty else np.nan,
        "mae_mean": float(yearly_df["mae"].mean()) if not yearly_df.empty else np.nan,
        "acc_mean": float(yearly_df["acc"].mean()) if not yearly_df.empty else np.nan,
        "bss_mean": float(yearly_df["bss"].mean()) if not yearly_df.empty else np.nan,
        "sie_abs_error_mean": float(yearly_df["sie_abs_error"].mean()) if not yearly_df.empty else np.nan,
        "iiee_mean": float(yearly_df["iiee"].mean()) if not yearly_df.empty else np.nan,
    }
    if not month_lead_df.empty:
        lead_summary = month_lead_df.groupby("lead", as_index=False)[["rmse", "acc", "mae", "iiee"]].mean()
        worst_lead = lead_summary.sort_values("rmse", ascending=False).iloc[0].to_dict()
        summary["worst_lead_by_rmse"] = _json_safe(worst_lead)
    if not regional_df.empty:
        regional_summary = regional_df.groupby(["region_id", "region_name"], as_index=False)["rmse"].mean()
        worst_region = regional_summary.sort_values("rmse", ascending=False).iloc[0].to_dict()
        summary["worst_region_by_rmse"] = _json_safe(worst_region)
    return summary


def _run_e2_incremental_variant(config: ExperimentConfig, bundle, variant: str, out_dir: Path):
    variant_dir = out_dir / variant
    _prepare_variant_dir(variant_dir, config.run_mode)
    progress_path = variant_dir / "progress.json"
    progress = _load_progress(progress_path)
    if config.run_mode == "fresh":
        progress = {
            "status": "running",
            "current_variant": variant,
            "completed_years": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
        }
        _save_progress(progress_path, progress)

    completed_years = set(progress.get("completed_years", []))
    years = list(range(config.test_start_year, config.test_end_year + 1))
    for held_out_year in years:
        if config.run_mode == "resume" and held_out_year in completed_years:
            continue
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] START variant={variant} held_out_year={held_out_year}",
            flush=True,
        )
        result = evaluate_model_for_year(variant, config, bundle, held_out_year)
        if result is None:
            continue
        _append_csv(variant_dir / "yearly_metrics.csv", [result["year_row"]])
        _append_csv(variant_dir / "month_lead_metrics.csv", result["month_lead_rows"])
        _append_csv(variant_dir / "regional_metrics.csv", result["region_rows"])
        _append_jsonl(variant_dir / "metadata.jsonl", result["metadata"])
        np.savez_compressed(
            variant_dir / "predictions" / f"{held_out_year}.npz",
            preds=result["preds"],
            truths=result["truths"],
        )
        completed_years.add(held_out_year)
        progress = {
            "status": "running",
            "current_variant": variant,
            "completed_years": sorted(completed_years),
            "started_at": progress.get("started_at"),
            "latest_year": held_out_year,
            "latest_rmse": result["year_row"]["rmse"],
        }
        _save_progress(progress_path, progress)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] DONE variant={variant} held_out_year={held_out_year} rmse={result['year_row']['rmse']:.6f}",
            flush=True,
        )

    yearly_df, month_lead_df, regional_df, metadata = _aggregate_variant_outputs(variant_dir)
    if not yearly_df.empty:
        yearly_df.to_csv(variant_dir / f"{variant}_metrics.csv", index=False)
    if not month_lead_df.empty:
        month_lead_df.to_csv(variant_dir / f"{variant}_month_lead_metrics.csv", index=False)
    if not regional_df.empty:
        regional_df.to_csv(variant_dir / f"{variant}_regional_metrics.csv", index=False)
    if metadata:
        _write_json(variant_dir / f"{variant}_metadata.json", metadata)
    final_progress = {
        "status": "completed",
        "current_variant": variant,
        "completed_years": sorted(completed_years),
        "started_at": progress.get("started_at"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_progress(progress_path, final_progress)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE variant={variant}", flush=True)
    return yearly_df


def _run_incremental_single_model(
    config: ExperimentConfig,
    bundle,
    model_name: str,
    out_dir: Path,
    years: list[int],
):
    model_dir = out_dir / model_name
    _prepare_variant_dir(model_dir, config.run_mode)
    progress_path = model_dir / "progress.json"
    progress = _load_progress(progress_path)
    if config.run_mode == "fresh":
        progress = {
            "status": "running",
            "current_model": model_name,
            "completed_years": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
        }
        _save_progress(progress_path, progress)

    completed_years = set(progress.get("completed_years", []))
    for held_out_year in years:
        if config.run_mode == "resume" and held_out_year in completed_years:
            continue
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] START model={model_name} held_out_year={held_out_year}",
            flush=True,
        )
        result = evaluate_model_for_year(model_name, config, bundle, held_out_year)
        if result is None:
            continue
        _append_csv(model_dir / "yearly_metrics.csv", [result["year_row"]])
        _append_csv(model_dir / "month_lead_metrics.csv", result["month_lead_rows"])
        _append_csv(model_dir / "regional_metrics.csv", result["region_rows"])
        _append_jsonl(model_dir / "metadata.jsonl", result["metadata"])
        np.savez_compressed(
            model_dir / "predictions" / f"{held_out_year}.npz",
            preds=result["preds"],
            truths=result["truths"],
        )
        completed_years.add(held_out_year)
        progress = {
            "status": "running",
            "current_model": model_name,
            "completed_years": sorted(completed_years),
            "started_at": progress.get("started_at"),
            "latest_year": held_out_year,
            "latest_rmse": result["year_row"]["rmse"],
        }
        _save_progress(progress_path, progress)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] DONE model={model_name} held_out_year={held_out_year} rmse={result['year_row']['rmse']:.6f}",
            flush=True,
        )

    yearly_df, month_lead_df, regional_df, metadata = _aggregate_variant_outputs(model_dir)
    if not yearly_df.empty:
        yearly_df.to_csv(model_dir / f"{model_name}_metrics.csv", index=False)
    if not month_lead_df.empty:
        month_lead_df.to_csv(model_dir / f"{model_name}_month_lead_metrics.csv", index=False)
    if not regional_df.empty:
        regional_df.to_csv(model_dir / f"{model_name}_regional_metrics.csv", index=False)
    if metadata:
        _write_json(model_dir / f"{model_name}_metadata.json", metadata)
    final_progress = {
        "status": "completed",
        "current_model": model_name,
        "completed_years": sorted(completed_years),
        "started_at": progress.get("started_at"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_progress(progress_path, final_progress)
    summary = _summarize_model_outputs(model_name, years, yearly_df, month_lead_df, regional_df)
    _write_json(model_dir / "summary.json", summary)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE model={model_name}", flush=True)
    return yearly_df, summary


def _run_incremental_seas5_model(
    config: ExperimentConfig,
    bundle,
    out_dir: Path,
    years: list[int],
    model_name: str = "seas5",
):
    model_dir = out_dir / model_name
    _prepare_variant_dir(model_dir, config.run_mode)
    progress_path = model_dir / "progress.json"
    progress = _load_progress(progress_path)
    if config.run_mode == "fresh":
        progress = {
            "status": "running",
            "current_model": model_name,
            "completed_years": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
        }
        _save_progress(progress_path, progress)

    if not config.seas5_processed_path.exists():
        if not (config.seas5_download_manifest_path.exists() and config.seas5_raw_blocks_dir.exists()):
            download_seas5(config)
        process_seas5(config)
    seas5_preds, init_times, lead_months = load_processed_seas5(config)
    bias_fields = None
    if model_name == "seas5_bias_corrected":
        if not config.seas5_bias_correction_path.exists():
            if not config.seas5_hindcast_processed_path.exists():
                manifest_path = config.seas5_hindcast_download_manifest_path
                if not manifest_path.exists():
                    download_seas5_hindcast(config)
                process_seas5_hindcast(config)
            fit_seas5_bias_correction(config)
        bias_fields, _, _ = load_seas5_bias_correction(config)

    completed_years = set(progress.get("completed_years", []))
    for held_out_year in years:
        if config.run_mode == "resume" and held_out_year in completed_years:
            continue
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] START model={model_name} held_out_year={held_out_year}",
            flush=True,
        )
        result = evaluate_seas5_for_year(
            config,
            bundle,
            seas5_preds,
            init_times,
            lead_months,
            held_out_year,
            model_name=model_name,
            bias_fields=bias_fields,
        )
        if result is None:
            continue
        _append_csv(model_dir / "yearly_metrics.csv", [result["year_row"]])
        _append_csv(model_dir / "month_lead_metrics.csv", result["month_lead_rows"])
        _append_csv(model_dir / "regional_metrics.csv", result["region_rows"])
        _append_jsonl(model_dir / "metadata.jsonl", result["metadata"])
        np.savez_compressed(
            model_dir / "predictions" / f"{held_out_year}.npz",
            preds=result["preds"],
            truths=result["truths"],
        )
        completed_years.add(held_out_year)
        progress = {
            "status": "running",
            "current_model": model_name,
            "completed_years": sorted(completed_years),
            "started_at": progress.get("started_at"),
            "latest_year": held_out_year,
            "latest_rmse": result["year_row"]["rmse"],
        }
        _save_progress(progress_path, progress)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] DONE model={model_name} held_out_year={held_out_year} rmse={result['year_row']['rmse']:.6f}",
            flush=True,
        )

    yearly_df, month_lead_df, regional_df, metadata = _aggregate_variant_outputs(model_dir)
    if not yearly_df.empty:
        yearly_df.to_csv(model_dir / f"{model_name}_metrics.csv", index=False)
    if not month_lead_df.empty:
        month_lead_df.to_csv(model_dir / f"{model_name}_month_lead_metrics.csv", index=False)
    if not regional_df.empty:
        regional_df.to_csv(model_dir / f"{model_name}_regional_metrics.csv", index=False)
    if metadata:
        _write_json(model_dir / f"{model_name}_metadata.json", metadata)
    final_progress = {
        "status": "completed",
        "current_model": model_name,
        "completed_years": sorted(completed_years),
        "started_at": progress.get("started_at"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_progress(progress_path, final_progress)
    summary = _summarize_model_outputs(model_name, years, yearly_df, month_lead_df, regional_df)
    _write_json(model_dir / "summary.json", summary)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE model={model_name}", flush=True)
    return yearly_df, summary


def _collect_compare_outputs(compare_dir: Path, models: list[str]):
    summary_rows = []
    lead_rows = []
    month_rows = []
    region_rows = []
    for model_name in models:
        model_dir = compare_dir / model_name
        yearly_df, month_lead_df, regional_df, _ = _aggregate_variant_outputs(model_dir)
        if yearly_df.empty:
            continue
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary_rows.append(json.load(f))
        else:
            summary_rows.append(_summarize_model_outputs(model_name, yearly_df["held_out_year"].tolist(), yearly_df, month_lead_df, regional_df))
        if not month_lead_df.empty:
            lead_summary = month_lead_df.groupby("lead", as_index=False)[["rmse", "mae", "acc", "iiee"]].mean()
            lead_summary.insert(0, "model", model_name)
            lead_rows.append(lead_summary)

            month_summary = month_lead_df.groupby("target_month", as_index=False)[["rmse", "mae", "iiee"]].mean()
            month_summary.insert(0, "model", model_name)
            month_rows.append(month_summary)
        if not regional_df.empty:
            regional_summary = regional_df.groupby(["region_id", "region_name"], as_index=False)["rmse"].mean()
            regional_summary.insert(0, "model", model_name)
            region_rows.append(regional_summary)

    summary_df = pd.DataFrame(summary_rows)
    lead_df = pd.concat(lead_rows, ignore_index=True) if lead_rows else pd.DataFrame()
    month_df = pd.concat(month_rows, ignore_index=True) if month_rows else pd.DataFrame()
    region_df = pd.concat(region_rows, ignore_index=True) if region_rows else pd.DataFrame()
    return summary_df, lead_df, month_df, region_df


def _write_compare_outputs(compare_dir: Path, models: list[str]):
    summary_df, lead_df, month_df, region_df = _collect_compare_outputs(compare_dir, models)
    if not summary_df.empty:
        summary_df.to_csv(compare_dir / "pilot_compare_summary.csv", index=False)
    if not lead_df.empty:
        lead_df.to_csv(compare_dir / "pilot_compare_lead_summary.csv", index=False)
    if not month_df.empty:
        month_df.to_csv(compare_dir / "pilot_compare_month_summary.csv", index=False)
    if not region_df.empty:
        region_df.to_csv(compare_dir / "pilot_compare_region_summary.csv", index=False)

    pairwise_rows = []
    if not summary_df.empty and "icecbr_full" in set(summary_df["model"]):
        target = summary_df.loc[summary_df["model"] == "icecbr_full"].iloc[0]
        for _, row in summary_df.iterrows():
            if row["model"] == "icecbr_full":
                continue
            pairwise_rows.append(
                {
                    "baseline_model": row["model"],
                    "target_model": "icecbr_full",
                    "delta_rmse": float(row["rmse_mean"] - target["rmse_mean"]),
                    "delta_mae": float(row["mae_mean"] - target["mae_mean"]),
                    "delta_acc": float(row["acc_mean"] - target["acc_mean"]),
                    "delta_bss": float(row["bss_mean"] - target["bss_mean"]),
                    "delta_sie_abs_error": float(row["sie_abs_error_mean"] - target["sie_abs_error_mean"]),
                    "baseline_beats_icecbr_rmse": bool(row["rmse_mean"] < target["rmse_mean"]),
                    "baseline_beats_icecbr_acc": bool(row["acc_mean"] > target["acc_mean"]),
                    "baseline_beats_icecbr_bss": bool(row["bss_mean"] > target["bss_mean"]),
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df.to_csv(compare_dir / "pilot_compare_pairwise_vs_icecbr.csv", index=False)
    return summary_df


def _write_forecast_compare_outputs(compare_dir: Path, models: list[str]):
    month_lead_frames = []
    yearly_frames = []
    region_frames = []
    for model_name in models:
        model_dir = compare_dir / model_name
        _, month_lead_df, regional_df, _ = _aggregate_variant_outputs(model_dir)
        yearly_path = model_dir / "yearly_metrics.csv"
        yearly_df = pd.read_csv(yearly_path) if yearly_path.exists() else pd.DataFrame()
        if not month_lead_df.empty:
            month_lead_frames.append(month_lead_df.copy())
        if not yearly_df.empty:
            yearly_frames.append(yearly_df.copy())
        if not regional_df.empty:
            region_frames.append(regional_df.copy())

    if not month_lead_frames:
        return pd.DataFrame()

    all_month_lead = pd.concat(month_lead_frames, ignore_index=True)
    valid_rows = all_month_lead[all_month_lead["rmse"].notna()].copy()
    key_cols = ["issue_year", "issue_month", "lead"]
    common_keys = (
        valid_rows.groupby(key_cols)["model"].nunique().reset_index(name="n_models")
    )
    common_keys = common_keys[common_keys["n_models"] == len(models)][key_cols]
    filtered_month_lead = valid_rows.merge(common_keys, on=key_cols, how="inner")

    summary_df = (
        filtered_month_lead.groupby("model", as_index=False)[["rmse", "mae", "acc", "iiee"]].mean()
        .rename(columns={"rmse": "rmse_mean", "mae": "mae_mean", "acc": "acc_mean", "iiee": "iiee_mean"})
    )
    yearly_all = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    extra_summary = (
        yearly_all.groupby("model", as_index=False)[["sie_abs_error"]].mean().rename(columns={"sie_abs_error": "sie_abs_error_mean"})
        if not yearly_all.empty and "sie_abs_error" in yearly_all.columns
        else pd.DataFrame(columns=["model", "sie_abs_error_mean"])
    )
    bss_summary = (
        yearly_all.groupby("model", as_index=False)[["bss"]].mean().rename(columns={"bss": "bss_mean"})
        if not yearly_all.empty and "bss" in yearly_all.columns
        else pd.DataFrame(columns=["model", "bss_mean"])
    )
    count_summary = (
        filtered_month_lead.groupby("model", as_index=False).size().rename(columns={"size": "n_common_samples"})
    )
    summary_df = summary_df.merge(extra_summary, on="model", how="left").merge(bss_summary, on="model", how="left").merge(count_summary, on="model")

    lead_df = filtered_month_lead.groupby(["model", "lead"], as_index=False)[["rmse", "mae", "acc", "iiee"]].mean()
    lead_counts = filtered_month_lead.groupby(["model", "lead"], as_index=False).size().rename(columns={"size": "n_samples"})
    lead_df = lead_df.merge(lead_counts, on=["model", "lead"])

    month_df = filtered_month_lead.groupby(["model", "target_month"], as_index=False)[["rmse", "mae", "iiee"]].mean()
    year_df = filtered_month_lead.groupby(["model", "held_out_year"], as_index=False)[["rmse", "mae", "acc", "iiee"]].mean()
    year_counts = filtered_month_lead.groupby(["model", "held_out_year"], as_index=False).size().rename(columns={"size": "n_samples"})
    year_df = year_df.merge(year_counts, on=["model", "held_out_year"])
    if not yearly_all.empty and {"model", "held_out_year", "bss", "sie_abs_error"}.issubset(yearly_all.columns):
        year_extras = yearly_all[["model", "held_out_year", "bss", "sie_abs_error"]]
        year_df = year_df.merge(year_extras, on=["model", "held_out_year"], how="left")

    common_keys["issue_ym"] = common_keys["issue_year"].astype(str) + "-" + common_keys["issue_month"].map(lambda m: f"{int(m):02d}")
    issue_dates = pd.to_datetime(
        common_keys["issue_year"].astype(str) + "-" + common_keys["issue_month"].map(lambda m: f"{int(m):02d}") + "-01"
    )
    common_keys["target_date"] = pd.to_datetime(
        [issue_date + pd.DateOffset(months=int(lead)) for issue_date, lead in zip(issue_dates, common_keys["lead"])]
    )
    common_keys["target_year"] = common_keys["target_date"].dt.year
    common_keys["target_month"] = common_keys["target_date"].dt.month
    common_keys["target_ym"] = common_keys["target_date"].dt.strftime("%Y-%m")
    common_keys = common_keys.drop(columns=["target_date"])

    common_keys.to_csv(compare_dir / "common_valid_samples.csv", index=False)
    filtered_month_lead.to_csv(compare_dir / "forecast_compare_month_lead_metrics.csv", index=False)
    summary_df.to_csv(compare_dir / "forecast_compare_summary.csv", index=False)
    lead_df.to_csv(compare_dir / "forecast_compare_lead_summary.csv", index=False)
    month_df.to_csv(compare_dir / "forecast_compare_month_summary.csv", index=False)
    year_df.to_csv(compare_dir / "forecast_compare_year_summary.csv", index=False)

    if year_df.empty:
        region_summary = pd.DataFrame()
    else:
        region_summary = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()
        if not region_summary.empty:
            region_summary = region_summary[region_summary["held_out_year"].isin(year_df["held_out_year"].unique())]
            region_summary = region_summary.groupby(["model", "region_id", "region_name"], as_index=False)[["rmse"]].mean()
            region_summary.to_csv(compare_dir / "forecast_compare_region_summary.csv", index=False)

    pairwise_rows = []
    model_set = set(summary_df["model"]) if not summary_df.empty else set()
    summary_lookup = {row["model"]: row for _, row in summary_df.iterrows()} if not summary_df.empty else {}
    compare_pairs = [("seas5", "a0"), ("seas5_bias_corrected", "seas5"), ("seas5_bias_corrected", "a0")]
    for lhs, rhs in compare_pairs:
        if lhs not in summary_lookup or rhs not in summary_lookup:
            continue
        left = summary_lookup[lhs]
        right = summary_lookup[rhs]
        pairwise_rows.append(
            {
                "lhs": lhs,
                "rhs": rhs,
                "delta_rmse": float(left["rmse_mean"] - right["rmse_mean"]),
                "delta_mae": float(left["mae_mean"] - right["mae_mean"]),
                "delta_acc": float(left["acc_mean"] - right["acc_mean"]),
                "delta_bss": float(left["bss_mean"] - right["bss_mean"]),
                "delta_sie_abs_error": float(left["sie_abs_error_mean"] - right["sie_abs_error_mean"]),
                "lhs_beats_rhs_rmse": bool(left["rmse_mean"] < right["rmse_mean"]),
            }
        )
    if pairwise_rows:
        pd.DataFrame(pairwise_rows).to_csv(compare_dir / "forecast_compare_pairwise.csv", index=False)
    return summary_df


def run_e2(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E2")
    variants = ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config)
    results = {}
    for name in variants:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START variant={name}", flush=True)
        results[name] = _run_e2_incremental_variant(config, bundle, name, out_dir)
    rows = []
    comparisons = []
    base_mean = results["v0"]["rmse"].mean()
    for idx, name in enumerate(variants):
        df = results[name]
        sig = {"test": "none", "stat": np.nan, "p_value": np.nan}
        if idx > 0:
            prev = variants[idx - 1]
            sig = paired_significance(results[prev]["rmse"].to_numpy(), df["rmse"].to_numpy())
            comparisons.append(
                {
                    "from_variant": prev,
                    "to_variant": name,
                    "delta_rmse": float(df["rmse"].mean() - results[prev]["rmse"].mean()),
                    "delta_acc": float(df["acc"].mean() - results[prev]["acc"].mean()),
                    "test": sig["test"],
                    "p_value": sig["p_value"],
                }
            )
        rows.append(
            {
                "variant": name,
                "rmse": float(df["rmse"].mean()),
                "delta_vs_v0_pct": float((df["rmse"].mean() - base_mean) / base_mean * 100.0),
                "test_vs_previous": sig["test"],
                "p_value_vs_previous": sig["p_value"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "E2_ablation.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "E2_adjacent_variant_tests.csv", index=False)
    return out


def run_v1_anchor_ablation(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "V1_anchor_ablation")
    variants = ["a0", "a1", "a2", "a3", "a4"]
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config)
    results = {}
    for name in variants:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START variant={name}", flush=True)
        results[name] = _run_e2_incremental_variant(config, bundle, name, out_dir)
    rows = []
    comparisons = []
    anchor_mean = results["a0"]["rmse"].mean()
    for idx, name in enumerate(variants):
        df = results[name]
        sig = {"test": "none", "stat": np.nan, "p_value": np.nan}
        if idx > 0:
            sig = paired_significance(results["a0"]["rmse"].to_numpy(), df["rmse"].to_numpy())
            comparisons.append(
                {
                    "anchor_variant": "a0",
                    "to_variant": name,
                    "delta_rmse_vs_a0": float(df["rmse"].mean() - results["a0"]["rmse"].mean()),
                    "delta_acc_vs_a0": float(df["acc"].mean() - results["a0"]["acc"].mean()),
                    "test": sig["test"],
                    "p_value": sig["p_value"],
                }
            )
        rows.append(
            {
                "variant": name,
                "rmse": float(df["rmse"].mean()),
                "delta_vs_a0_pct": float((df["rmse"].mean() - anchor_mean) / anchor_mean * 100.0),
                "test_vs_a0": sig["test"],
                "p_value_vs_a0": sig["p_value"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "V1_anchor_ablation.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "V1_anchor_tests.csv", index=False)
    return out


def run_agr_region_merge(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "AGR_region_merge")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    original_scheme = config.region_scheme
    schemes = list(config.region_merge_schemes)
    years = list(range(config.test_start_year, config.test_end_year + 1))
    scheme_rows = []

    for scheme in schemes:
        config.region_scheme = scheme
        scheme_dir = out_dir / scheme
        scheme_dir.mkdir(parents=True, exist_ok=True)
        bundle = load_bundle(config)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] START agr_region_merge scheme={scheme} n_regions={len(bundle.region_info.region_ids)}",
            flush=True,
        )
        for model_name in ["a0", "a1"]:
            _run_incremental_single_model(config, bundle, model_name, scheme_dir, years)

        a0_df = pd.read_csv(scheme_dir / "a0" / "yearly_metrics.csv")
        a1_df = pd.read_csv(scheme_dir / "a1" / "yearly_metrics.csv")
        a0_row = a0_df.iloc[0]
        a1_row = a1_df.iloc[0]
        scheme_rows.append(
            {
                "scheme": scheme,
                "n_regions": len(bundle.region_info.region_ids),
                "a0_rmse": float(a0_row["rmse"]),
                "a1_rmse": float(a1_row["rmse"]),
                "delta_rmse": float(a1_row["rmse"] - a0_row["rmse"]),
                "a0_acc": float(a0_row["acc"]),
                "a1_acc": float(a1_row["acc"]),
                "delta_acc": float(a1_row["acc"] - a0_row["acc"]),
                "a0_bss": float(a0_row["bss"]),
                "a1_bss": float(a1_row["bss"]),
                "delta_bss": float(a1_row["bss"] - a0_row["bss"]),
                "a0_sie_abs_error": float(a0_row["sie_abs_error"]),
                "a1_sie_abs_error": float(a1_row["sie_abs_error"]),
                "delta_sie_abs_error": float(a1_row["sie_abs_error"] - a0_row["sie_abs_error"]),
            }
        )

        a0_ml = pd.read_csv(scheme_dir / "a0" / "month_lead_metrics.csv")
        a1_ml = pd.read_csv(scheme_dir / "a1" / "month_lead_metrics.csv")
        ml = a0_ml.merge(a1_ml, on=["held_out_year", "lead", "target_month"], suffixes=("_a0", "_a1"))
        ml_summary = ml[["held_out_year", "lead", "target_month"]].copy()
        ml_summary["scheme"] = scheme
        ml_summary["delta_rmse"] = ml["rmse_a1"] - ml["rmse_a0"]
        ml_summary["delta_acc"] = ml["acc_a1"] - ml["acc_a0"]
        ml_summary.to_csv(scheme_dir / "AGR_vs_anchor_month_lead.csv", index=False)

        a0_reg = pd.read_csv(scheme_dir / "a0" / "regional_metrics.csv")
        a1_reg = pd.read_csv(scheme_dir / "a1" / "regional_metrics.csv")
        reg = a0_reg.merge(a1_reg, on=["held_out_year", "region_id", "region_name"], suffixes=("_a0", "_a1"))
        reg_summary = reg[["held_out_year", "region_id", "region_name"]].copy()
        reg_summary["scheme"] = scheme
        reg_summary["delta_rmse"] = reg["rmse_a1"] - reg["rmse_a0"]
        reg_summary.to_csv(scheme_dir / "AGR_vs_anchor_regions.csv", index=False)

    config.region_scheme = original_scheme
    out = pd.DataFrame(scheme_rows)
    out.to_csv(out_dir / "region_merge_agr_summary.csv", index=False)
    return out


def run_finalist_compare(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "finalist_compare")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    original_scheme = config.region_scheme
    years = list(range(config.test_start_year, config.test_end_year + 1))
    candidate_rows = []
    lead_frames = []
    month_frames = []
    region_frames = []

    for candidate_name, scheme, model_name in config.finalist_candidates:
        config.region_scheme = scheme
        bundle = load_bundle(config)
        candidate_dir = out_dir / candidate_name
        yearly_df, summary = _run_incremental_single_model(config, bundle, model_name, candidate_dir, years)
        month_lead_df = pd.read_csv(candidate_dir / model_name / "month_lead_metrics.csv")
        regional_df = pd.read_csv(candidate_dir / model_name / "regional_metrics.csv")

        candidate_rows.append(
            {
                "candidate_name": candidate_name,
                "region_scheme": scheme,
                "model": model_name,
                "n_regions": len(bundle.region_info.region_ids),
                "rmse_mean": float(yearly_df["rmse"].mean()),
                "mae_mean": float(yearly_df["mae"].mean()),
                "acc_mean": float(yearly_df["acc"].mean()),
                "bss_mean": float(yearly_df["bss"].mean()),
                "sie_abs_error_mean": float(yearly_df["sie_abs_error"].mean()),
                "iiee_mean": float(yearly_df["iiee"].mean()),
            }
        )

        lead_summary = month_lead_df.groupby("lead", as_index=False)[["rmse", "mae", "acc", "iiee"]].mean()
        lead_summary.insert(0, "candidate_name", candidate_name)
        lead_summary.insert(1, "region_scheme", scheme)
        lead_summary.insert(2, "model", model_name)
        lead_frames.append(lead_summary)

        month_summary = month_lead_df.groupby("target_month", as_index=False)[["rmse", "mae", "iiee"]].mean()
        month_summary.insert(0, "candidate_name", candidate_name)
        month_summary.insert(1, "region_scheme", scheme)
        month_summary.insert(2, "model", model_name)
        month_frames.append(month_summary)

        regional_summary = regional_df.groupby(["region_id", "region_name"], as_index=False)["rmse"].mean()
        regional_summary.insert(0, "candidate_name", candidate_name)
        regional_summary.insert(1, "region_scheme", scheme)
        regional_summary.insert(2, "model", model_name)
        region_frames.append(regional_summary)

        _write_json(candidate_dir / "candidate_summary.json", summary)

    config.region_scheme = original_scheme

    summary_df = pd.DataFrame(candidate_rows)
    lead_df = pd.concat(lead_frames, ignore_index=True) if lead_frames else pd.DataFrame()
    month_df = pd.concat(month_frames, ignore_index=True) if month_frames else pd.DataFrame()
    region_df = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()

    summary_df.to_csv(out_dir / "finalist_summary.csv", index=False)
    if not lead_df.empty:
        lead_df.to_csv(out_dir / "finalist_lead_summary.csv", index=False)
    if not month_df.empty:
        month_df.to_csv(out_dir / "finalist_month_summary.csv", index=False)
    if not region_df.empty:
        region_df.to_csv(out_dir / "finalist_region_summary.csv", index=False)

    pairwise_rows = []
    lookup = {row["candidate_name"]: row for row in candidate_rows}
    pairs = [
        ("merge6_a0", "nsidc18_a0"),
        ("merge6_a1", "merge6_a0"),
        ("merge6_a1", "nsidc18_a0"),
    ]
    for lhs, rhs in pairs:
        if lhs not in lookup or rhs not in lookup:
            continue
        left = lookup[lhs]
        right = lookup[rhs]
        pairwise_rows.append(
            {
                "lhs_candidate": lhs,
                "rhs_candidate": rhs,
                "delta_rmse": float(left["rmse_mean"] - right["rmse_mean"]),
                "delta_mae": float(left["mae_mean"] - right["mae_mean"]),
                "delta_acc": float(left["acc_mean"] - right["acc_mean"]),
                "delta_bss": float(left["bss_mean"] - right["bss_mean"]),
                "delta_sie_abs_error": float(left["sie_abs_error_mean"] - right["sie_abs_error_mean"]),
                "delta_iiee": float(left["iiee_mean"] - right["iiee_mean"]),
            }
        )
    pd.DataFrame(pairwise_rows).to_csv(out_dir / "finalist_pairwise.csv", index=False)
    return summary_df


def run_pilot_full(config: ExperimentConfig):
    original_start = config.test_start_year
    original_end = config.test_end_year
    config.test_start_year = config.pilot_start_year
    config.test_end_year = config.pilot_end_year
    if config.output_tag is None:
        config.output_tag = f"pilot_full_{config.pilot_start_year}_{config.pilot_end_year}"
    out_dir = _experiment_dir(config, "pilot_full")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config)
    years = list(range(config.test_start_year, config.test_end_year + 1))
    yearly_df, summary = _run_incremental_single_model(config, bundle, "icecbr_full", out_dir, years)
    yearly_df.to_csv(out_dir / "pilot_full_yearly_metrics.csv", index=False)
    _write_json(out_dir / "pilot_full_summary.json", summary)
    config.test_start_year = original_start
    config.test_end_year = original_end
    return yearly_df


def run_pilot_compare(config: ExperimentConfig):
    original_start = config.test_start_year
    original_end = config.test_end_year
    config.test_start_year = config.compare_start_year
    config.test_end_year = config.compare_end_year
    if config.output_tag is None:
        config.output_tag = f"pilot_compare_{config.compare_start_year}_{config.compare_end_year}"
    out_dir = _experiment_dir(config, "pilot_compare")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config)
    years = list(range(config.test_start_year, config.test_end_year + 1))
    models = list(config.compare_models)
    for model_name in models:
        _run_incremental_single_model(config, bundle, model_name, out_dir, years)
    summary_df = _write_compare_outputs(out_dir, models)
    config.test_start_year = original_start
    config.test_end_year = original_end
    return summary_df


def run_forecast_compare(config: ExperimentConfig):
    original_start = config.test_start_year
    original_end = config.test_end_year
    config.test_start_year = config.forecast_compare_start_year
    config.test_end_year = config.forecast_compare_end_year
    if config.output_tag is None:
        config.output_tag = f"forecast_compare_{config.forecast_compare_start_year}_{config.forecast_compare_end_year}"
    out_dir = _experiment_dir(config, "forecast_compare")
    if config.run_mode == "fresh" and out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config)
    years = list(range(config.test_start_year, config.test_end_year + 1))
    models = list(config.forecast_compare_models)
    for model_name in models:
        if model_name in {"seas5", "seas5_bias_corrected"}:
            _run_incremental_seas5_model(config, bundle, out_dir, years, model_name=model_name)
        else:
            _run_incremental_single_model(config, bundle, model_name, out_dir, years)
    summary_df = _write_forecast_compare_outputs(out_dir, models)
    config.test_start_year = original_start
    config.test_end_year = original_end
    return summary_df


def run_e3(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E3")
    evaluate_model_over_years("v2", config, output_dir=out_dir / "v2")
    evaluate_model_over_years("v3", config, output_dir=out_dir / "v3")
    with open(out_dir / "v2" / "v2_metadata.json", "r") as f:
        std_meta = json.load(f)
    with open(out_dir / "v3" / "v3_metadata.json", "r") as f:
        agr_meta = json.load(f)

    def collect(meta, label):
        rows = []
        for item in meta:
            payload = item["metadata"]
            neighbors = payload.get("global_neighbors", [])
            if not neighbors:
                continue
            years = [n["issue_year"] for n in neighbors]
            sims = [n["similarity"] for n in neighbors]
            adapts = [n["adaptability"] for n in neighbors]
            costs = [n["adaptation_parts"].get("cost", 0.0) for n in neighbors]
            rows.append(
                {
                    "mode": label,
                    "held_out_year": item["held_out_year"],
                    "issue_month": item["issue_month"],
                    "lead": item["lead"],
                    "avg_temporal_distance": float(np.mean([abs(item["held_out_year"] - y) for y in years])),
                    "avg_similarity": float(np.mean(sims)),
                    "avg_adaptability": float(np.mean(adapts)),
                    "avg_adaptation_cost": float(np.mean(costs)),
                }
            )
        return rows

    diag = pd.DataFrame(collect(std_meta, "standard") + collect(agr_meta, "agr"))
    diag.to_csv(out_dir / "E3_retrieval_diagnostics.csv", index=False)

    std_df = pd.read_csv(out_dir / "v2" / "v2_metrics.csv")
    agr_df = pd.read_csv(out_dir / "v3" / "v3_metrics.csv")
    comparison = pd.DataFrame(
        {
            "held_out_year": std_df["held_out_year"],
            "rmse_standard": std_df["rmse"],
            "rmse_agr": agr_df["rmse"],
            "improvement_pct": (std_df["rmse"] - agr_df["rmse"]) / std_df["rmse"] * 100.0,
        }
    )
    comparison.to_csv(out_dir / "E3_agr_summary.csv", index=False)
    return comparison


def run_e4(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E4")
    evaluate_model_over_years("v2", config, output_dir=out_dir / "v2")
    evaluate_model_over_years("v4", config, output_dir=out_dir / "v4")
    global_df = pd.read_csv(out_dir / "v2" / "v2_metrics.csv")
    csa_df = pd.read_csv(out_dir / "v4" / "v4_metrics.csv")
    out = pd.DataFrame(
        {
            "held_out_year": global_df["held_out_year"],
            "rmse_global": global_df["rmse"],
            "rmse_csa": csa_df["rmse"],
            "delta_pct": (global_df["rmse"] - csa_df["rmse"]) / global_df["rmse"] * 100.0,
        }
    )
    out.to_csv(out_dir / "E4_csa_summary.csv", index=False)

    with open(out_dir / "v4" / "v4_metadata.json", "r") as f:
        meta = json.load(f)
    region_rows = []
    boundary_rows = []
    for item in meta:
        region_sets = item["metadata"].get("regional_case_sets", {})
        region_ids = sorted(region_sets)
        for i, rid_a in enumerate(region_ids):
            set_a = {tuple(x) for x in region_sets[rid_a]}
            for rid_b in region_ids[i + 1 :]:
                set_b = {tuple(x) for x in region_sets[rid_b]}
                union = set_a | set_b
                jaccard = len(set_a & set_b) / len(union) if union else np.nan
                region_rows.append(
                    {
                        "held_out_year": item["held_out_year"],
                        "issue_month": item["issue_month"],
                        "lead": item["lead"],
                        "region_a": rid_a,
                        "region_b": rid_b,
                        "jaccard": jaccard,
                    }
                )
        pred_path = out_dir / "v4" / "v4_preds.npy"
        truth_path = out_dir / "v4" / "v4_truths.npy"
        if pred_path.exists() and truth_path.exists():
            preds = np.load(pred_path)
            truths = np.load(truth_path)
            grad_pred = np.nanmean(np.abs(np.diff(preds, axis=1))) + np.nanmean(np.abs(np.diff(preds, axis=2)))
            grad_truth = np.nanmean(np.abs(np.diff(truths, axis=1))) + np.nanmean(np.abs(np.diff(truths, axis=2)))
            boundary_rows.append({"gradient_pred": float(grad_pred), "gradient_truth": float(grad_truth)})
            break
    pd.DataFrame(region_rows).to_csv(out_dir / "E4_regional_jaccard.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(out_dir / "E4_boundary_smoothness.csv", index=False)
    return out


def run_e5(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E5")
    evaluate_model_over_years("v6", config, output_dir=out_dir / "v6")
    with open(out_dir / "v6" / "v6_metadata.json", "r") as f:
        meta = json.load(f)
    reachability = defaultdict(int)
    case_ages = defaultdict(list)
    for item in meta:
        for cid in _flatten_case_ids(item["metadata"]):
            reachability[str(cid)] += 1
            case_ages[str(cid)].append(item["issue_index"] - cid[0])
    competence = pd.DataFrame(
        {
            "case_id": list(reachability.keys()),
            "reachability": list(reachability.values()),
            "avg_case_age_months": [float(np.mean(case_ages[key])) for key in reachability],
        }
    )
    competence.to_csv(out_dir / "E5_case_competence.csv", index=False)

    size_rows = []
    original_start = config.test_start_year
    probe_year = max(config.test_start_year, 2015)
    for n_years in (10, 15, 20, 25, 30):
        start_year = max(config.start_year + n_years, probe_year)
        config.test_start_year = start_year
        probe = evaluate_model_over_years("v6", config, years=[probe_year], output_dir=out_dir / f"window_{n_years}")
        size_rows.append({"case_base_years": n_years, "probe_year": probe_year, "rmse": float(probe["rmse"].mean())})
    config.test_start_year = original_start
    pd.DataFrame(size_rows).to_csv(out_dir / "E5_case_base_size.csv", index=False)
    return competence


def run_e6(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E6")
    years = [2007, 2012, 2013, 2020]
    evaluate_model_over_years("v6", config, years=years, output_dir=out_dir / "v6")
    with open(out_dir / "v6" / "v6_metadata.json", "r") as f:
        meta = json.load(f)
    target_specs = {(2012, 6, 3), (2013, 6, 3), (2020, 1, 6), (2007, 6, 3)}
    out = []
    for item in meta:
        key = (item["held_out_year"], item["issue_month"], item["lead"])
        if key in target_specs:
            out.append(
                {
                    "query": key,
                    "retrieval_mode": item["metadata"].get("retrieval_mode"),
                    "adaptation_mode": item["metadata"].get("adaptation_mode"),
                    "regional": item["metadata"].get("regional", {}),
                    "regional_case_sets": item["metadata"].get("regional_case_sets", {}),
                }
            )
    _write_json(out_dir / "E6_case_studies.json", out)
    return out


def run_e7(config: ExperimentConfig):
    out_dir = _experiment_dir(config, "E7")
    rows = []
    original_k = config.top_k
    original_rho = config.retrieval_rho
    original_dim = config.sic_eof_dim
    probe_year = config.test_start_year
    for k in config.top_k_grid:
        for rho in config.rho_grid:
            for eof_dim in config.sic_eof_grid[:3]:
                config.top_k = int(k)
                config.retrieval_rho = float(rho)
                config.sic_eof_dim = int(eof_dim)
                df = evaluate_model_over_years("v6", config, years=[probe_year], output_dir=out_dir / f"k{k}_r{rho}_e{eof_dim}")
                rows.append(
                    {
                        "top_k": k,
                        "retrieval_rho": rho,
                        "sic_eof_dim": eof_dim,
                        "probe_year": probe_year,
                        "rmse": float(df["rmse"].mean()),
                    }
                )
    config.top_k = original_k
    config.retrieval_rho = original_rho
    config.sic_eof_dim = original_dim
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "E7_sensitivity.csv", index=False)
    return out


def run_smoke_test(config: ExperimentConfig):
    config.ensure_dirs()
    years = list(config.smoke_test_years)
    original_issue_months = config.issue_months
    original_lead_times = config.lead_times
    config.issue_months = tuple(config.smoke_issue_months)
    config.lead_times = tuple(config.smoke_lead_times)
    out_dir = _experiment_dir(config, "smoke", smoke=True)
    df = evaluate_model_over_years("v0", config, years=years, output_dir=out_dir)
    summary = {
        "smoke_only": True,
        "held_out_years": years,
        "issue_months": list(config.issue_months),
        "lead_times": list(config.lead_times),
        "rows": int(df.shape[0]),
        "rmse_mean": float(df["rmse"].mean()),
        "bootstrap": block_bootstrap_ci(df["rmse"].tolist(), n_boot=100),
    }
    _write_json(out_dir / "smoke_test_summary.json", summary)
    config.issue_months = original_issue_months
    config.lead_times = original_lead_times
    return summary
