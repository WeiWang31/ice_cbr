from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .config import ExperimentConfig
from .data import DatasetBundle, load_bundle
from .metrics import (
    brier_skill_score,
    calculate_acc,
    month_lead_iiee,
    month_lead_mae_mse,
    regional_rmse,
    sie_absolute_error,
)


def _lazy_cdsapi():
    import cdsapi

    return cdsapi


def _lazy_xarray():
    import xarray as xr

    return xr


def _lazy_pyproj():
    from pyproj import Transformer

    return Transformer


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _write_json(path: Path, payload):
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _month_span(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year = start_year
    month = start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def build_seas5_request(
    config: ExperimentConfig,
    years: list[int] | None = None,
    months: list[int] | None = None,
    lead_months: list[int] | None = None,
    system: str | None = None,
) -> dict:
    return {
        "originating_centre": config.seas5_originating_centre,
        "system": system or config.seas5_system,
        "variable": "sea_ice_cover",
        "product_type": "monthly_mean",
        "year": [str(y) for y in (years or list(range(config.seas5_start_year, config.seas5_end_year + 1)))],
        "month": [f"{m:02d}" for m in (months or list(range(1, 13)))],
        "leadtime_month": [str(lead) for lead in (lead_months or list(config.lead_times))],
        "data_format": "grib",
        "download_format": "unarchived",
    }


def _system_for_month(config: ExperimentConfig, year: int, month: int) -> str:
    if (year, month) >= (config.seas5_system_transition_year, config.seas5_system_transition_month):
        return config.seas5_system_post_transition
    return config.seas5_system


def _year_block_path(config: ExperimentConfig, year: int) -> Path:
    return config.seas5_raw_blocks_dir / f"seas5_{year}_6lead_monthly_sic.grib"


def _patch_block_path(config: ExperimentConfig, start_year: int, start_month: int, end_year: int, end_month: int) -> Path:
    return config.seas5_raw_blocks_dir / (
        f"seas5_{start_year}_{start_month:02d}_{end_year}_{end_month:02d}_6lead_monthly_sic.grib"
    )


def _open_block_dataarray(path: Path):
    xr = _lazy_xarray()
    ds = xr.open_dataset(path, engine="cfgrib")
    return _canonicalize_forecast_dataset(ds)


def _block_months(path: Path) -> list[tuple[int, int]]:
    da = _open_block_dataarray(path)
    times = pd.to_datetime(da["init_time"].values)
    return [(int(ts.year), int(ts.month)) for ts in times]


def _collect_existing_blocks(config: ExperimentConfig) -> list[dict]:
    blocks = []
    for path in sorted(config.seas5_raw_blocks_dir.glob("*.grib")):
        months = _block_months(path)
        if not months:
            continue
        blocks.append(
            {
                "path": path,
                "months": months,
                "start_year": months[0][0],
                "start_month": months[0][1],
                "end_year": months[-1][0],
                "end_month": months[-1][1],
            }
        )
    return blocks


def _group_contiguous_months(months: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if not months:
        return []
    groups = [[months[0]]]
    for year, month in months[1:]:
        prev_year, prev_month = groups[-1][-1]
        expected_year = prev_year + 1 if prev_month == 12 else prev_year
        expected_month = 1 if prev_month == 12 else prev_month + 1
        if (year, month) == (expected_year, expected_month):
            groups[-1].append((year, month))
        else:
            groups.append([(year, month)])
    return groups


def _download_plan(config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    target_months = _month_span(config.seas5_start_year, 1, config.seas5_end_year, 12)
    existing_blocks = _collect_existing_blocks(config)
    existing_months = {month for block in existing_blocks for month in block["months"]}
    missing_months = [month for month in target_months if month not in existing_months]
    missing_groups = []
    months_by_year: dict[int, list[tuple[int, int]]] = {}
    for year, month in missing_months:
        months_by_year.setdefault(year, []).append((year, month))
    for year in sorted(months_by_year):
        missing_groups.extend(_group_contiguous_months(months_by_year[year]))
    plan = []
    for group in missing_groups:
        start_year, start_month = group[0]
        end_year, end_month = group[-1]
        grouped_by_year: dict[int, list[int]] = {}
        for year, month in group:
            grouped_by_year.setdefault(year, []).append(month)
        if len(grouped_by_year) == 1 and list(grouped_by_year.values())[0] == list(range(1, 13)):
            path = _year_block_path(config, start_year)
        else:
            path = _patch_block_path(config, start_year, start_month, end_year, end_month)
        plan.append(
            {
                "start_year": start_year,
                "start_month": start_month,
                "end_year": end_year,
                "end_month": end_month,
                "years": [start_year],
                "months_by_year": grouped_by_year,
                "system": _system_for_month(config, start_year, start_month),
                "path": path,
            }
        )
    return plan, existing_blocks


def _load_download_manifest(config: ExperimentConfig) -> dict | None:
    path = config.seas5_download_manifest_path
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def download_seas5(config: ExperimentConfig) -> Path:
    config.ensure_dirs()
    plan, existing_blocks = _download_plan(config)
    print(
        f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] START seas5 patch_download fixed_range={config.seas5_start_year}-01_to_{config.seas5_end_year}-12 lead=1-6 existing_blocks={len(existing_blocks)} missing_segments={len(plan)}",
        flush=True,
    )
    cdsapi = _lazy_cdsapi()
    client = cdsapi.Client()

    downloaded_blocks = [
        {
            "path": block["path"],
            "start_year": block["start_year"],
            "start_month": block["start_month"],
            "end_year": block["end_year"],
            "end_month": block["end_month"],
            "system": _system_for_month(config, block["start_year"], block["start_month"]),
            "status": "reused",
        }
        for block in existing_blocks
    ]
    for block in plan:
        target = Path(block["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        request = build_seas5_request(
            config,
            years=block["years"],
            months=sorted({month for months in block["months_by_year"].values() for month in months}),
            lead_months=list(config.lead_times),
            system=block["system"],
        )
        print(
            f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] DOWNLOAD seas5 block range={block['start_year']}-{block['start_month']:02d}_to_{block['end_year']}-{block['end_month']:02d} system={block['system']}",
            flush=True,
        )
        client.retrieve("seasonal-monthly-single-levels", request, str(target))
        print(
            f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] DONE seas5 block path={target}",
            flush=True,
        )
        downloaded_blocks.append(
            {
                "path": target,
                "start_year": block["start_year"],
                "start_month": block["start_month"],
                "end_year": block["end_year"],
                "end_month": block["end_month"],
                "system": block["system"],
                "status": "downloaded",
            }
        )

    completed_months = sorted({month for block in _collect_existing_blocks(config) for month in block["months"]})
    remaining_missing = [
        month
        for month in _month_span(config.seas5_start_year, 1, config.seas5_end_year, 12)
        if month not in set(completed_months)
    ]
    manifest = {
        "dataset": "seasonal-monthly-single-levels",
        "originating_centre": config.seas5_originating_centre,
        "system": config.seas5_system,
        "variable": "sea_ice_cover",
        "product_type": "monthly_mean",
        "leadtime_month": list(config.lead_times),
        "base_system": config.seas5_system,
        "post_transition_system": config.seas5_system_post_transition,
        "system_transition_year": config.seas5_system_transition_year,
        "system_transition_month": config.seas5_system_transition_month,
        "start_year": config.seas5_start_year,
        "start_month": 1,
        "end_year": config.seas5_end_year,
        "end_month": 12,
        "missing_months": remaining_missing,
        "blocks": downloaded_blocks,
    }
    _write_json(config.seas5_download_manifest_path, manifest)
    print(
        f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] DONE seas5 download manifest={config.seas5_download_manifest_path}",
        flush=True,
    )
    return config.seas5_download_manifest_path


def _target_lon_lat(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    xr = _lazy_xarray()
    Transformer = _lazy_pyproj()
    ds = xr.open_dataset(config.region_nc_path)
    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return lon.astype(np.float32), lat.astype(np.float32)


def _canonicalize_forecast_dataset(ds):
    if isinstance(ds, list):
        ds = ds[0]
    if "sea_ice_cover" in ds.data_vars:
        da = ds["sea_ice_cover"]
    elif "siconc" in ds.data_vars:
        da = ds["siconc"]
    else:
        da = next(iter(ds.data_vars.values()))

    if "time" in da.coords and "time" not in da.dims:
        da = da.expand_dims(time=np.atleast_1d(da.coords["time"].values))
    if "step" in da.coords and "step" not in da.dims:
        da = da.expand_dims(step=np.atleast_1d(da.coords["step"].values))

    rename_map = {}
    if "time" in da.dims:
        rename_map["time"] = "init_time"
    if "step" in da.dims:
        rename_map["step"] = "lead"
    if rename_map:
        da = da.rename(rename_map)

    if "number" in da.dims:
        da = da.mean("number")

    if "lead" not in da.dims and "forecastMonth" in da.dims:
        da = da.rename({"forecastMonth": "lead"})

    da = da.astype(np.float32)
    if "latitude" in da.coords:
        da = da.sortby("latitude")
    if "longitude" in da.coords:
        lon = da["longitude"].values
        if np.nanmax(lon) > 180.0:
            wrapped = ((lon + 180.0) % 360.0) - 180.0
            da = da.assign_coords(longitude=wrapped).sortby("longitude")
    return da


def _extract_lead_months(lead_coord) -> np.ndarray:
    values = np.atleast_1d(np.asarray(lead_coord.values))
    if np.issubdtype(values.dtype, np.timedelta64):
        days = (values / np.timedelta64(1, "D")).astype(np.float64)
        months = np.rint(days / 30.0).astype(np.int32)
        return np.where(months < 1, np.arange(1, len(values) + 1), months)
    return values.astype(np.int32)


def _month_delta(start, end) -> int:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return (end.year - start.year) * 12 + (end.month - start.month)


def _processed_file_is_valid(config: ExperimentConfig) -> bool:
    path = config.seas5_processed_path
    if not path.exists():
        return False
    try:
        payload = np.load(path, allow_pickle=False)
        preds = payload["preds"]
        lead_months = payload["lead_months"].astype(np.int32)
    except Exception:
        return False
    expected = np.asarray(config.lead_times, dtype=np.int32)
    if preds.ndim != 4 or preds.shape[-1] != len(expected):
        return False
    if not np.array_equal(lead_months, expected):
        return False
    return bool(np.isfinite(preds).any())


def _select_effective_monthly_leads(da, expected_leads: list[int]):
    xr = _lazy_xarray()
    if "valid_time" not in da.coords:
        raise ValueError("SEAS5 field is missing valid_time coordinate")

    init_times = pd.to_datetime(da["init_time"].values)
    valid_times = da["valid_time"].values.astype("datetime64[ns]")
    raw_values = da.values
    valid_mask = np.isfinite(raw_values).any(axis=(2, 3))
    lead_values = np.atleast_1d(np.asarray(da["lead"].values))

    filtered = []
    selected_valid_times = []
    selected_step_days = []
    for init_idx, init_time in enumerate(init_times):
        step_indices = np.where(valid_mask[init_idx])[0]
        if len(step_indices) != len(expected_leads):
            raise ValueError(
                f"Expected {len(expected_leads)} valid SEAS5 steps for init_time={init_time.date()}, found {len(step_indices)}"
            )
        month_deltas = np.asarray(
            [_month_delta(init_time, valid_times[init_idx, step_idx]) for step_idx in step_indices],
            dtype=np.int32,
        )
        order = np.argsort(month_deltas)
        step_indices = step_indices[order]
        month_deltas = month_deltas[order]
        if month_deltas.tolist() != list(expected_leads):
            raise ValueError(
                f"Unexpected lead month mapping for init_time={init_time.date()}: {month_deltas.tolist()}"
            )
        filtered.append(raw_values[init_idx, step_indices, :, :])
        selected_valid_times.append(valid_times[init_idx, step_indices])
        if np.issubdtype(lead_values.dtype, np.timedelta64):
            selected_step_days.append(
                [int(lead_values[step_idx] / np.timedelta64(1, "D")) for step_idx in step_indices]
            )
        else:
            selected_step_days.append([int(lead_values[step_idx]) for step_idx in step_indices])

    filtered_values = np.stack(filtered, axis=0).astype(np.float32)
    filtered_valid_times = np.stack(selected_valid_times, axis=0)
    filtered_da = xr.DataArray(
        filtered_values,
        dims=("init_time", "lead", "latitude", "longitude"),
        coords={
            "init_time": da["init_time"].values,
            "lead": np.asarray(expected_leads, dtype=np.int32),
            "latitude": da["latitude"].values,
            "longitude": da["longitude"].values,
            "valid_time": (("init_time", "lead"), filtered_valid_times),
        },
        name=da.name,
    )
    summary = {
        "n_init_times": int(len(init_times)),
        "selected_step_days_unique": sorted({tuple(days) for days in selected_step_days}),
    }
    return filtered_da, summary


def _regrid_field(
    src_values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (src_lat, src_lon),
        src_values,
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.stack([target_lat.ravel(), target_lon.ravel()], axis=-1)
    out = interpolator(points).reshape(target_lat.shape)
    return out.astype(np.float32)


def _raw_paths_from_manifest(config: ExperimentConfig) -> list[Path]:
    manifest = _load_download_manifest(config)
    return _raw_paths_from_manifest_payload(
        manifest,
        config.seas5_raw_blocks_dir,
        config.seas5_start_year,
        config.seas5_end_year,
    )


def _raw_paths_from_manifest_payload(
    manifest: dict | None,
    raw_blocks_dir: Path,
    start_year: int,
    end_year: int,
) -> list[Path]:
    if manifest and manifest.get("blocks"):
        paths = []
        for block in manifest["blocks"]:
            start = (int(block["start_year"]), int(block["start_month"]))
            end = (int(block["end_year"]), int(block["end_month"]))
            if end < (start_year, 1) or start > (end_year, 12):
                continue
            path = Path(block["path"])
            if path.exists():
                paths.append(path)
        return paths
    paths = []
    for path in sorted(raw_blocks_dir.glob("*.grib")):
        months = _block_months(path)
        if months and months[-1] >= (start_year, 1):
            paths.append(path)
    return paths


def _hindcast_year_block_path(config: ExperimentConfig, year: int) -> Path:
    return config.seas5_hindcast_raw_blocks_dir / f"seas5_hindcast_{year}_6lead_monthly_sic.grib"


def _process_seas5_common(
    config: ExperimentConfig,
    raw_paths: list[Path],
    processed_path: Path,
    processing_manifest_path: Path,
) -> Path:
    if not raw_paths:
        raise FileNotFoundError("No SEAS5 raw files found.")

    xr = _lazy_xarray()
    selection_summaries = []
    target_lon, target_lat = _target_lon_lat(config)
    target_shape = target_lat.shape
    block_preds = []
    block_init_times = []
    for path in raw_paths:
        ds = xr.open_dataset(path, engine="cfgrib")
        da = _canonicalize_forecast_dataset(ds)
        filtered_da, selection_summary = _select_effective_monthly_leads(da, list(config.lead_times))
        src_lat = filtered_da["latitude"].values.astype(np.float64)
        src_lon = filtered_da["longitude"].values.astype(np.float64)
        current_init_times = pd.to_datetime(filtered_da["init_time"].values)
        current_preds = np.full(
            (len(current_init_times), target_shape[0], target_shape[1], len(config.lead_times)),
            np.nan,
            dtype=np.float32,
        )
        for init_idx in range(len(current_init_times)):
            for lead_idx, _ in enumerate(config.lead_times):
                field = filtered_da.isel(init_time=init_idx, lead=lead_idx).values
                current_preds[init_idx, :, :, lead_idx] = _regrid_field(
                    field,
                    src_lat=src_lat,
                    src_lon=src_lon,
                    target_lat=target_lat,
                    target_lon=target_lon,
                )
        block_preds.append(current_preds)
        block_init_times.append(current_init_times.to_numpy())
        selection_summaries.append({"raw_path": str(path), **selection_summary})

    init_times = pd.to_datetime(np.concatenate(block_init_times))
    preds = np.concatenate(block_preds, axis=0)
    order = np.argsort(init_times.values)
    init_times = init_times[order]
    preds = preds[order]
    _, unique_idx = np.unique(init_times.values, return_index=True)
    unique_idx = np.sort(unique_idx)
    init_times = init_times[unique_idx]
    preds = preds[unique_idx]
    lead_months = np.asarray(config.lead_times, dtype=np.int32)

    np.savez_compressed(
        processed_path,
        preds=preds,
        init_times=np.array([ts.strftime("%Y-%m-%d") for ts in init_times], dtype="U10"),
        lead_months=lead_months.astype(np.int32),
    )
    processing_manifest = {
        "processed_path": processed_path,
        "n_init_times": int(len(init_times)),
        "n_leads": int(len(lead_months)),
        "first_init_time": str(init_times[0].date()) if len(init_times) else None,
        "last_init_time": str(init_times[-1].date()) if len(init_times) else None,
        "lead_months": lead_months.tolist(),
        "raw_paths": raw_paths,
        "selection_summaries": selection_summaries,
    }
    _write_json(processing_manifest_path, processing_manifest)
    return processed_path


def process_seas5(config: ExperimentConfig) -> Path:
    config.ensure_dirs()
    if config.run_mode == "resume" and _processed_file_is_valid(config):
        return config.seas5_processed_path
    raw_paths = _raw_paths_from_manifest(config)
    return _process_seas5_common(
        config,
        raw_paths=raw_paths,
        processed_path=config.seas5_processed_path,
        processing_manifest_path=config.seas5_processing_manifest_path,
    )


def download_seas5_hindcast(config: ExperimentConfig) -> Path:
    config.ensure_dirs()
    cdsapi = _lazy_cdsapi()
    client = cdsapi.Client()
    downloaded_blocks = []
    missing_years = []
    for year in range(config.seas5_hindcast_start_year, config.seas5_hindcast_end_year + 1):
        target = _hindcast_year_block_path(config, year)
        if target.exists():
            downloaded_blocks.append(
                {
                    "path": target,
                    "start_year": year,
                    "start_month": 1,
                    "end_year": year,
                    "end_month": 12,
                    "system": config.seas5_system,
                    "status": "reused",
                }
            )
            continue
        request = build_seas5_request(
            config,
            years=[year],
            months=list(range(1, 13)),
            lead_months=list(config.lead_times),
            system=config.seas5_system,
        )
        print(
            f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] DOWNLOAD seas5 hindcast year={year}",
            flush=True,
        )
        client.retrieve("seasonal-monthly-single-levels", request, str(target))
        downloaded_blocks.append(
            {
                "path": target,
                "start_year": year,
                "start_month": 1,
                "end_year": year,
                "end_month": 12,
                "system": config.seas5_system,
                "status": "downloaded",
            }
        )
    for year in range(config.seas5_hindcast_start_year, config.seas5_hindcast_end_year + 1):
        if not _hindcast_year_block_path(config, year).exists():
            missing_years.append(year)
    manifest = {
        "dataset": "seasonal-monthly-single-levels",
        "originating_centre": config.seas5_originating_centre,
        "system": config.seas5_system,
        "variable": "sea_ice_cover",
        "product_type": "monthly_mean",
        "leadtime_month": list(config.lead_times),
        "start_year": config.seas5_hindcast_start_year,
        "start_month": 1,
        "end_year": config.seas5_hindcast_end_year,
        "end_month": 12,
        "missing_years": missing_years,
        "blocks": downloaded_blocks,
    }
    _write_json(config.seas5_hindcast_download_manifest_path, manifest)
    return config.seas5_hindcast_download_manifest_path


def process_seas5_hindcast(config: ExperimentConfig) -> Path:
    config.ensure_dirs()
    raw_paths = _raw_paths_from_manifest_payload(
        _load_json(config.seas5_hindcast_download_manifest_path),
        config.seas5_hindcast_raw_blocks_dir,
        config.seas5_hindcast_start_year,
        config.seas5_hindcast_end_year,
    )
    if not raw_paths:
        raise FileNotFoundError("No SEAS5 hindcast raw files found. Run download_seas5_hindcast first.")
    return _process_seas5_common(
        config,
        raw_paths=raw_paths,
        processed_path=config.seas5_hindcast_processed_path,
        processing_manifest_path=config.seas5_hindcast_processing_manifest_path,
    )


def _load_processed_seas5_file(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed SEAS5 file: {path}")
    payload = np.load(path, allow_pickle=False)
    preds = payload["preds"].astype(np.float32)
    init_times = pd.to_datetime(payload["init_times"])
    lead_months = payload["lead_months"].astype(np.int32)
    return preds, init_times, lead_months


def load_processed_seas5(config: ExperimentConfig) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    return _load_processed_seas5_file(config.seas5_processed_path)


def load_processed_seas5_hindcast(config: ExperimentConfig) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    return _load_processed_seas5_file(config.seas5_hindcast_processed_path)


def fit_seas5_bias_correction(config: ExperimentConfig) -> Path:
    config.ensure_dirs()
    if not config.seas5_hindcast_processed_path.exists():
        manifest = _load_json(config.seas5_hindcast_download_manifest_path)
        if not manifest or manifest.get("missing_years"):
            raise FileNotFoundError(
                "Missing complete SEAS5 hindcast data. Download/process hindcast before fitting bias correction."
            )
        process_seas5_hindcast(config)

    hindcast_preds, init_times, lead_months = load_processed_seas5_hindcast(config)
    bundle = load_bundle(config)
    height, width = hindcast_preds.shape[1], hindcast_preds.shape[2]
    n_months = 12
    n_leads = len(config.lead_times)
    sum_fields = np.zeros((n_months, n_leads, height, width), dtype=np.float32)
    count_fields = np.zeros((n_months, n_leads, height, width), dtype=np.uint16)
    sample_counts = np.zeros((n_months, n_leads), dtype=np.int32)

    lead_lookup = {int(lead): idx for idx, lead in enumerate(lead_months.tolist())}
    for init_idx, init_time in enumerate(pd.DatetimeIndex(init_times)):
        issue_index = (int(init_time.year) - config.start_year) * 12 + (int(init_time.month) - 1)
        if issue_index < 0 or issue_index >= bundle.sic.shape[0]:
            continue
        month_idx = int(init_time.month) - 1
        for lead_pos, lead in enumerate(config.lead_times):
            src_lead_idx = lead_lookup.get(int(lead))
            if src_lead_idx is None:
                continue
            target_index = issue_index + int(lead)
            if target_index >= bundle.sic.shape[0]:
                continue
            pred = hindcast_preds[init_idx, :, :, src_lead_idx]
            obs = bundle.sic[target_index]
            valid = np.isfinite(pred) & np.isfinite(obs)
            if not np.any(valid):
                continue
            diff = pred - obs
            sum_fields[month_idx, lead_pos][valid] += diff[valid]
            count_fields[month_idx, lead_pos][valid] += 1
            sample_counts[month_idx, lead_pos] += 1

    bias_fields = np.full_like(sum_fields, np.nan, dtype=np.float32)
    valid_counts = count_fields > 0
    np.divide(sum_fields, count_fields, out=bias_fields, where=valid_counts)
    if not np.isfinite(bias_fields).any():
        raise ValueError("Failed to compute any finite SEAS5 bias correction fields.")

    np.savez_compressed(
        config.seas5_bias_correction_path,
        bias_fields=bias_fields,
        sample_counts=sample_counts,
        lead_months=np.asarray(config.lead_times, dtype=np.int32),
        init_months=np.arange(1, 13, dtype=np.int32),
        hindcast_start_year=np.asarray([config.seas5_hindcast_start_year], dtype=np.int32),
        hindcast_end_year=np.asarray([config.seas5_hindcast_end_year], dtype=np.int32),
    )
    return config.seas5_bias_correction_path


def load_seas5_bias_correction(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not config.seas5_bias_correction_path.exists():
        raise FileNotFoundError(f"Missing SEAS5 bias correction file: {config.seas5_bias_correction_path}")
    payload = np.load(config.seas5_bias_correction_path, allow_pickle=False)
    bias_fields = payload["bias_fields"].astype(np.float32)
    sample_counts = payload["sample_counts"].astype(np.int32)
    lead_months = payload["lead_months"].astype(np.int32)
    return bias_fields, sample_counts, lead_months


def evaluate_seas5_for_year(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    seas5_preds: np.ndarray,
    init_times: pd.DatetimeIndex,
    lead_months: np.ndarray,
    held_out_year: int,
    model_name: str = "seas5",
    bias_fields: np.ndarray | None = None,
):
    months = list(config.issue_months)
    leads = list(config.lead_times)
    year_preds = np.full((len(months),) + bundle.sic.shape[1:] + (len(leads),), np.nan, dtype=np.float32)
    year_truths = np.full_like(year_preds, np.nan)
    year_meta = []

    init_lookup = {(ts.year, ts.month): idx for idx, ts in enumerate(pd.DatetimeIndex(init_times))}
    lead_lookup = {int(lead): idx for idx, lead in enumerate(lead_months.tolist())}

    for month_idx, month in enumerate(months):
        init_idx = init_lookup.get((held_out_year, month))
        if init_idx is None:
            continue
        issue_index = (held_out_year - config.start_year) * 12 + (month - 1)
        for lead_idx, lead in enumerate(leads):
            src_lead_idx = lead_lookup.get(int(lead))
            target_index = issue_index + lead
            if src_lead_idx is None or target_index >= bundle.sic.shape[0]:
                continue
            pred = seas5_preds[init_idx, :, :, src_lead_idx].copy()
            if bias_fields is not None:
                pred = np.clip(pred - bias_fields[month - 1, lead_idx], 0.0, 1.0)
            year_preds[month_idx, :, :, lead_idx] = pred
            year_truths[month_idx, :, :, lead_idx] = bundle.sic[target_index]
            year_meta.append(
                {
                    "held_out_year": held_out_year,
                    "issue_month": month,
                    "lead": lead,
                    "issue_index": issue_index,
                    "target_index": target_index,
                    "metadata": {"source": model_name, "init_time": str(init_times[init_idx].date())},
                }
            )

    if not np.isfinite(year_preds).any():
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
    train_end = (held_out_year - config.start_year) * 12
    train_indices = np.arange(train_end)
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

    return {
        "year_row": year_row,
        "month_lead_rows": month_lead_rows,
        "region_rows": region_rows,
        "metadata": year_meta,
        "preds": year_preds,
        "truths": year_truths,
    }


def run_seas5_baseline(config: ExperimentConfig):
    from .pipelines import (
        _append_csv,
        _append_jsonl,
        _aggregate_variant_outputs,
        _experiment_dir,
        _load_progress,
        _prepare_variant_dir,
        _save_progress,
    )

    out_dir = _experiment_dir(config, "seas5_baseline")
    if config.run_mode == "fresh" and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(config)
    if not config.seas5_processed_path.exists():
        if not _raw_paths_from_manifest(config):
            download_seas5(config)
        process_seas5(config)
    seas5_preds, init_times, lead_months = load_processed_seas5(config)

    obs_last_year = int(bundle.time_index[-1].year)
    years = sorted({int(ts.year) for ts in init_times if ts.year <= obs_last_year and ts.year >= config.seas5_start_year})

    model_dir = out_dir / "seas5"
    _prepare_variant_dir(model_dir, config.run_mode)
    progress_path = model_dir / "progress.json"
    progress = _load_progress(progress_path)
    if config.run_mode == "fresh":
        progress = {
            "status": "running",
            "current_model": "seas5",
            "completed_years": [],
            "started_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        _save_progress(progress_path, progress)

    completed_years = set(progress.get("completed_years", []))
    for held_out_year in years:
        if config.run_mode == "resume" and held_out_year in completed_years:
            continue
        result = evaluate_seas5_for_year(config, bundle, seas5_preds, init_times, lead_months, held_out_year)
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
        _save_progress(
            progress_path,
            {
                "status": "running",
                "current_model": "seas5",
                "completed_years": sorted(completed_years),
                "started_at": progress.get("started_at"),
                "latest_year": held_out_year,
                "latest_rmse": result["year_row"]["rmse"],
            },
        )

    yearly_df, month_lead_df, regional_df, metadata = _aggregate_variant_outputs(model_dir)
    if not yearly_df.empty:
        yearly_df.to_csv(model_dir / "seas5_metrics.csv", index=False)
    if not month_lead_df.empty:
        month_lead_df.to_csv(model_dir / "seas5_month_lead_metrics.csv", index=False)
    if not regional_df.empty:
        regional_df.to_csv(model_dir / "seas5_regional_metrics.csv", index=False)
    if metadata:
        _write_json(model_dir / "seas5_metadata.json", metadata)

    summary = {
        "model": "seas5",
        "rmse_mean": float(yearly_df["rmse"].mean()) if not yearly_df.empty else np.nan,
        "mae_mean": float(yearly_df["mae"].mean()) if not yearly_df.empty else np.nan,
        "acc_mean": float(yearly_df["acc"].mean()) if not yearly_df.empty else np.nan,
        "bss_mean": float(yearly_df["bss"].mean()) if not yearly_df.empty else np.nan,
        "sie_abs_error_mean": float(yearly_df["sie_abs_error"].mean()) if not yearly_df.empty else np.nan,
        "iiee_mean": float(yearly_df["iiee"].mean()) if not yearly_df.empty else np.nan,
        "held_out_years": years,
    }
    _write_json(model_dir / "summary.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "seas5_summary.csv", index=False)
    _save_progress(
        progress_path,
        {
            "status": "completed",
            "current_model": "seas5",
            "completed_years": sorted(completed_years),
            "started_at": progress.get("started_at"),
            "finished_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
    )

    finalist_dir = config.results_dir / "finalist_compare_2015_2023" / "finalist_compare"
    if finalist_dir.exists():
        compare_rows = [dict(candidate_name="seas5", **summary)]
        eval_start = min(years) if years else config.seas5_start_year
        eval_end = max(years) if years else config.seas5_start_year
        candidate_map = {
            "nsidc18_a0": finalist_dir / "nsidc18_a0" / "a0" / "yearly_metrics.csv",
            "merge6_a1": finalist_dir / "merge6_a1" / "a1" / "yearly_metrics.csv",
        }
        for candidate_name, path in candidate_map.items():
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["held_out_year"].between(eval_start, eval_end)]
            compare_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model": candidate_name,
                    "rmse_mean": float(df["rmse"].mean()),
                    "mae_mean": float(df["mae"].mean()),
                    "acc_mean": float(df["acc"].mean()),
                    "bss_mean": float(df["bss"].mean()),
                    "sie_abs_error_mean": float(df["sie_abs_error"].mean()),
                    "iiee_mean": float(df["iiee"].mean()),
                }
            )
        pd.DataFrame(compare_rows).to_csv(out_dir / "seas5_vs_internal_models.csv", index=False)

    return yearly_df
