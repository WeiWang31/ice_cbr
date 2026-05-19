from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from .config import ExperimentConfig


def _lazy_xarray():
    import xarray as xr

    return xr


@dataclass
class RegionInfo:
    region_ids: list[int]
    region_names: dict[int, str]
    region_masks: dict[int, np.ndarray]
    fusion_weights: dict[int, np.ndarray]
    ocean_mask: np.ndarray
    surface_mask: np.ndarray


@dataclass
class DatasetBundle:
    sic: np.ndarray  # (T,H,W)
    land_mask: np.ndarray  # (H,W) bool; True for ocean/valid sea-ice cells
    background: dict[str, np.ndarray]  # var -> (T,H,W)
    region_info: RegionInfo
    time_index: pd.DatetimeIndex


@dataclass
class ExportRegionWorkspace:
    mask: np.ndarray
    sic_matrix: np.ndarray
    bg_matrix: np.ndarray
    region_mean_sic: np.ndarray
    hist: np.ndarray


@dataclass
class ExportWorkspace:
    month_numbers: np.ndarray
    pan_sie: np.ndarray
    rolling_bg: np.ndarray
    regions: dict[int, ExportRegionWorkspace]


def region_merge_definitions() -> dict[str, dict[int, tuple[str, list[int]]]]:
    return {
        "nsidc18": {},
        "merge6": {
            101: ("central_arctic", [1]),
            102: ("pacific_gateway", [2, 3, 13, 18]),
            103: ("siberian_east", [4, 5]),
            104: ("atlantic_sector", [6, 7, 8]),
            105: ("canadian_gateway", [9, 10, 11, 12]),
            106: ("subarctic_marginal", [14, 15, 16, 17]),
        },
        "merge4": {
            201: ("central_arctic", [1]),
            202: ("pacific_sector", [2, 3, 4, 13, 14, 15, 16, 18]),
            203: ("atlantic_sector", [5, 6, 7, 8, 9, 10, 17]),
            204: ("canadian_gateway", [11, 12]),
        },
    }


def _apply_region_scheme(
    config: ExperimentConfig,
    region_grid: np.ndarray,
    ocean_mask: np.ndarray,
    region_names: dict[int, str],
    valid_ids: list[int],
) -> tuple[list[int], dict[int, str], dict[int, np.ndarray]]:
    scheme = config.region_scheme
    merge_defs = region_merge_definitions()
    if scheme not in merge_defs:
        raise ValueError(f"Unknown region_scheme: {scheme}")

    if scheme == "nsidc18":
        region_masks = {}
        kept_names = {}
        for rid in valid_ids:
            mask = (region_grid == rid) & ocean_mask
            if mask.sum() == 0:
                continue
            region_masks[rid] = mask
            kept_names[rid] = region_names[rid]
        return sorted(region_masks), kept_names, region_masks

    region_masks = {}
    kept_names = {}
    for merged_id, (merged_name, source_ids) in merge_defs[scheme].items():
        source_ids = [rid for rid in source_ids if rid in valid_ids]
        if not source_ids:
            continue
        mask = np.isin(region_grid, source_ids) & ocean_mask
        if mask.sum() == 0:
            continue
        region_masks[merged_id] = mask
        kept_names[merged_id] = merged_name
    return sorted(region_masks), kept_names, region_masks


def hwt_to_thw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D HWT array, got {arr.shape}")
    return np.moveaxis(arr, -1, 0)


def parse_flag_metadata(var) -> tuple[list[int], dict[int, str]]:
    flag_values = var.attrs.get("flag_values")
    flag_meanings = var.attrs.get("flag_meanings")
    if flag_values is None or flag_meanings is None:
        raise ValueError("Region file missing flag_values or flag_meanings")
    values = [int(v) for v in np.array(flag_values).tolist()]
    meanings = str(flag_meanings).split()
    names = {v: m for v, m in zip(values, meanings)}
    return values, names


def load_region_info(config: ExperimentConfig) -> RegionInfo:
    xr = _lazy_xarray()
    ds = xr.open_dataset(config.region_nc_path)
    if "sea_ice_region" in ds.data_vars:
        reg = ds["sea_ice_region"]
    elif "sea_ice_region_surface_mask" in ds.data_vars:
        reg = ds["sea_ice_region_surface_mask"]
    else:
        raise ValueError("Region file missing sea_ice_region and sea_ice_region_surface_mask")

    if "sea_ice_region_surface_mask" in ds.data_vars:
        surface = ds["sea_ice_region_surface_mask"]
    else:
        surface = reg

    flag_values, region_names = parse_flag_metadata(reg)
    region_grid = reg.values.astype(np.int32)
    surface_grid = surface.values.astype(np.int32)
    ocean_mask = surface_grid < 30

    valid_ids = [rid for rid in flag_values if rid < 30]
    if not config.include_unassigned_region:
        valid_ids = [rid for rid in valid_ids if rid != 0]

    region_ids, region_names, region_masks = _apply_region_scheme(
        config=config,
        region_grid=region_grid,
        ocean_mask=ocean_mask,
        region_names=region_names,
        valid_ids=valid_ids,
    )

    fusion_weights: dict[int, np.ndarray] = {}
    raw_weights = []
    for rid in region_ids:
        mask = region_masks[rid]
        raw = gaussian_filter(mask.astype(np.float32), sigma=1.5)
        raw = np.where(ocean_mask, raw, 0.0)
        raw_weights.append(raw)
        fusion_weights[rid] = raw

    if not fusion_weights:
        raise ValueError("No valid ocean regions were parsed from NSIDC region file")

    denom = np.sum(np.stack(list(fusion_weights.values()), axis=0), axis=0)
    denom = np.where(denom > 0, denom, 1.0)
    for rid in list(fusion_weights):
        fusion_weights[rid] = fusion_weights[rid] / denom

    return RegionInfo(
        region_ids=region_ids,
        region_names=region_names,
        region_masks=region_masks,
        fusion_weights=fusion_weights,
        ocean_mask=ocean_mask,
        surface_mask=surface_grid,
    )


def load_background_fields(config: ExperimentConfig) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in config.background_variables:
        if name.startswith("normalized_ohc") or name.startswith("normalized_mld"):
            path = config.oras5_normalized_dir / name
        else:
            path = config.era5_normalized_dir / name
        fields[name] = hwt_to_thw(np.load(path))
    return fields


def load_bundle(config: ExperimentConfig) -> DatasetBundle:
    sic = hwt_to_thw(np.load(config.sic_path))
    land_mask = np.load(config.land_mask_path).astype(bool)
    background = load_background_fields(config)
    region_info = load_region_info(config)
    time_index = pd.date_range(
        start=f"{config.start_year}-01-01", periods=sic.shape[0], freq="MS"
    )
    return DatasetBundle(
        sic=sic,
        land_mask=land_mask,
        background=background,
        region_info=region_info,
        time_index=time_index,
    )


def compute_monthly_climatology(series: np.ndarray, train_indices: Iterable[int]) -> np.ndarray:
    train_indices = np.array(list(train_indices), dtype=int)
    climatology = np.zeros((12,) + series.shape[1:], dtype=np.float32)
    for month in range(12):
        month_idx = train_indices[train_indices % 12 == month]
        climatology[month] = np.mean(series[month_idx], axis=0)
    return climatology


def compute_anomalies(series: np.ndarray, train_indices: Iterable[int]) -> np.ndarray:
    climatology = compute_monthly_climatology(series, train_indices)
    anomalies = np.empty_like(series, dtype=np.float32)
    for t in range(series.shape[0]):
        anomalies[t] = series[t] - climatology[t % 12]
    return anomalies


def monthly_sie(sic: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    binary = (sic[:, mask] > threshold).astype(np.float32)
    return binary.sum(axis=1)


def fit_region_pca(anoms: np.ndarray, region_mask: np.ndarray, n_components: int) -> PCA:
    X = anoms[:, region_mask]
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)
    return pca


def trailing_slope(values: np.ndarray) -> float:
    x = np.arange(values.shape[0], dtype=np.float32)
    x = x - x.mean()
    denom = np.sum(x**2)
    if denom == 0:
        return 0.0
    y = values - values.mean()
    return float(np.sum(x * y) / denom)


def build_export_workspace(bundle: DatasetBundle, config: ExperimentConfig) -> ExportWorkspace:
    pan_sie = monthly_sie(bundle.sic, bundle.region_info.ocean_mask, config.target_threshold)
    rolling_bg = pd.Series(pan_sie).rolling(window=120, min_periods=12).mean().to_numpy()
    month_numbers = np.array([ts.month - 1 for ts in bundle.time_index], dtype=np.int32)

    regions: dict[int, ExportRegionWorkspace] = {}
    for rid in bundle.region_info.region_ids:
        mask = bundle.region_info.region_masks[rid] & bundle.land_mask
        sic_matrix = bundle.sic[:, mask].astype(np.float32, copy=False)
        bg_matrix = np.stack(
            [bg[:, mask].mean(axis=1) for bg in bundle.background.values()],
            axis=1,
        ).astype(np.float32, copy=False)
        region_mean_sic = sic_matrix.mean(axis=1).astype(np.float32, copy=False)
        hist = np.empty(bundle.sic.shape[0], dtype=np.float32)
        for t in range(bundle.sic.shape[0]):
            hist[t] = trailing_slope(region_mean_sic[max(0, t - config.history_window + 1): t + 1])
        regions[rid] = ExportRegionWorkspace(
            mask=mask,
            sic_matrix=sic_matrix,
            bg_matrix=bg_matrix,
            region_mean_sic=region_mean_sic,
            hist=hist,
        )

    return ExportWorkspace(
        month_numbers=month_numbers,
        pan_sie=pan_sie.astype(np.float32, copy=False),
        rolling_bg=rolling_bg.astype(np.float32, copy=False),
        regions=regions,
    )


def build_feature_store(bundle: DatasetBundle, config: ExperimentConfig, train_indices: Iterable[int]) -> dict:
    train_indices = np.array(sorted(set(train_indices)), dtype=int)
    sic_anoms = compute_anomalies(bundle.sic, train_indices)
    bg_anoms = {
        name: compute_anomalies(field, train_indices) for name, field in bundle.background.items()
    }
    pan_sie = monthly_sie(bundle.sic, bundle.region_info.ocean_mask, config.target_threshold)
    rolling_bg = pd.Series(pan_sie).rolling(window=120, min_periods=12).mean().to_numpy()
    feature_store = {
        "sic_anoms": sic_anoms,
        "bg_anoms": bg_anoms,
        "pan_sie": pan_sie,
        "rolling_bg": rolling_bg,
        "regions": {},
    }
    for rid in bundle.region_info.region_ids:
        mask = bundle.region_info.region_masks[rid] & bundle.land_mask
        pca = fit_region_pca(sic_anoms[train_indices], mask, config.sic_eof_dim)
        region_sie = monthly_sie(bundle.sic, mask, config.target_threshold)
        region_mean_sic = bundle.sic[:, mask].mean(axis=1)
        region_bg = np.stack([bg[:, mask].mean(axis=1) for bg in bg_anoms.values()], axis=1)
        bg_n_components = min(config.bg_eof_dim, region_bg.shape[0], region_bg.shape[1])
        bg_pca = PCA(n_components=bg_n_components, random_state=config.random_state)
        bg_pca.fit(region_bg[train_indices])
        feature_store["regions"][rid] = {
            "mask": mask,
            "sic_pca": pca,
            "region_sie": region_sie,
            "region_mean_sic": region_mean_sic,
            "region_bg": region_bg,
            "bg_pca": bg_pca,
        }
    return feature_store


def build_case_record(
    t: int,
    lead: int,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    feature_store: dict,
) -> dict:
    issue_date = bundle.time_index[t]
    target_index = t + lead
    pan_sie = feature_store["pan_sie"]
    rolling_bg = feature_store["rolling_bg"]
    record = {
        "issue_index": t,
        "target_index": target_index,
        "lead": lead,
        "year": issue_date.year,
        "month": issue_date.month,
        "regime": "pre-2007" if issue_date.year < config.regime_split_year else "post-2007",
        "delta_sie": float(pan_sie[t] - np.nanmean(pan_sie[max(0, t - 12): t + 1])),
        "global_solution": bundle.sic[target_index],
        "regional": {},
        "rolling_bg": float(rolling_bg[t]) if np.isfinite(rolling_bg[t]) else float(pan_sie[t]),
    }
    for rid, region in feature_store["regions"].items():
        mask = region["mask"]
        sic_proj = region["sic_pca"].transform(feature_store["sic_anoms"][t, mask][None, :])[0]
        region_bg = region["bg_pca"].transform(region["region_bg"][t][None, :])[0]
        hist_vals = region["region_mean_sic"][max(0, t - config.history_window + 1): t + 1]
        hist = float(trailing_slope(hist_vals))
        local_trend_vals = region["region_mean_sic"][max(0, t - 11): t + 1]
        record["regional"][rid] = {
            "z_ice": sic_proj.astype(np.float32),
            "z_bg": region_bg.astype(np.float32),
            "h": hist,
            "local_trend": float(trailing_slope(local_trend_vals)),
            "solution": bundle.sic[target_index],
        }
    return record


def build_case_base(
    train_indices: Iterable[int],
    leads: Iterable[int],
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> tuple[list[dict], dict]:
    train_indices = np.array(sorted(set(train_indices)), dtype=int)
    feature_store = build_feature_store(bundle, config, train_indices)
    train_set = set(train_indices.tolist())
    cases = []
    for t in train_indices:
        for lead in leads:
            target = t + lead
            if target in train_set:
                cases.append(build_case_record(t, lead, bundle, config, feature_store))
    return cases, feature_store


def build_query_record(
    t: int,
    lead: int,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    feature_store: dict,
) -> dict:
    return build_case_record(t, lead, bundle, config, feature_store)


def iter_test_queries(test_year: int, bundle: DatasetBundle, config: ExperimentConfig):
    for month in config.issue_months:
        t = (test_year - config.start_year) * 12 + (month - 1)
        for lead in config.lead_times:
            target = t + lead
            if target < bundle.sic.shape[0]:
                yield t, lead
