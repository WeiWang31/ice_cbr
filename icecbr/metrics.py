from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def calculate_iiee_single(y_obs: np.ndarray, y_sim: np.ndarray, cell_area_km2: float = 625.0):
    union = y_sim + y_obs
    union = union.copy()
    union[union == 2] = 1
    intersection = y_sim * y_obs
    iiee_area = union - intersection
    oe_area = y_sim - intersection
    ue_area = y_obs - intersection
    iiee = np.sum(iiee_area == 1) * cell_area_km2 / 1e6
    oe = np.sum(oe_area == 1) * cell_area_km2 / 1e6
    ue = np.sum(ue_area == 1) * cell_area_km2 / 1e6
    return oe, ue, iiee


def month_lead_mae_mse(
    y_prediction: np.ndarray,
    y_true: np.ndarray,
    ocean_mask: np.ndarray,
    start_date: str,
):
    lead_time = y_true.shape[-1]
    mae_all = np.zeros((lead_time, 12), dtype=np.float64)
    mse_all = np.zeros((lead_time, 12), dtype=np.float64)
    month_distribution = np.zeros((lead_time, 12), dtype=np.float64)
    pd_start_date = pd.to_datetime(start_date)
    for i in range(y_true.shape[0]):
        sample_start_date = pd_start_date + pd.DateOffset(months=i)
        for j in range(y_true.shape[-1]):
            sample_date = sample_start_date + pd.DateOffset(months=j)
            month = sample_date.month - 1
            y_obs = y_true[i, :, :, j][ocean_mask]
            y_prd = y_prediction[i, :, :, j][ocean_mask]
            valid = np.isfinite(y_obs) & np.isfinite(y_prd)
            if not np.any(valid):
                continue
            mae_all[j, month] += np.mean(np.abs(y_obs[valid] - y_prd[valid]))
            mse_all[j, month] += np.mean((y_obs[valid] - y_prd[valid]) ** 2)
            month_distribution[j, month] += 1
    month_distribution = np.where(month_distribution > 0, month_distribution, 1.0)
    return mae_all / month_distribution, mse_all / month_distribution


def month_lead_iiee(
    y_prediction: np.ndarray,
    y_true: np.ndarray,
    start_date: str,
    threshold: float = 0.15,
    cell_area_km2: float = 625.0,
):
    lead_time = y_true.shape[-1]
    iiee_all = np.zeros((lead_time, 12), dtype=np.float64)
    oe_all = np.zeros((lead_time, 12), dtype=np.float64)
    ue_all = np.zeros((lead_time, 12), dtype=np.float64)
    month_distribution = np.zeros((lead_time, 12), dtype=np.float64)
    pd_start_date = pd.to_datetime(start_date)
    for i in range(y_true.shape[0]):
        sample_start_date = pd_start_date + pd.DateOffset(months=i)
        for j in range(y_true.shape[-1]):
            sample_date = sample_start_date + pd.DateOffset(months=j)
            month = sample_date.month - 1
            y_obs = (y_true[i, :, :, j] > threshold).astype(np.int32)
            y_prd = (y_prediction[i, :, :, j] > threshold).astype(np.int32)
            oe, ue, iiee = calculate_iiee_single(y_obs, y_prd, cell_area_km2=cell_area_km2)
            iiee_all[j, month] += iiee
            oe_all[j, month] += oe
            ue_all[j, month] += ue
            month_distribution[j, month] += 1
    month_distribution = np.where(month_distribution > 0, month_distribution, 1.0)
    return (
        iiee_all / month_distribution,
        oe_all / month_distribution,
        ue_all / month_distribution,
    )


def calculate_acc(preds: np.ndarray, obs: np.ndarray, ocean_mask: np.ndarray, eps: float = 1e-10):
    batch, _, _, leads = preds.shape
    acc = np.full(leads, np.nan, dtype=np.float64)
    for c in range(leads):
        p = preds[:, :, :, c]
        o = obs[:, :, :, c]
        p_anom = p - np.mean(p, axis=0, keepdims=True)
        o_anom = o - np.mean(o, axis=0, keepdims=True)
        num = np.sum(p_anom * o_anom, axis=0)
        den = np.sqrt(np.sum(p_anom**2, axis=0) * np.sum(o_anom**2, axis=0)) + eps
        r_grid = np.where(den > eps, num / den, np.nan)
        acc[c] = np.nanmean(r_grid[ocean_mask])
    return acc


def sie_absolute_error(preds: np.ndarray, obs: np.ndarray, ocean_mask: np.ndarray, threshold: float = 0.15):
    pred_ocean = preds[:, ocean_mask, :]
    obs_ocean = obs[:, ocean_mask, :]
    valid = np.isfinite(pred_ocean) & np.isfinite(obs_ocean)
    pred_sie = ((pred_ocean > threshold) & valid).sum(axis=1)
    obs_sie = ((obs_ocean > threshold) & valid).sum(axis=1)
    return np.abs(pred_sie - obs_sie)


def brier_skill_score(preds: np.ndarray, obs: np.ndarray, climatology_prob: np.ndarray, ocean_mask: np.ndarray, threshold: float = 0.15):
    y = (obs > threshold).astype(np.float32)
    p = np.clip(preds, 0.0, 1.0)
    p_ocean = p[:, ocean_mask, :]
    y_ocean = y[:, ocean_mask, :]
    valid = np.isfinite(p_ocean) & np.isfinite(y_ocean)
    bs = np.mean(((p_ocean - y_ocean) ** 2)[valid])
    clim = np.broadcast_to(climatology_prob[None, :, :, None], obs.shape)
    clim_ocean = clim[:, ocean_mask, :]
    valid_ref = np.isfinite(clim_ocean) & np.isfinite(y_ocean)
    bs_ref = np.mean(((clim_ocean - y_ocean) ** 2)[valid_ref])
    if bs_ref == 0:
        return np.nan
    return 1.0 - bs / bs_ref


def regional_rmse(preds: np.ndarray, obs: np.ndarray, region_masks: dict[int, np.ndarray]):
    out = {}
    for rid, mask in region_masks.items():
        diff = preds[:, mask, :] - obs[:, mask, :]
        out[rid] = float(np.sqrt(np.nanmean(diff**2)))
    return out


@dataclass
class MetricSummary:
    rmse: float
    mae: float
    acc: float
    iiee: float
    bss: float
