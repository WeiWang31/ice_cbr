from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from .config import ExperimentConfig
from .data import (
    DatasetBundle,
    ExportWorkspace,
    build_case_base,
    build_case_record,
    build_export_workspace,
    build_feature_store,
    build_query_record,
    compute_monthly_climatology,
    monthly_sie,
    trailing_slope,
)


def _lazy_lowess():
    from statsmodels.nonparametric.smoothers_lowess import lowess

    return lowess


def flatten_field(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return field[mask].astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / denom)


def softmax_weights(scores: np.ndarray, rho: float = 2.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    exps = np.exp(rho * shifted)
    denom = np.sum(exps)
    if denom == 0:
        return np.full_like(scores, 1.0 / len(scores))
    return exps / denom


def revise_cases(
    adapted_fields: np.ndarray,
    weights: np.ndarray,
    *,
    active_mask: np.ndarray,
    variance_quantile: float,
    downweight_strength: float,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adapted_fields = np.asarray(adapted_fields, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)

    n_cases = adapted_fields.shape[0]
    if n_cases == 0:
        return adapted_fields, weights, {
            "enabled": True,
            "disagreement_threshold": 0.0,
            "disagreement_fraction": 0.0,
            "num_downweighted": 0,
            "case_deviation_scores": [],
            "case_reliability_scores": [],
            "downweight_factors": [],
        }
    if n_cases == 1 or not np.any(active_mask):
        normalized = weights / np.sum(weights) if np.sum(weights) > 0 else np.full_like(weights, 1.0 / len(weights))
        return np.clip(adapted_fields, 0.0, 1.0), normalized, {
            "enabled": True,
            "disagreement_threshold": 0.0,
            "disagreement_fraction": 0.0,
            "num_downweighted": 0,
            "case_deviation_scores": [0.0 for _ in range(n_cases)],
            "case_reliability_scores": [1.0 for _ in range(n_cases)],
            "downweight_factors": [1.0 for _ in range(n_cases)],
        }

    variance_map = np.var(adapted_fields, axis=0)
    active_variance = variance_map[active_mask]
    threshold = float(np.quantile(active_variance, variance_quantile)) if active_variance.size else 0.0
    disagreement_mask = active_mask & (variance_map >= threshold)
    if not np.any(disagreement_mask):
        disagreement_mask = active_mask.copy()

    disagreement_fraction = float(np.mean(disagreement_mask[active_mask])) if np.any(active_mask) else 0.0
    median_field = np.median(adapted_fields, axis=0)
    deviation_scores = np.mean(
        np.abs(adapted_fields[:, disagreement_mask] - median_field[disagreement_mask][None, :]),
        axis=1,
    )
    scale = float(np.median(deviation_scores))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(deviation_scores))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    reliability_scores = np.exp(-downweight_strength * (deviation_scores / scale))
    downweight_factors = np.clip(reliability_scores, min_weight, 1.0)
    revised_weights = weights * downweight_factors

    if np.sum(revised_weights) <= 0:
        revised_weights = weights.copy()

    revised_weights = revised_weights / np.sum(revised_weights)
    revised_fields = np.clip(adapted_fields, 0.0, 1.0)
    reliability_stats = {
        "min": float(np.min(downweight_factors)),
        "max": float(np.max(downweight_factors)),
        "mean": float(np.mean(downweight_factors)),
    }

    metadata = {
        "enabled": True,
        "disagreement_threshold": threshold,
        "disagreement_fraction": disagreement_fraction,
        "num_downweighted": int(np.sum(downweight_factors < 0.999)),
        "case_deviation_scores": deviation_scores.astype(float).tolist(),
        "case_reliability_scores": downweight_factors.astype(float).tolist(),
        "reliability_distribution": reliability_stats,
        "original_weights": weights.astype(float).tolist(),
        "revised_weights": revised_weights.astype(float).tolist(),
        "downweight_factors": downweight_factors.astype(float).tolist(),
    }
    return revised_fields, revised_weights.astype(np.float64, copy=False), metadata


def _exp_score(value: float, scale: float) -> float:
    scale = max(float(scale), 1e-6)
    value = max(float(value), 0.0)
    return float(np.exp(-value / scale))


def _bounded_score(value: float, center: float, tolerance: float) -> float:
    tolerance = max(float(tolerance), 1e-6)
    return float(np.exp(-abs(float(value) - float(center)) / tolerance))


def score_retained_case(
    retain_metadata: dict[str, Any],
    *,
    config: ExperimentConfig,
    issue_year: int,
    current_year: int,
) -> dict[str, Any]:
    weights = config.retain_admission_weights
    error = retain_metadata.get("forecast_error", {})
    adaptation = retain_metadata.get("adaptation_magnitude") or {}

    rmse = float(error.get("rmse", 0.0))
    disagreement = float(retain_metadata.get("disagreement_level", 0.0))
    adaptation_mag = float(adaptation.get("weighted_mean_abs_delta", adaptation.get("mean_abs_delta", 0.0)))
    age_years = max(float(current_year - issue_year), 0.0)

    components = {
        "error": _exp_score(rmse, config.retain_admission_error_scale),
        "disagreement": _exp_score(disagreement, config.retain_admission_disagreement_scale),
        "recency": _exp_score(age_years, config.retained_case_recency_tau_years),
        "adaptation": _bounded_score(
            adaptation_mag,
            config.retain_admission_preferred_adaptation,
            config.retain_admission_adaptation_tolerance,
        ),
    }
    weight_sum = float(sum(float(weights.get(key, 0.0)) for key in components)) or 1.0
    score = 0.0
    for key, value in components.items():
        score += float(weights.get(key, 0.0)) * float(value)
    score /= weight_sum

    mode = config.retain_admission_mode
    if not config.enable_selective_retain or mode == "none":
        admitted = True
    elif mode == "strict":
        admitted = score >= float(config.retain_admission_threshold_strict)
    else:
        admitted = score >= float(config.retain_admission_threshold_permissive)

    return {
        "score": float(score),
        "components": {k: float(v) for k, v in components.items()},
        "mode": mode,
        "admitted": bool(admitted),
        "rmse": rmse,
        "disagreement": disagreement,
        "adaptation_magnitude": adaptation_mag,
        "age_years": age_years,
    }


def _retain_selector_enabled(config: ExperimentConfig) -> bool:
    return bool(config.enable_selective_retain and config.enable_representative_retained_memory)


def retained_case_priority(
    retained_case: dict[str, Any],
    *,
    current_year: int,
    recency_tau_years: float,
) -> float:
    retain_metadata = retained_case.get("retain_metadata") or {}
    admission = retain_metadata.get("admission") or {}
    usefulness = float(retain_metadata.get("retained_usefulness_score", admission.get("score", 0.0)))
    issue_year = int(retained_case.get("year", current_year))
    age_years = max(float(current_year - issue_year), 0.0)
    recency = _exp_score(age_years, recency_tau_years)
    return float(0.7 * usefulness + 0.3 * recency)


def summarize_retain_diagnostics(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "generated": 0,
            "admitted": 0,
            "rejected": 0,
            "representative_evicted": 0,
            "evicted": 0,
            "average_admission_score": 0.0,
            "average_active_retained_cases": 0.0,
            "average_retrieved_retained_cases": 0.0,
            "average_retained_contribution": 0.0,
            "peak_active_retained_cases": 0,
        }
    scores = [float(e.get("admission_score", 0.0)) for e in events]
    active_counts = [int(e.get("active_retained_cases", 0)) for e in events]
    retrieved_counts = [float(e.get("retained_retrieval_count", 0.0)) for e in events]
    retained_contrib = [float(e.get("retained_contribution", 0.0)) for e in events]
    return {
        "generated": int(len(events)),
        "admitted": int(sum(1 for e in events if e.get("admitted"))),
        "rejected": int(sum(1 for e in events if not e.get("admitted"))),
        "representative_evicted": int(sum(1 for e in events if e.get("representative_decision") == "evict_existing")),
        "evicted": int(max((int(e.get("evicted_count", 0)) for e in events), default=0)),
        "average_admission_score": float(np.mean(scores)),
        "average_active_retained_cases": float(np.mean(active_counts)),
        "average_retrieved_retained_cases": float(np.mean(retrieved_counts)),
        "average_retained_contribution": float(np.mean(retained_contrib)),
        "peak_active_retained_cases": int(max(active_counts)),
    }


def _retain_store_path(config: ExperimentConfig, model_name: str) -> Path:
    namespace = config.retain_namespace or "default"
    root = config.results_dir
    if config.output_tag:
        root = root / config.output_tag
    return root / namespace / "retained_cases" / f"{model_name}.jsonl"


def _deserialize_retained_case(payload: dict[str, Any]) -> dict[str, Any]:
    case = dict(payload)
    case["global_solution"] = np.asarray(case["global_solution"], dtype=np.float32)
    regional = {}
    for key, region in case["regional"].items():
        rid = int(key)
        regional[rid] = {
            "z_ice": np.asarray(region["z_ice"], dtype=np.float32),
            "z_bg": np.asarray(region["z_bg"], dtype=np.float32),
            "h": float(region["h"]),
            "local_trend": float(region["local_trend"]),
            "solution": np.asarray(region["solution"], dtype=np.float32),
        }
    case["regional"] = regional
    return case


@dataclass
class PredictionResult:
    prediction: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    retrieval: str
    adaptation: str
    reuse: str
    retain: str
    aggregation: str = "default"
    time_decay: bool = False
    climate_gate: bool = False


class BaseForecastModel:
    name = "base"

    def fit(self, train_indices: np.ndarray, bundle: DatasetBundle, config: ExperimentConfig):
        raise NotImplementedError

    def predict(self, issue_index: int, lead: int) -> PredictionResult:
        raise NotImplementedError

    def retain_case(
        self,
        issue_index: int,
        lead: int,
        prediction_result: PredictionResult,
        truth_field: np.ndarray,
    ) -> dict[str, Any] | None:
        return None

    def append_retained_case(self, retained_case: dict[str, Any]) -> None:
        return None

    def retain_diagnostics_snapshot(self) -> dict[str, Any]:
        return {}


class ClimatologyModel(BaseForecastModel):
    name = "climatology"

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.monthly = np.zeros((12,) + bundle.sic.shape[1:], dtype=np.float32)
        for month in range(12):
            idx = [i for i in train_indices if i % 12 == month]
            self.monthly[month] = np.mean(bundle.sic[np.array(idx)], axis=0)

    def predict(self, issue_index, lead):
        target_month = (issue_index + lead) % 12
        return PredictionResult(self.monthly[target_month], {"source": "monthly_mean"})


class DampedPersistenceModel(BaseForecastModel):
    name = "damped_persistence"

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.climatology = ClimatologyModel()
        self.climatology.fit(train_indices, bundle, config)
        self.damping = {lead: max(0.0, 1.0 - 0.15 * (lead - 1)) for lead in config.lead_times}

    def predict(self, issue_index, lead):
        climatology = self.climatology.predict(issue_index, lead).prediction
        issue_month = issue_index % 12
        issue_clim = self.climatology.monthly[issue_month]
        anomaly = self.bundle.sic[issue_index] - issue_clim
        pred = climatology + self.damping[lead] * anomaly
        return PredictionResult(np.clip(pred, 0.0, 1.0), {"damping": self.damping[lead]})


class LinearTrendPersistenceModel(BaseForecastModel):
    name = "linear_trend_persistence"

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.climatology = ClimatologyModel()
        self.climatology.fit(train_indices, bundle, config)
        self.persistence = DampedPersistenceModel()
        self.persistence.fit(train_indices, bundle, config)
        x = np.arange(bundle.sic.shape[0], dtype=np.float32)
        pan_sie = monthly_sie(bundle.sic, bundle.land_mask, config.target_threshold)
        self.global_trend = np.polyfit(x[train_indices], pan_sie[train_indices], 1)

    def predict(self, issue_index, lead):
        pred = self.persistence.predict(issue_index, lead).prediction
        current = np.polyval(self.global_trend, issue_index)
        future = np.polyval(self.global_trend, issue_index + lead)
        delta = (future - current) / max(self.bundle.land_mask.sum(), 1)
        return PredictionResult(np.clip(pred + delta, 0.0, 1.0), {"trend_delta": float(delta)})


class FeatureRegressorModel(BaseForecastModel):
    def __init__(self, estimator):
        self.estimator = estimator
        self.name = estimator.__class__.__name__.lower()

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.estimators = {}
        for lead in config.lead_times:
            X, y = [], []
            train_set = set(train_indices.tolist())
            for t in train_indices:
                if t < config.history_window - 1 or (t + lead) not in train_set:
                    continue
                X.append(flatten_field(bundle.sic[t], bundle.land_mask))
                y.append(flatten_field(bundle.sic[t + lead], bundle.land_mask))
            estimator = self.estimator.__class__(**self.estimator.get_params())
            estimator.fit(np.asarray(X), np.asarray(y))
            self.estimators[lead] = estimator

    def predict(self, issue_index, lead):
        pred = self.estimators[lead].predict(flatten_field(self.bundle.sic[issue_index], self.bundle.land_mask)[None, :])[0]
        field = np.zeros_like(self.bundle.sic[0])
        field[self.bundle.land_mask] = pred
        return PredictionResult(np.clip(field, 0.0, 1.0), {"estimator": self.name})


class ConfigurableCBRModel(BaseForecastModel):
    name = "configurable_cbr"

    def __init__(self, spec: ModelSpec, name: str):
        self.spec = spec
        self.name = name

    def _needs_global_trend(self) -> bool:
        return self.spec.adaptation == "global_linear"

    def _needs_region_trends(self) -> bool:
        return self.spec.adaptation == "regional_nonlinear"

    def _needs_region_linear_trends(self) -> bool:
        return self.spec.adaptation == "regional_monthly_linear"

    def _needs_competence(self) -> bool:
        return self.spec.retain == "competence"

    def _uses_time_decay(self) -> bool:
        return self.spec.time_decay

    def _uses_climate_gate(self) -> bool:
        return self.spec.climate_gate

    def _revise_case_fields(self, adapted_fields: list[np.ndarray], weights: np.ndarray, active_mask: np.ndarray):
        if not self.config.enable_revise:
            return (
                np.asarray(adapted_fields, dtype=np.float32),
                np.asarray(weights, dtype=np.float64),
                None,
            )
        return revise_cases(
            np.asarray(adapted_fields, dtype=np.float32),
            np.asarray(weights, dtype=np.float64),
            active_mask=active_mask,
            variance_quantile=float(self.config.revise_variance_quantile),
            downweight_strength=float(self.config.revise_downweight_strength),
            min_weight=float(self.config.revise_min_weight),
        )

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.train_indices = np.array(sorted(train_indices))
        self.case_base, self.feature_store = build_case_base(
            train_indices=self.train_indices,
            leads=config.lead_times,
            bundle=bundle,
            config=config,
        )
        self.case_by_lead = defaultdict(list)
        for case in self.case_base:
            self.case_by_lead[case["lead"]].append(case)
        self.case_weights = defaultdict(lambda: 1.0)
        self.case_lookup = {(int(case["issue_index"]), int(case["lead"])): case for case in self.case_base}
        self.archive_case_keys = set(self.case_lookup)
        self.retained_case_keys = set()
        self.retained_case_meta = {}
        self.retained_case_order = []
        self.retain_stats = {
            "generated": 0,
            "admitted": 0,
            "rejected": 0,
            "representative_evicted": 0,
            "evicted": 0,
            "active_retained": 0,
            "retrieval_frequency": 0,
            "retained_contribution_total": 0.0,
            "prediction_count": 0,
        }
        if self.config.enable_retain and self.config.retain_reload:
            self._load_retained_cases()
        self.pan_sie = None
        self.global_trend = None
        self.lowess = None
        self.region_trends = None
        self.region_linear_trends = None
        if self._needs_global_trend() or self._needs_competence():
            self.pan_sie = monthly_sie(bundle.sic, bundle.land_mask, config.target_threshold)
        if self._needs_global_trend():
            self.global_trend = self._fit_global_trend()
        if self._needs_region_trends():
            self.lowess = _lazy_lowess()
            self.region_trends = self._fit_region_trends()
        if self._needs_region_linear_trends():
            self.region_linear_trends = self._fit_region_linear_trends()
        if self._needs_competence():
            self._estimate_competence()

    def _load_retained_cases(self) -> None:
        store_path = _retain_store_path(self.config, self.name)
        if not store_path.exists():
            return
        with open(store_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                self.append_retained_case(_deserialize_retained_case(payload))

    def _retained_usefulness(self, case: dict) -> float:
        retain_meta = case.get("retain_metadata", {})
        admission = retain_meta.get("admission") or {}
        return float(admission.get("score", 0.0))

    def _retained_case_boost(self, query_idx: int, case: dict) -> float:
        if not self.config.enable_retained_case_boost or case.get("source") != "retained":
            return 1.0
        usefulness = self._retained_usefulness(case)
        age_years = max(
            float(self.bundle.time_index[query_idx].year - self.bundle.time_index[int(case["issue_index"])].year),
            0.0,
        )
        recency = _exp_score(age_years, self.config.retained_case_recency_tau_years)
        base = usefulness * recency
        boost = 1.0 + min(
            float(self.config.retained_case_boost_max),
            float(self.config.retained_case_boost_strength) * base,
        )
        return float(boost)

    def _selective_retain_enabled(self) -> bool:
        return _retain_selector_enabled(self.config)

    def _active_retained_cases_for_lead(self, lead: int) -> list[dict[str, Any]]:
        return [
            self.case_lookup[key]
            for key in self.retained_case_keys
            if int(key[1]) == int(lead) and key in self.case_lookup
        ]

    def _representative_priority(self, retained_case: dict[str, Any]) -> float:
        current_year = int(self.bundle.time_index[self.train_indices.max()].year) if self.train_indices.size else self.config.start_year
        return retained_case_priority(
            retained_case,
            current_year=current_year,
            recency_tau_years=float(self.config.retained_case_recency_tau_years),
        )

    def _finalize_retained_candidate(self, retained_case: dict[str, Any]) -> dict[str, Any]:
        retain_metadata = retained_case.setdefault("retain_metadata", {})
        admission = retain_metadata.setdefault("admission", {})
        selection = retain_metadata.setdefault(
            "selection",
            {
                "representative_decision": "disabled",
            },
        )
        if not self._selective_retain_enabled():
            selection["final_decision"] = "admit"
            selection["admitted"] = bool(admission.get("admitted", True))
            return {"admitted": bool(admission.get("admitted", True)), "replaced_key": None}

        admission["admitted"] = True
        selection["admitted"] = True
        selection["final_decision"] = "admit"
        return {"admitted": True, "replaced_key": None}

    def _evict_retained_case(self, key: tuple[int, int]) -> bool:
        case = self.case_lookup.get(key)
        if case is None or key not in self.retained_case_keys:
            return False
        self.retained_case_keys.discard(key)
        self.retained_case_meta.pop(key, None)
        self.case_lookup.pop(key, None)
        self.case_weights.pop(key, None)
        self.case_base = [
            existing
            for existing in self.case_base
            if (int(existing["issue_index"]), int(existing["lead"])) != key
        ]
        lead = int(key[1])
        if lead in self.case_by_lead:
            self.case_by_lead[lead] = [
                existing
                for existing in self.case_by_lead[lead]
                if (int(existing["issue_index"]), int(existing["lead"])) != key
            ]
        self.retained_case_order = [item for item in self.retained_case_order if item != key]
        self.retain_stats["evicted"] += 1
        self.retain_stats["active_retained"] = len(self.retained_case_keys)
        return True

    def _enforce_retained_memory_limits(self) -> list[tuple[int, int]]:
        if not self.config.enable_bounded_retain_memory or not self.retained_case_keys:
            return []
        evicted: list[tuple[int, int]] = []

        def _candidate_sort_key(key: tuple[int, int]):
            meta = self.retained_case_meta.get(key, {})
            score = float(meta.get("score", 0.0))
            issue_idx = int(key[0])
            return (score, issue_idx)

        if self.config.retained_case_cap_per_lead is not None:
            per_lead = int(self.config.retained_case_cap_per_lead)
            for lead in self.config.lead_times:
                keys = [key for key in self.retained_case_keys if int(key[1]) == int(lead)]
                while len(keys) > per_lead:
                    victim = min(keys, key=_candidate_sort_key)
                    if not self._evict_retained_case(victim):
                        break
                    evicted.append(victim)
                    keys.remove(victim)
        if self.config.retained_case_cap_per_month is not None:
            per_month = int(self.config.retained_case_cap_per_month)
            month_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for key in self.retained_case_keys:
                month = int(self.bundle.time_index[int(key[0])].month)
                month_groups[month].append(key)
            for month, keys in month_groups.items():
                while len(keys) > per_month:
                    victim = min(keys, key=_candidate_sort_key)
                    if not self._evict_retained_case(victim):
                        break
                    evicted.append(victim)
                    keys.remove(victim)
        if self.config.retained_case_cap is not None:
            global_cap = int(self.config.retained_case_cap)
            while len(self.retained_case_keys) > global_cap:
                victim = min(self.retained_case_keys, key=_candidate_sort_key)
                if not self._evict_retained_case(victim):
                    break
                evicted.append(victim)
        self.retain_stats["active_retained"] = len(self.retained_case_keys)
        return evicted

    def _record_retained_usage(self, metadata: dict[str, Any]) -> None:
        if not metadata:
            return
        usage = metadata.get("retained_case_usage") or {}
        self.retain_stats["prediction_count"] += 1
        self.retain_stats["retrieval_frequency"] += int(usage.get("retained_count", 0))
        self.retain_stats["retained_contribution_total"] += float(usage.get("retained_contribution", 0.0))

    def retain_diagnostics_snapshot(self) -> dict[str, Any]:
        predictions = max(int(self.retain_stats.get("prediction_count", 0)), 1)
        return {
            "generated": int(self.retain_stats.get("generated", 0)),
            "admitted": int(self.retain_stats.get("admitted", 0)),
            "rejected": int(self.retain_stats.get("rejected", 0)),
            "representative_evicted": int(self.retain_stats.get("representative_evicted", 0)),
            "evicted": int(self.retain_stats.get("evicted", 0)),
            "active_retained": int(len(self.retained_case_keys)),
            "average_retrieved_retained_cases": float(self.retain_stats.get("retrieval_frequency", 0)) / predictions,
            "average_retained_contribution": float(self.retain_stats.get("retained_contribution_total", 0.0)) / predictions,
        }

    def _case_error_summary(self, prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
        diff = prediction[self.bundle.land_mask] - truth[self.bundle.land_mask]
        return {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }

    def _summarize_adaptation_magnitude(self, issue_index: int, lead: int, metadata: dict[str, Any]) -> dict[str, Any]:
        if "global_neighbors" in metadata:
            magnitudes = []
            weighted = 0.0
            total_weight = 0.0
            for item in metadata["global_neighbors"]:
                case = self.case_lookup.get(tuple(item["case_id"]))
                if case is None:
                    continue
                raw = case["global_solution"]
                adapted = self._apply_adaptation(raw, issue_index, case["issue_index"], lead)
                magnitude = float(np.mean(np.abs(adapted[self.bundle.land_mask] - raw[self.bundle.land_mask])))
                weight = float(item.get("revised_weight", item.get("weight", 0.0)))
                magnitudes.append(magnitude)
                weighted += weight * magnitude
                total_weight += weight
            return {
                "mode": "global",
                "mean_abs_delta": float(np.mean(magnitudes)) if magnitudes else 0.0,
                "weighted_mean_abs_delta": float(weighted / total_weight) if total_weight > 0 else 0.0,
            }
        if "regional" in metadata:
            regional = {}
            collected = []
            for rid, items in metadata["regional"].items():
                rid_int = int(rid)
                mask = self.bundle.region_info.region_masks[rid_int]
                magnitudes = []
                weighted = 0.0
                total_weight = 0.0
                for item in items:
                    case = self.case_lookup.get(tuple(item["case_id"]))
                    if case is None:
                        continue
                    raw_field = case["regional"][rid_int]["solution"]
                    adapted = self._adapt_region_solution(raw_field, rid_int, issue_index, case["issue_index"], lead)
                    magnitude = float(np.mean(np.abs(adapted[mask] - raw_field[mask])))
                    weight = float(item.get("revised_weight", item.get("weight", 0.0)))
                    magnitudes.append(magnitude)
                    collected.append(magnitude)
                    weighted += weight * magnitude
                    total_weight += weight
                regional[rid_int] = {
                    "mean_abs_delta": float(np.mean(magnitudes)) if magnitudes else 0.0,
                    "weighted_mean_abs_delta": float(weighted / total_weight) if total_weight > 0 else 0.0,
                }
            return {
                "mode": "compositional",
                "mean_abs_delta": float(np.mean(collected)) if collected else 0.0,
                "regional": regional,
            }
        return {"mode": "unknown", "mean_abs_delta": 0.0}

    def append_retained_case(self, retained_case: dict[str, Any]) -> dict[str, Any]:
        key = (int(retained_case["issue_index"]), int(retained_case["lead"]))
        if key in self.case_lookup:
            return {"admitted": False, "duplicate": True}
        admission = (retained_case.get("retain_metadata") or {}).get("admission") or {}
        decision = self._finalize_retained_candidate(retained_case)
        if self.config.enable_selective_retain and not bool(decision.get("admitted", admission.get("admitted", True))):
            return decision
        self.case_base.append(retained_case)
        self.case_by_lead[int(retained_case["lead"])].append(retained_case)
        self.case_lookup[key] = retained_case
        self.case_weights[key] = 1.0
        self.retained_case_keys.add(key)
        self.retained_case_meta[key] = {
            "score": float(admission.get("score", 0.0)),
            "admitted": bool(admission.get("admitted", True)),
        }
        self.retained_case_order.append(key)
        selection = retained_case.setdefault("retain_metadata", {}).setdefault("selection", {})
        selection.setdefault("representative_decision", "disabled")
        if self._selective_retain_enabled() and self.config.enable_representative_retained_memory:
            cap = int(self.config.retain_representative_cap_per_lead)
            lead_keys = [item for item in self.retained_case_keys if int(item[1]) == int(retained_case["lead"])]
            if len(lead_keys) > cap:
                victim = min(
                    lead_keys,
                    key=lambda item: (self._representative_priority(self.case_lookup[item]), int(item[0])),
                )
                if victim != key and self._evict_retained_case(victim):
                    self.retain_stats["representative_evicted"] += 1
                    selection["representative_decision"] = "evict_existing"
                    selection["representative_victim"] = [int(victim[0]), int(victim[1])]
                elif victim == key and self._evict_retained_case(key):
                    self.retain_stats["representative_evicted"] += 1
                    admission["admitted"] = False
                    selection["admitted"] = False
                    selection["final_decision"] = "reject_representative"
                    selection["representative_decision"] = "reject_new_case"
                    selection["representative_victim"] = [int(key[0]), int(key[1])]
                    self.retain_stats["rejected"] += 1
                    self.retain_stats["active_retained"] = len(self.retained_case_keys)
                    return {"admitted": False, "representative_victim": key}
                else:
                    selection["representative_decision"] = "keep"
        self.retain_stats["active_retained"] = len(self.retained_case_keys)
        self._enforce_retained_memory_limits()
        self.retain_stats["admitted"] += 1
        selection["active_retained_cases_after"] = int(len(self.retained_case_keys))
        return {"admitted": True, "replaced_key": None}

    def retain_case(
        self,
        issue_index: int,
        lead: int,
        prediction_result: PredictionResult,
        truth_field: np.ndarray,
    ) -> dict[str, Any] | None:
        if not self.config.enable_retain:
            return None
        self.retain_stats["generated"] += 1
        retained = build_case_record(issue_index, lead, self.bundle, self.config, self.feature_store)
        retained["global_solution"] = np.asarray(truth_field, dtype=np.float32)
        for rid in retained["regional"]:
            retained["regional"][rid]["solution"] = np.asarray(truth_field, dtype=np.float32)
        retained["source"] = "retained"
        retain_metadata = {
            "predicted_solution": np.asarray(prediction_result.prediction, dtype=np.float32),
            "forecast_error": self._case_error_summary(prediction_result.prediction, truth_field),
            "disagreement_level": float(prediction_result.metadata.get("revise", {}).get("disagreement_fraction", 0.0)),
            "adaptation_magnitude": self._summarize_adaptation_magnitude(issue_index, lead, prediction_result.metadata),
            "revise_statistics": prediction_result.metadata.get("revise"),
        }
        admission = score_retained_case(
            retain_metadata,
            config=self.config,
            issue_year=int(self.bundle.time_index[issue_index].year),
            current_year=int(self.bundle.time_index[self.train_indices.max()].year),
        )
        retain_metadata["admission"] = admission
        retain_metadata["selection"] = {
            "representative_decision": "pending" if self.config.enable_representative_retained_memory else "disabled",
        }
        retained["retain_metadata"] = {
            **retain_metadata,
            "retained_usefulness_score": float(admission["score"]),
        }
        return retained

    def _fit_global_trend(self):
        x = np.arange(self.bundle.sic.shape[0], dtype=np.float32)
        return np.polyfit(x[self.train_indices], self.pan_sie[self.train_indices], 1)

    def _fit_region_trends(self):
        trends = {}
        years = np.array([d.year + (d.month - 1) / 12.0 for d in self.bundle.time_index])
        month_num = np.array([d.month for d in self.bundle.time_index], dtype=np.int32)
        for rid in self.bundle.region_info.region_ids:
            region_sie = self.feature_store["regions"][rid]["region_mean_sic"]
            region_monthly = {}
            for month in range(1, 13):
                mask = month_num == month
                region_monthly[month] = self.lowess(region_sie[mask], years[mask], frac=0.3, return_sorted=True)
            trends[rid] = region_monthly
        return trends

    def _fit_region_linear_trends(self):
        trends = {}
        month_num = np.array([d.month for d in self.bundle.time_index], dtype=np.int32)
        train_idx = self.train_indices.astype(int, copy=False)
        for rid in self.bundle.region_info.region_ids:
            region_mean_sic = self.feature_store["regions"][rid]["region_mean_sic"]
            region_monthly = {}
            for month in range(1, 13):
                idx = train_idx[month_num[train_idx] == month]
                x = idx.astype(np.float64, copy=False)
                y = region_mean_sic[idx].astype(np.float64, copy=False)
                if idx.size <= 1:
                    slope = 0.0
                    intercept = float(y[0]) if idx.size == 1 else 0.0
                else:
                    slope, intercept = np.polyfit(x, y, 1)
                region_monthly[month] = np.array([slope, intercept], dtype=np.float64)
            trends[rid] = region_monthly
        return trends

    def _trend_surface_value(self, rid: int, issue_index: int):
        issue_date = self.bundle.time_index[issue_index]
        year_value = issue_date.year + (issue_date.month - 1) / 12.0
        xy = self.region_trends[rid][issue_date.month]
        return float(np.interp(year_value, xy[:, 0], xy[:, 1]))

    def _linear_trend_surface_value(self, rid: int, target_index: int):
        target_date = self.bundle.time_index[target_index]
        coeffs = self.region_linear_trends[rid][target_date.month]
        return float(np.polyval(coeffs, target_index))

    def _competence_weight(self, case: dict) -> float:
        return self.case_weights[(case["issue_index"], case["lead"])]

    def _similarity(self, query_region: dict, case_region: dict):
        ice = cosine_similarity(query_region["z_ice"], case_region["z_ice"])
        bg = cosine_similarity(query_region["z_bg"], case_region["z_bg"])
        hist = 1.0 / (1.0 + abs(query_region["h"] - case_region["h"]))
        w = self.config.sim_weights
        total = w["ice"] * ice + w["bg"] * bg + w["hist"] * hist
        return total, {"ice": ice, "bg": bg, "hist": hist}

    def _global_similarity(self, query: dict, case: dict):
        region_scores = []
        region_parts = {}
        for rid in self.bundle.region_info.region_ids:
            score, parts = self._similarity(query["regional"][rid], case["regional"][rid])
            region_scores.append(score)
            region_parts[rid] = parts
        return float(np.mean(region_scores)), region_parts

    def _adaptability(self, query: dict, case: dict, rid: int):
        if self.spec.retrieval != "agr":
            return 1.0, {"time": 0.0, "regime": 0.0, "trend": 0.0, "cost": 0.0}
        query_idx = query["issue_index"]
        n_years = max(1, self.config.test_end_year - self.config.start_year + 1)
        d_time = abs(query["year"] - case["year"]) / n_years
        rolling_scale = max(abs(query["rolling_bg"]), abs(case["rolling_bg"]), 1.0)
        d_regime = abs(query["rolling_bg"] - case["rolling_bg"]) / rolling_scale
        d_trend = abs(query["regional"][rid]["local_trend"] - case["regional"][rid]["local_trend"])
        aw = self.config.adapt_weights
        adapt_cost = aw["time"] * d_time + aw["regime"] * d_regime + aw["trend"] * d_trend
        adaptability = float(np.exp(-adapt_cost / (2 * self.config.adaptability_tau**2)))
        return adaptability, {"time": d_time, "regime": d_regime, "trend": d_trend, "cost": adapt_cost}

    def _global_adaptability(self, query: dict, case: dict):
        values = []
        parts = {}
        for rid in self.bundle.region_info.region_ids:
            adaptability, detail = self._adaptability(query, case, rid)
            values.append(adaptability)
            parts[rid] = detail
        return float(np.mean(values)), parts

    def _time_decay_weight(self, query_idx: int, case_idx: int) -> float:
        if not self._uses_time_decay():
            return 1.0
        query_year = int(self.bundle.time_index[query_idx].year)
        case_year = int(self.bundle.time_index[case_idx].year)
        year_diff = abs(query_year - case_year)
        tau = max(float(self.config.retrieval_time_decay_tau_years), 1e-6)
        return float(np.exp(-year_diff / tau))

    def _climate_gate_weight(self, query_idx: int, case_idx: int) -> float:
        if not self._uses_climate_gate():
            return 1.0
        query_bg = self.feature_store["rolling_bg"][query_idx]
        if not np.isfinite(query_bg):
            query_bg = self.feature_store["pan_sie"][query_idx]
        case_bg = self.feature_store["rolling_bg"][case_idx]
        if not np.isfinite(case_bg):
            case_bg = self.feature_store["pan_sie"][case_idx]
        scale = max(abs(query_bg), abs(case_bg), 1.0)
        tau = max(float(self.config.climate_gate_tau), 1e-6)
        return float(np.exp(-abs(query_bg - case_bg) / (scale * tau)))

    def _adapt_field_global_linear(self, field: np.ndarray, query_idx: int, case_idx: int):
        current = np.polyval(self.global_trend, query_idx)
        past = np.polyval(self.global_trend, case_idx)
        delta = (current - past) / max(self.bundle.land_mask.sum(), 1)
        return np.clip(field + delta, 0.0, 1.0)

    def _adapt_field_regional(self, field: np.ndarray, query_idx: int, case_idx: int):
        out = field.copy()
        for rid in self.bundle.region_info.region_ids:
            query_surface = self._trend_surface_value(rid, query_idx)
            case_surface = self._trend_surface_value(rid, case_idx)
            mask = self.bundle.region_info.region_masks[rid]
            out[mask] = np.clip(out[mask] - case_surface + query_surface, 0.0, 1.0)
        return out

    def _adapt_field_regional_monthly_linear(self, field: np.ndarray, query_idx: int, case_idx: int, lead: int):
        out = field.copy()
        query_target_idx = query_idx + lead
        case_target_idx = case_idx + lead
        for rid in self.bundle.region_info.region_ids:
            query_surface = self._linear_trend_surface_value(rid, query_target_idx)
            case_surface = self._linear_trend_surface_value(rid, case_target_idx)
            mask = self.bundle.region_info.region_masks[rid]
            out[mask] = np.clip(out[mask] - case_surface + query_surface, 0.0, 1.0)
        return out

    def _adapt_field_region_only(self, field: np.ndarray, rid: int, query_idx: int, case_idx: int):
        mask = self.bundle.region_info.region_masks[rid]
        out = np.zeros_like(field, dtype=np.float32)
        query_surface = self._trend_surface_value(rid, query_idx)
        case_surface = self._trend_surface_value(rid, case_idx)
        out[mask] = np.clip(field[mask] - case_surface + query_surface, 0.0, 1.0)
        return out

    def _adapt_region_solution(self, field: np.ndarray, rid: int, query_idx: int, case_idx: int, lead: int):
        mask = self.bundle.region_info.region_masks[rid]
        out = np.zeros_like(field, dtype=np.float32)
        if self.spec.adaptation == "none":
            out[mask] = field[mask]
            return out
        if self.spec.adaptation == "global_linear":
            current = np.polyval(self.global_trend, query_idx)
            past = np.polyval(self.global_trend, case_idx)
            delta = (current - past) / max(self.bundle.land_mask.sum(), 1)
            out[mask] = np.clip(field[mask] + delta, 0.0, 1.0)
            return out
        if self.spec.adaptation == "regional_monthly_linear":
            query_surface = self._linear_trend_surface_value(rid, query_idx + lead)
            case_surface = self._linear_trend_surface_value(rid, case_idx + lead)
            out[mask] = np.clip(field[mask] - case_surface + query_surface, 0.0, 1.0)
            return out
        if self.spec.adaptation == "regional_nonlinear":
            return self._adapt_field_region_only(field, rid, query_idx, case_idx)
        raise ValueError(f"Unknown adaptation mode: {self.spec.adaptation}")

    def _apply_adaptation(self, field: np.ndarray, query_idx: int, case_idx: int, lead: int):
        if self.spec.adaptation == "none":
            return field
        if self.spec.adaptation == "global_linear":
            return self._adapt_field_global_linear(field, query_idx, case_idx)
        if self.spec.adaptation == "regional_monthly_linear":
            return self._adapt_field_regional_monthly_linear(field, query_idx, case_idx, lead)
        if self.spec.adaptation == "regional_nonlinear":
            return self._adapt_field_regional(field, query_idx, case_idx)
        raise ValueError(f"Unknown adaptation mode: {self.spec.adaptation}")

    def _rank_global_candidates(self, query: dict, candidates: list[dict]):
        ranked = []
        for case in candidates:
            sim, sim_parts = self._global_similarity(query, case)
            adaptability, adapt_parts = self._global_adaptability(query, case)
            score = sim
            if self.spec.retrieval == "agr":
                score *= adaptability
            score *= self._time_decay_weight(query["issue_index"], case["issue_index"])
            score *= self._climate_gate_weight(query["issue_index"], case["issue_index"])
            competence = self._competence_weight(case)
            score *= competence
            retained_boost = self._retained_case_boost(query["issue_index"], case)
            score *= retained_boost
            ranked.append(
                {
                    "case": case,
                    "score": float(score),
                    "similarity": float(sim),
                    "adaptability": float(adaptability),
                    "similarity_parts": sim_parts,
                    "adaptation_parts": adapt_parts,
                    "competence_weight": float(competence),
                    "retained_boost": float(retained_boost),
                    "retained_usefulness_score": float(self._retained_usefulness(case)) if case.get("source") == "retained" else 0.0,
                    "retained_case": bool(case.get("source") == "retained"),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[: self.config.top_k]

    def _rank_regional_candidates(self, query: dict, candidates: list[dict], rid: int):
        ranked = []
        for case in candidates:
            sim, sim_parts = self._similarity(query["regional"][rid], case["regional"][rid])
            adaptability, adapt_parts = self._adaptability(query, case, rid)
            score = sim
            if self.spec.retrieval == "agr":
                score *= adaptability
            score *= self._time_decay_weight(query["issue_index"], case["issue_index"])
            score *= self._climate_gate_weight(query["issue_index"], case["issue_index"])
            competence = self._competence_weight(case)
            score *= competence
            retained_boost = self._retained_case_boost(query["issue_index"], case)
            score *= retained_boost
            ranked.append(
                {
                    "case": case,
                    "score": float(score),
                    "similarity": float(sim),
                    "adaptability": float(adaptability),
                    "similarity_parts": sim_parts,
                    "adaptation_parts": adapt_parts,
                    "competence_weight": float(competence),
                    "retained_boost": float(retained_boost),
                    "retained_usefulness_score": float(self._retained_usefulness(case)) if case.get("source") == "retained" else 0.0,
                    "retained_case": bool(case.get("source") == "retained"),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[: self.config.top_k]

    def _predict_global(self, query: dict, candidates: list[dict]):
        ranked = self._rank_global_candidates(query, candidates)
        weights = softmax_weights(np.array([item["score"] for item in ranked]), rho=self.config.retrieval_rho)
        adapted_fields = []
        chosen_meta = []
        for weight, item in zip(weights, ranked):
            case = item["case"]
            field = self._apply_adaptation(case["global_solution"], query["issue_index"], case["issue_index"], query["lead"])
            adapted_fields.append(field)
            chosen_meta.append(
                {
                    "case_id": (case["issue_index"], case["lead"]),
                    "issue_year": case["year"],
                    "score": item["score"],
                    "weight": float(weight),
                    "original_weight": float(weight),
                    "similarity": item["similarity"],
                    "adaptability": item["adaptability"],
                    "similarity_parts": item["similarity_parts"],
                    "adaptation_parts": item["adaptation_parts"],
                    "competence_weight": item["competence_weight"],
                    "retained_case": item["retained_case"],
                    "retained_boost_multiplier": item["retained_boost"],
                    "retained_usefulness_score": item["retained_usefulness_score"],
                }
            )
        revised_fields, revised_weights, revise_meta = self._revise_case_fields(
            adapted_fields,
            weights,
            self.bundle.land_mask,
        )
        pred = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        downweight = (
            np.asarray(revise_meta["downweight_factors"], dtype=np.float64)
            if revise_meta is not None
            else np.ones(len(chosen_meta), dtype=np.float64)
        )
        for idx, (weight, field) in enumerate(zip(revised_weights, revised_fields)):
            pred += float(weight) * field
            chosen_meta[idx]["revised_weight"] = float(weight)
            chosen_meta[idx]["revised_downweight_factor"] = float(downweight[idx])
            chosen_meta[idx]["reliability_score"] = float(downweight[idx])
            chosen_meta[idx]["suppressed_by_revise"] = bool(downweight[idx] < 0.999)
            chosen_meta[idx]["removed"] = False
        metadata = {
            "retrieval_mode": self.spec.retrieval,
            "adaptation_mode": self.spec.adaptation,
            "reuse_mode": self.spec.reuse,
            "retain_mode": self.spec.retain,
            "global_neighbors": chosen_meta,
        }
        retained_count = sum(1 for item in chosen_meta if item["retained_case"])
        retained_contribution = float(sum(item["revised_weight"] for item in chosen_meta if item["retained_case"]))
        metadata["retained_case_usage"] = {
            "retained_count": int(retained_count),
            "retained_contribution": retained_contribution,
        }
        if revise_meta is not None:
            metadata["revise"] = revise_meta
        self._record_retained_usage(metadata)
        return PredictionResult(np.clip(pred, 0.0, 1.0), metadata)

    def _predict_compositional(self, query: dict, candidates: list[dict]):
        regional_pred = {}
        regional_meta = {}
        case_sets = {}
        for rid in self.bundle.region_info.region_ids:
            ranked = self._rank_regional_candidates(query, candidates, rid)
            weights = softmax_weights(np.array([item["score"] for item in ranked]), rho=self.config.retrieval_rho)
            region_fields = []
            chosen_meta = []
            chosen_ids = []
            for weight, item in zip(weights, ranked):
                case = item["case"]
                adapted = self._adapt_region_solution(
                    case["regional"][rid]["solution"],
                    rid,
                    query["issue_index"],
                    case["issue_index"],
                    query["lead"],
                )
                region_fields.append(adapted)
                cid = (case["issue_index"], case["lead"])
                chosen_ids.append(cid)
                chosen_meta.append(
                    {
                        "case_id": cid,
                        "issue_year": case["year"],
                        "score": item["score"],
                        "weight": float(weight),
                        "original_weight": float(weight),
                        "similarity": item["similarity"],
                        "adaptability": item["adaptability"],
                        "similarity_parts": item["similarity_parts"],
                        "adaptation_parts": item["adaptation_parts"],
                        "competence_weight": item["competence_weight"],
                        "retained_case": item["retained_case"],
                        "retained_boost_multiplier": item["retained_boost"],
                        "retained_usefulness_score": item["retained_usefulness_score"],
                    }
                )
            revised_fields, revised_weights, revise_meta = self._revise_case_fields(
                region_fields,
                weights,
                self.bundle.region_info.region_masks[rid],
            )
            region_field = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            downweight = (
                np.asarray(revise_meta["downweight_factors"], dtype=np.float64)
                if revise_meta is not None
                else np.ones(len(chosen_meta), dtype=np.float64)
            )
            for idx, (weight, adapted) in enumerate(zip(revised_weights, revised_fields)):
                region_field += float(weight) * adapted
                chosen_meta[idx]["revised_weight"] = float(weight)
                chosen_meta[idx]["revised_downweight_factor"] = float(downweight[idx])
                chosen_meta[idx]["reliability_score"] = float(downweight[idx])
                chosen_meta[idx]["suppressed_by_revise"] = bool(downweight[idx] < 0.999)
                chosen_meta[idx]["removed"] = False
            if revise_meta is not None:
                revise_meta["region_id"] = int(rid)
                revise_meta["region_name"] = self.bundle.region_info.region_names.get(rid, str(rid))
                revise_meta["active_grid_count"] = int(np.sum(self.bundle.region_info.region_masks[rid]))
                regional_meta[f"{rid}_revise"] = revise_meta
            regional_pred[rid] = region_field
            regional_meta[rid] = chosen_meta
            case_sets[rid] = chosen_ids

        fused = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        for rid, field in regional_pred.items():
            fused += self.bundle.region_info.fusion_weights[rid] * field
        metadata = {
            "retrieval_mode": self.spec.retrieval,
            "adaptation_mode": self.spec.adaptation,
            "reuse_mode": self.spec.reuse,
            "retain_mode": self.spec.retain,
            "regional": {rid: regional_meta[rid] for rid in self.bundle.region_info.region_ids},
            "regional_case_sets": case_sets,
        }
        retained_count = sum(
            1 for rid in self.bundle.region_info.region_ids for item in regional_meta[rid] if item["retained_case"]
        )
        retained_contribution = float(
            np.mean(
                [
                    sum(item["revised_weight"] for item in regional_meta[rid] if item["retained_case"])
                    for rid in self.bundle.region_info.region_ids
                ]
            )
        ) if self.bundle.region_info.region_ids else 0.0
        metadata["retained_case_usage"] = {
            "retained_count": int(retained_count),
            "retained_contribution": retained_contribution,
        }
        revise_entries = [regional_meta[key] for key in regional_meta if isinstance(key, str) and key.endswith("_revise")]
        if revise_entries:
            metadata["revise"] = {"regional": revise_entries}
            total_cells = sum(entry.get("active_grid_count", 0) for entry in revise_entries)
            disagreement_fraction = 0.0
            if total_cells > 0:
                disagreement_fraction = float(
                    sum(entry.get("disagreement_fraction", 0.0) * entry.get("active_grid_count", 0) for entry in revise_entries)
                    / total_cells
                )
            metadata["revise"]["disagreement_fraction"] = disagreement_fraction
        self._record_retained_usage(metadata)
        return PredictionResult(np.clip(fused, 0.0, 1.0), metadata)

    def _predict_internal(self, query: dict) -> PredictionResult:
        lead = query["lead"]
        candidates = self.case_by_lead[lead]
        if self.spec.reuse == "global":
            return self._predict_global(query, candidates)
        if self.spec.reuse == "compositional":
            return self._predict_compositional(query, candidates)
        raise ValueError(f"Unknown reuse mode: {self.spec.reuse}")

    def _estimate_competence(self):
        recent_start = max(
            self.train_indices.min(),
            self.train_indices.max() - self.config.competence_recent_years * 12,
        )
        liability = defaultdict(list)
        reachability = defaultdict(int)
        coverage = defaultdict(int)
        for t in self.train_indices:
            if t < recent_start or t < self.config.history_window - 1:
                continue
            for lead in self.config.lead_times:
                if t + lead >= self.bundle.sic.shape[0]:
                    continue
                query = build_query_record(t, lead, self.bundle, self.config, self.feature_store)
                pred = self._predict_internal(query)
                err = np.mean(
                    (pred.prediction[self.bundle.land_mask] - self.bundle.sic[t + lead][self.bundle.land_mask]) ** 2
                )
                if self.spec.reuse == "global":
                    for item in pred.metadata.get("global_neighbors", []):
                        cid = tuple(item["case_id"])
                        liability[cid].append(err)
                        reachability[cid] += 1
                        if err < self.config.competence_error_threshold:
                            coverage[cid] += 1
                else:
                    for region_items in pred.metadata.get("regional", {}).values():
                        for item in region_items:
                            cid = tuple(item["case_id"])
                            liability[cid].append(err)
                            reachability[cid] += 1
                            if err < self.config.competence_error_threshold:
                                coverage[cid] += 1

        for case in self.case_base:
            cid = (case["issue_index"], case["lead"])
            case["liability"] = float(np.mean(liability[cid])) if liability[cid] else 0.0
            age = self.train_indices.max() - case["issue_index"]
            case["temporal_liability"] = float(age / max(1, len(self.train_indices)))
            case["reachability"] = int(reachability[cid])
            case["coverage"] = int(coverage[cid])
            total = case["liability"] + self.config.competence_beta * case["temporal_liability"]
            case["total_liability"] = float(total)
            case["retain_status"] = "pivotal_downweight" if case["coverage"] > 0 and total > 0 else "active"
            self.case_weights[cid] = 1.0 / (1.0 + total)

    def predict(self, issue_index: int, lead: int) -> PredictionResult:
        query = build_query_record(issue_index, lead, self.bundle, self.config, self.feature_store)
        return self._predict_internal(query)


class StandardAnalogKNNExportModel(BaseForecastModel):
    """Export-only fast path for regional SIC+BG+Hist retrieval without reuse."""

    name = "analog_regional_sic_bg_hist_export"

    def __init__(self):
        self.workspace: ExportWorkspace | None = None
        self._bundle_id: int | None = None

    def _ensure_workspace(self, bundle: DatasetBundle, config: ExperimentConfig):
        bundle_id = id(bundle)
        if self.workspace is None or self._bundle_id != bundle_id:
            self.workspace = build_export_workspace(bundle, config)
            self._bundle_id = bundle_id

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.train_indices = np.array(sorted(train_indices), dtype=int)
        self._ensure_workspace(bundle, config)
        self.issue_index = int(self.train_indices.size)
        self.region_ids = tuple(bundle.region_info.region_ids)
        self.region_tables = {}
        month_numbers = self.workspace.month_numbers
        train_count = self.issue_index
        selected_months = month_numbers[: train_count + 1]

        for rid in self.region_ids:
            region = self.workspace.regions[rid]

            sic_monthly = _prefix_monthly_mean(region.sic_matrix, train_count, month_numbers)
            sic_selected = region.sic_matrix[: train_count + 1] - sic_monthly[selected_months]
            sic_train = sic_selected[:train_count]
            sic_n_components = min(config.sic_eof_dim, sic_train.shape[0], sic_train.shape[1])
            sic_pca = PCA(n_components=sic_n_components, random_state=0)
            sic_pca.fit(sic_train)
            sic_proj = sic_pca.transform(sic_selected).astype(np.float32, copy=False)

            bg_monthly = _prefix_monthly_mean(region.bg_matrix, train_count, month_numbers)
            bg_selected = region.bg_matrix[: train_count + 1] - bg_monthly[selected_months]
            bg_train = bg_selected[:train_count]
            bg_n_components = min(config.bg_eof_dim, bg_train.shape[0], bg_train.shape[1])
            bg_pca = PCA(n_components=bg_n_components, random_state=config.random_state)
            bg_pca.fit(bg_train)
            bg_proj = bg_pca.transform(bg_selected).astype(np.float32, copy=False)

            self.region_tables[rid] = {
                "sic_proj": sic_proj,
                "sic_norm": np.linalg.norm(sic_proj, axis=1),
                "bg_proj": bg_proj,
                "bg_norm": np.linalg.norm(bg_proj, axis=1),
                "hist": region.hist[: train_count + 1],
            }

        self.case_tables = {}
        for lead in config.lead_times:
            candidate_count = max(self.issue_index - lead, 0)
            case_issue_indices = np.arange(candidate_count, dtype=int)
            self.case_tables[lead] = {
                "issue_indices": case_issue_indices,
                "solutions": bundle.sic[case_issue_indices + lead].astype(np.float32, copy=False),
            }

    def predict(self, issue_index: int, lead: int) -> PredictionResult:
        case_table = self.case_tables[lead]
        candidate_indices = case_table["issue_indices"]
        if candidate_indices.size == 0:
            empty = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            return PredictionResult(empty, {})

        weights_cfg = self.config.sim_weights
        n_regions = len(self.region_ids)
        scores = np.zeros(candidate_indices.shape[0], dtype=np.float64)
        eps = 1e-10

        for rid in self.region_ids:
            region = self.region_tables[rid]
            q_ice = region["sic_proj"][issue_index]
            q_bg = region["bg_proj"][issue_index]
            q_hist = float(region["hist"][issue_index])

            q_ice_norm = float(np.linalg.norm(q_ice))
            q_bg_norm = float(np.linalg.norm(q_bg))

            cand_ice = region["sic_proj"][candidate_indices]
            cand_bg = region["bg_proj"][candidate_indices]
            cand_hist = region["hist"][candidate_indices]

            ice = (cand_ice @ q_ice) / (region["sic_norm"][candidate_indices] * q_ice_norm + eps)
            bg = (cand_bg @ q_bg) / (region["bg_norm"][candidate_indices] * q_bg_norm + eps)
            hist = 1.0 / (1.0 + np.abs(cand_hist - q_hist))
            scores += weights_cfg["ice"] * ice + weights_cfg["bg"] * bg + weights_cfg["hist"] * hist

        scores /= n_regions
        top_k = min(self.config.top_k, candidate_indices.shape[0])
        ranked_idx = np.argsort(-scores, kind="stable")[:top_k]
        weights = softmax_weights(scores[ranked_idx], rho=self.config.retrieval_rho)

        pred = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        for weight, field in zip(weights, case_table["solutions"][ranked_idx]):
            pred += float(weight) * field
        return PredictionResult(np.clip(pred, 0.0, 1.0), {})


class GlobalAnalogueExportModel(BaseForecastModel):
    """Export-only fast path for global analogue baselines without regional retrieval."""

    def __init__(self, mode: str):
        if mode not in {"sic_only", "sic_bg_hist"}:
            raise ValueError(f"Unknown global analogue mode: {mode}")
        self.mode = mode
        self.workspace: ExportWorkspace | None = None
        self._bundle_id: int | None = None
        self.global_mask: np.ndarray | None = None
        self.global_sic_matrix: np.ndarray | None = None
        self.global_bg_matrix: np.ndarray | None = None
        self.global_mean_sic: np.ndarray | None = None
        self.global_hist: np.ndarray | None = None

    def _ensure_workspace(self, bundle: DatasetBundle, config: ExperimentConfig):
        bundle_id = id(bundle)
        if self.workspace is None or self._bundle_id != bundle_id:
            self.workspace = build_export_workspace(bundle, config)
            self._bundle_id = bundle_id
            self.global_mask = bundle.region_info.ocean_mask & bundle.land_mask
            self.global_sic_matrix = bundle.sic[:, self.global_mask].astype(np.float32, copy=False)
            self.global_mean_sic = self.global_sic_matrix.mean(axis=1).astype(np.float32, copy=False)
            self.global_hist = np.empty(bundle.sic.shape[0], dtype=np.float32)
            for t in range(bundle.sic.shape[0]):
                self.global_hist[t] = trailing_slope(
                    self.global_mean_sic[max(0, t - config.history_window + 1): t + 1]
                )
            self.global_bg_matrix = np.stack(
                [bg[:, self.global_mask].mean(axis=1) for bg in bundle.background.values()],
                axis=1,
            ).astype(np.float32, copy=False)

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.train_indices = np.array(sorted(train_indices), dtype=int)
        self._ensure_workspace(bundle, config)
        self.issue_index = int(self.train_indices.size)

        month_numbers = self.workspace.month_numbers
        train_count = self.issue_index
        selected_months = month_numbers[: train_count + 1]

        sic_monthly = _prefix_monthly_mean(self.global_sic_matrix, train_count, month_numbers)
        sic_selected = self.global_sic_matrix[: train_count + 1] - sic_monthly[selected_months]
        sic_train = sic_selected[:train_count]
        sic_n_components = min(config.sic_eof_dim, sic_train.shape[0], sic_train.shape[1])
        sic_pca = PCA(n_components=sic_n_components, random_state=0)
        sic_pca.fit(sic_train)
        self.global_sic_proj = sic_pca.transform(sic_selected).astype(np.float32, copy=False)
        self.global_sic_norm = np.linalg.norm(self.global_sic_proj, axis=1)

        if self.mode == "sic_bg_hist":
            bg_monthly = _prefix_monthly_mean(self.global_bg_matrix, train_count, month_numbers)
            bg_selected = self.global_bg_matrix[: train_count + 1] - bg_monthly[selected_months]
            bg_train = bg_selected[:train_count]
            bg_n_components = min(config.bg_eof_dim, bg_train.shape[0], bg_train.shape[1])
            bg_pca = PCA(n_components=bg_n_components, random_state=config.random_state)
            bg_pca.fit(bg_train)
            self.global_bg_proj = bg_pca.transform(bg_selected).astype(np.float32, copy=False)
            self.global_bg_norm = np.linalg.norm(self.global_bg_proj, axis=1)

        self.case_tables = {}
        for lead in config.lead_times:
            candidate_count = max(self.issue_index - lead, 0)
            case_issue_indices = np.arange(candidate_count, dtype=int)
            self.case_tables[lead] = {
                "issue_indices": case_issue_indices,
                "solutions": bundle.sic[case_issue_indices + lead].astype(np.float32, copy=False),
            }

    def predict(self, issue_index: int, lead: int) -> PredictionResult:
        case_table = self.case_tables[lead]
        candidate_indices = case_table["issue_indices"]
        if candidate_indices.size == 0:
            empty = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            return PredictionResult(empty, {})

        eps = 1e-10
        q_ice = self.global_sic_proj[issue_index]
        q_ice_norm = float(np.linalg.norm(q_ice))
        cand_ice = self.global_sic_proj[candidate_indices]
        scores = (cand_ice @ q_ice) / (self.global_sic_norm[candidate_indices] * q_ice_norm + eps)

        if self.mode == "sic_bg_hist":
            weights_cfg = self.config.sim_weights
            q_bg = self.global_bg_proj[issue_index]
            q_bg_norm = float(np.linalg.norm(q_bg))
            cand_bg = self.global_bg_proj[candidate_indices]
            q_hist = float(self.global_hist[issue_index])
            cand_hist = self.global_hist[candidate_indices]

            ice = scores
            bg = (cand_bg @ q_bg) / (self.global_bg_norm[candidate_indices] * q_bg_norm + eps)
            hist = 1.0 / (1.0 + np.abs(cand_hist - q_hist))
            scores = weights_cfg["ice"] * ice + weights_cfg["bg"] * bg + weights_cfg["hist"] * hist

        top_k = min(self.config.top_k, candidate_indices.shape[0])
        ranked_idx = np.argsort(-scores, kind="stable")[:top_k]
        weights = softmax_weights(scores[ranked_idx], rho=self.config.retrieval_rho)

        pred = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        for weight, field in zip(weights, case_table["solutions"][ranked_idx]):
            pred += float(weight) * field
        return PredictionResult(np.clip(pred, 0.0, 1.0), {})


def _prefix_monthly_mean(matrix: np.ndarray, train_count: int, month_numbers: np.ndarray) -> np.ndarray:
    monthly = np.empty((12, matrix.shape[1]), dtype=np.float32)
    for month in range(12):
        rows = np.flatnonzero(month_numbers[:train_count] == month)
        monthly[month] = np.mean(matrix[rows], axis=0)
    return monthly


class A0ExpandingWindowExportModel(BaseForecastModel):
    """Export-only fast path that preserves a0's monthly expanding-window semantics."""

    name = "a0_export"

    def __init__(self, model_name: str = "a0"):
        self.model_name = model_name
        self.workspace: ExportWorkspace | None = None
        self._bundle_id: int | None = None
        self.region_bg_residual_models: dict[int, dict[int, Ridge | None]] = {}

    def _aggregation_strategy(self) -> str:
        strategy = MODEL_SPECS[self.model_name].aggregation
        if strategy != "default":
            return strategy
        if self.model_name == "a0":
            return "global_casewise"
        return "global_casewise"

    def _ensure_workspace(self, bundle: DatasetBundle, config: ExperimentConfig):
        bundle_id = id(bundle)
        if self.workspace is None or self._bundle_id != bundle_id:
            self.workspace = build_export_workspace(bundle, config)
            self._bundle_id = bundle_id

    def fit(self, train_indices, bundle, config):
        self.bundle = bundle
        self.config = config
        self.train_indices = np.array(sorted(train_indices), dtype=int)
        self.issue_index = int(self.train_indices.size)
        self.revise_baseline_monthly = None
        self.active_case_keys = set()
        self.retained_case_keys = set()
        self.retained_case_meta = {}
        self.retained_case_records = {}
        self.retain_stats = {
            "generated": 0,
            "admitted": 0,
            "rejected": 0,
            "representative_evicted": 0,
            "evicted": 0,
            "active_retained": 0,
            "retrieval_frequency": 0,
            "retained_contribution_total": 0.0,
            "prediction_count": 0,
        }
        self._ensure_workspace(bundle, config)
        self.region_ids = tuple(bundle.region_info.region_ids)
        self.region_tables = {}
        self.retrieval_mode = MODEL_SPECS[self.model_name].retrieval
        self.reuse_mode = MODEL_SPECS[self.model_name].reuse
        self.adaptation_mode = MODEL_SPECS[self.model_name].adaptation
        self.use_time_decay = MODEL_SPECS[self.model_name].time_decay
        self.use_climate_gate = MODEL_SPECS[self.model_name].climate_gate
        month_numbers = self.workspace.month_numbers
        train_count = self.issue_index
        selected_months = month_numbers[: train_count + 1]

        for rid in self.region_ids:
            region = self.workspace.regions[rid]

            sic_monthly = _prefix_monthly_mean(region.sic_matrix, train_count, month_numbers)
            sic_selected = region.sic_matrix[: train_count + 1] - sic_monthly[selected_months]
            sic_train = sic_selected[:train_count]
            sic_n_components = min(config.sic_eof_dim, sic_train.shape[0], sic_train.shape[1])
            sic_pca = PCA(n_components=sic_n_components, random_state=0)
            sic_pca.fit(sic_train)
            sic_proj = sic_pca.transform(sic_selected).astype(np.float32, copy=False)

            bg_monthly = _prefix_monthly_mean(region.bg_matrix, train_count, month_numbers)
            bg_selected = region.bg_matrix[: train_count + 1] - bg_monthly[selected_months]
            bg_train = bg_selected[:train_count]
            bg_n_components = min(config.bg_eof_dim, bg_train.shape[0], bg_train.shape[1])
            bg_pca = PCA(n_components=bg_n_components, random_state=config.random_state)
            bg_pca.fit(bg_train)
            bg_proj = bg_pca.transform(bg_selected).astype(np.float32, copy=False)

            self.region_tables[rid] = {
                "sic_proj": sic_proj,
                "sic_norm": np.linalg.norm(sic_proj, axis=1),
                "bg_proj": bg_proj,
                "bg_norm": np.linalg.norm(bg_proj, axis=1),
                "hist": region.hist[: train_count + 1],
                "local_trend": np.array(
                    [
                        trailing_slope(region.region_mean_sic[max(0, t - 11): t + 1])
                        for t in range(train_count + 1)
                    ],
                    dtype=np.float32,
                ),
            }

        self.case_tables = {}
        for lead in config.lead_times:
            candidate_count = max(self.issue_index - lead, 0)
            case_issue_indices = np.arange(candidate_count, dtype=int)
            self.case_tables[lead] = {
                "issue_indices": case_issue_indices,
                "solutions": bundle.sic[case_issue_indices + lead].astype(np.float32, copy=False),
                "is_retained": np.zeros(candidate_count, dtype=bool),
                "retained_scores": np.zeros(candidate_count, dtype=np.float32),
            }
            for idx in case_issue_indices:
                self.active_case_keys.add((int(idx), int(lead)))

        x = np.arange(train_count, dtype=np.float64)
        y = self.workspace.pan_sie[:train_count].astype(np.float64, copy=False)
        sx = np.sum(x)
        sy = np.sum(y)
        sxx = np.sum(x * x)
        sxy = np.sum(x * y)
        denom = train_count * sxx - sx * sx
        if denom == 0:
            slope = 0.0
            intercept = float(np.mean(y))
        else:
            slope = (train_count * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / train_count
        self.global_trend = np.array([slope, intercept], dtype=np.float64)
        self.region_linear_trends = None
        self.region_bg_residual_models = {}
        if self.adaptation_mode == "regional_monthly_linear":
            self.region_linear_trends = {}
            target_month_numbers = month_numbers[self.train_indices]
            for rid in self.region_ids:
                region_workspace = self.workspace.regions[rid]
                region_mean_sic = region_workspace.region_mean_sic
                region_monthly = {}
                for month in range(12):
                    idx = self.train_indices[target_month_numbers == month]
                    x = idx.astype(np.float64, copy=False)
                    y = region_mean_sic[idx].astype(np.float64, copy=False)
                    if idx.size <= 1:
                        slope = 0.0
                        intercept = float(y[0]) if idx.size == 1 else 0.0
                    else:
                        slope, intercept = np.polyfit(x, y, 1)
                    region_monthly[month] = np.array([slope, intercept], dtype=np.float64)
                self.region_linear_trends[rid] = region_monthly
                if self._uses_bg_residual():
                    lead_models: dict[int, Ridge | None] = {}
                    bg_proj = self.region_tables[rid]["bg_proj"]
                    for lead in self.config.lead_times:
                        issue_idx = np.arange(max(train_count - lead + 1, 0), dtype=int)
                        if issue_idx.size < 2:
                            lead_models[int(lead)] = None
                            continue
                        target_idx = issue_idx + int(lead)
                        x_train = bg_proj[issue_idx].astype(np.float64, copy=False)
                        y_train = np.array(
                            [
                                region_mean_sic[tgt] - self._linear_trend_surface_value(rid, int(tgt))
                                for tgt in target_idx
                            ],
                            dtype=np.float64,
                        )
                        if x_train.ndim != 2 or x_train.shape[1] == 0:
                            lead_models[int(lead)] = None
                            continue
                        model = Ridge(alpha=float(self.config.reuse_bg_residual_alpha))
                        model.fit(x_train, y_train)
                        lead_models[int(lead)] = model
                    self.region_bg_residual_models[rid] = lead_models
        if self.config.enable_retain and self.config.retain_reload:
            self._load_retained_cases()

    def _adapt_field_global_linear(self, field: np.ndarray, case_idx: int) -> np.ndarray:
        current = np.polyval(self.global_trend, self.issue_index)
        past = np.polyval(self.global_trend, case_idx)
        delta = (current - past) / max(self.bundle.land_mask.sum(), 1)
        return np.clip(field + delta, 0.0, 1.0)

    def _linear_trend_surface_value(self, rid: int, target_idx: int) -> float:
        coeffs = self.region_linear_trends[rid][target_idx % 12]
        return float(np.polyval(coeffs, target_idx))

    def _sigmoid_gate_k(self) -> float | None:
        if self.model_name.endswith("_sigmoid_k8"):
            return 8.0
        if self.model_name.endswith("_sigmoid_k20"):
            return 20.0
        return None

    def _uses_bg_gate(self) -> bool:
        return self.model_name in {
            "a0_mrlinear_bggate",
            "a0_mrlinear_bgcombo",
        }

    def _uses_bg_residual(self) -> bool:
        return self.model_name in {
            "a0_mrlinear_bgresid",
            "a0_mrlinear_bgcombo",
        }

    def _bg_gate_weight(self, rid: int, case_idx: int) -> float:
        if not self._uses_bg_gate():
            return 1.0
        region = self.region_tables[rid]
        query_bg = region["bg_proj"][self.issue_index].astype(np.float64, copy=False)
        case_bg = region["bg_proj"][case_idx].astype(np.float64, copy=False)
        dim = max(int(query_bg.shape[0]), 1)
        distance = float(np.linalg.norm(query_bg - case_bg) / np.sqrt(dim))
        tau = max(float(self.config.reuse_bg_gate_tau), 1e-6)
        return float(np.exp(-distance / tau))

    def _bg_residual_delta(self, rid: int, case_idx: int, lead: int) -> float:
        if not self._uses_bg_residual():
            return 0.0
        model = self.region_bg_residual_models.get(rid, {}).get(int(lead))
        if model is None:
            return 0.0
        region = self.region_tables[rid]
        query_bg = region["bg_proj"][self.issue_index].astype(np.float64, copy=False)[None, :]
        case_bg = region["bg_proj"][case_idx].astype(np.float64, copy=False)[None, :]
        query_resid = float(model.predict(query_bg)[0])
        case_resid = float(model.predict(case_bg)[0])
        return float(self.config.reuse_bg_residual_weight) * (query_resid - case_resid)

    def _apply_sigmoid_gated_shift(self, patch: np.ndarray, delta: float) -> np.ndarray:
        gate_k = self._sigmoid_gate_k()
        patch = np.asarray(patch, dtype=np.float32)
        if gate_k is None:
            return np.clip(patch + delta, 0.0, 1.0)
        weights = 1.0 / (1.0 + np.exp(-float(gate_k) * (patch.astype(np.float64) - 0.15)))
        shifted = patch.astype(np.float64) + weights * float(delta)
        return np.clip(shifted, 0.0, 1.0).astype(np.float32)

    def _adapt_field(self, field: np.ndarray, case_idx: int, lead: int) -> np.ndarray:
        if self.adaptation_mode == "global_linear":
            return self._adapt_field_global_linear(field, case_idx)
        if self.adaptation_mode == "regional_monthly_linear":
            out = field.copy()
            query_target_idx = self.issue_index + lead
            case_target_idx = case_idx + lead
            for rid in self.region_ids:
                mask = self.bundle.region_info.region_masks[rid]
                query_surface = self._linear_trend_surface_value(rid, query_target_idx)
                case_surface = self._linear_trend_surface_value(rid, case_target_idx)
                delta_trend = query_surface - case_surface
                delta = self._bg_gate_weight(rid, case_idx) * delta_trend
                delta += self._bg_residual_delta(rid, case_idx, lead)
                out[mask] = self._apply_sigmoid_gated_shift(out[mask], delta)
            return out
        return field

    def _aggregate_region_scores(
        self,
        issue_index: int,
        lead: int,
        candidate_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        scores = np.zeros(candidate_indices.shape[0], dtype=np.float64)
        regional_scores: dict[int, np.ndarray] = {}
        for rid in self.region_ids:
            region_score = self._candidate_scores(rid, issue_index, lead, candidate_indices)
            regional_scores[int(rid)] = region_score
            scores += region_score
        scores /= len(self.region_ids)
        return scores, regional_scores

    def _region_soft_weights(self, scores: np.ndarray, rho: float) -> np.ndarray:
        return softmax_weights(np.asarray(scores, dtype=np.float64), rho=rho)

    def _region_topm_weights(self, scores: np.ndarray, m: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return scores
        keep = min(int(m), scores.size)
        ranked_idx = np.argsort(-scores, kind="stable")[:keep]
        weights = np.zeros_like(scores, dtype=np.float64)
        weights[ranked_idx] = softmax_weights(scores[ranked_idx], rho=self.config.retrieval_rho)
        return weights

    def _aggregate_postagg_regional_monthly(
        self,
        raw_pred: np.ndarray,
        chosen_cases: np.ndarray,
        weights: np.ndarray,
        lead: int,
    ) -> np.ndarray:
        pred = np.asarray(raw_pred, dtype=np.float32).copy()
        query_target_idx = self.issue_index + lead
        for rid in self.region_ids:
            mask = self.bundle.region_info.region_masks[rid]
            query_surface = self._linear_trend_surface_value(rid, query_target_idx)
            case_surfaces = np.array(
                [self._linear_trend_surface_value(rid, int(case_idx) + lead) for case_idx in chosen_cases],
                dtype=np.float64,
            )
            source_surface = float(np.sum(np.asarray(weights, dtype=np.float64) * case_surfaces))
            pred[mask] = self._apply_sigmoid_gated_shift(pred[mask], query_surface - source_surface)
        return pred

    def _aggregate_region_gated(
        self,
        adapted_fields: np.ndarray,
        chosen_cases: np.ndarray,
        global_weights: np.ndarray,
        selected_region_scores: dict[int, np.ndarray],
        strategy: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        fused = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        regional_meta: dict[str, Any] = {}
        global_weights = np.asarray(global_weights, dtype=np.float64)
        adapted_fields = np.asarray(adapted_fields, dtype=np.float32)
        for rid in self.region_ids:
            raw_scores = np.asarray(selected_region_scores[int(rid)], dtype=np.float64)
            if strategy == "global_region_gated_sharp":
                gate = self._region_soft_weights(raw_scores, rho=2.0 * self.config.retrieval_rho)
            elif strategy == "global_region_gated_top3":
                gate = self._region_topm_weights(raw_scores, m=3)
            else:
                gate = self._region_soft_weights(raw_scores, rho=self.config.retrieval_rho)
            combined = global_weights * gate
            if np.sum(combined) <= 0:
                regional_weights = global_weights / np.sum(global_weights)
            else:
                regional_weights = combined / np.sum(combined)
            mask = self.bundle.region_info.region_masks[rid]
            region_field = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            for weight, field in zip(regional_weights, adapted_fields):
                region_field[mask] += float(weight) * field[mask]
            fused += self.bundle.region_info.fusion_weights[rid] * region_field
            regional_meta[str(rid)] = {
                "case_issue_indices": chosen_cases.astype(int).tolist(),
                "region_scores": raw_scores.astype(float).tolist(),
                "global_weights": global_weights.astype(float).tolist(),
                "regional_weights": regional_weights.astype(float).tolist(),
            }
        return np.clip(fused, 0.0, 1.0), {"mode": strategy, "regional": regional_meta}

    def _revise_case_fields(self, adapted_fields: list[np.ndarray], weights: np.ndarray, active_mask: np.ndarray):
        if not self.config.enable_revise:
            return (
                np.asarray(adapted_fields, dtype=np.float32),
                np.asarray(weights, dtype=np.float64),
                None,
            )
        return revise_cases(
            np.asarray(adapted_fields, dtype=np.float32),
            np.asarray(weights, dtype=np.float64),
            active_mask=active_mask,
            variance_quantile=float(self.config.revise_variance_quantile),
            downweight_strength=float(self.config.revise_downweight_strength),
            min_weight=float(self.config.revise_min_weight),
        )

    def _load_retained_cases(self) -> None:
        store_path = _retain_store_path(self.config, self.model_name)
        if not store_path.exists():
            return
        with open(store_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.append_retained_case(_deserialize_retained_case(json.loads(line)))

    def _retained_usefulness_for_key(self, key: tuple[int, int]) -> float:
        meta = self.retained_case_meta.get(key, {})
        return float(meta.get("score", 0.0))

    def _retained_case_boost_vector(self, issue_index: int, lead: int, candidate_indices: np.ndarray) -> np.ndarray:
        if not self.config.enable_retained_case_boost:
            return np.ones(candidate_indices.shape[0], dtype=np.float64)
        boosts = np.ones(candidate_indices.shape[0], dtype=np.float64)
        for idx, case_idx in enumerate(candidate_indices):
            key = (int(case_idx), int(lead))
            if key not in self.retained_case_keys:
                continue
            usefulness = self._retained_usefulness_for_key(key)
            age_years = max(
                float(self.bundle.time_index[issue_index].year - self.bundle.time_index[int(case_idx)].year),
                0.0,
            )
            recency = _exp_score(age_years, self.config.retained_case_recency_tau_years)
            boosts[idx] = 1.0 + min(
                float(self.config.retained_case_boost_max),
                float(self.config.retained_case_boost_strength) * usefulness * recency,
            )
        return boosts

    def _selective_retain_enabled(self) -> bool:
        return _retain_selector_enabled(self.config)

    def _active_retained_cases_for_lead(self, lead: int) -> list[dict[str, Any]]:
        return [
            self.retained_case_records[key]
            for key in self.retained_case_keys
            if int(key[1]) == int(lead) and key in self.retained_case_records
        ]

    def _representative_priority_for_case(self, retained_case: dict[str, Any]) -> float:
        current_year = int(self.bundle.time_index[self.issue_index].year)
        return retained_case_priority(
            retained_case,
            current_year=current_year,
            recency_tau_years=float(self.config.retained_case_recency_tau_years),
        )

    def _finalize_retained_candidate(self, retained_case: dict[str, Any]) -> dict[str, Any]:
        retain_metadata = retained_case.setdefault("retain_metadata", {})
        admission = retain_metadata.setdefault("admission", {})
        selection = retain_metadata.setdefault(
            "selection",
            {
                "representative_decision": "disabled",
            },
        )
        if not self._selective_retain_enabled():
            selection["final_decision"] = "admit"
            selection["admitted"] = bool(admission.get("admitted", True))
            return {"admitted": bool(admission.get("admitted", True)), "replaced_key": None}

        admission["admitted"] = True
        selection["admitted"] = True
        selection["final_decision"] = "admit"
        return {"admitted": True, "replaced_key": None}

    def _evict_retained_case(self, key: tuple[int, int]) -> bool:
        if key not in self.retained_case_keys:
            return False
        lead = int(key[1])
        table = self.case_tables.get(lead)
        if table is None:
            return False
        mask = ~(
            (table["issue_indices"] == int(key[0]))
            & table["is_retained"]
        )
        if bool(np.all(mask)):
            return False
        table["issue_indices"] = table["issue_indices"][mask]
        table["solutions"] = table["solutions"][mask]
        table["is_retained"] = table["is_retained"][mask]
        table["retained_scores"] = table["retained_scores"][mask]
        self.retained_case_keys.discard(key)
        self.retained_case_meta.pop(key, None)
        self.retained_case_records.pop(key, None)
        self.active_case_keys.discard(key)
        self.retain_stats["evicted"] += 1
        self.retain_stats["active_retained"] = len(self.retained_case_keys)
        return True

    def _enforce_retained_memory_limits(self) -> list[tuple[int, int]]:
        if not self.config.enable_bounded_retain_memory or not self.retained_case_keys:
            return []
        evicted: list[tuple[int, int]] = []

        def _candidate_sort_key(key: tuple[int, int]):
            meta = self.retained_case_meta.get(key, {})
            return (float(meta.get("score", 0.0)), int(key[0]))

        if self.config.retained_case_cap_per_lead is not None:
            per_lead = int(self.config.retained_case_cap_per_lead)
            for lead in self.config.lead_times:
                keys = [key for key in self.retained_case_keys if int(key[1]) == int(lead)]
                while len(keys) > per_lead:
                    victim = min(keys, key=_candidate_sort_key)
                    if not self._evict_retained_case(victim):
                        break
                    evicted.append(victim)
                    keys.remove(victim)
        if self.config.retained_case_cap_per_month is not None:
            per_month = int(self.config.retained_case_cap_per_month)
            month_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for key in self.retained_case_keys:
                month = int(self.bundle.time_index[int(key[0])].month)
                month_groups[month].append(key)
            for keys in month_groups.values():
                while len(keys) > per_month:
                    victim = min(keys, key=_candidate_sort_key)
                    if not self._evict_retained_case(victim):
                        break
                    evicted.append(victim)
                    keys.remove(victim)
        if self.config.retained_case_cap is not None:
            global_cap = int(self.config.retained_case_cap)
            while len(self.retained_case_keys) > global_cap:
                victim = min(self.retained_case_keys, key=_candidate_sort_key)
                if not self._evict_retained_case(victim):
                    break
                evicted.append(victim)
        self.retain_stats["active_retained"] = len(self.retained_case_keys)
        return evicted

    def _record_retained_usage(self, metadata: dict[str, Any]) -> None:
        usage = metadata.get("retained_case_usage") or {}
        self.retain_stats["prediction_count"] += 1
        self.retain_stats["retrieval_frequency"] += int(usage.get("retained_count", 0))
        self.retain_stats["retained_contribution_total"] += float(usage.get("retained_contribution", 0.0))

    def retain_diagnostics_snapshot(self) -> dict[str, Any]:
        predictions = max(int(self.retain_stats.get("prediction_count", 0)), 1)
        return {
            "generated": int(self.retain_stats.get("generated", 0)),
            "admitted": int(self.retain_stats.get("admitted", 0)),
            "rejected": int(self.retain_stats.get("rejected", 0)),
            "representative_evicted": int(self.retain_stats.get("representative_evicted", 0)),
            "evicted": int(self.retain_stats.get("evicted", 0)),
            "active_retained": int(len(self.retained_case_keys)),
            "average_retrieved_retained_cases": float(self.retain_stats.get("retrieval_frequency", 0)) / predictions,
            "average_retained_contribution": float(self.retain_stats.get("retained_contribution_total", 0.0)) / predictions,
        }

    def append_retained_case(self, retained_case: dict[str, Any]) -> dict[str, Any]:
        key = (int(retained_case["issue_index"]), int(retained_case["lead"]))
        if key in self.active_case_keys:
            return {"admitted": False, "duplicate": True}
        admission = (retained_case.get("retain_metadata") or {}).get("admission") or {}
        decision = self._finalize_retained_candidate(retained_case)
        if self.config.enable_selective_retain and not bool(decision.get("admitted", admission.get("admitted", True))):
            return decision
        lead = int(retained_case["lead"])
        table = self.case_tables.setdefault(
            lead,
            {
                "issue_indices": np.empty((0,), dtype=int),
                "solutions": np.empty((0,) + self.bundle.sic.shape[1:], dtype=np.float32),
                "is_retained": np.empty((0,), dtype=bool),
                "retained_scores": np.empty((0,), dtype=np.float32),
            },
        )
        table["issue_indices"] = np.concatenate([table["issue_indices"], np.array([key[0]], dtype=int)])
        table["solutions"] = np.concatenate(
            [table["solutions"], retained_case["global_solution"][None, ...].astype(np.float32)],
            axis=0,
        )
        table["is_retained"] = np.concatenate([table["is_retained"], np.array([True], dtype=bool)])
        table["retained_scores"] = np.concatenate(
            [table["retained_scores"], np.array([float(admission.get("score", 0.0))], dtype=np.float32)]
        )
        self.retained_case_keys.add(key)
        self.active_case_keys.add(key)
        self.retained_case_records[key] = retained_case
        self.retained_case_meta[key] = {
            "score": float(admission.get("score", 0.0)),
            "admitted": bool(admission.get("admitted", True)),
        }
        selection = retained_case.setdefault("retain_metadata", {}).setdefault("selection", {})
        selection.setdefault("representative_decision", "disabled")
        if self._selective_retain_enabled() and self.config.enable_representative_retained_memory:
            cap = int(self.config.retain_representative_cap_per_lead)
            lead_keys = [item for item in self.retained_case_keys if int(item[1]) == lead]
            if len(lead_keys) > cap:
                victim = min(
                    lead_keys,
                    key=lambda item: (
                        self._representative_priority_for_case(self.retained_case_records[item]),
                        int(item[0]),
                    ),
                )
                if victim != key and self._evict_retained_case(victim):
                    self.retain_stats["representative_evicted"] += 1
                    selection["representative_decision"] = "evict_existing"
                    selection["representative_victim"] = [int(victim[0]), int(victim[1])]
                elif victim == key and self._evict_retained_case(key):
                    self.retain_stats["representative_evicted"] += 1
                    admission["admitted"] = False
                    selection["admitted"] = False
                    selection["final_decision"] = "reject_representative"
                    selection["representative_decision"] = "reject_new_case"
                    selection["representative_victim"] = [int(key[0]), int(key[1])]
                    self.retain_stats["rejected"] += 1
                    self.retain_stats["active_retained"] = len(self.retained_case_keys)
                    return {"admitted": False, "representative_victim": key}
                else:
                    selection["representative_decision"] = "keep"
        self.retain_stats["active_retained"] = len(self.retained_case_keys)
        self._enforce_retained_memory_limits()
        self.retain_stats["admitted"] += 1
        selection["active_retained_cases_after"] = int(len(self.retained_case_keys))
        return {"admitted": True, "replaced_key": None}

    def _case_error_summary(self, prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
        diff = prediction[self.bundle.land_mask] - truth[self.bundle.land_mask]
        return {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }

    def _build_runtime_case(self, issue_index: int, lead: int, truth_field: np.ndarray) -> dict[str, Any]:
        issue_date = self.bundle.time_index[issue_index]
        pan_sie = self.workspace.pan_sie
        rolling_bg = self.workspace.rolling_bg
        record = {
            "issue_index": int(issue_index),
            "target_index": int(issue_index + lead),
            "lead": int(lead),
            "year": int(issue_date.year),
            "month": int(issue_date.month),
            "regime": "pre-2007" if issue_date.year < self.config.regime_split_year else "post-2007",
            "delta_sie": float(pan_sie[issue_index] - np.nanmean(pan_sie[max(0, issue_index - 12): issue_index + 1])),
            "global_solution": np.asarray(truth_field, dtype=np.float32),
            "regional": {},
            "rolling_bg": float(rolling_bg[issue_index]) if np.isfinite(rolling_bg[issue_index]) else float(pan_sie[issue_index]),
        }
        for rid in self.region_ids:
            region = self.region_tables[rid]
            record["regional"][rid] = {
                "z_ice": region["sic_proj"][issue_index].astype(np.float32),
                "z_bg": region["bg_proj"][issue_index].astype(np.float32),
                "h": float(region["hist"][issue_index]),
                "local_trend": float(region["local_trend"][issue_index]),
                "solution": np.asarray(truth_field, dtype=np.float32),
            }
        return record

    def _summarize_adaptation_magnitude(self, lead: int, metadata: dict[str, Any]) -> dict[str, Any]:
        trace = metadata.get("retain_trace") or {}
        if trace.get("mode") == "global":
            magnitudes = []
            weighted = 0.0
            total_weight = 0.0
            for case_idx, weight in zip(trace.get("case_issue_indices", []), trace.get("revised_weights", [])):
                raw = self.bundle.sic[int(case_idx) + lead]
                adapted = self._adapt_field(raw, int(case_idx), lead)
                magnitude = float(np.mean(np.abs(adapted[self.bundle.land_mask] - raw[self.bundle.land_mask])))
                magnitudes.append(magnitude)
                weighted += float(weight) * magnitude
                total_weight += float(weight)
            return {
                "mode": "global",
                "mean_abs_delta": float(np.mean(magnitudes)) if magnitudes else 0.0,
                "weighted_mean_abs_delta": float(weighted / total_weight) if total_weight > 0 else 0.0,
            }
        if trace.get("mode") == "compositional":
            regional = {}
            collected = []
            for rid_key, payload in trace.get("regional", {}).items():
                rid = int(rid_key)
                mask = self.bundle.region_info.region_masks[rid]
                magnitudes = []
                weighted = 0.0
                total_weight = 0.0
                for case_idx, weight in zip(payload.get("case_issue_indices", []), payload.get("revised_weights", [])):
                    raw = self.bundle.sic[int(case_idx) + lead]
                    if self.adaptation_mode == "regional_monthly_linear":
                        query_surface = self._linear_trend_surface_value(rid, self.issue_index + lead)
                        case_surface = self._linear_trend_surface_value(rid, int(case_idx) + lead)
                        adapted = raw.copy()
                        adapted[mask] = np.clip(raw[mask] - case_surface + query_surface, 0.0, 1.0)
                    else:
                        adapted = self._adapt_field(raw, int(case_idx), lead)
                    magnitude = float(np.mean(np.abs(adapted[mask] - raw[mask])))
                    magnitudes.append(magnitude)
                    collected.append(magnitude)
                    weighted += float(weight) * magnitude
                    total_weight += float(weight)
                regional[rid] = {
                    "mean_abs_delta": float(np.mean(magnitudes)) if magnitudes else 0.0,
                    "weighted_mean_abs_delta": float(weighted / total_weight) if total_weight > 0 else 0.0,
                }
            return {"mode": "compositional", "mean_abs_delta": float(np.mean(collected)) if collected else 0.0, "regional": regional}
        return {"mode": "unknown", "mean_abs_delta": 0.0}

    def retain_case(
        self,
        issue_index: int,
        lead: int,
        prediction_result: PredictionResult,
        truth_field: np.ndarray,
    ) -> dict[str, Any] | None:
        if not self.config.enable_retain:
            return None
        self.retain_stats["generated"] += 1
        retained = self._build_runtime_case(issue_index, lead, truth_field)
        retained["source"] = "retained"
        retain_metadata = {
            "predicted_solution": np.asarray(prediction_result.prediction, dtype=np.float32),
            "forecast_error": self._case_error_summary(prediction_result.prediction, truth_field),
            "disagreement_level": float(prediction_result.metadata.get("revise", {}).get("disagreement_fraction", 0.0)),
            "adaptation_magnitude": self._summarize_adaptation_magnitude(lead, prediction_result.metadata),
            "revise_statistics": prediction_result.metadata.get("revise"),
        }
        admission = score_retained_case(
            retain_metadata,
            config=self.config,
            issue_year=int(self.bundle.time_index[issue_index].year),
            current_year=int(self.bundle.time_index[self.issue_index].year),
        )
        retain_metadata["admission"] = admission
        retain_metadata["selection"] = {
            "representative_decision": "pending" if self.config.enable_representative_retained_memory else "disabled",
        }
        retained["retain_metadata"] = {
            **retain_metadata,
            "retained_usefulness_score": float(admission["score"]),
        }
        return retained

    def _candidate_scores(self, rid: int, issue_index: int, lead: int, candidate_indices: np.ndarray) -> np.ndarray:
        region = self.region_tables[rid]
        q_ice = region["sic_proj"][issue_index]
        q_bg = region["bg_proj"][issue_index]
        q_hist = float(region["hist"][issue_index])

        q_ice_norm = float(np.linalg.norm(q_ice))
        q_bg_norm = float(np.linalg.norm(q_bg))
        cand_ice = region["sic_proj"][candidate_indices]
        cand_bg = region["bg_proj"][candidate_indices]
        cand_hist = region["hist"][candidate_indices]
        eps = 1e-10

        ice = (cand_ice @ q_ice) / (region["sic_norm"][candidate_indices] * q_ice_norm + eps)
        bg = (cand_bg @ q_bg) / (region["bg_norm"][candidate_indices] * q_bg_norm + eps)
        hist = 1.0 / (1.0 + np.abs(cand_hist - q_hist))
        weights_cfg = self.config.sim_weights
        score = weights_cfg["ice"] * ice + weights_cfg["bg"] * bg + weights_cfg["hist"] * hist

        if self.retrieval_mode == "agr":
            years = self.bundle.time_index.year
            query_year = int(years[issue_index])
            case_years = years[candidate_indices]
            n_years = max(1, self.config.test_end_year - self.config.start_year + 1)
            d_time = np.abs(query_year - case_years) / n_years
            query_rolling = self.workspace.rolling_bg[issue_index]
            if not np.isfinite(query_rolling):
                query_rolling = self.workspace.pan_sie[issue_index]
            case_rolling = self.workspace.rolling_bg[candidate_indices].copy()
            invalid = ~np.isfinite(case_rolling)
            if np.any(invalid):
                case_rolling[invalid] = self.workspace.pan_sie[candidate_indices[invalid]]
            query_rolling = float(query_rolling)
            rolling_scale = np.maximum.reduce(
                [
                    np.abs(np.full(candidate_indices.shape[0], query_rolling, dtype=np.float32)),
                    np.abs(case_rolling),
                    np.ones(candidate_indices.shape[0], dtype=np.float32),
                ]
            )
            d_regime = np.abs(query_rolling - case_rolling) / rolling_scale
            q_local = float(region["local_trend"][issue_index])
            d_trend = np.abs(q_local - region["local_trend"][candidate_indices])
            aw = self.config.adapt_weights
            adapt_cost = aw["time"] * d_time + aw["regime"] * d_regime + aw["trend"] * d_trend
            adaptability = np.exp(-adapt_cost / (2 * self.config.adaptability_tau**2))
            score = score * adaptability
        if self.use_time_decay:
            years = self.bundle.time_index.year
            query_year = int(years[issue_index])
            case_years = years[candidate_indices]
            tau = max(float(self.config.retrieval_time_decay_tau_years), 1e-6)
            score = score * np.exp(-np.abs(query_year - case_years) / tau)
        if self.use_climate_gate:
            query_rolling = self.workspace.rolling_bg[issue_index]
            if not np.isfinite(query_rolling):
                query_rolling = self.workspace.pan_sie[issue_index]
            case_rolling = self.workspace.rolling_bg[candidate_indices].copy()
            invalid = ~np.isfinite(case_rolling)
            if np.any(invalid):
                case_rolling[invalid] = self.workspace.pan_sie[candidate_indices[invalid]]
            scale = np.maximum.reduce(
                [
                    np.abs(np.full(candidate_indices.shape[0], query_rolling, dtype=np.float32)),
                    np.abs(case_rolling),
                    np.ones(candidate_indices.shape[0], dtype=np.float32),
                ]
            )
            tau = max(float(self.config.climate_gate_tau), 1e-6)
            score = score * np.exp(-np.abs(query_rolling - case_rolling) / (scale * tau))
        if candidate_indices.size:
            score = score * self._retained_case_boost_vector(issue_index, lead, candidate_indices)
        return score.astype(np.float64, copy=False)

    def _predict_global(self, issue_index: int, lead: int, candidate_indices: np.ndarray, solutions: np.ndarray) -> PredictionResult:
        aggregation_strategy = self._aggregation_strategy()
        scores, regional_scores = self._aggregate_region_scores(issue_index, lead, candidate_indices)
        top_k = min(self.config.top_k, candidate_indices.shape[0])
        ranked_idx = np.argsort(-scores, kind="stable")[:top_k]
        weights = softmax_weights(scores[ranked_idx], rho=self.config.retrieval_rho)
        chosen_cases = candidate_indices[ranked_idx]
        adapted_fields = [self._adapt_field(field, int(case_idx), lead) for case_idx, field in zip(chosen_cases, solutions[ranked_idx])]
        revised_fields, revised_weights, revise_meta = self._revise_case_fields(
            adapted_fields,
            weights,
            self.bundle.land_mask,
        )
        revised_fields_arr = np.asarray(revised_fields, dtype=np.float32)
        pred = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        region_gate_meta = None
        if aggregation_strategy == "global_postagg" and self.adaptation_mode == "regional_monthly_linear":
            raw_pred = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            for weight, field in zip(revised_weights, solutions[ranked_idx]):
                raw_pred += float(weight) * field
            pred = self._aggregate_postagg_regional_monthly(raw_pred, chosen_cases, revised_weights, lead)
        elif aggregation_strategy.startswith("global_region_gated"):
            selected_region_scores = {rid: regional_scores[rid][ranked_idx] for rid in self.region_ids}
            pred, region_gate_meta = self._aggregate_region_gated(
                revised_fields_arr,
                chosen_cases,
                revised_weights,
                selected_region_scores,
                aggregation_strategy,
            )
        else:
            for weight, field in zip(revised_weights, revised_fields_arr):
                pred += float(weight) * field
        metadata = {}
        if revise_meta is not None:
            metadata["revise"] = revise_meta
        if region_gate_meta is not None:
            metadata["region_gate"] = region_gate_meta
        metadata["aggregation_strategy"] = aggregation_strategy
        retained_mask = np.array([(int(case_idx), int(lead)) in self.retained_case_keys for case_idx in chosen_cases], dtype=bool)
        metadata["retained_case_usage"] = {
            "retained_count": int(np.sum(retained_mask)),
            "retained_contribution": float(np.sum(revised_weights[retained_mask])) if retained_mask.size else 0.0,
        }
        metadata["retain_trace"] = {
            "mode": "global",
            "case_issue_indices": chosen_cases.astype(int).tolist(),
            "original_weights": weights.astype(float).tolist(),
            "revised_weights": revised_weights.astype(float).tolist(),
        }
        self._record_retained_usage(metadata)
        return PredictionResult(np.clip(pred, 0.0, 1.0), metadata)

    def _predict_compositional(self, issue_index: int, lead: int, candidate_indices: np.ndarray, solutions: np.ndarray) -> PredictionResult:
        fused = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
        delta_scale = max(self.bundle.land_mask.sum(), 1)
        current = np.polyval(self.global_trend, self.issue_index)
        revise_entries = []
        retain_regional = {}
        for rid in self.region_ids:
            scores = self._candidate_scores(rid, issue_index, lead, candidate_indices)
            top_k = min(self.config.top_k, candidate_indices.shape[0])
            ranked_idx = np.argsort(-scores, kind="stable")[:top_k]
            weights = softmax_weights(scores[ranked_idx], rho=self.config.retrieval_rho)

            region_field = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            mask = self.bundle.region_info.region_masks[rid]
            chosen_cases = candidate_indices[ranked_idx]
            adapted_fields = []
            for case_idx, field in zip(chosen_cases, solutions[ranked_idx]):
                if self.adaptation_mode == "regional_monthly_linear":
                    query_surface = self._linear_trend_surface_value(rid, self.issue_index + lead)
                    case_surface = self._linear_trend_surface_value(rid, int(case_idx) + lead)
                    adapted_patch = np.clip(field[mask] - case_surface + query_surface, 0.0, 1.0)
                else:
                    past = np.polyval(self.global_trend, int(case_idx))
                    delta = (current - past) / delta_scale
                    adapted_patch = np.clip(field[mask] + delta, 0.0, 1.0)
                adapted_field = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
                adapted_field[mask] = adapted_patch
                adapted_fields.append(adapted_field)
            revised_fields, revised_weights, revise_meta = self._revise_case_fields(
                adapted_fields,
                weights,
                mask,
            )
            for weight, adapted_field in zip(revised_weights, revised_fields):
                region_field += float(weight) * adapted_field
            if revise_meta is not None:
                revise_meta["region_id"] = int(rid)
                revise_meta["active_grid_count"] = int(np.sum(mask))
                revise_entries.append(revise_meta)
            retain_regional[str(rid)] = {
                "case_issue_indices": chosen_cases.astype(int).tolist(),
                "original_weights": weights.astype(float).tolist(),
                "revised_weights": revised_weights.astype(float).tolist(),
            }
            fused += self.bundle.region_info.fusion_weights[rid] * region_field
        metadata = {}
        if revise_entries:
            total_cells = sum(entry.get("active_grid_count", 0) for entry in revise_entries)
            disagreement_fraction = 0.0
            if total_cells > 0:
                disagreement_fraction = float(
                    sum(entry.get("disagreement_fraction", 0.0) * entry.get("active_grid_count", 0) for entry in revise_entries)
                    / total_cells
                )
            metadata["revise"] = {"regional": revise_entries, "disagreement_fraction": disagreement_fraction}
        retained_count = sum(
            1
            for payload in retain_regional.values()
            for case_idx in payload["case_issue_indices"]
            if (int(case_idx), int(lead)) in self.retained_case_keys
        )
        retained_contribution = float(
            np.mean(
                [
                    sum(
                        float(weight)
                        for case_idx, weight in zip(payload["case_issue_indices"], payload["revised_weights"])
                        if (int(case_idx), int(lead)) in self.retained_case_keys
                    )
                    for payload in retain_regional.values()
                ]
            )
        ) if retain_regional else 0.0
        metadata["retained_case_usage"] = {
            "retained_count": int(retained_count),
            "retained_contribution": retained_contribution,
        }
        metadata["retain_trace"] = {"mode": "compositional", "regional": retain_regional}
        self._record_retained_usage(metadata)
        return PredictionResult(np.clip(fused, 0.0, 1.0), metadata)

    def predict(self, issue_index: int, lead: int) -> PredictionResult:
        if issue_index != self.issue_index:
            raise ValueError(f"Expected issue_index {self.issue_index}, got {issue_index}")
        case_table = self.case_tables[lead]
        candidate_indices = case_table["issue_indices"]
        if candidate_indices.size == 0:
            empty = np.zeros_like(self.bundle.sic[0], dtype=np.float32)
            return PredictionResult(empty, {})
        if self.reuse_mode == "global":
            return self._predict_global(issue_index, lead, candidate_indices, case_table["solutions"])
        return self._predict_compositional(issue_index, lead, candidate_indices, case_table["solutions"])


MODEL_SPECS = {
    "analog_regional_sic_bg_hist": ModelSpec("standard", "none", "global", "static"),
    "standard_analog_knn": ModelSpec("standard", "none", "global", "static"),
    "analog_knn_global_detrend": ModelSpec("standard", "global_linear", "global", "static"),
    "icecbr_lite": ModelSpec("agr", "regional_nonlinear", "global", "static"),
    "icecbr_full": ModelSpec("agr", "regional_nonlinear", "compositional", "competence"),
    "v0": ModelSpec("standard", "none", "global", "static"),
    "v1": ModelSpec("standard", "global_linear", "global", "static"),
    "v2": ModelSpec("standard", "regional_nonlinear", "global", "static"),
    "v3": ModelSpec("agr", "regional_nonlinear", "global", "static"),
    "v4": ModelSpec("standard", "regional_nonlinear", "compositional", "static"),
    "v5": ModelSpec("agr", "regional_nonlinear", "compositional", "static"),
    "v6": ModelSpec("agr", "regional_nonlinear", "compositional", "competence"),
    "a0": ModelSpec("standard", "global_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear_bggate": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear_bgresid": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear_bgcombo": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear_postagg": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_postagg"),
    "a0_mrlinear_sigmoid_k8": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear_sigmoid_k20": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise"),
    "a0_mrlinear_postagg_sigmoid_k8": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_postagg"),
    "a0_mrlinear_postagg_sigmoid_k20": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_postagg"),
    "a0_mrlinear_gate_soft": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_region_gated_soft"),
    "a0_mrlinear_gate_sharp": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_region_gated_sharp"),
    "a0_mrlinear_gate_top3": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_region_gated_top3"),
    "a0_mrlinear_recency": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise", time_decay=True),
    "a0_mrlinear_climategate": ModelSpec("standard", "regional_monthly_linear", "global", "static", aggregation="global_casewise", climate_gate=True),
    "a1": ModelSpec("agr", "global_linear", "global", "static"),
    "a2": ModelSpec("standard", "global_linear", "compositional", "static"),
    "a3": ModelSpec("agr", "global_linear", "compositional", "static"),
    "a4": ModelSpec("agr", "global_linear", "compositional", "competence"),
}


def build_model(name: str) -> BaseForecastModel:
    if name == "climatology":
        return ClimatologyModel()
    if name == "damped_persistence":
        return DampedPersistenceModel()
    if name == "linear_trend_persistence":
        return LinearTrendPersistenceModel()
    if name == "analog_global_sic":
        return GlobalAnalogueExportModel(mode="sic_only")
    if name == "analog_global_sic_bg_hist":
        return GlobalAnalogueExportModel(mode="sic_bg_hist")
    if name == "analog_regional_sic_bg_hist":
        return StandardAnalogKNNExportModel()
    if name == "ridge":
        return FeatureRegressorModel(Ridge(alpha=1.0))
    if name == "random_forest":
        return FeatureRegressorModel(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    if name in {
        "a0_mrlinear_bggate",
        "a0_mrlinear_bgresid",
        "a0_mrlinear_bgcombo",
        "a0_mrlinear_postagg",
        "a0_mrlinear_sigmoid_k8",
        "a0_mrlinear_sigmoid_k20",
        "a0_mrlinear_postagg_sigmoid_k8",
        "a0_mrlinear_postagg_sigmoid_k20",
        "a0_mrlinear_gate_soft",
        "a0_mrlinear_gate_sharp",
        "a0_mrlinear_gate_top3",
    }:
        return A0ExpandingWindowExportModel(model_name=name)
    if name in MODEL_SPECS:
        return ConfigurableCBRModel(MODEL_SPECS[name], name=name)
    raise KeyError(f"Unknown model name: {name}")


def build_export_model(name: str) -> BaseForecastModel:
    if name in {
        "a0",
        "a0_mrlinear",
        "a0_mrlinear_bggate",
        "a0_mrlinear_bgresid",
        "a0_mrlinear_bgcombo",
        "a0_mrlinear_postagg",
        "a0_mrlinear_sigmoid_k8",
        "a0_mrlinear_sigmoid_k20",
        "a0_mrlinear_postagg_sigmoid_k8",
        "a0_mrlinear_postagg_sigmoid_k20",
        "a0_mrlinear_gate_soft",
        "a0_mrlinear_gate_sharp",
        "a0_mrlinear_gate_top3",
        "a0_mrlinear_recency",
        "a0_mrlinear_climategate",
        "a2",
        "a3",
    }:
        return A0ExpandingWindowExportModel(model_name=name)
    if name in {"analog_regional_sic_bg_hist", "standard_analog_knn"}:
        return StandardAnalogKNNExportModel()
    if name == "analog_global_sic":
        return GlobalAnalogueExportModel(mode="sic_only")
    if name == "analog_global_sic_bg_hist":
        return GlobalAnalogueExportModel(mode="sic_bg_hist")
    return build_model(name)
