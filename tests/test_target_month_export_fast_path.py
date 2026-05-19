from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from icecbr import models as models_module
from icecbr.config import ExperimentConfig
from icecbr.data import DatasetBundle, RegionInfo, load_background_fields
from icecbr.models import (
    _retain_store_path,
    build_export_model,
    build_model,
    revise_cases,
    score_retained_case,
)
from icecbr.pipelines import _export_internal_model_target_months, _target_index_range


def _build_synthetic_bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    n_times, height, width = 48, 2, 2
    sic = rng.random((n_times, height, width), dtype=np.float32)
    land_mask = np.ones((height, width), dtype=bool)
    background = {
        "bg": rng.random((n_times, height, width), dtype=np.float32),
    }
    region_masks = {
        1: np.array([[True, True], [False, False]]),
        2: np.array([[False, False], [True, True]]),
    }
    fusion_weights = {
        1: region_masks[1].astype(np.float32),
        2: region_masks[2].astype(np.float32),
    }
    region_info = RegionInfo(
        region_ids=[1, 2],
        region_names={1: "north", 2: "south"},
        region_masks=region_masks,
        fusion_weights=fusion_weights,
        ocean_mask=land_mask.copy(),
        surface_mask=np.zeros((height, width), dtype=np.int32),
    )
    time_index = pd.date_range(start="1979-01-01", periods=n_times, freq="MS")
    return DatasetBundle(
        sic=sic,
        land_mask=land_mask,
        background=background,
        region_info=region_info,
        time_index=time_index,
    )


def _reference_export(model_name: str, config: ExperimentConfig, bundle: DatasetBundle) -> dict:
    target_indices = list(_target_index_range(config, bundle))
    target_dates = bundle.time_index[target_indices]
    leads = list(config.lead_times)
    height, width = bundle.sic.shape[1:]
    n_targets = len(target_indices)
    n_leads = len(leads)

    unique_issue_indices = sorted({target_idx - lead for target_idx in target_indices for lead in leads})
    unique_issue_indices = [idx for idx in unique_issue_indices if idx >= 0]

    pred_cache: dict[tuple[int, int], np.ndarray] = {}
    for issue_index in unique_issue_indices:
        train_indices = np.arange(issue_index)
        model = build_model(model_name)
        model.fit(train_indices, bundle, config)
        for lead in leads:
            target_index = issue_index + lead
            if target_index >= bundle.sic.shape[0]:
                continue
            result = model.predict(issue_index, lead)
            pred_cache[(issue_index, lead)] = result.prediction.astype(np.float32)

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
    return {
        "preds": preds,
        "truths": truths,
        "target_ym": target_ym,
        "issue_ym": issue_ym,
        "lead_months": np.asarray(leads, dtype=np.int32),
    }


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


class TargetMonthExportFastPathTest(unittest.TestCase):
    def test_load_background_fields_uses_fixed_normalized_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            era_norm = root / "ERA5" / "normalized"
            oras_norm = root / "ORAS5" / "normalized"
            era_norm.mkdir(parents=True)
            oras_norm.mkdir(parents=True)

            anomaly_name = "normalized_psl_1979-01_2023-12_anomaly.npy"
            oras_name = "normalized_ohc700_1979-01_2023-12_anomaly.npy"

            anomaly = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            oras = np.arange(12, 24, dtype=np.float32).reshape(2, 2, 3)
            np.save(era_norm / anomaly_name, anomaly)
            np.save(oras_norm / oras_name, oras)

            config = ExperimentConfig(
                era5_normalized_dir=era_norm,
                oras5_normalized_dir=oras_norm,
                background_variables=(anomaly_name, oras_name),
            )
            fields = load_background_fields(config)

            self.assertEqual(set(fields), {anomaly_name, oras_name})
            np.testing.assert_array_equal(fields[anomaly_name], np.moveaxis(anomaly, -1, 0))
            np.testing.assert_array_equal(fields[oras_name], np.moveaxis(oras, -1, 0))

    def test_revise_cases_downweights_outlier_case(self):
        active_mask = np.ones((2, 2), dtype=bool)
        fields = np.array(
            [
                [[0.2, 0.2], [0.2, 0.2]],
                [[0.2, 0.2], [0.2, 0.2]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        weights = np.array([0.3, 0.3, 0.4], dtype=np.float64)
        revised_fields, revised_weights, metadata = revise_cases(
            fields,
            weights,
            active_mask=active_mask,
            variance_quantile=0.9,
            downweight_strength=2.0,
            min_weight=0.05,
        )

        np.testing.assert_allclose(revised_fields, fields, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(float(np.sum(revised_weights)), 1.0)
        self.assertLess(revised_weights[2], weights[2])
        self.assertGreaterEqual(metadata["num_downweighted"], 1)
        self.assertEqual(len(metadata["case_reliability_scores"]), 3)
        self.assertLess(metadata["case_reliability_scores"][2], metadata["case_reliability_scores"][0])
        self.assertTrue(metadata["downweight_factors"][2] < 0.999)

    def test_retain_case_appends_to_case_base(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            enable_retain=True,
        )

        model = build_model("a0_mrlinear")
        model.fit(np.arange(36), bundle, config)
        initial_len = len(model.case_base)
        result = model.predict(36, 1)
        retained_case = model.retain_case(36, 1, result, bundle.sic[37])

        self.assertIsNotNone(retained_case)
        self.assertEqual(retained_case["source"], "retained")
        self.assertIn("forecast_error", retained_case["retain_metadata"])
        model.append_retained_case(retained_case)
        self.assertEqual(len(model.case_base), initial_len + 1)
        self.assertIn((36, 1), model.case_lookup)

    def test_selective_retain_scores_and_rejects_noisy_case_in_strict_mode(self):
        config = ExperimentConfig(enable_selective_retain=True, retain_admission_mode="strict")
        good = score_retained_case(
            {
                "forecast_error": {"rmse": 0.03},
                "disagreement_level": 0.05,
                "adaptation_magnitude": {"weighted_mean_abs_delta": 0.03},
            },
            config=config,
            issue_year=2022,
            current_year=2023,
        )
        bad = score_retained_case(
            {
                "forecast_error": {"rmse": 0.18},
                "disagreement_level": 0.6,
                "adaptation_magnitude": {"weighted_mean_abs_delta": 0.12},
            },
            config=config,
            issue_year=2010,
            current_year=2023,
        )
        self.assertGreater(good["score"], bad["score"])
        self.assertTrue(good["admitted"])
        self.assertFalse(bad["admitted"])

    def test_bounded_retain_memory_evicts_retained_only(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            enable_retain=True,
            enable_selective_retain=True,
            retain_admission_mode="permissive",
            enable_bounded_retain_memory=True,
            retained_case_cap=1,
            retained_case_cap_per_lead=1,
        )
        model = build_model("a0_mrlinear")
        model.fit(np.arange(36), bundle, config)
        archive_len = len(model.case_base)
        first = model.retain_case(36, 1, model.predict(36, 1), bundle.sic[37])
        second = model.retain_case(37, 1, model.predict(37, 1), bundle.sic[38])
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        model.append_retained_case(first)
        model.append_retained_case(second)
        self.assertEqual(len(model.retained_case_keys), 1)
        self.assertGreaterEqual(len(model.case_base), archive_len)
        self.assertEqual(len(model.case_base), archive_len + 1)

    def test_export_representative_memory_caps_per_lead(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            enable_retain=True,
            enable_selective_retain=True,
            enable_representative_retained_memory=True,
            retain_representative_cap_per_lead=1,
        )
        model = build_model("a0_mrlinear")
        model.fit(np.arange(36), bundle, config)
        first = model.retain_case(36, 1, model.predict(36, 1), bundle.sic[37])
        second = model.retain_case(37, 1, model.predict(37, 1), bundle.sic[38])
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        first["retain_metadata"]["admission"]["score"] = 0.4
        first["retain_metadata"]["retained_usefulness_score"] = 0.4
        second["retain_metadata"]["admission"]["score"] = 0.9
        second["retain_metadata"]["retained_usefulness_score"] = 0.9
        model.append_retained_case(first)
        decision = model.append_retained_case(second)

        self.assertTrue(decision["admitted"])
        self.assertEqual(len(model.retained_case_keys), 1)
        self.assertEqual(model.retain_stats["representative_evicted"], 1)

    def test_retained_case_boost_changes_prediction_when_enabled(self):
        bundle = _build_synthetic_bundle()
        common = dict(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=40,
            enable_retain=True,
            enable_selective_retain=True,
            retain_admission_mode="permissive",
            retain_online_growth=True,
        )
        base_model = build_model("a0_mrlinear")
        base_model.fit(np.arange(36), bundle, ExperimentConfig(**common))
        retained_case = base_model.retain_case(36, 1, base_model.predict(36, 1), bundle.sic[37])
        base_model.append_retained_case(retained_case)
        base_pred = base_model.predict(37, 1)

        boost_model = build_model("a0_mrlinear")
        boost_model.fit(np.arange(36), bundle, ExperimentConfig(**common, enable_retained_case_boost=True))
        retained_case_boost = boost_model.retain_case(36, 1, boost_model.predict(36, 1), bundle.sic[37])
        boost_model.append_retained_case(retained_case_boost)
        boost_pred = boost_model.predict(37, 1)

        base_neighbors = base_pred.metadata.get("global_neighbors", [])
        boosted_neighbors = boost_pred.metadata.get("global_neighbors", [])
        self.assertTrue(any(item.get("retained_boost_multiplier", 1.0) == 1.0 for item in base_neighbors))
        self.assertTrue(any(item.get("retained_boost_multiplier", 1.0) > 1.0 for item in boosted_neighbors if item.get("retained_case")))

    def test_retain_reload_loads_sidecar_cases(self):
        bundle = _build_synthetic_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            common = dict(
                root_dir=Path(tmpdir),
                results_dir=Path(tmpdir),
                start_year=1979,
                target_eval_start_year=1982,
                target_eval_end_year=1982,
                lead_times=(1, 2),
                issue_months=tuple(range(1, 13)),
                background_variables=("bg",),
                sic_eof_dim=2,
                bg_eof_dim=1,
                top_k=3,
                enable_retain=True,
                output_tag="retain_test",
                retain_namespace="target_month_export",
            )
            config = ExperimentConfig(**common)
            model = build_model("a0_mrlinear")
            model.fit(np.arange(36), bundle, config)
            retained_case = model.retain_case(36, 1, model.predict(36, 1), bundle.sic[37])
            store_path = _retain_store_path(config, "a0_mrlinear")
            store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(store_path, "w") as f:
                f.write(json.dumps(_json_safe(retained_case)))
                f.write("\n")

            reload_config = ExperimentConfig(**common, retain_reload=True)
            reloaded = build_model("a0_mrlinear")
            reloaded.fit(np.arange(36), bundle, reload_config)
            self.assertIn((36, 1), reloaded.case_lookup)

    def test_export_with_retain_writes_sidecar_cases(self):
        bundle = _build_synthetic_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                root_dir=Path(tmpdir),
                results_dir=Path(tmpdir),
                output_tag="retain_export",
                start_year=1979,
                target_eval_start_year=1982,
                target_eval_end_year=1982,
                lead_times=(1, 2),
                issue_months=tuple(range(1, 13)),
                background_variables=("bg",),
                sic_eof_dim=2,
                bg_eof_dim=1,
                top_k=3,
                enable_retain=True,
            )

            actual = _export_internal_model_target_months("a0_mrlinear", config, bundle)
            lookup_config = ExperimentConfig(
                root_dir=Path(tmpdir),
                results_dir=Path(tmpdir),
                output_tag="retain_export",
                retain_namespace="target_month_export",
            )
            store_path = _retain_store_path(lookup_config, "a0_mrlinear")
            self.assertTrue(store_path.exists())
            with open(store_path, "r") as f:
                retained_lines = [line for line in f if line.strip()]
            self.assertGreater(len(retained_lines), 0)
            self.assertTrue(np.isfinite(actual["preds"]).any())
            self.assertIn("retain_diagnostics", actual)
            self.assertIn("size_trace", actual["retain_diagnostics"])

    def test_revise_disabled_keeps_a0_export_identical(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            enable_revise=False,
        )

        expected = _reference_export("a0", config, bundle)
        actual = _export_internal_model_target_months("a0", config, bundle)

        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a0_mrlinear_revise_export_matches_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            enable_revise=True,
        )

        expected = _reference_export("a0_mrlinear", config, bundle)
        actual = _export_internal_model_target_months("a0_mrlinear", config, bundle)

        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a0_export_matches_reference_and_skips_region_trends(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        model = build_model("a0")
        model.fit(np.arange(36), bundle, config)
        self.assertIsNotNone(model.global_trend)
        self.assertIsNone(model.lowess)
        self.assertIsNone(model.region_trends)
        result = model.predict(36, 1)
        self.assertNotIn("revise", result.metadata)

        expected = _reference_export("a0", config, bundle)
        actual = _export_internal_model_target_months("a0", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a0_mrlinear_export_matches_reference_and_uses_linear_region_trends(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        model = build_model("a0_mrlinear")
        model.fit(np.arange(36), bundle, config)
        self.assertIsNone(model.global_trend)
        self.assertIsNone(model.lowess)
        self.assertIsNone(model.region_trends)
        self.assertIsNotNone(model.region_linear_trends)

        expected = _reference_export("a0_mrlinear", config, bundle)
        actual = _export_internal_model_target_months("a0_mrlinear", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a0_mrlinear_postagg_export_matches_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        expected = _reference_export("a0_mrlinear_postagg", config, bundle)
        actual = _export_internal_model_target_months("a0_mrlinear_postagg", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_gated_reuse_exports_match_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        for model_name in ("a0_mrlinear_gate_soft", "a0_mrlinear_gate_sharp", "a0_mrlinear_gate_top3"):
            expected = _reference_export(model_name, config, bundle)
            actual = _export_internal_model_target_months(model_name, config, bundle)

            np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
            np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
            np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
            np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
            np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_sigmoid_reuse_exports_match_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        for model_name in (
            "a0_mrlinear_sigmoid_k8",
            "a0_mrlinear_sigmoid_k20",
            "a0_mrlinear_postagg_sigmoid_k8",
            "a0_mrlinear_postagg_sigmoid_k20",
        ):
            expected = _reference_export(model_name, config, bundle)
            actual = _export_internal_model_target_months(model_name, config, bundle)

            np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
            np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
            np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
            np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
            np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a0_mrlinear_recency_export_matches_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            retrieval_time_decay_tau_years=2.0,
        )

        expected = _reference_export("a0_mrlinear_recency", config, bundle)
        actual = _export_internal_model_target_months("a0_mrlinear_recency", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a0_mrlinear_climategate_export_matches_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            climate_gate_tau=0.5,
        )

        expected = _reference_export("a0_mrlinear_climategate", config, bundle)
        actual = _export_internal_model_target_months("a0_mrlinear_climategate", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_time_decay_prefers_recent_cases(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            test_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            retrieval_time_decay_tau_years=1.0,
        )

        model = build_model("a0_mrlinear_recency")
        model.fit(np.arange(36), bundle, config)
        recent_case = {"issue_index": 35, "lead": 1}
        old_case = {"issue_index": 0, "lead": 1}
        recent_weight = model._time_decay_weight(36, recent_case["issue_index"])
        old_weight = model._time_decay_weight(36, old_case["issue_index"])
        self.assertGreater(recent_weight, old_weight)

    def test_climate_gate_prefers_closer_background_state(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
            climate_gate_tau=0.5,
        )

        model = build_model("a0_mrlinear_climategate")
        model.fit(np.arange(36), bundle, config)
        query_idx = 36
        gate_scores = []
        for case_idx in (5, 20, 30):
            gate_scores.append(
                (
                    abs(
                        (model.feature_store["rolling_bg"][query_idx] if np.isfinite(model.feature_store["rolling_bg"][query_idx]) else model.feature_store["pan_sie"][query_idx])
                        - (model.feature_store["rolling_bg"][case_idx] if np.isfinite(model.feature_store["rolling_bg"][case_idx]) else model.feature_store["pan_sie"][case_idx])
                    ),
                    model._climate_gate_weight(query_idx, case_idx),
                )
            )
        gate_scores.sort(key=lambda x: x[0])
        self.assertGreaterEqual(gate_scores[0][1], gate_scores[-1][1])

    def test_analog_regional_sic_bg_hist_skips_region_trend_setup(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        model = build_model("analog_regional_sic_bg_hist")
        model.fit(np.arange(36), bundle, config)
        self.assertFalse(hasattr(model, "global_trend"))
        self.assertFalse(hasattr(model, "lowess"))
        self.assertFalse(hasattr(model, "region_trends"))
        self.assertIsNotNone(model.workspace)

    def test_postagg_local_detrend_differs_from_casewise_with_clipping(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=2,
        )

        model = build_model("a0_mrlinear_postagg")
        model.fit(np.arange(36), bundle, config)
        chosen_cases = np.array([0, 1], dtype=int)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        lead = 1
        query_target_idx = model.issue_index + lead
        original_surface_value = model._linear_trend_surface_value

        def fake_surface(rid: int, target_idx: int) -> float:
            if target_idx == query_target_idx:
                return 1.0
            if target_idx == chosen_cases[0] + lead:
                return 0.0
            if target_idx == chosen_cases[1] + lead:
                return 1.0
            return original_surface_value(rid, target_idx)

        model._linear_trend_surface_value = fake_surface
        try:
            raw_fields = np.array(
                [
                    [[0.4, 0.4], [0.4, 0.4]],
                    [[0.6, 0.6], [0.6, 0.6]],
                ],
                dtype=np.float32,
            )
            raw_pred = np.zeros_like(raw_fields[0])
            for weight, field in zip(weights, raw_fields):
                raw_pred += float(weight) * field
            postagg = model._aggregate_postagg_regional_monthly(raw_pred, chosen_cases, weights, lead)
            casewise = np.zeros_like(raw_fields[0])
            for weight, case_idx, field in zip(weights, chosen_cases, raw_fields):
                casewise += float(weight) * model._adapt_field(field, int(case_idx), lead)
        finally:
            model._linear_trend_surface_value = original_surface_value

        self.assertFalse(np.allclose(postagg, casewise))

    def test_region_gating_produces_region_specific_weights(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        model = build_model("a0_mrlinear_gate_soft")
        model.fit(np.arange(36), bundle, config)
        adapted_fields = np.array(
            [
                [[0.2, 0.2], [0.8, 0.8]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.8, 0.8], [0.2, 0.2]],
                [[0.3, 0.3], [0.7, 0.7]],
            ],
            dtype=np.float32,
        )
        chosen_cases = np.array([1, 2, 3, 4], dtype=int)
        global_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        selected_region_scores = {
            1: np.array([3.0, 1.0, 0.0, -1.0], dtype=np.float64),
            2: np.array([0.0, 1.0, 3.0, -1.0], dtype=np.float64),
        }

        _, meta = model._aggregate_region_gated(
            adapted_fields,
            chosen_cases,
            global_weights,
            selected_region_scores,
            "global_region_gated_soft",
        )

        north = np.array(meta["regional"]["1"]["regional_weights"], dtype=np.float64)
        south = np.array(meta["regional"]["2"]["regional_weights"], dtype=np.float64)
        self.assertFalse(np.allclose(north, south))
        self.assertAlmostEqual(float(np.sum(north)), 1.0)
        self.assertAlmostEqual(float(np.sum(south)), 1.0)

        _, sparse_meta = model._aggregate_region_gated(
            adapted_fields,
            chosen_cases,
            global_weights,
            selected_region_scores,
            "global_region_gated_top3",
        )
        sparse_north = np.array(sparse_meta["regional"]["1"]["regional_weights"], dtype=np.float64)
        self.assertTrue(np.all(sparse_north >= 0.0))
        self.assertEqual(int(np.sum(sparse_north == 0.0)), 1)

    def test_sigmoid_gate_weights_behave_as_expected(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=2,
        )

        smooth_model = build_model("a0_mrlinear_sigmoid_k8")
        smooth_model.fit(np.arange(36), bundle, config)
        sharp_model = build_model("a0_mrlinear_sigmoid_k20")
        sharp_model.fit(np.arange(36), bundle, config)

        patch = np.array([[0.0, 0.15], [0.5, 0.6]], dtype=np.float32)
        delta = 0.3
        smooth_shifted = smooth_model._apply_sigmoid_gated_shift(patch, delta)
        sharp_shifted = sharp_model._apply_sigmoid_gated_shift(patch, delta)

        self.assertLess(float(smooth_shifted[0, 0] - patch[0, 0]), 0.1)
        self.assertAlmostEqual(float(smooth_shifted[0, 1] - patch[0, 1]), 0.15, places=6)
        self.assertGreater(float(smooth_shifted[1, 1] - patch[1, 1]), 0.29)
        self.assertLess(float(smooth_shifted[1, 0] - patch[1, 0]), float(sharp_shifted[1, 0] - patch[1, 0]))

    def test_sigmoid_reuse_differs_from_linear_baseline(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=2,
        )

        linear = build_export_model("a0_mrlinear")
        linear.fit(np.arange(36), bundle, config)
        nonlinear = build_model("a0_mrlinear_sigmoid_k20")
        nonlinear.fit(np.arange(36), bundle, config)

        patch = np.array([[0.05, 0.2], [0.85, 0.5]], dtype=np.float32)
        linear_shifted = linear._apply_sigmoid_gated_shift(patch, 0.3)
        nonlinear_shifted = nonlinear._apply_sigmoid_gated_shift(patch, 0.3)

        self.assertFalse(np.allclose(linear_shifted, nonlinear_shifted))

    def test_bg_gate_scales_existing_trend_shift(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=2,
        )

        baseline = build_export_model("a0_mrlinear")
        baseline.fit(np.arange(36), bundle, config)
        gated = build_export_model("a0_mrlinear_bggate")
        gated.fit(np.arange(36), bundle, config)

        patch = np.full((2, 2), 0.4, dtype=np.float32)
        baseline._bg_gate_weight = lambda rid, case_idx: 1.0
        gated._bg_gate_weight = lambda rid, case_idx: 0.5
        baseline._linear_trend_surface_value = lambda rid, target_idx: 0.6 if target_idx == baseline.issue_index + 1 else 0.2
        gated._linear_trend_surface_value = baseline._linear_trend_surface_value

        baseline_shifted = baseline._adapt_field(patch, case_idx=10, lead=1)
        gated_shifted = gated._adapt_field(patch, case_idx=10, lead=1)

        self.assertGreater(float(np.mean(baseline_shifted - patch)), float(np.mean(gated_shifted - patch)))
        self.assertAlmostEqual(float(np.mean(gated_shifted - patch)), 0.5 * float(np.mean(baseline_shifted - patch)), places=6)

    def test_bg_residual_adds_extra_shift(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=2,
        )

        residual = build_export_model("a0_mrlinear_bgresid")
        residual.fit(np.arange(36), bundle, config)
        patch = np.full((2, 2), 0.4, dtype=np.float32)
        residual._linear_trend_surface_value = lambda rid, target_idx: 0.6 if target_idx == residual.issue_index + 1 else 0.2
        residual._bg_residual_delta = lambda rid, case_idx, lead: 0.1

        shifted = residual._adapt_field(patch, case_idx=10, lead=1)
        self.assertAlmostEqual(float(np.mean(shifted - patch)), 0.5, places=6)

    def test_bg_combo_combines_gate_and_residual(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1,),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=2,
        )

        combo = build_export_model("a0_mrlinear_bgcombo")
        combo.fit(np.arange(36), bundle, config)
        patch = np.full((2, 2), 0.2, dtype=np.float32)
        combo._linear_trend_surface_value = lambda rid, target_idx: 0.5 if target_idx == combo.issue_index + 1 else 0.1
        combo._bg_gate_weight = lambda rid, case_idx: 0.25
        combo._bg_residual_delta = lambda rid, case_idx, lead: 0.1

        shifted = combo._adapt_field(patch, case_idx=10, lead=1)
        self.assertAlmostEqual(float(np.mean(shifted - patch)), 0.2, places=6)

    def test_a2_export_matches_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        expected = _reference_export("a2", config, bundle)
        actual = _export_internal_model_target_months("a2", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_a3_export_matches_reference(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            test_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        expected = _reference_export("a3", config, bundle)
        actual = _export_internal_model_target_months("a3", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_analog_regional_sic_bg_hist_fast_path_matches_reference_export(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        original_lazy_lowess = models_module._lazy_lowess
        models_module._lazy_lowess = lambda: (lambda y, x, frac=0.3, return_sorted=True: np.column_stack((x, y)))
        try:
            expected = _reference_export("analog_regional_sic_bg_hist", config, bundle)
            actual = _export_internal_model_target_months("analog_regional_sic_bg_hist", config, bundle)
        finally:
            models_module._lazy_lowess = original_lazy_lowess

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_analog_global_sic_fast_path_matches_reference_export(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        expected = _reference_export("analog_global_sic", config, bundle)
        actual = _export_internal_model_target_months("analog_global_sic", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_analog_global_sic_bg_hist_fast_path_matches_reference_export(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        expected = _reference_export("analog_global_sic_bg_hist", config, bundle)
        actual = _export_internal_model_target_months("analog_global_sic_bg_hist", config, bundle)

        np.testing.assert_array_equal(actual["target_ym"], expected["target_ym"])
        np.testing.assert_array_equal(actual["issue_ym"], expected["issue_ym"])
        np.testing.assert_array_equal(actual["lead_months"], expected["lead_months"])
        np.testing.assert_allclose(actual["truths"], expected["truths"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual["preds"], expected["preds"], atol=1e-6, rtol=0.0)

    def test_bg_reuse_variants_export_smoke(self):
        bundle = _build_synthetic_bundle()
        config = ExperimentConfig(
            start_year=1979,
            target_eval_start_year=1982,
            target_eval_end_year=1982,
            lead_times=(1, 2),
            issue_months=tuple(range(1, 13)),
            background_variables=("bg",),
            sic_eof_dim=2,
            bg_eof_dim=1,
            top_k=3,
        )

        for model_name in ("a0_mrlinear_bggate", "a0_mrlinear_bgresid", "a0_mrlinear_bgcombo"):
            actual = _export_internal_model_target_months(model_name, config, bundle)
            self.assertEqual(actual["preds"].shape, actual["truths"].shape)
            self.assertTrue(np.isfinite(actual["preds"]).any())


if __name__ == "__main__":
    unittest.main()
