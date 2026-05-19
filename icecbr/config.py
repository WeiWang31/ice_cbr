from __future__ import annotations

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
from typing import Sequence


PAPER_CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get("ICECBR_DATA_DIR", PAPER_CODE_ROOT.parent / "data_preprocess")
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("ICECBR_OUTPUT_DIR", PAPER_CODE_ROOT / "outputs")
).expanduser()


@dataclass
class ExperimentConfig:
    root_dir: Path = PAPER_CODE_ROOT
    data_dir: Path = DEFAULT_DATA_DIR
    sic_path: Path = data_dir / "SIC" / "g2202_197901_202312.npy"
    land_mask_path: Path = data_dir / "g2202_land.npy"
    region_nc_path: Path = data_dir / "NSIDC-0780" / "NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
    seas5_dir: Path = data_dir / "SEAS5"
    seas5_raw_path: Path = seas5_dir / "raw" / "seas5_2017_2023_monthly_sic.grib"
    seas5_raw_blocks_dir: Path = seas5_dir / "raw" / "blocks"
    seas5_download_manifest_path: Path = seas5_dir / "download_manifest.json"
    seas5_processing_manifest_path: Path = seas5_dir / "processing_manifest.json"
    seas5_processed_path: Path = seas5_dir / "processed" / "seas5_2017_2023_6lead_on_nsidc.npz"
    seas5_hindcast_raw_blocks_dir: Path = seas5_dir / "hindcast" / "raw" / "blocks"
    seas5_hindcast_download_manifest_path: Path = seas5_dir / "hindcast" / "download_manifest.json"
    seas5_hindcast_processing_manifest_path: Path = seas5_dir / "hindcast" / "processing_manifest.json"
    seas5_hindcast_processed_path: Path = seas5_dir / "processed" / "seas5_hindcast_1993_2016_6lead_on_nsidc.npz"
    seas5_bias_correction_path: Path = seas5_dir / "processed" / "seas5_lead_month_bias_correction_1993_2016.npz"
    results_dir: Path = DEFAULT_OUTPUT_DIR
    cache_dir: Path = DEFAULT_OUTPUT_DIR / "cache"
    figure_dir: Path = DEFAULT_OUTPUT_DIR / "figures"
    smoke_results_dir: Path = DEFAULT_OUTPUT_DIR / "smoke"

    era5_normalized_dir: Path = data_dir / "ERA5" / "normalized"
    oras5_normalized_dir: Path = data_dir / "ORAS5" / "normalized"

    start_year: int = 1979
    test_start_year: int = 1995
    test_end_year: int = 2023
    lead_times: Sequence[int] = (1, 2, 3, 4, 5, 6)
    issue_months: Sequence[int] = tuple(range(1, 13))
    history_window: int = 3
    target_threshold: float = 0.15
    cell_area_km2: float = 625.0

    sic_eof_dim: int = 5
    bg_eof_dim: int = 3
    top_k: int = 7
    retrieval_rho: float = 2.0
    retrieval_time_decay_tau_years: float = 5.0
    climate_gate_tau: float = 0.5
    reuse_bg_gate_tau: float = 0.5
    reuse_bg_residual_alpha: float = 1.0
    reuse_bg_residual_weight: float = 1.0
    enable_revise: bool = False
    revise_variance_quantile: float = 0.9
    revise_downweight_strength: float = 2.0
    revise_min_weight: float = 0.05
    enable_retain: bool = False
    retain_online_growth: bool = True
    retain_persist_jsonl: bool = True
    retain_reload: bool = False
    retain_namespace: str | None = None
    enable_selective_retain: bool = False
    enable_bounded_retain_memory: bool = False
    enable_retained_case_boost: bool = False
    enable_representative_retained_memory: bool = False
    retain_admission_mode: str = "none"
    retain_admission_threshold_permissive: float = 0.45
    retain_admission_threshold_strict: float = 0.60
    retain_admission_error_scale: float = 0.10
    retain_admission_disagreement_scale: float = 0.25
    retain_admission_preferred_adaptation: float = 0.03
    retain_admission_adaptation_tolerance: float = 0.03
    retained_case_cap: int | None = None
    retained_case_cap_per_lead: int | None = None
    retained_case_cap_per_month: int | None = None
    retain_representative_cap_per_lead: int = 2
    retained_case_recency_tau_years: float = 4.0
    retained_case_boost_strength: float = 0.10
    retained_case_boost_max: float = 0.15
    retain_log_rejections: bool = True
    top_k_grid: Sequence[int] = (3, 5, 7, 10, 15)
    rho_grid: Sequence[float] = (1.0, 2.0, 3.0, 5.0)
    sic_eof_grid: Sequence[int] = (5, 10, 15, 20, 30)
    region_counts_grid: Sequence[int] = (1, 4, 8, 12)
    history_grid: Sequence[int] = (1, 2, 3, 6)

    sim_weights: dict = field(
        default_factory=lambda: {"ice": 0.5, "bg": 0.3, "hist": 0.2}
    )
    retain_admission_weights: dict = field(
        default_factory=lambda: {
            "error": 0.40,
            "disagreement": 0.20,
            "recency": 0.30,
            "adaptation": 0.10,
        }
    )
    adapt_weights: dict = field(
        default_factory=lambda: {"time": 0.3, "regime": 0.4, "trend": 0.3}
    )
    adaptability_tau: float = 0.5
    competence_beta: float = 0.5
    competence_recent_years: int = 5
    competence_error_threshold: float = 0.02
    regime_split_year: int = 2007
    include_unassigned_region: bool = False
    region_scheme: str = "nsidc18"
    region_merge_schemes: Sequence[str] = ("nsidc18", "merge6", "merge4")
    random_state: int = 42
    run_mode: str = "fresh"
    output_tag: str | None = None
    selected_models: Sequence[str] | None = None
    pilot_start_year: int = 2015
    pilot_end_year: int = 2023
    compare_start_year: int = 2015
    compare_end_year: int = 2023
    target_eval_start_year: int = 2016
    target_eval_end_year: int = 2023
    compare_models: Sequence[str] = (
        "climatology",
        "damped_persistence",
        "linear_trend_persistence",
        "analog_regional_sic_bg_hist",
        "icecbr_full",
    )
    target_compare_models: Sequence[str] = (
        "climatology",
        "damped_persistence",
        "linear_trend_persistence",
        "analog_regional_sic_bg_hist",
        "a0",
    )
    target_export_models: Sequence[str] = (
        "climatology",
        "damped_persistence",
        "linear_trend_persistence",
        "analog_regional_sic_bg_hist",
        "a0",
    )
    forecast_compare_start_year: int = 2017
    forecast_compare_end_year: int = 2023
    forecast_compare_models: Sequence[str] = (
        "climatology",
        "damped_persistence",
        "linear_trend_persistence",
        "analog_regional_sic_bg_hist",
        "a0",
        "seas5",
        "seas5_bias_corrected",
    )
    finalist_candidates: Sequence[tuple[str, str, str]] = (
        ("nsidc18_a0", "nsidc18", "a0"),
        ("merge6_a0", "merge6", "a0"),
        ("merge6_a1", "merge6", "a1"),
    )
    seas5_start_year: int = 2017
    seas5_end_year: int = 2023
    seas5_hindcast_start_year: int = 1993
    seas5_hindcast_end_year: int = 2016
    seas5_system: str = "5"
    seas5_system_transition_year: int = 2022
    seas5_system_transition_month: int = 11
    seas5_system_post_transition: str = "51"
    seas5_originating_centre: str = "ecmwf"
    smoke_test_years: Sequence[int] = (1995,)
    smoke_issue_months: Sequence[int] = (1, 6)
    smoke_lead_times: Sequence[int] = (1, 3)

    background_variables: Sequence[str] = (
        "normalized_psl_1979-01_2023-12_anomaly.npy",
        "normalized_tas_1979-01_2023-12_anomaly.npy",
        "normalized_tos_1979-01_2023-12_anomaly.npy",
        "normalized_zg500_1979-01_2023-12_anomaly.npy",
        "normalized_ohc300_1979-01_2023-12_anomaly.npy",
        "normalized_mld001_1979-01_2023-12_anomaly.npy",
    )

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.smoke_results_dir.mkdir(parents=True, exist_ok=True)
        self.seas5_dir.mkdir(parents=True, exist_ok=True)
        self.seas5_raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.seas5_raw_blocks_dir.mkdir(parents=True, exist_ok=True)
        self.seas5_hindcast_raw_blocks_dir.mkdir(parents=True, exist_ok=True)
        self.seas5_processed_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
        return out


DEFAULT_CONFIG = ExperimentConfig()
