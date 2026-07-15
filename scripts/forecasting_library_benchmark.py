#!/usr/bin/env python3
"""Benchmark CartoBoost against explicit forecasting libraries.

The default fixture is synthetic but domain-shaped: daily pickup/dropoff lane
demand with zone IDs, route distance, airport-lane structure, borough codes,
weekly effects, and deterministic event spikes. The real-data path aggregates
NYC TLC trip records into the same lane-demand shape.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import urllib.request
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import __version__  # noqa: E402
from cartoboost import _native  # noqa: E402
from cartoboost import croston_forecast, sba_forecast, tsb_forecast  # noqa: E402
from cartoboost.config import Backend  # noqa: E402
from cartoboost.forecasting.global_models import CartoBoostLagForecaster  # noqa: E402
from cartoboost.forecasting import (  # noqa: E402
    AutoStatsBank,
    DCRNNForecaster,
    GraphTemporalFrame,
    LSTTNForecaster,
    LaneNeuralPanelForecaster,
    MarketPanelFrame,
    MarketStructureForecaster,
    PiecewiseLinearSeasonalForecaster,
)
from cartoboost.forecasting.schema import ForecastFrame  # noqa: E402
from cartoboost.metrics import competition_forecast_metrics  # noqa: E402
from cartoboost.metrics.rank_portfolio import (
    portfolio_summary as native_portfolio_summary,
)  # noqa: E402
from cartoboost.metrics.rank_portfolio import (
    rank_hit_rates as native_rank_hit_rates,
)
from cartoboost.metrics.rank_portfolio import (
    rank_probability_calibration as native_rank_probability_calibration,
)
from cartoboost.metrics.wrmsse import aggregate_equal_level_wrmsse, rmsse_scale, wrmsse  # noqa: E402

DEFAULT_CACHE_DIR = ROOT / "data" / "nyc_taxi"
DEFAULT_FORECASTING_CACHE_DIR = ROOT / "data" / "forecasting_benchmarks"

BASE_CARTOBOOST_LAGS = [1, 7, 14, 21, 28]
BASE_CARTOBOOST_ROLLING_WINDOWS = [7, 14, 28]
EXOGENOUS_FEATURE_COLUMNS = [
    "date_dayofweek",
    "date_day",
    "date_dayofyear",
    "date_month",
    "date_elapsed_days",
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
STATIC_COVARIATES = [
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
M5_EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
M5_SNAP_COLUMNS = ["snap_CA", "snap_TX", "snap_WI"]
M5_KNOWN_FUTURE_COVARIATES = [
    "m5_event_name_1_code",
    "m5_event_type_1_code",
    "m5_event_name_2_code",
    "m5_event_type_2_code",
    "m5_snap_CA",
    "m5_snap_TX",
    "m5_snap_WI",
    "m5_sell_price",
]
M5_HIERARCHY_COVARIATES = [
    "m5_state_code",
    "m5_store_code",
    "m5_cat_code",
    "m5_dept_code",
    "m5_item_code",
]
FUNCTIME_MODELS = ["functime_snaive", "functime_ridge", "functime_lightgbm"]
STATSFORECAST_MODELS = [
    "statsforecast_seasonal_naive",
    "statsforecast_autoets",
    "statsforecast_autoarima",
    "statsforecast_autotheta",
    "statsforecast_autoces",
    "statsforecast_dynamic_optimized_theta",
    "statsforecast_autotbats",
]
PROPHET_MODELS = ["prophet_additive"]
EXTERNAL_TREE_MODELS = ["xgboost_lag", "lightgbm_lag"]
CARTOBOOST_BENCHMARK_MODELS = [
    "cartoboost_lag",
    "cartoboost_auto_forecast",
]
INTERMITTENT_BENCHMARK_MODELS = [
    "croston",
    "sba",
    "tsb",
]
SEASONAL_NAIVE_BENCHMARK_MODEL = "seasonal_naive"
NEURAL_PANEL_BENCHMARK_MODEL = "cartoboost_neural_panel"
PIECEWISE_LINEAR_BENCHMARK_MODEL = "cartoboost_piecewise_linear_seasonal"
FORECASTING_LIBRARY_MODELS = {
    "functime": FUNCTIME_MODELS,
    "statsforecast": STATSFORECAST_MODELS,
    "prophet": PROPHET_MODELS,
    "external_trees": EXTERNAL_TREE_MODELS,
}
MODEL_LIBRARIES = {
    SEASONAL_NAIVE_BENCHMARK_MODEL: "baseline",
    **{model: "cartoboost" for model in CARTOBOOST_BENCHMARK_MODELS},
    **{model: "intermittent" for model in INTERMITTENT_BENCHMARK_MODELS},
    NEURAL_PANEL_BENCHMARK_MODEL: "cartoboost",
    PIECEWISE_LINEAR_BENCHMARK_MODEL: "cartoboost",
    **{model: "functime" for model in FUNCTIME_MODELS},
    **{model: "statsforecast" for model in STATSFORECAST_MODELS},
    **{model: "prophet" for model in PROPHET_MODELS},
    **{model: "external_trees" for model in EXTERNAL_TREE_MODELS},
}
FORECASTING_LIBRARY_BASELINES = [
    *FUNCTIME_MODELS,
    *STATSFORECAST_MODELS,
    *PROPHET_MODELS,
    *EXTERNAL_TREE_MODELS,
]
SCALABLE_FORECASTING_LIBRARY_BASELINES = [
    *FUNCTIME_MODELS,
    *EXTERNAL_TREE_MODELS,
]
AIRPORT_ZONE_IDS = {1, 132, 138}
M1_GROUPS = ["Yearly", "Quarterly", "Monthly"]
M1_GROUP_INFO = {
    "Yearly": {
        "record_id": "4656193",
        "zip_name": "m1_yearly_dataset.zip",
        "tsf_name": "m1_yearly_dataset.tsf",
        "horizon": 6,
        "season_length": 1,
    },
    "Quarterly": {
        "record_id": "4656154",
        "zip_name": "m1_quarterly_dataset.zip",
        "tsf_name": "m1_quarterly_dataset.tsf",
        "horizon": 8,
        "season_length": 4,
    },
    "Monthly": {
        "record_id": "4656159",
        "zip_name": "m1_monthly_dataset.zip",
        "tsf_name": "m1_monthly_dataset.tsf",
        "horizon": 18,
        "season_length": 12,
    },
}
M3_GROUPS = ["Yearly", "Quarterly", "Monthly", "Other"]
M4_GROUPS = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
M6_ASSETS_URL = "https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv"
AUTO_ENSEMBLE_CANDIDATE = "cartoboost_validation_weighted_ensemble"
AUTO_SELECTION_MIN_RELATIVE_GAIN = 0.03
AUTO_SELECTION_ROBUST_RELATIVE_TOLERANCE = 0.05
LOW_SUPPORT_AUTO_MIN_RELATIVE_GAIN = 0.08
RAW_AUTO_OVERRIDE_MIN_RELATIVE_GAIN = 0.15
M6_RAW_AUTO_DISPLACEMENT_MIN_RELATIVE_GAIN = 0.005
M6_INVESTMENT_RPS_TIEBREAK_WEIGHT = 1.0e-4
CALENDAR_PROFILE_ELAPSED_PHASE_PERIOD = 14
NATIVE_AUTO_RAW_KEEP_RELATIVE_GAIN = 0.50
NON_M_LAG_DOMINANCE_EARLY_STOP_RELATIVE_GAIN = 0.15
VALIDATION_CACHE_STATS_KEY = "__validation_cache_stats__"
SYNTHETIC_PROBLEMS = [
    "taxi_weekly",
    "airport_calendar_events",
    "route_mix_shift",
    "borough_monthly_pulses",
]
PROPHET_CLASS: Any | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CartoBoost global lag forecasts against forecasting libraries."
    )
    parser.add_argument(
        "--source",
        choices=["polars", "duckdb", "nyc-taxi", "m", "m1", "m3", "m4", "m5", "m6"],
        default="polars",
    )
    parser.add_argument("--output", default="artifacts/forecasting_library_benchmark_polars.json")
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument(
        "--problem",
        choices=SYNTHETIC_PROBLEMS,
        default="taxi_weekly",
        help="Synthetic problem to run when --source is polars or duckdb.",
    )
    parser.add_argument(
        "--suite",
        nargs="?",
        const="synthetic",
        choices=["synthetic", "committed"],
        default=None,
        help=(
            "Run all synthetic forecasting problems and report aggregate rankings. "
            "Use 'committed' for the fixed committed sample suite."
        ),
    )
    parser.add_argument(
        "--no-hyperopt",
        action="store_true",
        help="Benchmark integrity marker: model menus/settings are fixed and deterministic.",
    )
    parser.add_argument(
        "--suite-folds",
        type=int,
        default=3,
        help="Rolling-origin folds per synthetic problem when --suite is set.",
    )
    parser.add_argument("--lanes", type=int, default=36)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument(
        "--rolling-origin-folds",
        type=int,
        default=1,
        help=(
            "For real taxi demand, score this many leakage-safe outer rolling-origin "
            "folds instead of one final holdout."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--years",
        default=None,
        help=(
            "Comma-separated TLC years. When supplied, overrides --year and keeps "
            "every requested year in the chronological benchmark panel."
        ),
    )
    parser.add_argument("--months", default="1", help="Comma-separated month numbers, e.g. 1,2,3")
    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument(
        "--taxi-frequency",
        choices=["daily", "monthly"],
        default="daily",
        help=(
            "Time bucket for TLC lane panels. Use monthly for all-observed-lane, multi-year "
            "graphs so the complete panel remains tractable."
        ),
    )
    parser.add_argument(
        "--all-observed-lanes",
        action="store_true",
        help="Use every observed directed pickup-to-dropoff lane instead of only --lanes.",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=None,
        help=(
            "Map TLC pickup and dropoff zones to their centroid H3 cells at this resolution "
            "for market-structure graph endpoints."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--m1-group",
        default="Monthly",
        choices=M1_GROUPS,
    )
    parser.add_argument(
        "--m1-series-limit",
        type=int,
        default=96,
        help="Maximum M1 series to score; use 0 for every series in the selected group.",
    )
    parser.add_argument(
        "--m1-suite",
        action="store_true",
        help="Run all three public M1 groups: Yearly, Quarterly, and Monthly.",
    )
    parser.add_argument(
        "--m3-group",
        default="Monthly",
        choices=M3_GROUPS,
    )
    parser.add_argument(
        "--m3-series-limit",
        type=int,
        default=96,
        help="Maximum M3 series to score; use 0 for every series in the selected group.",
    )
    parser.add_argument(
        "--m3-suite",
        action="store_true",
        help="Run all four M3 groups: Yearly, Quarterly, Monthly, and Other.",
    )
    parser.add_argument(
        "--m4-group",
        default="Hourly",
        choices=M4_GROUPS,
    )
    parser.add_argument(
        "--m4-suite",
        action="store_true",
        help="Run all six M4 groups: Hourly, Daily, Weekly, Monthly, Quarterly, and Yearly.",
    )
    parser.add_argument(
        "--m4-series-limit",
        type=int,
        default=96,
        help=(
            "Maximum M4 series per group to score locally; use 0 for every series. "
            "The full group dataset is still downloaded."
        ),
    )
    parser.add_argument(
        "--m5-data-dir",
        type=Path,
        default=DEFAULT_FORECASTING_CACHE_DIR / "m5",
        help=(
            "Directory containing Kaggle M5 files. Requires calendar.csv and either "
            "sales_train_evaluation.csv or sales_train_validation.csv."
        ),
    )
    parser.add_argument(
        "--m5-series-limit",
        type=int,
        default=0,
        help="Maximum M5 item-store series to score; use 0 for the full bottom-level corpus.",
    )
    parser.add_argument(
        "--m5-history-days",
        type=int,
        default=365,
        help=(
            "Most recent M5 daily columns to materialize before the 28-day holdout; "
            "use 0 for every available day."
        ),
    )
    parser.add_argument(
        "--m6-assets-path",
        type=Path,
        default=DEFAULT_FORECASTING_CACHE_DIR / "m6" / "assets_m6.csv",
        help="Path to the M6 assets CSV with symbol/date/price columns.",
    )
    parser.add_argument(
        "--m6-series-limit",
        type=int,
        default=0,
        help="Maximum M6 symbols to score; use 0 for every symbol in the assets file.",
    )
    parser.add_argument(
        "--m6-horizon",
        type=int,
        default=28,
        help="Daily return holdout horizon for the M6 point-forecast proxy.",
    )
    parser.add_argument(
        "--model-roster",
        choices=[
            "full",
            "scalable",
            "cartoboost",
            "intermittent",
            "piecewise",
            "prophet-comparison",
            "neural-panel",
        ],
        default="full",
        help=(
            "Forecast model roster. Use scalable for full M5-style panels where "
            "per-series Prophet/StatsForecast models are impractical. Use piecewise "
            "for only the native piecewise-linear model, or prophet-comparison for "
            "the native piecewise-linear model and Prophet only. Use intermittent "
            "for Croston, SBA, and TSB. Use neural-panel for seasonal naive, "
            "CartoBoost lag, and the Rust-native lane neural model."
        ),
    )
    parser.add_argument(
        "--neural-panel-splits",
        action="store_true",
        help=(
            "Run the NeuralPanel taxi-lane split suite: rolling-origin, cold-lane, "
            "cold-origin, and sparse-tail. Writes split metrics, timing, command, and "
            "artifact path metadata to --output."
        ),
    )
    parser.add_argument(
        "--market-structure-splits",
        action="store_true",
        help=(
            "Run the learned market-structure taxi lane suite. Requires real NYC taxi data "
            "with daily effective-fare and zone geometry inputs."
        ),
    )
    parser.add_argument(
        "--lsttn-h3-splits",
        action="store_true",
        help="Train native LSTTN on H3-cell demand with every observed directed OD edge.",
    )
    parser.add_argument("--lsttn-epochs", type=int, default=80)
    parser.add_argument("--lsttn-hidden-size", type=int, default=16)
    parser.add_argument(
        "--market-scale-only",
        action="store_true",
        help=(
            "Run only the native learned market graph and last-value reference. "
            "Use for full all-observed-lane scale tests; dense pairwise baselines and "
            "repeated diagnostic refits are intentionally excluded."
        ),
    )
    parser.add_argument(
        "--allow-full-m5-roster",
        action="store_true",
        help=(
            "Allow the full per-series library roster on the unbounded M5 corpus. "
            "Without this flag, full M5 roster runs require a positive --m5-series-limit."
        ),
    )
    parser.add_argument(
        "--no-candidate-selection",
        action="store_true",
        help="Skip inner-origin shared candidate selection for very large panels.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--cartoboost-n-estimators", type=int, default=180)
    parser.add_argument(
        "--cartoboost-auto-n-estimators",
        type=int,
        default=None,
        help=(
            "Override the auto CartoBoost estimator count. By default auto uses "
            "a quality floor independent of --cartoboost-n-estimators."
        ),
    )
    parser.add_argument("--cartoboost-learning-rate", type=float, default=0.06)
    parser.add_argument("--cartoboost-max-depth", type=int, default=4)
    parser.add_argument("--cartoboost-min-samples-leaf", type=int, default=8)
    args = parser.parse_args()
    normalize_competition_source(args)
    validate_args(args)
    if not args.market_structure_splits and "prophet" in forecasting_library_models_for_roster(
        args.model_roster
    ):
        ensure_prophet_class()

    cartoboost_config = {
        "n_estimators": args.cartoboost_n_estimators,
        "auto_n_estimators": args.cartoboost_auto_n_estimators,
        "learning_rate": args.cartoboost_learning_rate,
        "max_depth": args.cartoboost_max_depth,
        "min_samples_leaf": args.cartoboost_min_samples_leaf,
        "split_policy": "structured",
    }

    benchmark_start = perf_counter()
    if args.suite:
        return run_synthetic_suite(args, cartoboost_config, benchmark_start)
    if args.m1_suite:
        return run_m1_suite(args, cartoboost_config, benchmark_start)
    if args.m3_suite:
        return run_m3_suite(args, cartoboost_config, benchmark_start)
    if args.m4_suite:
        return run_m4_suite(args, cartoboost_config, benchmark_start)

    load_start = perf_counter()
    table, dataset = load_dataset(args)
    dataset["dataset_hash"] = canonical_dataset_hash(table)
    dataset_source_hashes = source_file_hashes(dataset)
    load_seconds = perf_counter() - load_start
    if args.neural_panel_splits:
        return run_neural_panel_split_suite(
            args,
            table=table,
            dataset=dataset,
            source_file_hashes=dataset_source_hashes,
            load_seconds=load_seconds,
            cartoboost_config=cartoboost_config,
            benchmark_start=benchmark_start,
        )
    if args.market_structure_splits:
        return run_market_structure_taxi_suite(
            args,
            table=table,
            dataset=dataset,
            source_file_hashes=dataset_source_hashes,
            load_seconds=load_seconds,
            cartoboost_config=cartoboost_config,
            benchmark_start=benchmark_start,
        )
    if args.lsttn_h3_splits:
        return run_lsttn_h3_taxi_suite(
            args,
            table=table,
            dataset=dataset,
            source_file_hashes=dataset_source_hashes,
            load_seconds=load_seconds,
            benchmark_start=benchmark_start,
        )
    benchmark_horizon = int(dataset.get("horizon", args.horizon))
    season_length = int(dataset.get("season_length", 7))
    model_names = benchmark_model_names(args.model_roster)
    if args.rolling_origin_folds > 1:
        if args.source != "nyc-taxi":
            raise ValueError(
                "--rolling-origin-folds > 1 is currently supported for --source nyc-taxi"
            )
        split_results, metrics, quality, timing, scored = score_rolling_origin_problem(
            table,
            horizon=benchmark_horizon,
            season_length=season_length,
            folds=args.rolling_origin_folds,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=args.source,
        )
    else:
        split_results = None
        metrics, quality, timing, scored = score_models(
            table,
            horizon=benchmark_horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=args.source,
            candidate_selection=not args.no_candidate_selection,
        )
    total_seconds = perf_counter() - benchmark_start
    timing = {
        "total_seconds": total_seconds,
        "load_seconds": load_seconds,
        **timing,
    }
    plots = (
        write_forecast_plots(
            scored,
            args.plot_dir,
            prefix=args.source,
            models=model_names,
        )
        if args.plot_dir
        else []
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "requested_source": getattr(args, "requested_source", args.source),
        "dataset_hash": dataset["dataset_hash"],
        "source_file_hashes": dataset_source_hashes,
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "geotemporal_lane_demand_forecasting_libraries",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": forecasting_library_models_for_roster(args.model_roster),
        "model_libraries": MODEL_LIBRARIES,
        "dataset": dataset,
        "models": model_names,
        "model_roster": args.model_roster,
        "model_settings": cartoboost_model_settings(cartoboost_config),
        "metrics": metrics,
        "quality": quality,
        "official_metrics": benchmark_objective_artifacts(
            args.source,
            train_table=table,
            scored=scored,
            model_names=benchmark_model_names(args.model_roster),
            season_length=season_length,
            cartoboost_config=cartoboost_config,
        ),
        "timing": timing,
        "rolling_origin": (
            {
                "folds": args.rolling_origin_folds,
                "splits": split_results,
            }
            if split_results is not None
            else None
        ),
        "resource_usage": resource_usage_snapshot(),
        "plots": plots,
    }
    payload["comparability_audit"] = forecasting_comparability_audit(
        args=args,
        model_names=model_names,
        metrics=metrics,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"quality": quality, "timing": timing}, indent=2, sort_keys=True))
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.lanes <= 0:
        raise ValueError("--lanes must be positive")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    years = getattr(args, "years", None)
    if years is not None:
        parse_taxi_years(years, fallback_year=getattr(args, "year", 2024))
    if getattr(args, "all_observed_lanes", False) and args.source != "nyc-taxi":
        raise ValueError("--all-observed-lanes requires --source nyc-taxi")
    h3_resolution = getattr(args, "h3_resolution", None)
    if h3_resolution is not None:
        if args.source != "nyc-taxi":
            raise ValueError("--h3-resolution requires --source nyc-taxi")
        if not 0 <= h3_resolution <= 15:
            raise ValueError("--h3-resolution must be between 0 and 15")
    if getattr(args, "market_scale_only", False) and not getattr(
        args, "market_structure_splits", False
    ):
        raise ValueError("--market-scale-only requires --market-structure-splits")
    if getattr(args, "rolling_origin_folds", 1) <= 0:
        raise ValueError("--rolling-origin-folds must be positive")
    if args.source in {"polars", "duckdb"} and args.days <= args.horizon + max(28, args.horizon):
        raise ValueError("--days must leave at least 28 training days before the holdout")
    if args.cartoboost_n_estimators <= 0:
        raise ValueError("--cartoboost-n-estimators must be positive")
    auto_n_estimators = getattr(args, "cartoboost_auto_n_estimators", None)
    if auto_n_estimators is not None and auto_n_estimators <= 0:
        raise ValueError("--cartoboost-auto-n-estimators must be positive when provided")
    if args.cartoboost_max_depth <= 0:
        raise ValueError("--cartoboost-max-depth must be positive")
    if args.cartoboost_min_samples_leaf <= 0:
        raise ValueError("--cartoboost-min-samples-leaf must be positive")
    if args.suite and args.source == "nyc-taxi":
        raise ValueError("--suite is only supported for synthetic polars or duckdb sources")
    if args.suite and args.source in {"m1", "m3", "m4"}:
        raise ValueError(
            "use --source m1, --source m3, or --source m4 without --suite; "
            "M1/M3/M4 groups are already benchmark datasets"
        )
    if args.suite and args.source in {"m5", "m6"}:
        raise ValueError("--suite is only supported for synthetic polars or duckdb sources")
    if getattr(args, "m1_suite", False) and args.source != "m1":
        raise ValueError("--m1-suite requires --source m1")
    if getattr(args, "m3_suite", False) and args.source != "m3":
        raise ValueError("--m3-suite requires --source m3")
    if getattr(args, "m4_suite", False) and args.source != "m4":
        raise ValueError("--m4-suite requires --source m4")
    if args.m4_series_limit < 0:
        raise ValueError("--m4-series-limit must be non-negative; use 0 for every M4 series")
    if getattr(args, "m1_series_limit", 0) < 0:
        raise ValueError("--m1-series-limit must be non-negative; use 0 for every M1 series")
    if getattr(args, "m3_series_limit", 0) < 0:
        raise ValueError("--m3-series-limit must be non-negative; use 0 for every M3 series")
    if args.m5_series_limit < 0:
        raise ValueError("--m5-series-limit must be non-negative; use 0 for every M5 series")
    if args.m5_history_days < 0:
        raise ValueError("--m5-history-days must be non-negative; use 0 for every M5 day")
    if args.m6_series_limit < 0:
        raise ValueError("--m6-series-limit must be non-negative; use 0 for every M6 symbol")
    if args.m6_horizon <= 0:
        raise ValueError("--m6-horizon must be positive")
    if (
        args.source == "m5"
        and args.model_roster == "full"
        and args.m5_series_limit == 0
        and not args.allow_full_m5_roster
    ):
        raise ValueError(
            "--source m5 --model-roster full requires a positive --m5-series-limit for "
            "scientific comparison samples. To run the full per-series roster on all "
            "30,490 M5 bottom-level series, pass --allow-full-m5-roster and expect a "
            "long heavyweight benchmark."
        )
    if args.suite_folds <= 0:
        raise ValueError("--suite-folds must be positive")
    if args.suite and args.days <= args.horizon * args.suite_folds + 60:
        raise ValueError("--suite requires enough days for rolling origins and 60 training days")


def normalize_competition_source(args: argparse.Namespace) -> argparse.Namespace:
    requested_source = getattr(args, "source", None)
    args.requested_source = requested_source
    if requested_source == "m":
        args.source = "m1"
        args.source_alias = "m"
    else:
        args.source_alias = None
    return args


def parse_taxi_years(value: str | None, *, fallback_year: int) -> list[int]:
    """Parse a deterministic, chronological TLC year selection."""
    if value is None:
        return [int(fallback_year)]
    years = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            raise ValueError("--years must contain comma-separated calendar years")
        try:
            year = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid TLC year {token!r} in --years") from exc
        if year < 2009 or year > 2100:
            raise ValueError("--years values must be between 2009 and 2100")
        years.append(year)
    if not years:
        raise ValueError("--years must select at least one year")
    return sorted(set(years))


def canonical_dataset_hash(table: Any) -> str:
    frame = table
    columns = sorted(str(column) for column in frame.columns)
    sort_columns = [
        column
        for column in ["lane_id", "series_id", "date", "timestamp", "horizon"]
        if column in frame.columns
    ]
    if sort_columns and hasattr(frame, "sort"):
        frame = frame.sort(sort_columns)
    frame = frame.select(columns)
    buffer = io.StringIO()
    frame.write_csv(buffer)
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def aggregate_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(value) for value in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_file_hashes(dataset: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in ["sales_file", "calendar_file", "prices_file", "assets_file", "tsf_file"]:
        value = dataset.get(key)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            hashes[key] = file_sha256(path)
    return hashes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def benchmark_integrity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "no_hyperopt": bool(args.no_hyperopt),
        "seed": int(args.seed),
        "model_roster": args.model_roster,
        "candidate_selection": not args.no_candidate_selection,
        "threading": {
            "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
    }


def forecasting_comparability_audit(
    *,
    args: argparse.Namespace,
    model_names: list[str],
    metrics: dict[str, dict[str, float]] | None = None,
    grouped_results: dict[str, Any] | None = None,
    split_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed_models = completed_forecasting_models(
        model_names,
        metrics=metrics,
        grouped_results=grouped_results,
        split_results=split_results,
    )
    completed_cartoboost = [
        model for model in completed_models if MODEL_LIBRARIES.get(model) == "cartoboost"
    ]
    completed_libraries = [
        model for model in completed_models if MODEL_LIBRARIES.get(model) != "cartoboost"
    ]
    return {
        "same_forecast_rows": True,
        "same_horizon": True,
        "same_metrics": ["mae", "rmse", "wape", "r2", "mase", "smape"],
        "primary_metric": "rmse",
        "selection_metric": "rmse",
        "candidate_selection": bool(not args.no_candidate_selection),
        "selection_uses_outer_test_labels": False,
        "model_roster": args.model_roster,
        "requested_models": list(model_names),
        "completed_models": completed_models,
        "completed_cartoboost_models": completed_cartoboost,
        "completed_forecasting_library_models": completed_libraries,
        "skipped_requested_models": [
            model for model in model_names if model not in set(completed_models)
        ],
        "best_cartoboost_compared_to_best_library": bool(
            completed_cartoboost and completed_libraries
        ),
    }


def completed_forecasting_models(
    model_names: list[str],
    *,
    metrics: dict[str, dict[str, float]] | None = None,
    grouped_results: dict[str, Any] | None = None,
    split_results: dict[str, Any] | None = None,
) -> list[str]:
    completed: set[str] = set()
    if metrics:
        completed.update(metrics)
    if grouped_results:
        for result in grouped_results.values():
            completed.update(result.get("metrics", {}))
    if split_results:
        for result in split_results.values():
            completed.update(result.get("metrics", {}))
    return [model for model in model_names if model in completed]


def invocation_metadata() -> dict[str, Any]:
    argv = [str(part) for part in sys.argv]
    return {
        "argv": argv,
        "command": " ".join(shlex.quote(part) for part in argv),
    }


def resource_usage_snapshot() -> dict[str, Any]:
    if resource is None:
        return {
            "process_cpu_seconds": float(process_time()),
            "peak_rss_mb": windows_peak_rss_mb(),
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_mb": peak_rss_mb(usage.ru_maxrss),
    }


def peak_rss_mb(raw_maxrss: int) -> float:
    if platform.system() == "Darwin":
        return float(raw_maxrss) / (1024.0 * 1024.0)
    return float(raw_maxrss) / 1024.0


def windows_peak_rss_mb() -> float:
    if platform.system() != "Windows":
        return 1.0
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return 1.0
    return 1.0


def benchmark_model_names(roster: str) -> list[str]:
    if roster == "cartoboost":
        return list(CARTOBOOST_BENCHMARK_MODELS)
    if roster == "intermittent":
        return list(INTERMITTENT_BENCHMARK_MODELS)
    if roster == "piecewise":
        return [PIECEWISE_LINEAR_BENCHMARK_MODEL]
    if roster == "prophet-comparison":
        return [PIECEWISE_LINEAR_BENCHMARK_MODEL, "prophet_additive"]
    if roster == "neural-panel":
        return [
            SEASONAL_NAIVE_BENCHMARK_MODEL,
            "cartoboost_lag",
            NEURAL_PANEL_BENCHMARK_MODEL,
        ]
    if roster == "scalable":
        return [
            "cartoboost_lag",
            "cartoboost_auto_forecast",
            *SCALABLE_FORECASTING_LIBRARY_BASELINES,
        ]
    return [*CARTOBOOST_BENCHMARK_MODELS, *FORECASTING_LIBRARY_BASELINES]


def forecasting_library_models_for_roster(roster: str) -> dict[str, list[str]]:
    if roster in {"cartoboost", "intermittent", "piecewise", "neural-panel"}:
        return {}
    if roster == "prophet-comparison":
        return {"prophet": PROPHET_MODELS}
    if roster == "scalable":
        return {
            "functime": FUNCTIME_MODELS,
            "external_trees": EXTERNAL_TREE_MODELS,
        }
    return FORECASTING_LIBRARY_MODELS


def run_synthetic_suite(
    args: argparse.Namespace,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for problem in SYNTHETIC_PROBLEMS:
        problem_args = argparse.Namespace(**vars(args))
        problem_args.problem = problem
        load_start = perf_counter()
        table, dataset = load_synthetic_fixture(problem_args)
        dataset["dataset_hash"] = canonical_dataset_hash(table)
        load_seconds = perf_counter() - load_start
        split_results, metrics, quality, timing, scored = score_rolling_origin_problem(
            table,
            horizon=args.horizon,
            season_length=7,
            folds=args.suite_folds,
            cartoboost_config=cartoboost_config,
            model_names=benchmark_model_names(args.model_roster),
            source="synthetic",
        )
        plots = (
            write_forecast_plots(
                scored,
                args.plot_dir,
                prefix=problem,
                models=benchmark_model_names(args.model_roster),
            )
            if args.plot_dir
            else []
        )
        results[problem] = {
            "dataset": dataset,
            "splits": split_results,
            "metrics": metrics,
            "quality": quality,
            "plots": plots,
        }
        timings[problem] = {
            "load_seconds": load_seconds,
            **timing,
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "requested_source": getattr(args, "requested_source", args.source),
        "dataset_hash": aggregate_hash(
            result["dataset"]["dataset_hash"] for result in results.values()
        ),
        "source_file_hashes": {},
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "geotemporal_lane_demand_forecasting_library_suite",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": FORECASTING_LIBRARY_MODELS,
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "problems": SYNTHETIC_PROBLEMS,
            "series": args.lanes,
            "days": args.days,
            "horizon": args.horizon,
            "season_length": 7,
            "folds": args.suite_folds,
            "seed": args.seed,
            "domain": "synthetic NYC taxi-style forecasting problem suite",
            "split_type": "rolling_origin_last_windows",
            "static_covariates": STATIC_COVARIATES,
        },
        "models": benchmark_model_names(args.model_roster),
        "model_settings": cartoboost_model_settings(cartoboost_config),
        "problems": results,
        "aggregate_quality": aggregate_suite_quality(results),
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            "problems": timings,
        },
        "resource_usage": resource_usage_snapshot(),
    }
    payload["comparability_audit"] = forecasting_comparability_audit(
        args=args,
        model_names=benchmark_model_names(args.model_roster),
        grouped_results=results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate_quality"], indent=2, sort_keys=True))
    return 0


def run_m4_suite(
    args: argparse.Namespace,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for group in M4_GROUPS:
        group_args = argparse.Namespace(**vars(args))
        group_args.m4_group = group
        load_start = perf_counter()
        table, dataset = load_m4_fixture(group_args)
        dataset["dataset_hash"] = canonical_dataset_hash(table)
        load_seconds = perf_counter() - load_start
        metrics, quality, timing, _scored = score_models(
            table,
            horizon=int(dataset["horizon"]),
            season_length=int(dataset["season_length"]),
            cartoboost_config=cartoboost_config,
            model_names=benchmark_model_names(args.model_roster),
            source="m4",
        )
        results[group] = {
            "dataset": dataset,
            "metrics": metrics,
            "official_metrics": benchmark_objective_artifacts(
                "m4",
                train_table=table,
                scored=_scored,
                model_names=benchmark_model_names(args.model_roster),
                season_length=int(dataset["season_length"]),
                cartoboost_config=cartoboost_config,
            ),
            "quality": quality,
        }
        timings[group] = {
            "load_seconds": load_seconds,
            **timing,
        }

    aggregate_quality = aggregate_suite_quality(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "requested_source": getattr(args, "requested_source", args.source),
        "dataset_hash": aggregate_hash(
            result["dataset"]["dataset_hash"] for result in results.values()
        ),
        "source_file_hashes": {},
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "m4_forecasting_library_group_suite",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": FORECASTING_LIBRARY_MODELS,
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "groups": M4_GROUPS,
            "source": "m4",
            "domain": "M4 forecasting competition train panels",
            "split_type": "last_official_horizon_from_training_panel",
            "series_limit_per_group": (None if args.m4_series_limit == 0 else args.m4_series_limit),
            "static_covariates": STATIC_COVARIATES,
        },
        "models": benchmark_model_names(args.model_roster),
        "model_settings": cartoboost_model_settings(cartoboost_config),
        "groups": results,
        "aggregate_quality": aggregate_quality,
        "official_metrics": aggregate_m_series_suite_official_metrics("m4", M4_GROUPS, results),
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            "groups": timings,
        },
        "resource_usage": resource_usage_snapshot(),
    }
    payload["comparability_audit"] = forecasting_comparability_audit(
        args=args,
        model_names=benchmark_model_names(args.model_roster),
        grouped_results=results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate_quality, indent=2, sort_keys=True))
    return 0


def run_m1_suite(
    args: argparse.Namespace,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for group in M1_GROUPS:
        group_args = argparse.Namespace(**vars(args))
        group_args.m1_group = group
        load_start = perf_counter()
        table, dataset = load_m1_fixture(group_args)
        dataset["dataset_hash"] = canonical_dataset_hash(table)
        load_seconds = perf_counter() - load_start
        metrics, quality, timing, scored = score_models(
            table,
            horizon=int(dataset["horizon"]),
            season_length=int(dataset["season_length"]),
            cartoboost_config=cartoboost_config,
            model_names=benchmark_model_names(args.model_roster),
            source="m1",
        )
        results[group] = {
            "dataset": dataset,
            "metrics": metrics,
            "official_metrics": benchmark_objective_artifacts(
                "m1",
                train_table=table,
                scored=scored,
                model_names=benchmark_model_names(args.model_roster),
                season_length=int(dataset["season_length"]),
                cartoboost_config=cartoboost_config,
            ),
            "quality": quality,
        }
        timings[group] = {
            "load_seconds": load_seconds,
            **timing,
        }

    aggregate_quality = aggregate_suite_quality(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "requested_source": getattr(args, "requested_source", args.source),
        "dataset_hash": aggregate_hash(
            result["dataset"]["dataset_hash"] for result in results.values()
        ),
        "source_file_hashes": {
            group: source_file_hashes(result["dataset"]) for group, result in results.items()
        },
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "m1_forecasting_library_group_suite",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": forecasting_library_models_for_roster(args.model_roster),
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "groups": M1_GROUPS,
            "source": "m1",
            "domain": "M1 forecasting competition train panels from public TSF archives",
            "split_type": "last_official_horizon_from_full_public_series",
            "series_limit_per_group": (None if args.m1_series_limit == 0 else args.m1_series_limit),
            "static_covariates": STATIC_COVARIATES,
        },
        "models": benchmark_model_names(args.model_roster),
        "model_settings": cartoboost_model_settings(cartoboost_config),
        "groups": results,
        "aggregate_quality": aggregate_quality,
        "official_metrics": aggregate_m_series_suite_official_metrics("m1", M1_GROUPS, results),
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            "groups": timings,
        },
        "resource_usage": resource_usage_snapshot(),
    }
    payload["comparability_audit"] = forecasting_comparability_audit(
        args=args,
        model_names=benchmark_model_names(args.model_roster),
        grouped_results=results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate_quality, indent=2, sort_keys=True))
    return 0


def run_m3_suite(
    args: argparse.Namespace,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for group in M3_GROUPS:
        group_args = argparse.Namespace(**vars(args))
        group_args.m3_group = group
        load_start = perf_counter()
        table, dataset = load_m3_fixture(group_args)
        dataset["dataset_hash"] = canonical_dataset_hash(table)
        load_seconds = perf_counter() - load_start
        metrics, quality, timing, scored = score_models(
            table,
            horizon=int(dataset["horizon"]),
            season_length=int(dataset["season_length"]),
            cartoboost_config=cartoboost_config,
            model_names=benchmark_model_names(args.model_roster),
            source="m3",
        )
        results[group] = {
            "dataset": dataset,
            "metrics": metrics,
            "official_metrics": benchmark_objective_artifacts(
                "m3",
                train_table=table,
                scored=scored,
                model_names=benchmark_model_names(args.model_roster),
                season_length=int(dataset["season_length"]),
                cartoboost_config=cartoboost_config,
            ),
            "quality": quality,
        }
        timings[group] = {
            "load_seconds": load_seconds,
            **timing,
        }

    aggregate_quality = aggregate_suite_quality(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "requested_source": getattr(args, "requested_source", args.source),
        "dataset_hash": aggregate_hash(
            result["dataset"]["dataset_hash"] for result in results.values()
        ),
        "source_file_hashes": {},
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "m3_forecasting_library_group_suite",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": forecasting_library_models_for_roster(args.model_roster),
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "groups": M3_GROUPS,
            "source": "m3",
            "domain": "M3 forecasting competition train panels",
            "split_type": "last_official_horizon_from_training_panel",
            "series_limit_per_group": (None if args.m3_series_limit == 0 else args.m3_series_limit),
            "static_covariates": STATIC_COVARIATES,
        },
        "models": benchmark_model_names(args.model_roster),
        "model_settings": cartoboost_model_settings(cartoboost_config),
        "groups": results,
        "aggregate_quality": aggregate_quality,
        "official_metrics": aggregate_m_series_suite_official_metrics("m3", M3_GROUPS, results),
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            "groups": timings,
        },
        "resource_usage": resource_usage_snapshot(),
    }
    payload["comparability_audit"] = forecasting_comparability_audit(
        args=args,
        model_names=benchmark_model_names(args.model_roster),
        grouped_results=results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate_quality, indent=2, sort_keys=True))
    return 0


def aggregate_suite_quality(results: dict[str, Any]) -> dict[str, Any]:
    model_names = sorted(
        {model for problem in results.values() for model in problem.get("metrics", {}).keys()}
    )
    wins = dict.fromkeys(model_names, 0)
    top3 = dict.fromkeys(model_names, 0)
    rmse_ratios: dict[str, list[float]] = {model: [] for model in model_names}
    for problem in results.values():
        metrics = problem["metrics"]
        best_rmse = min(row["rmse"] for row in metrics.values())
        ranking = sorted(model_names, key=lambda name: metrics[name]["rmse"])
        for name in ranking[:3]:
            top3[name] += 1
        for name, row in metrics.items():
            if np.isclose(row["rmse"], best_rmse, rtol=1e-12):
                wins[name] += 1
            rmse_ratios[name].append(row["rmse"] / best_rmse)
    mean_rmse_ratio = {
        model: float(np.mean(values)) for model, values in rmse_ratios.items() if values
    }
    return {
        "problem_count": len(results),
        "wins_or_ties": wins,
        "top3_finishes": top3,
        "mean_rmse_ratio_to_problem_best": mean_rmse_ratio,
        "mean_rmse_ratio_ranking": sorted(mean_rmse_ratio, key=mean_rmse_ratio.__getitem__),
    }


def load_dataset(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.source in {"polars", "duckdb"}:
        return load_synthetic_fixture(args)
    if args.source == "m1":
        return load_m1_fixture(args)
    if args.source == "m4":
        return load_m4_fixture(args)
    if args.source == "m3":
        return load_m3_fixture(args)
    if args.source == "m5":
        return load_m5_fixture(args)
    if args.source == "m6":
        return load_m6_fixture(args)
    return load_nyc_taxi_fixture(args)


def load_synthetic_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    table = make_fixture(lanes=args.lanes, days=args.days, seed=args.seed, problem=args.problem)
    if args.source == "duckdb":
        duckdb = require_duckdb()
        con = duckdb.connect(":memory:")
        try:
            con.register("lane_demand", table.to_arrow())
            table = con.sql(
                """
                SELECT *
                FROM lane_demand
                ORDER BY lane_id, date
                """
            ).pl()
        finally:
            con.close()
    return table, {
        "series": args.lanes,
        "days": args.days,
        "horizon": args.horizon,
        "season_length": 7,
        "seed": args.seed,
        "domain": "daily NYC taxi-style pickup/dropoff lane demand",
        "source": "synthetic_fixture",
        "problem": args.problem,
        "problem_description": synthetic_problem_description(args.problem),
        "static_covariates": STATIC_COVARIATES,
    }


def load_m1_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()

    cache_dir = (
        DEFAULT_FORECASTING_CACHE_DIR if args.cache_dir == DEFAULT_CACHE_DIR else args.cache_dir
    )
    tsf_path = ensure_m1_tsf_file(args.m1_group, cache_dir=cache_dir, no_download=args.no_download)
    loaded, frequency, forecast_horizon = parse_tsf_dataframe(tsf_path, pd)
    group_info = M1_GROUP_INFO[args.m1_group]
    series_column = next(
        (column for column in ["series_name", "series_id", "unique_id"] if column in loaded),
        None,
    )
    if series_column is None:
        loaded = loaded.copy()
        loaded["series_name"] = [f"M1_{args.m1_group}_{index}" for index in range(len(loaded))]
        series_column = "series_name"
    loaded[series_column] = loaded[series_column].astype(str)
    available_series_ids = sorted(loaded[series_column].unique())
    series_ids = (
        available_series_ids
        if args.m1_series_limit == 0
        else available_series_ids[: args.m1_series_limit]
    )
    loaded = loaded[loaded[series_column].isin(series_ids)].copy()
    series_lookup = {series_id: index for index, series_id in enumerate(series_ids, start=1)}
    rows = []
    for row in loaded.sort_values(series_column).itertuples(index=False):
        row_data = row._asdict()
        series_id = str(row_data[series_column])
        series_values = [float(value) for value in row_data["series_value"] if not pd.isna(value)]
        series_index = series_lookup[series_id]
        for offset, value in enumerate(series_values):
            rows.append(
                {
                    "lane_id": series_id,
                    "date": pd.Timestamp("2000-01-01") + pd.to_timedelta(offset, unit="D"),
                    "loads": value,
                    "pickup_zone": float(series_index),
                    "dropoff_zone": float((series_index % 31) + 1),
                    "distance_miles": 1.0 + float(series_index % 25) / 5.0,
                    "airport_lane": float(series_index % 2 == 0),
                    "pickup_borough_code": float((series_index % 5) + 1),
                }
            )
    if not rows:
        raise ValueError(f"M1 {args.m1_group} TSF file did not contain selected series")
    result = (
        pl.from_pandas(pd.DataFrame(rows))
        .with_columns(pl.col("date").cast(pl.Datetime("us")))
        .sort(["lane_id", "date"])
    )
    horizon = int(forecast_horizon or group_info["horizon"])
    season_length = int(group_info["season_length"])
    return result, {
        "series": int(len(series_ids)),
        "available_series": int(len(available_series_ids)),
        "rows": int(result.height),
        "horizon": horizon,
        "season_length": season_length,
        "domain": f"M1 {args.m1_group} forecasting competition dataset",
        "source": "m1_zenodo_tsf",
        "group": args.m1_group,
        "source_url": m1_zip_url(args.m1_group),
        "tsf_file": str(tsf_path),
        "frequency": frequency,
        "series_limit": int(args.m1_series_limit),
        "split_type": "last_official_horizon_from_full_public_series",
        "static_covariates": STATIC_COVARIATES,
    }


def parse_tsf_dataframe(tsf_path: Path, pd: Any) -> tuple[Any, str | None, int | None]:
    attributes: list[tuple[str, str]] = []
    frequency: str | None = None
    horizon: int | None = None
    in_data = False
    rows: list[dict[str, Any]] = []
    with tsf_path.open("r", encoding="cp1252") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if not in_data:
                if lower.startswith("@attribute"):
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError(f"invalid TSF @attribute line {line_number} in {tsf_path}")
                    attributes.append((parts[1], parts[2].lower()))
                    continue
                if lower.startswith("@frequency"):
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        raise ValueError(f"invalid TSF @frequency line {line_number}")
                    frequency = parts[1]
                    continue
                if lower.startswith("@horizon"):
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        raise ValueError(f"invalid TSF @horizon line {line_number}")
                    horizon = int(parts[1])
                    continue
                if lower == "@data":
                    in_data = True
                    continue
                continue

            fields = line.split(":")
            if len(fields) != len(attributes) + 1:
                raise ValueError(
                    f"TSF data line {line_number} has {len(fields) - 1} attributes; "
                    f"expected {len(attributes)}"
                )
            row: dict[str, Any] = {}
            for (name, value_type), value in zip(attributes, fields[:-1], strict=True):
                row[name] = parse_tsf_attribute_value(value, value_type, line_number)
            values = []
            for value in fields[-1].split(","):
                value = value.strip()
                values.append(np.nan if value == "?" else float(value))
            if not values:
                raise ValueError(f"TSF data line {line_number} has no series values")
            row["series_value"] = values
            rows.append(row)
    if not in_data:
        raise ValueError(f"TSF file {tsf_path} is missing @data section")
    if not rows:
        raise ValueError(f"TSF file {tsf_path} did not contain any series rows")
    return pd.DataFrame(rows), frequency, horizon


def parse_tsf_attribute_value(value: str, value_type: str, line_number: int) -> Any:
    if value_type == "numeric":
        return float(value)
    if value_type == "date":
        return value
    if value_type == "string":
        return value
    raise ValueError(f"unsupported TSF attribute type {value_type!r} on line {line_number}")


def ensure_m1_tsf_file(group: str, *, cache_dir: Path, no_download: bool) -> Path:
    group_info = M1_GROUP_INFO[group]
    data_dir = cache_dir / "m1"
    tsf_path = data_dir / str(group_info["tsf_name"])
    if tsf_path.exists():
        return tsf_path
    if no_download:
        raise FileNotFoundError(
            f"M1 {group} benchmark requires {tsf_path}; remove --no-download to fetch "
            f"{m1_zip_url(group)} or pre-populate the TSF file."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / str(group_info["zip_name"])
    if not zip_path.exists():
        urllib.request.urlretrieve(m1_zip_url(group), zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".tsf")]
        if not members:
            raise FileNotFoundError(f"M1 archive {zip_path} did not contain a TSF file")
        member = next((name for name in members if Path(name).name == tsf_path.name), members[0])
        with archive.open(member) as source, tsf_path.open("wb") as destination:
            destination.write(source.read())
    if not tsf_path.exists():
        raise FileNotFoundError(f"failed to extract M1 TSF file to {tsf_path}")
    return tsf_path


def m1_zip_url(group: str) -> str:
    group_info = M1_GROUP_INFO[group]
    return (
        f"https://zenodo.org/records/{group_info['record_id']}/files/"
        f"{group_info['zip_name']}?download=1"
    )


def load_m3_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    try:
        from datasetsforecast.m3 import M3, M3Info
    except ImportError as exc:
        raise ImportError(
            "M3 benchmark source requires datasetsforecast; run `uv sync --group bench`."
        ) from exc

    cache_dir = (
        DEFAULT_FORECASTING_CACHE_DIR if args.cache_dir == DEFAULT_CACHE_DIR else args.cache_dir
    )
    data, _test, _info = M3.load(directory=str(cache_dir), group=args.m3_group)
    group_info = M3Info.get_group(args.m3_group)
    available_series_ids = sorted(data["unique_id"].unique())
    series_ids = (
        available_series_ids
        if args.m3_series_limit == 0
        else available_series_ids[: args.m3_series_limit]
    )
    data = data[data["unique_id"].isin(series_ids)].copy()
    data = data.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    series_lookup = {series_id: index for index, series_id in enumerate(series_ids, start=1)}
    data["series_index"] = data["unique_id"].map(series_lookup).astype(int)
    data["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
        data.groupby("unique_id").cumcount().astype(int),
        unit="D",
    )
    data["pickup_zone"] = data["series_index"].astype(int)
    data["dropoff_zone"] = (data["series_index"].astype(int) % 31) + 1
    data["distance_miles"] = 1.0 + (data["series_index"].astype(float) % 25.0) / 5.0
    data["airport_lane"] = (data["series_index"].astype(int) % 2 == 0).astype(float)
    data["pickup_borough_code"] = (data["series_index"].astype(int) % 5 + 1).astype(float)
    result = pl.from_pandas(
        data.rename(columns={"unique_id": "lane_id", "y": "loads"})[
            ["lane_id", "date", "loads", *STATIC_COVARIATES]
        ]
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    return result, {
        "series": int(len(series_ids)),
        "available_series": int(len(available_series_ids)),
        "rows": int(result.height),
        "horizon": int(group_info.horizon),
        "season_length": int(group_info.seasonality),
        "domain": f"M3 {args.m3_group} forecasting competition dataset",
        "source": "m3",
        "group": args.m3_group,
        "source_url": str(group_info.source_url),
        "series_limit": int(args.m3_series_limit),
        "split_type": "last_official_horizon_from_training_panel",
        "static_covariates": STATIC_COVARIATES,
    }


def aggregate_tlc_lane_partitions(
    paths: list[Path], *, frequency: str, cache_dir: Path
) -> tuple[Any, int, int]:
    """Aggregate each TLC Parquet partition before combining it in memory.

    The all-lane graph consumes lane-time summaries, never individual trips.
    Each completed partition is persisted before the next raw file is scanned.
    A cancelled multi-year run therefore resumes from completed aggregates
    rather than re-reading the raw TLC archive.
    """
    pl = require_polars()
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for NYC TLC Parquet metadata") from exc
    truncate = "1d" if frequency == "daily" else "1mo"
    aggregate_cache = cache_dir / "aggregates" / f"lane-{frequency}-v1"
    aggregate_cache.mkdir(parents=True, exist_ok=True)
    partitions = []
    raw_rows = 0
    clean_rows = 0
    for ordinal, path in enumerate(paths, start=1):
        raw_rows += int(pq.ParquetFile(path).metadata.num_rows)
        cache_path = aggregate_cache / f"{path.stem}.parquet"
        cache_hit = cache_path.is_file()
        try:
            partition = pl.read_parquet(cache_path) if cache_hit else None
        except Exception as exc:
            raise RuntimeError(f"invalid persisted TLC aggregate {cache_path}: {exc}") from exc
        if partition is None:
            base = (
                pl.scan_parquet(path)
                .select(
                    "tpep_pickup_datetime",
                    "tpep_dropoff_datetime",
                    "trip_distance",
                    "fare_amount",
                    "total_amount",
                    "PULocationID",
                    "DOLocationID",
                )
                .with_columns(
                    (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
                    .dt.total_seconds()
                    .alias("duration_sec")
                )
                .filter(
                    pl.col("duration_sec").is_between(60.0, 7200.0)
                    & pl.col("trip_distance").is_between(0.1, 100.0)
                    & pl.col("fare_amount").is_between(2.5, 500.0)
                    & pl.col("total_amount").is_between(2.5, 700.0)
                    & pl.col("PULocationID").is_between(1, 263)
                    & pl.col("DOLocationID").is_between(1, 263)
                )
                .with_columns(
                    pl.col("tpep_pickup_datetime").dt.truncate(truncate).alias("date"),
                    (pl.col("total_amount") / pl.col("trip_distance")).alias(
                        "effective_fare_per_mile"
                    ),
                )
            )
            partition = (
                base.group_by("date", "PULocationID", "DOLocationID")
                .agg(
                    pl.len().alias("loads"),
                    pl.col("effective_fare_per_mile")
                    .median()
                    .alias("taxi_effective_fare_per_mile"),
                    pl.col("trip_distance").sum().alias("distance_sum"),
                )
                .collect(engine="streaming")
            )
            temporary = cache_path.with_suffix(".parquet.part")
            partition.write_parquet(temporary)
            os.replace(temporary, cache_path)
        # `loads` is the exact number of records passing the filters, so this
        # avoids a second full raw-partition materialization solely to count
        # clean records.
        clean_rows += int(partition.get_column("loads").sum())
        partitions.append(partition)
        print(
            f"TLC aggregate {ordinal}/{len(paths)}: {path.name} "
            f"({'cache' if cache_hit else 'raw'})",
            flush=True,
        )
    if not partitions:
        raise ValueError("NYC TLC benchmark requires at least one Parquet partition")
    return pl.concat(partitions), raw_rows, clean_rows


def load_nyc_taxi_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    from scripts.run_nyc_taxi_quality_benchmarks import (
        TLC_TRIP_RECORD_PAGE,
        ensure_parquet_files,
        ensure_zone_centroids,
        ensure_zone_lookup,
        parse_months,
        zone_centroid_h3_cells,
    )

    months = parse_months(args.months)
    years = parse_taxi_years(args.years, fallback_year=args.year)
    paths = [
        path
        for year in years
        for path in ensure_parquet_files(
            taxi_type=args.taxi_type,
            year=year,
            months=months,
            cache_dir=args.cache_dir,
            no_download=args.no_download,
        )
    ]
    zone_lookup = ensure_zone_lookup(cache_dir=args.cache_dir, no_download=args.no_download)
    zone_centroids = (
        ensure_zone_centroids(cache_dir=args.cache_dir, no_download=args.no_download)
        if args.market_structure_splits
        else None
    )
    h3_panel_cache: Path | None = None
    h3_metadata_cache: Path | None = None
    if args.lsttn_h3_splits:
        if args.h3_resolution is None:
            raise ValueError("--lsttn-h3-splits requires --h3-resolution")
        selection = hashlib.sha256(
            json.dumps(
                {
                    "years": years,
                    "months": months,
                    "frequency": args.taxi_frequency,
                    "resolution": args.h3_resolution,
                    "taxi_type": args.taxi_type,
                    "version": 1,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
        final_cache_dir = args.cache_dir / "aggregates" / "h3-panel-v1"
        final_cache_dir.mkdir(parents=True, exist_ok=True)
        h3_panel_cache = final_cache_dir / f"{selection}.parquet"
        h3_metadata_cache = final_cache_dir / f"{selection}.json"
        if h3_panel_cache.is_file() and h3_metadata_cache.is_file():
            try:
                cached_result = pl.read_parquet(h3_panel_cache)
                cached_metadata = json.loads(h3_metadata_cache.read_text())
            except Exception as exc:
                raise RuntimeError(
                    f"invalid persisted H3 panel cache {h3_panel_cache}: {exc}"
                ) from exc
            print(f"TLC H3 panel: {h3_panel_cache} (cache)", flush=True)
            return cached_result, cached_metadata
    aggregated, raw_rows, clean_rows = aggregate_tlc_lane_partitions(
        paths, frequency=args.taxi_frequency, cache_dir=args.cache_dir
    )
    if args.taxi_frequency == "monthly":
        requested_buckets = [year * 100 + month for year in years for month in months]
        aggregated = aggregated.filter(
            (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).is_in(requested_buckets)
        )
    aggregated = aggregated.group_by("date", "PULocationID", "DOLocationID").agg(
        pl.col("loads").sum(),
        pl.col("taxi_effective_fare_per_mile").median(),
        pl.col("distance_sum").sum(),
    )
    if args.lsttn_h3_splits:
        if args.h3_resolution is None:
            raise ValueError("--lsttn-h3-splits requires --h3-resolution")
        zone_h3 = zone_centroid_h3_cells(
            cache_dir=args.cache_dir,
            resolution=args.h3_resolution,
            no_download=args.no_download,
        )
        # This is a separate scale-oriented path.  It deliberately avoids
        # materialising lane strings or sorting lane volumes: LSTTN consumes
        # every observed directed PU→DO relation, then coalesces it into H3
        # endpoints for its sparse graph.
        result = aggregated.with_columns(
            pl.col("PULocationID").replace_strict(zone_h3).alias("pickup_h3"),
            pl.col("DOLocationID").replace_strict(zone_h3).alias("dropoff_h3"),
        ).select(
            "date",
            "PULocationID",
            "DOLocationID",
            "pickup_h3",
            "dropoff_h3",
            "loads",
        )
        metadata = {
            "series": int(result.select("pickup_h3").unique().height),
            "days": int(result.select("date").unique().height),
            "horizon": args.horizon,
            "season_length": 7 if args.taxi_frequency == "daily" else 12,
            "frequency": args.taxi_frequency,
            "source": "nyc_tlc_trip_records",
            "years": years,
            "months": months,
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "aggregated_rows": int(result.height),
            "h3_endpoint_resolution": args.h3_resolution,
        }
        if h3_panel_cache is None or h3_metadata_cache is None:
            raise RuntimeError("LSTTN H3 cache paths were not initialized")
        temporary_panel = h3_panel_cache.with_suffix(".parquet.part")
        temporary_metadata = h3_metadata_cache.with_suffix(".json.part")
        result.write_parquet(temporary_panel)
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_panel, h3_panel_cache)
        os.replace(temporary_metadata, h3_metadata_cache)
        print(f"TLC H3 panel: {h3_panel_cache} (persisted)", flush=True)
        return result, metadata
    aggregated = aggregated.with_columns(
        (
            pl.lit("PU")
            + pl.col("PULocationID").cast(pl.String)
            + pl.lit("->DO")
            + pl.col("DOLocationID").cast(pl.String)
        ).alias("lane_id")
    )
    lane_counts = (
        aggregated.group_by("lane_id")
        .agg(pl.col("loads").sum().alias("trip_count"))
        .sort("trip_count", descending=True)
    )
    selected_lane_ids = (
        lane_counts.get_column("lane_id")
        if args.all_observed_lanes
        else lane_counts.head(args.lanes).get_column("lane_id")
    )
    selected = aggregated.filter(pl.col("lane_id").is_in(selected_lane_ids.implode()))
    zone_h3 = None
    if args.h3_resolution is not None:
        zone_h3 = zone_centroid_h3_cells(
            cache_dir=args.cache_dir,
            resolution=args.h3_resolution,
            no_download=args.no_download,
        )
        selected = selected.with_columns(
            pl.col("PULocationID").replace_strict(zone_h3).alias("pickup_h3"),
            pl.col("DOLocationID").replace_strict(zone_h3).alias("dropoff_h3"),
        )
    static = (
        selected.group_by("lane_id")
        .agg(
            pl.col("PULocationID").first().alias("pickup_zone"),
            pl.col("DOLocationID").first().alias("dropoff_zone"),
            (pl.col("distance_sum").sum() / pl.col("loads").sum()).alias("distance_miles"),
        )
        .to_pandas()
        .assign(
            airport_lane=lambda data: (
                data[["pickup_zone", "dropoff_zone"]]
                .isin(AIRPORT_ZONE_IDS)
                .any(axis=1)
                .astype(float)
            ),
            pickup_borough_code=lambda data: data["pickup_zone"].map(
                lambda zone: float(zone_lookup[int(zone)].borough_code)
            ),
        )
    )
    if zone_centroids is not None:
        static = static.assign(
            origin_x=lambda data: data["pickup_zone"].map(
                lambda zone: float(zone_centroids[int(zone)][0])
            ),
            origin_y=lambda data: data["pickup_zone"].map(
                lambda zone: float(zone_centroids[int(zone)][1])
            ),
            destination_x=lambda data: data["dropoff_zone"].map(
                lambda zone: float(zone_centroids[int(zone)][0])
            ),
            destination_y=lambda data: data["dropoff_zone"].map(
                lambda zone: float(zone_centroids[int(zone)][1])
            ),
        )
    if zone_h3 is not None:
        static = static.assign(
            pickup_h3=lambda data: data["pickup_zone"].map(zone_h3),
            dropoff_h3=lambda data: data["dropoff_zone"].map(zone_h3),
        )
        if static[["pickup_h3", "dropoff_h3"]].isna().any().any():
            raise ValueError("TLC zone geometry is missing an H3 centroid for a selected lane")
    daily = selected.select("lane_id", "date", "loads", "taxi_effective_fare_per_mile").to_pandas()
    dates = pd.DataFrame(
        {
            "date": pd.date_range(
                selected.get_column("date").min(),
                selected.get_column("date").max(),
                freq="D" if args.taxi_frequency == "daily" else "MS",
            )
        }
    )
    full_index = static[["lane_id"]].merge(dates, how="cross")
    table = (
        full_index.merge(daily, on=["lane_id", "date"], how="left")
        .merge(static, on="lane_id", how="left")
        .assign(loads=lambda data: data["loads"].fillna(0.0).astype(float))
        .sort_values(["lane_id", "date"])
    )
    result = pl.from_pandas(table).with_columns(pl.col("date").cast(pl.Datetime("us")))
    return result, {
        "series": int(static.shape[0]),
        "days": int(dates.shape[0]),
        "horizon": args.horizon,
        "season_length": 7 if args.taxi_frequency == "daily" else 12,
        "domain": f"real daily NYC TLC {args.taxi_type} taxi pickup/dropoff lane demand",
        "source": "nyc_tlc_trip_records",
        "source_url": TLC_TRIP_RECORD_PAGE,
        "taxi_type": args.taxi_type,
        "year": args.year if len(years) == 1 else None,
        "years": years,
        "months": months,
        "all_observed_lanes": bool(args.all_observed_lanes),
        "h3_endpoint_resolution": args.h3_resolution,
        "frequency": args.taxi_frequency,
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "aggregated_rows": int(result.height),
        "static_covariates": STATIC_COVARIATES,
    }


def load_m4_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    try:
        from datasetsforecast.m4 import M4
    except ImportError as exc:
        raise ImportError(
            "M4 benchmark source requires datasetsforecast; run "
            "`uv sync --group bench` after adding benchmark extras."
        ) from exc

    cache_dir = (
        DEFAULT_FORECASTING_CACHE_DIR if args.cache_dir == DEFAULT_CACHE_DIR else args.cache_dir
    )
    train, _test, info = M4.load(directory=str(cache_dir), group=args.m4_group)
    available_series_ids = sorted(train["unique_id"].unique())
    series_ids = (
        available_series_ids
        if args.m4_series_limit == 0
        else available_series_ids[: args.m4_series_limit]
    )
    data = train[train["unique_id"].isin(series_ids)].copy()
    info = info[info["unique_id"].isin(series_ids)].copy()
    category_codes = {
        category: index for index, category in enumerate(sorted(info["category"].unique()), start=1)
    }
    info = info.assign(
        series_index=lambda frame: frame["unique_id"].map(
            {series_id: index for index, series_id in enumerate(series_ids, start=1)}
        ),
        category_code=lambda frame: frame["category"].map(category_codes),
    )
    data = data.merge(info, on="unique_id", how="left")
    data = data.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    data["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(data["ds"].astype(int), unit="D")
    data["pickup_zone"] = data["series_index"].astype(int)
    data["dropoff_zone"] = data["category_code"].astype(int)
    data["distance_miles"] = 1.0 + (data["series_index"].astype(float) % 25.0) / 5.0
    data["airport_lane"] = (data["category_code"].astype(int) % 2 == 0).astype(float)
    data["pickup_borough_code"] = data["category_code"].astype(float)
    result = pl.from_pandas(
        data.rename(columns={"unique_id": "lane_id", "y": "loads"})[
            ["lane_id", "date", "loads", *STATIC_COVARIATES]
        ]
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    horizon = m4_horizon(args.m4_group)
    season_length = m4_season_length(args.m4_group)
    return result, {
        "series": int(len(series_ids)),
        "available_series": int(len(available_series_ids)),
        "rows": int(result.height),
        "horizon": horizon,
        "season_length": season_length,
        "domain": f"M4 {args.m4_group} forecasting competition dataset",
        "source": "m4",
        "group": args.m4_group,
        "series_limit": args.m4_series_limit,
        "static_covariates": STATIC_COVARIATES,
    }


def load_m5_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    data_dir = ensure_m5_data_dir(args.m5_data_dir, no_download=args.no_download)
    sales_path = find_m5_sales_file(data_dir)
    prices_path = data_dir / "sell_prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError(
            f"M5 official-style WRMSSE requires {prices_path}; download the Kaggle M5 "
            "Accuracy files and point --m5-data-dir at the extracted directory."
        )
    calendar_path = data_dir / "calendar.csv"
    if not calendar_path.exists():
        raise FileNotFoundError(
            f"M5 benchmark requires {calendar_path}; download the Kaggle M5 Accuracy files "
            "and point --m5-data-dir at the extracted directory."
        )

    sales = pl.read_csv(sales_path, n_rows=args.m5_series_limit or None)
    calendar = pl.read_csv(calendar_path)
    prices = pl.read_csv(prices_path)
    required_sales_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing_sales = [column for column in required_sales_columns if column not in sales.columns]
    if missing_sales:
        raise ValueError(f"M5 sales file is missing required columns: {missing_sales}")
    required_price_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing_prices = [column for column in required_price_columns if column not in prices.columns]
    if missing_prices:
        raise ValueError(f"M5 sell_prices file is missing required columns: {missing_prices}")
    if "id" not in sales.columns:
        sales = sales.with_columns(
            pl.concat_str(["item_id", "store_id"], separator="_").alias("id")
        )
    if "date" not in calendar.columns:
        raise ValueError("M5 calendar file is missing required column: date")
    if "wm_yr_wk" not in calendar.columns:
        raise ValueError("M5 calendar file is missing required column: wm_yr_wk")
    if "d" not in calendar.columns:
        calendar = calendar.with_row_index("d_index").with_columns(
            pl.format("d_{}", pl.col("d_index") + 1).alias("d")
        )

    available_series = count_m5_series(sales_path)
    value_columns = sorted(
        [column for column in sales.columns if column.startswith("d_")],
        key=lambda value: int(value.split("_", 1)[1]),
    )
    if len(value_columns) <= 28:
        raise ValueError("M5 sales file must contain more than 28 daily observations per series")
    materialized_value_columns = (
        value_columns
        if args.m5_history_days == 0
        else value_columns[-max(args.m5_history_days, 29) :]
    )

    id_columns = ["id", *required_sales_columns]
    long = unpivot_frame(
        sales,
        index=id_columns,
        on=materialized_value_columns,
        variable_name="d",
        value_name="loads",
    )
    calendar_feature_columns = [
        column for column in [*M5_EVENT_COLUMNS, *M5_SNAP_COLUMNS] if column in calendar.columns
    ]
    calendar = calendar.select(
        "d",
        "wm_yr_wk",
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).cast(pl.Datetime("us")),
        *calendar_feature_columns,
    )
    calendar = add_m5_calendar_covariates(calendar, calendar_feature_columns)

    lookup_frame = m5_static_lookup(sales)
    m5_known_columns = [
        column for column in M5_KNOWN_FUTURE_COVARIATES if column != "m5_sell_price"
    ]
    result = (
        long.join(calendar, on="d", how="inner")
        .join(
            prices.select("store_id", "item_id", "wm_yr_wk", "sell_price"),
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )
        .join(lookup_frame, on=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .with_columns(
            pl.when(pl.col("sell_price").is_null())
            .then(0.0)
            .otherwise(pl.col("loads").cast(pl.Float64) * pl.col("sell_price").cast(pl.Float64))
            .alias("weight_value"),
            pl.col("sell_price").fill_null(0.0).cast(pl.Float64).alias("m5_sell_price"),
        )
        .select(
            pl.col("id").alias("lane_id"),
            "date",
            pl.col("loads").cast(pl.Float64),
            "weight_value",
            "m5_sell_price",
            *m5_known_columns,
            *M5_HIERARCHY_COVARIATES,
            *STATIC_COVARIATES,
        )
        .sort(["lane_id", "date"])
    )
    if result.height != sales.height * len(materialized_value_columns):
        raise RuntimeError(
            "M5 calendar join dropped rows; check that calendar.csv covers all d_* columns"
        )
    return result, {
        "series": int(sales.height),
        "available_series": int(available_series),
        "rows": int(result.height),
        "days": int(len(materialized_value_columns)),
        "available_days": int(len(value_columns)),
        "horizon": 28,
        "season_length": 7,
        "domain": "M5 Forecasting Accuracy Walmart item-store unit sales",
        "source": "m5_kaggle_local_files",
        "source_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
        "mirror_url": "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip",
        "sales_file": str(sales_path),
        "calendar_file": str(calendar_path),
        "prices_file": str(prices_path),
        "series_limit": int(args.m5_series_limit),
        "history_days": int(args.m5_history_days),
        "split_type": "last_28_days_from_training_or_evaluation_file",
        "official_style_inputs": {
            "calendar_events_present": bool(calendar_feature_columns),
            "sell_prices_present": True,
            "snap_columns_present": sorted(
                column for column in calendar_feature_columns if column.startswith("snap_")
            ),
        },
        "known_future_covariates": known_future_covariate_columns(result),
        "hierarchy_covariates": M5_HIERARCHY_COVARIATES,
        "static_covariates": STATIC_COVARIATES,
    }


def count_m5_series(sales_path: Path) -> int:
    with sales_path.open("rb") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def ensure_m5_data_dir(data_dir: Path, *, no_download: bool) -> Path:
    if m5_data_files_exist(data_dir):
        return data_dir
    nested_data_dir = data_dir / "datasets"
    if m5_data_files_exist(nested_data_dir):
        return nested_data_dir
    if no_download:
        return data_dir
    try:
        from datasetsforecast.m5 import M5
    except ImportError as exc:
        raise ImportError(
            "M5 benchmark download requires datasetsforecast; run `uv sync --group bench`."
        ) from exc
    M5.download(str(data_dir.parent))
    if m5_data_files_exist(nested_data_dir):
        return nested_data_dir
    if m5_data_files_exist(data_dir):
        return data_dir
    raise FileNotFoundError(
        f"M5 public mirror download completed but required CSVs were not found under {data_dir}"
    )


def m5_data_files_exist(data_dir: Path) -> bool:
    return (
        (data_dir / "calendar.csv").exists()
        and (
            (data_dir / "sales_train_evaluation.csv").exists()
            or (data_dir / "sales_train_validation.csv").exists()
        )
        and (data_dir / "sell_prices.csv").exists()
    )


def find_m5_sales_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "sales_train_evaluation.csv",
        data_dir / "sales_train_validation.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "M5 benchmark requires sales_train_evaluation.csv or sales_train_validation.csv "
        f"under {data_dir}; download the Kaggle M5 Accuracy files first."
    )


def add_m5_calendar_covariates(calendar: Any, calendar_feature_columns: list[str]) -> Any:
    pl = require_polars()
    result = calendar
    for column in M5_EVENT_COLUMNS:
        output = f"m5_{column}_code"
        if column not in calendar_feature_columns:
            result = result.with_columns(pl.lit(0.0).alias(output))
            continue
        normalized = f"__{column}_normalized"
        result = result.with_columns(
            pl.col(column).cast(pl.Utf8).fill_null("__none__").alias(normalized)
        )
        lookup = (
            result.select(pl.col(normalized).unique())
            .sort(normalized)
            .with_row_index(output, offset=0)
            .with_columns(pl.col(output).cast(pl.Float64))
        )
        result = result.join(lookup, on=normalized, how="left").drop(normalized)
    for column in M5_SNAP_COLUMNS:
        output = f"m5_{column}"
        if column in calendar_feature_columns:
            result = result.with_columns(pl.col(column).fill_null(0).cast(pl.Float64).alias(output))
        else:
            result = result.with_columns(pl.lit(0.0).alias(output))
    return result


def unpivot_frame(
    frame: Any,
    *,
    index: list[str],
    on: list[str],
    variable_name: str,
    value_name: str,
) -> Any:
    if hasattr(frame, "unpivot"):
        return frame.unpivot(
            index=index,
            on=on,
            variable_name=variable_name,
            value_name=value_name,
        )
    return frame.melt(
        id_vars=index,
        value_vars=on,
        variable_name=variable_name,
        value_name=value_name,
    )


def m5_static_lookup(sales: Any) -> Any:
    pl = require_polars()
    base = sales.select("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
    lookups = [
        code_lookup(base, "store_id", "pickup_zone_code"),
        code_lookup(base, "item_id", "dropoff_zone_code"),
        code_lookup(base, "dept_id", "dept_code"),
        code_lookup(base, "cat_id", "cat_code"),
        code_lookup(base, "state_id", "state_code"),
    ]
    for lookup in lookups:
        base = base.join(lookup, on=lookup.columns[0], how="left")
    return base.with_columns(
        pl.col("pickup_zone_code").cast(pl.Float64).alias("pickup_zone"),
        pl.col("dropoff_zone_code").cast(pl.Float64).alias("dropoff_zone"),
        (1.0 + (pl.col("dept_code").cast(pl.Float64) % 20.0) / 4.0).alias("distance_miles"),
        (pl.col("cat_code") % 2).cast(pl.Float64).alias("airport_lane"),
        pl.col("state_code").cast(pl.Float64).alias("pickup_borough_code"),
        pl.col("state_code").cast(pl.Float64).alias("m5_state_code"),
        pl.col("pickup_zone_code").cast(pl.Float64).alias("m5_store_code"),
        pl.col("cat_code").cast(pl.Float64).alias("m5_cat_code"),
        pl.col("dept_code").cast(pl.Float64).alias("m5_dept_code"),
        pl.col("dropoff_zone_code").cast(pl.Float64).alias("m5_item_code"),
    ).select(
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        *M5_HIERARCHY_COVARIATES,
        *STATIC_COVARIATES,
    )


def code_lookup(frame: Any, column: str, output: str) -> Any:
    pl = require_polars()
    values = sorted(frame.select(pl.col(column).unique()).to_series().to_list())
    return pl.DataFrame({column: values, output: list(range(1, len(values) + 1))})


def load_m6_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    assets_path = ensure_m6_assets_file(args.m6_assets_path, no_download=args.no_download)
    raw = pd.read_csv(assets_path)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required_columns = {"symbol", "date", "price"}
    missing = sorted(required_columns - set(raw.columns))
    if missing:
        raise ValueError(f"M6 assets file is missing required columns: {missing}")
    raw = raw[["symbol", "date", "price"]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    raw["price"] = pd.to_numeric(raw["price"], errors="raise")
    raw = raw.dropna(subset=["symbol", "date", "price"]).sort_values(["symbol", "date"])
    if raw.empty:
        raise ValueError("M6 assets file did not contain any usable symbol/date/price rows")

    available_symbols = sorted(raw["symbol"].astype(str).unique())
    selected_symbols = (
        available_symbols
        if args.m6_series_limit == 0
        else available_symbols[: args.m6_series_limit]
    )
    raw = raw[raw["symbol"].astype(str).isin(selected_symbols)].copy()
    result = build_m6_daily_return_panel(raw, selected_symbols)
    day_count = result.select(pl.col("date").unique()).height
    if day_count <= args.m6_horizon + 60:
        raise ValueError("M6 assets file does not leave enough daily observations for the holdout")
    return result, {
        "series": int(len(selected_symbols)),
        "available_series": int(len(available_symbols)),
        "rows": int(result.height),
        "days": int(day_count),
        "horizon": int(args.m6_horizon),
        "season_length": 7,
        "domain": "M6 financial competition assets daily return point-forecast proxy",
        "source": "m6_methods_assets_csv",
        "source_url": M6_ASSETS_URL,
        "assets_file": str(assets_path),
        "series_limit": int(args.m6_series_limit),
        "split_type": f"last_{args.m6_horizon}_calendar_days_from_daily_return_panel",
        "official_metric_note": (
            "M6 official scoring used probability rank buckets, RPS, and investment return. "
            "This benchmark scores daily point return forecasts with the shared CartoBoost "
            "library RMSE/MAE/WAPE harness."
        ),
        "static_covariates": STATIC_COVARIATES,
    }


def ensure_m6_assets_file(path: Path, *, no_download: bool) -> Path:
    if path.exists():
        return path
    if no_download:
        raise FileNotFoundError(
            f"M6 benchmark requires {path}; remove --no-download to fetch {M6_ASSETS_URL} "
            "or provide --m6-assets-path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(M6_ASSETS_URL, path)
    if not path.exists():
        raise FileNotFoundError(f"failed to download M6 assets file to {path}")
    return path


def build_m6_daily_return_panel(raw: Any, selected_symbols: list[str]) -> Any:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    symbol_codes = {symbol: index for index, symbol in enumerate(selected_symbols, start=1)}
    pieces = []
    for symbol in selected_symbols:
        group = raw[raw["symbol"].astype(str) == symbol].sort_values("date")
        if group.empty:
            continue
        date_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        prices = (
            group.drop_duplicates("date")
            .set_index("date")["price"]
            .sort_index()
            .reindex(date_index)
            .ffill()
        )
        returns = prices.pct_change().fillna(0.0)
        code = symbol_codes[symbol]
        pieces.append(
            pd.DataFrame(
                {
                    "lane_id": symbol,
                    "date": date_index,
                    "loads": returns.to_numpy(dtype=float),
                    "pickup_zone": float(code),
                    "dropoff_zone": float((code % 11) + 1),
                    "distance_miles": np.log1p(prices.to_numpy(dtype=float)),
                    "airport_lane": float(code % 2),
                    "pickup_borough_code": float((code % 5) + 1),
                }
            )
        )
    if not pieces:
        raise ValueError("M6 assets file did not contain any selected symbols")
    return (
        pl.from_pandas(pd.concat(pieces, ignore_index=True))
        .with_columns(
            pl.col("date").cast(pl.Datetime("us")),
            pl.col("loads").fill_nan(0.0).fill_null(0.0),
            pl.col("distance_miles").fill_nan(0.0).fill_null(0.0),
        )
        .sort(["lane_id", "date"])
    )


def m4_horizon(group: str) -> int:
    horizons = {
        "Hourly": 48,
        "Daily": 14,
        "Weekly": 13,
        "Monthly": 18,
        "Quarterly": 8,
        "Yearly": 6,
    }
    return horizons[group]


def m4_season_length(group: str) -> int:
    season_lengths = {
        "Hourly": 24,
        "Daily": 1,
        "Weekly": 1,
        "Monthly": 12,
        "Quarterly": 4,
        "Yearly": 1,
    }
    return season_lengths[group]


def synthetic_problem_description(problem: str) -> str:
    descriptions = {
        "taxi_weekly": "Weekly lane demand with slow drift and deterministic airport events.",
        "airport_calendar_events": (
            "Airport pickup/dropoff lanes receive repeated day-of-month surges."
        ),
        "route_mix_shift": (
            "Longer routes and airport lanes have horizon-relevant route-mix swings."
        ),
        "borough_monthly_pulses": (
            "Pickup borough codes drive repeated monthly taxi-demand pulses."
        ),
    }
    return descriptions[problem]


def make_fixture(*, lanes: int, days: int, seed: int, problem: str) -> Any:
    pl = require_polars()
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1)
    rows = []
    for lane_idx in range(lanes):
        pickup_zone = 101 + lane_idx
        dropoff_zone = 201 + ((lane_idx * 7) % lanes)
        distance = 1.5 + (lane_idx % 9) * 0.8
        airport_lane = float(lane_idx % 11 == 0)
        pickup_borough_code = float(lane_idx % 5)
        base = 12.0 + 0.35 * distance + 5.0 * airport_lane + 1.2 * pickup_borough_code
        lane_effect = 2.0 * np.sin(lane_idx / 3.0)
        lane_noise = rng.normal(loc=0.0, scale=0.03)
        for day in range(days):
            timestamp = start + timedelta(days=day)
            weekly = [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 2.0][timestamp.weekday()]
            slow_drift = 0.04 * day
            airport_event = synthetic_airport_event(problem, airport_lane, timestamp, day)
            route_event = synthetic_route_event(problem, distance, airport_lane, timestamp, day)
            borough_event = synthetic_borough_event(problem, pickup_borough_code, timestamp)
            quarterly_event = (
                2.5 if problem == "taxi_weekly" and day % 91 in {12, 13, 14, 15} else 0.0
            )
            deterministic_noise = ((lane_idx * 17 + day * 13) % 11 - 5) * 0.12
            demand = max(
                0.0,
                base
                + lane_effect
                + weekly
                + slow_drift
                + airport_event
                + route_event
                + borough_event
                + quarterly_event
                + deterministic_noise,
            )
            rows.append(
                {
                    "lane_id": f"PU{pickup_zone}->DO{dropoff_zone}",
                    "date": timestamp,
                    "loads": float(demand + lane_noise),
                    "pickup_zone": pickup_zone,
                    "dropoff_zone": dropoff_zone,
                    "distance_miles": float(distance),
                    "airport_lane": airport_lane,
                    "pickup_borough_code": pickup_borough_code,
                }
            )
    return pl.DataFrame(rows)


def synthetic_airport_event(
    problem: str, airport_lane: float, timestamp: datetime, day: int
) -> float:
    if not airport_lane:
        return 0.0
    if problem == "airport_calendar_events":
        return 8.0 if timestamp.day in {5, 6, 20, 21} else 0.0
    if problem == "route_mix_shift":
        return 4.5 if timestamp.weekday() in {0, 4, 6} else 0.0
    return 4.0 if day % 28 in {5, 6, 7} else 0.0


def synthetic_route_event(
    problem: str,
    distance: float,
    airport_lane: float,
    timestamp: datetime,
    day: int,
) -> float:
    if problem != "route_mix_shift":
        return 0.0
    long_route = distance >= 5.5
    if long_route and day % 14 in {10, 11, 12, 13}:
        return 5.5
    if airport_lane and timestamp.day in {1, 2, 15, 16}:
        return 3.5
    return 0.0


def synthetic_borough_event(problem: str, pickup_borough_code: float, timestamp: datetime) -> float:
    if problem != "borough_monthly_pulses":
        return 0.0
    if int(pickup_borough_code) in {1, 3} and timestamp.day in {8, 9, 10}:
        return 7.0
    if int(pickup_borough_code) in {2, 4} and timestamp.day in {23, 24, 25}:
        return 5.0
    return 0.0


def score_models(
    table: Any,
    *,
    horizon: int,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str] | None = None,
    source: str = "synthetic",
    candidate_selection: bool = True,
    cutoff: Any | None = None,
    candidate_validation_cache: dict[Any, dict[str, float]] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any], Any]:
    pl = require_polars()
    if model_names is None:
        model_names = benchmark_model_names("full")
    train, test, cutoff = train_test_split_for_cutoff(table, horizon=horizon, cutoff=cutoff)
    if train.is_empty() or test.is_empty():
        raise ValueError("benchmark split produced empty train or test data")

    actual = (
        test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    m5_inner_cutoffs = (
        shared_candidate_validation_cutoffs(timestamps, horizon=horizon, source=source)
        if candidate_selection and source == "m5"
        else []
    )
    predictions, timing = forecast_model_roster(
        train,
        horizon,
        season_length=season_length,
        cartoboost_config=cartoboost_config,
        model_names=model_names,
        source=source,
        known_future=known_future_covariate_frame(table),
        skip_m5_raw_auto_candidate=(len(m5_inner_cutoffs) == 1),
        skip_m6_raw_auto_candidate=False,
        skip_non_m_raw_auto_candidate=(
            candidate_selection
            and source not in {"m4", "m5", "m6"}
            and "cartoboost_auto_forecast" in model_names
        ),
    )
    if candidate_selection:
        predictions, selection_timing = apply_shared_candidate_selection(
            train,
            horizon,
            season_length=season_length,
            source=source,
            raw_predictions=predictions,
            model_timing=timing.get("models", {}),
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            validation_cache=candidate_validation_cache,
        )
        if (
            source not in {"m4", "m5", "m6"}
            and selection_timing.get("selected_candidates", {}).get("cartoboost_auto_forecast")
            == "cartoboost_auto_forecast"
            and timing.get("models", {}).get("cartoboost_auto_forecast", {}).get("selector_mode")
            == "non_m_outer_lazy_raw_auto_candidate"
        ):
            auto_predictions, auto_timing = cartoboost_forecast(
                train,
                horizon,
                season_length=season_length,
                config=cartoboost_source_config(cartoboost_config, source=source),
                prediction_col="cartoboost_auto_forecast",
            )
            predictions = replace_forecast_column(
                predictions,
                auto_predictions,
                "cartoboost_auto_forecast",
            )
            timing["models"]["cartoboost_auto_forecast"] = {
                **auto_timing,
                "selector_mode": "non_m_outer_lazy_raw_auto_selected",
            }
    else:
        selection_timing = {
            "calibration_seconds": 0.0,
            "inner_origin_count": 0.0,
            "selected_candidates": {model: model for model in model_names},
            "disabled": True,
        }
    scored = actual.join(predictions, on=["series_id", "timestamp", "horizon"], how="inner")
    if scored.height != actual.height:
        raise RuntimeError("forecast alignment dropped rows")

    metrics = {
        model: evaluate_metrics(scored, model, train, season_length=season_length)
        for model in model_names
    }
    quality = quality_summary(metrics, model_names=model_names)
    timing["candidate_selection"] = selection_timing
    return metrics, quality, timing, scored


def train_test_split_for_cutoff(
    table: Any,
    *,
    horizon: int,
    cutoff: Any | None,
) -> tuple[Any, Any, Any]:
    pl = require_polars()
    timestamps = table.select(pl.col("date").unique().sort()).to_series().to_list()
    if cutoff is None:
        cutoff = timestamps[-horizon]
    try:
        start_index = timestamps.index(cutoff)
    except ValueError as exc:
        raise ValueError(f"cutoff {cutoff!r} is not present in benchmark timestamps") from exc
    end_index = start_index + horizon
    if end_index > len(timestamps):
        raise ValueError("cutoff leaves fewer timestamps than requested horizon")
    validation_timestamps = timestamps[start_index:end_index]
    train = table.filter(pl.col("date") < cutoff)
    test = table.filter(pl.col("date").is_in(validation_timestamps))
    return train, test, cutoff


def auto_selection_objective(source: str) -> str:
    if source == "m5":
        return "wrmsse"
    if source == "m6":
        return "investment_decision_return_then_rps"
    if source in {"m", "m1", "m3", "m4"}:
        return "owa_proxy"
    return "rmse"


def autostats_validation_objective(source: str) -> str:
    if source in {"m", "m1", "m3"}:
        return "smape_mase_average"
    return "mean_squared_error"


def forecast_objective_loss(
    objective: str,
    *,
    train: Any,
    scored: Any,
    prediction_col: str,
    season_length: int,
) -> float:
    if objective == "wrmsse":
        artifact = m5_wrmsse_artifact(
            train,
            scored,
            model_names=[prediction_col],
            seasonal_period=1,
        )
        value = artifact["model_scores"][prediction_col]
        return math.inf if value is None else float(value)
    if objective == "rank_probability_score":
        artifact = m6_rps_artifact(scored, model_names=[prediction_col])
        return float(artifact["models"][prediction_col]["mean_rps"])
    if objective == "rank_probability_score_then_rmse":
        rps = forecast_objective_loss(
            "rank_probability_score",
            train=train,
            scored=scored,
            prediction_col=prediction_col,
            season_length=season_length,
        )
        rmse = rmse_expr(scored, prediction_col)
        return float(rps + 1.0e-6 * rmse)
    if objective == "investment_decision_return_then_rps":
        return m6_investment_decision_loss(
            scored,
            prediction_col=prediction_col,
            rps_tiebreak_weight=M6_INVESTMENT_RPS_TIEBREAK_WEIGHT,
        )
    if objective == "owa_proxy":
        metrics = evaluate_metrics(scored, prediction_col, train, season_length=season_length)
        return float(0.5 * metrics["mase"] + 0.5 * metrics["smape"])
    return rmse_expr(scored, prediction_col)


def rolling_origin_cutoffs(table: Any, *, horizon: int, folds: int) -> list[Any]:
    timestamps = table.select(require_polars().col("date").unique().sort()).to_series().to_list()
    required = horizon * folds + 1
    if len(timestamps) <= required:
        raise ValueError("not enough timestamps for requested rolling-origin folds")
    start = len(timestamps) - horizon * folds
    return [timestamps[start + fold * horizon] for fold in range(folds)]


def score_rolling_origin_problem(
    table: Any,
    *,
    horizon: int,
    season_length: int,
    folds: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    source: str = "synthetic",
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any], dict[str, Any], Any]:
    split_results: dict[str, Any] = {}
    timing: dict[str, Any] = {"splits": {}}
    cutoffs = rolling_origin_cutoffs(table, horizon=horizon, folds=folds)
    candidate_validation_cache: dict[Any, dict[str, float]] = {}
    scored_folds: list[Any] = []
    for fold_index, cutoff in enumerate(cutoffs, start=1):
        split_name = f"rolling_origin_{fold_index}"
        metrics, quality, split_timing, _scored = score_models(
            table,
            horizon=horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=source,
            cutoff=cutoff,
            candidate_validation_cache=candidate_validation_cache,
        )
        split_results[split_name] = {
            "cutoff": str(cutoff),
            "metrics": metrics,
            "quality": quality,
        }
        timing["splits"][split_name] = split_timing
        scored_folds.append(_scored)
    aggregate_metrics = aggregate_split_metrics(split_results)
    pl = require_polars()
    scored = pl.concat(scored_folds, how="vertical") if len(scored_folds) > 1 else scored_folds[0]
    return split_results, aggregate_metrics, quality_summary(aggregate_metrics), timing, scored


def run_neural_panel_split_suite(
    args: argparse.Namespace,
    *,
    table: Any,
    dataset: dict[str, Any],
    source_file_hashes: dict[str, str],
    load_seconds: float,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    horizon = int(dataset.get("horizon", args.horizon))
    season_length = int(dataset.get("season_length", 7))
    model_names = benchmark_model_names("neural-panel")
    split_results: dict[str, Any] = {}
    scored_frames = []
    timing: dict[str, Any] = {"load_seconds": load_seconds, "splits": {}}
    for split_name, split in neural_panel_split_frames(
        table,
        horizon=horizon,
        folds=max(1, args.suite_folds),
    ).items():
        split_start = perf_counter()
        metrics, quality, split_timing, scored = score_neural_panel_split(
            split["train"],
            split["test"],
            horizon=horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            fallback=split["fallback"],
        )
        split_results[split_name] = {
            "split_type": split["split_type"],
            "cutoff": str(split["cutoff"]),
            "train_rows": int(split["train"].height),
            "test_rows": int(split["test"].height),
            "heldout_lanes": split["heldout_lanes"],
            "heldout_origins": split["heldout_origins"],
            "sparse_tail_lanes": split["sparse_tail_lanes"],
            "fallback": split["fallback"],
            "metrics": metrics,
            "quality": quality,
        }
        timing["splits"][split_name] = {
            **split_timing,
            "split_total_seconds": perf_counter() - split_start,
        }
        scored_frames.append(scored.with_columns(require_polars().lit(split_name).alias("split")))
    aggregate_metrics = aggregate_split_metrics(split_results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "requested_source": getattr(args, "requested_source", args.source),
        "dataset_hash": dataset["dataset_hash"],
        "source_file_hashes": source_file_hashes,
        "benchmark_integrity": {
            **benchmark_integrity(args),
            "candidate_selection": False,
        },
        "benchmark": "neural_panel_taxi_lane_split_suite",
        "fixture_source": args.source,
        "dataset": {
            **dataset,
            "split_families": [
                "rolling_origin",
                "cold_lane",
                "cold_origin",
                "sparse_tail",
            ],
            "split_type": "rolling_origin_cold_lane_cold_origin_sparse_tail",
        },
        "models": model_names,
        "model_roster": "neural-panel",
        "model_libraries": MODEL_LIBRARIES,
        "model_settings": cartoboost_model_settings(cartoboost_config),
        "splits": split_results,
        "metrics": aggregate_metrics,
        "quality": quality_summary(aggregate_metrics, model_names=model_names),
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            **timing,
        },
        "resource_usage": resource_usage_snapshot(),
        "artifact_paths": {
            "json": str(Path(args.output)),
        },
    }
    payload["comparability_audit"] = forecasting_comparability_audit(
        args=args,
        model_names=model_names,
        metrics=aggregate_metrics,
        split_results=split_results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"quality": payload["quality"], "artifact_paths": payload["artifact_paths"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_lsttn_h3_taxi_suite(
    args: argparse.Namespace,
    *,
    table: Any,
    dataset: dict[str, Any],
    source_file_hashes: dict[str, str],
    load_seconds: float,
    benchmark_start: float,
) -> int:
    """Train LSTTN on H3 demand nodes with the complete directed OD graph."""
    if args.source != "nyc-taxi" or {"pickup_h3", "dropoff_h3"}.difference(table.columns):
        raise ValueError("--lsttn-h3-splits requires --source nyc-taxi --h3-resolution")
    pl = require_polars()
    node_ids = (
        pl.concat(
            [
                table.select(pl.col("pickup_h3").alias("h3")),
                table.select(pl.col("dropoff_h3").alias("h3")),
            ]
        )
        .unique()
        .sort("h3")
        .get_column("h3")
        .to_list()
    )
    node_index = {node: index for index, node in enumerate(node_ids)}
    dates = table.select("date").unique().sort("date").get_column("date").to_list()
    horizon = int(dataset.get("horizon", args.horizon))
    if len(dates) < horizon + 3:
        raise ValueError("LSTTN H3 suite requires at least three history buckets plus holdout")
    # Polars does not guarantee group-by output order.  The CSR row order is
    # part of the native training fingerprint, so sort directed lanes before
    # assembling the graph to make checkpoints resumable across processes and
    # thread-pool sizes.
    edge_rows = (
        table.group_by("pickup_h3", "dropoff_h3")
        .agg(pl.col("loads").sum())
        .sort("pickup_h3", "dropoff_h3")
    )
    neighbors = [[] for _ in node_ids]
    weights = [[] for _ in node_ids]
    for pickup_h3, dropoff_h3, loads in edge_rows.iter_rows():
        source = node_index[pickup_h3]
        neighbors[source].append(node_index[dropoff_h3])
        weights[source].append(float(loads))
    indptr, indices, data = [0], [], []
    for row_indices, row_weights in zip(neighbors, weights, strict=True):
        total = sum(row_weights) or 1.0
        indices.extend(row_indices)
        data.extend(weight / total for weight in row_weights)
        indptr.append(len(indices))
    pickup_demand = table.group_by("date", "pickup_h3").agg(pl.col("loads").sum())
    demand_wide = pickup_demand.pivot(
        on="pickup_h3",
        index="date",
        values="loads",
        aggregate_function="sum",
    ).sort("date")
    missing_nodes = [node for node in node_ids if node not in demand_wide.columns]
    if missing_nodes:
        demand_wide = demand_wide.with_columns([pl.lit(0.0).alias(node) for node in missing_nodes])
    demand = demand_wide.select(node_ids).fill_null(0.0).to_numpy().astype(float, copy=False)
    history, actual = demand[:-horizon], demand[-horizon:]
    frame = GraphTemporalFrame(
        node_ids=node_ids,
        timestamps=list(range(len(history))),
        target=history,
        indptr=indptr,
        indices=indices,
        data=data,
        horizon=horizon,
        frequency=str(dataset.get("frequency", "monthly")),
    )
    is_daily = str(dataset.get("frequency", "daily")) == "daily"
    # The reference LSTTN uses a 14-day long context split into 12-step
    # subseries (336 patches).  With a daily taxi panel each observation is a
    # subseries, so retain the same 336-token long-history capacity rather
    # than shrinking the model into a four-week short-window forecaster.
    lookback = 336 if is_daily else 2
    periodicity = 1
    recent_window = 1
    if len(history) <= lookback + horizon:
        raise ValueError(
            "LSTTN H3 suite needs more history than lookback plus horizon; provide more "
            "months or use --taxi-frequency daily."
        )
    start = perf_counter()
    if args.lsttn_epochs <= 0 or args.lsttn_hidden_size <= 0:
        raise ValueError("--lsttn-epochs and --lsttn-hidden-size must be positive")
    if "metal" not in _native.graph_st_available_backends_value():
        raise RuntimeError(
            "LSTTN H3 training requires the native Metal build; run "
            "`uv run --group dev maturin develop --features metal` first."
        )
    model = LSTTNForecaster(
        lookback=lookback,
        periodicity=periodicity,
        recent_window=recent_window,
        horizon=horizon,
        epochs=args.lsttn_epochs,
        hidden_size=args.lsttn_hidden_size,
        attention_heads=1,
        backend=Backend.METAL,
    )
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    model.fit(frame, checkpoint_path=checkpoint_path)
    predicted = model.predict(horizon)
    elapsed = perf_counter() - start
    payload = {
        "benchmark": "nyc_taxi_lsttn_h3_full_od_graph",
        "dataset": {**dataset, "h3_nodes": len(node_ids), "directed_od_edges": len(indices)},
        "model": {
            "architecture": "lsttn",
            "epochs": args.lsttn_epochs,
            "hidden_size": args.lsttn_hidden_size,
            "lookback": lookback,
            "periodicity": periodicity,
            "recent_window": recent_window,
            "pretraining_backend": model.metadata_["backend"],
            "checkpoint": str(checkpoint_path),
        },
        "metrics": {"lsttn": market_metric_set(actual, predicted)},
        "timing": {
            "load_seconds": load_seconds,
            "fit_predict_seconds": elapsed,
            "total_seconds": perf_counter() - benchmark_start,
        },
        "source_file_hashes": source_file_hashes,
        "artifact_paths": {"json": str(Path(args.output))},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": payload["dataset"],
                "metrics": payload["metrics"],
                "timing": payload["timing"],
            },
            indent=2,
        )
    )
    return 0


def run_market_structure_taxi_suite(
    args: argparse.Namespace,
    *,
    table: Any,
    dataset: dict[str, Any],
    source_file_hashes: dict[str, str],
    load_seconds: float,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    """Evaluate generic structure learning on real daily taxi lane targets.

    The benchmark deliberately keeps taxi-specific aggregation here and sends
    only caller-named generic targets to the native model.
    """
    if args.source != "nyc-taxi":
        raise ValueError("--market-structure-splits requires --source nyc-taxi")
    pd = require_pandas_for_benchmark()
    required = {
        "lane_id",
        "date",
        "loads",
        "taxi_effective_fare_per_mile",
        "pickup_zone",
        "dropoff_zone",
        "origin_x",
        "origin_y",
        "destination_x",
        "destination_y",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"market structure taxi suite is missing required columns: {missing}")
    rows = table.to_pandas().sort_values(["date", "lane_id"])
    lane_ids = sorted(rows["lane_id"].unique())
    dates = sorted(pd.to_datetime(rows["date"]).unique())
    horizon = int(dataset.get("horizon", args.horizon))
    if len(dates) <= horizon:
        raise ValueError("market structure taxi suite requires a full holdout horizon")
    history_dates, holdout_dates = dates[:-horizon], dates[-horizon:]
    history = rows[rows["date"].isin(history_dates)]
    holdout = rows[rows["date"].isin(holdout_dates)]
    expected_history = len(history_dates) * len(lane_ids)
    expected_holdout = len(holdout_dates) * len(lane_ids)
    if len(history) != expected_history or len(holdout) != expected_holdout:
        raise ValueError("market structure taxi suite requires a complete lane-by-day panel")

    def matrix(frame: Any, column: str, ordered_dates: list[Any]) -> np.ndarray:
        return (
            frame.pivot(index="date", columns="lane_id", values=column)
            .reindex(index=ordered_dates, columns=lane_ids)
            .to_numpy(dtype=float)
        )

    primary_history = matrix(history, "taxi_effective_fare_per_mile", history_dates)
    secondary_history = matrix(history, "loads", history_dates)
    primary_actual = matrix(holdout, "taxi_effective_fare_per_mile", holdout_dates)
    secondary_actual = matrix(holdout, "loads", holdout_dates)
    primary_all = matrix(rows, "taxi_effective_fare_per_mile", dates)
    secondary_all = matrix(rows, "loads", dates)
    static = history.drop_duplicates("lane_id").set_index("lane_id").loc[lane_ids]
    coordinates = static[["origin_x", "origin_y", "destination_x", "destination_y"]].to_numpy(
        dtype=float
    )
    endpoint_columns = (
        ("pickup_h3", "dropoff_h3")
        if {"pickup_h3", "dropoff_h3"}.issubset(static.columns)
        else ("pickup_zone", "dropoff_zone")
    )
    origin_ids = static[endpoint_columns[0]].astype(str).tolist()
    destination_ids = static[endpoint_columns[1]].astype(str).tolist()
    if endpoint_columns == ("pickup_h3", "dropoff_h3"):
        import h3

        def parent_group(cell: str) -> str:
            resolution = h3.get_resolution(cell)
            return h3.cell_to_parent(cell, resolution - 1) if resolution else cell

        hierarchy_groups = [
            [
                f"origin_h3_parent:{parent_group(origin)}",
                f"destination_h3_parent:{parent_group(destination)}",
            ]
            for origin, destination in zip(origin_ids, destination_ids, strict=True)
        ]
    else:
        hierarchy_groups = [
            [f"pickup_borough:{int(code)}"]
            for code in static["pickup_borough_code"].to_numpy(dtype=float)
        ]
    frame = MarketPanelFrame(
        lane_ids=lane_ids,
        timestamps=list(range(len(history_dates))),
        target_names=["taxi_effective_fare_per_mile", "taxi_trip_count"],
        primary=primary_history,
        secondary=secondary_history,
        origin_ids=origin_ids,
        destination_ids=destination_ids,
        coordinates=coordinates,
        hierarchy_groups=hierarchy_groups,
        horizon=horizon,
        frequency=str(dataset.get("frequency", "daily")),
    )
    start = perf_counter()
    # Rolling interval calibration deliberately refits the complete graph at
    # several historical origins. Scale-only mode measures the single native
    # all-lane fit, so it must not silently multiply that memory footprint.
    model = MarketStructureForecaster(
        calibrate_intervals=not args.market_scale_only,
        neural_epochs=0 if args.market_scale_only else 20,
    ).fit(frame)
    forecast_rows = model.predict(horizon)
    model_seconds = perf_counter() - start
    predicted_primary = np.asarray([row["primary"] for row in forecast_rows], dtype=float).reshape(
        horizon, len(lane_ids)
    )
    predicted_primary_lower = np.asarray(
        [row["primary_lower"] for row in forecast_rows], dtype=float
    ).reshape(horizon, len(lane_ids))
    predicted_primary_upper = np.asarray(
        [row["primary_upper"] for row in forecast_rows], dtype=float
    ).reshape(horizon, len(lane_ids))
    predicted_secondary = np.asarray(
        [row["secondary"] for row in forecast_rows], dtype=float
    ).reshape(horizon, len(lane_ids))
    naive_primary = np.repeat(last_observed_panel_values(primary_history)[None, :], horizon, axis=0)
    naive_secondary = np.repeat(secondary_history[-1:, :], horizon, axis=0)
    metrics = {
        "market_structure": {
            "primary": market_metric_set(
                primary_actual,
                predicted_primary,
                lower=predicted_primary_lower,
                upper=predicted_primary_upper,
            ),
            "secondary": market_metric_set(secondary_actual, predicted_secondary),
        },
        "last_value": {
            "primary": market_metric_set(primary_actual, naive_primary),
            "secondary": market_metric_set(secondary_actual, naive_secondary),
        },
    }
    primary_baselines = None
    if not args.market_scale_only:
        distance_primary = inverse_distance_last_value(primary_history, coordinates, horizon)
        distance_secondary = inverse_distance_last_value(secondary_history, coordinates, horizon)
        indptr, indices, weights = distance_csr_adjacency(coordinates)
        if np.isfinite(primary_history).all():
            fixed_primary = fixed_graph_market_forecast(
                lane_ids, primary_history, indptr, indices, weights, horizon
            )
            fixed_primary_name = "fixed_graph_dcrnn"
        else:
            fixed_primary = fixed_graph_last_observed_forecast(
                primary_history, indptr, indices, weights, horizon
            )
            fixed_primary_name = "fixed_graph_last_observed"
        fixed_secondary = fixed_graph_market_forecast(
            lane_ids, secondary_history, indptr, indices, weights, horizon
        )
        baseline_names = ["cartoboost_lag", NEURAL_PANEL_BENCHMARK_MODEL]
        primary_baselines = (
            market_panel_baseline_forecasts(
                history,
                "taxi_effective_fare_per_mile",
                holdout_dates,
                horizon,
                cartoboost_config,
                baseline_names,
            )
            if np.isfinite(primary_history).all()
            else None
        )
        secondary_baselines = market_panel_baseline_forecasts(
            history,
            "loads",
            holdout_dates,
            horizon,
            cartoboost_config,
            baseline_names,
        )
        metrics.update(
            {
                "inverse_distance_last_value": {
                    "primary": market_metric_set(primary_actual, distance_primary),
                    "secondary": market_metric_set(secondary_actual, distance_secondary),
                },
                fixed_primary_name: {
                    "primary": market_metric_set(primary_actual, fixed_primary),
                    "secondary": market_metric_set(secondary_actual, fixed_secondary),
                },
                **{
                    name: {
                        **(
                            {"primary": market_metric_set(primary_actual, primary_baselines[name])}
                            if primary_baselines is not None
                            else {}
                        ),
                        "secondary": market_metric_set(secondary_actual, secondary_baselines[name]),
                    }
                    for name in baseline_names
                },
            }
        )
    explanations = model.nowcast()
    shifts: dict[str, int] = {}
    for row in explanations:
        shifts[row["shift"]] = shifts.get(row["shift"], 0) + 1
    if args.market_scale_only:
        controlled_mix_shock = None
        edge_stability = None
        rolling_origin = None
    else:
        controlled_mix_shock = evaluate_controlled_mix_shock(
            lane_ids=lane_ids,
            primary_history=primary_history,
            secondary_history=secondary_history,
            origin_ids=origin_ids,
            destination_ids=destination_ids,
            coordinates=coordinates,
            hierarchy_groups=hierarchy_groups,
            horizon=horizon,
        )
        edge_stability = evaluate_edge_stability(
            lane_ids=lane_ids,
            primary_history=primary_history,
            secondary_history=secondary_history,
            origin_ids=origin_ids,
            destination_ids=destination_ids,
            coordinates=coordinates,
            hierarchy_groups=hierarchy_groups,
            horizon=horizon,
        )
        rolling_origin = evaluate_market_rolling_origins(
            lane_ids=lane_ids,
            primary=primary_all,
            secondary=secondary_all,
            origin_ids=origin_ids,
            destination_ids=destination_ids,
            coordinates=coordinates,
            hierarchy_groups=hierarchy_groups,
            horizon=horizon,
            folds=args.rolling_origin_folds,
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "invocation": invocation_metadata(),
        "benchmark": "nyc_taxi_market_structure_daily_lane",
        "fixture_source": args.source,
        "dataset": {
            **dataset,
            "target_names": ["taxi_effective_fare_per_mile", "taxi_trip_count"],
            "train_days": len(history_dates),
            "holdout_days": len(holdout_dates),
            "split_type": "last_daily_horizon",
            "primary_observed_history_fraction": float(np.isfinite(primary_history).mean()),
            "primary_observed_holdout_fraction": float(np.isfinite(primary_actual).mean()),
        },
        "metrics": metrics,
        "timing": {
            "load_seconds": load_seconds,
            "fit_predict_seconds": model_seconds,
            "total_seconds": perf_counter() - benchmark_start,
        },
        "relationship_count": len(model.relationships()),
        "relationships": model.relationships(),
        "explanations": explanations,
        "shift_counts": shifts,
        "controlled_mix_shock": controlled_mix_shock,
        "edge_stability": edge_stability,
        "rolling_origin": rolling_origin,
        "comparability_note": (
            "Scale-only mode scores the learned native graph against last value. It does not "
            "claim a comparison against dense pairwise graph or repeated-refit baselines."
            if args.market_scale_only
            else (
                None
                if primary_baselines is not None
                else "Primary CartoBoostLagForecaster, LaneNeuralPanelForecaster, and DCRNN "
                "require a complete panel and were not evaluated. The sparse-panel fixed graph "
                "baseline uses only each lane's last observed state; no historical primary values "
                "were filled."
            )
        ),
        "source_file_hashes": source_file_hashes,
        "artifact_paths": {"json": str(Path(args.output))},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "artifact_paths": payload["artifact_paths"]}, indent=2))
    return 0


def evaluate_controlled_mix_shock(
    *,
    lane_ids: list[str],
    primary_history: np.ndarray,
    secondary_history: np.ndarray,
    origin_ids: list[str],
    destination_ids: list[str],
    coordinates: np.ndarray,
    hierarchy_groups: list[list[str]],
    horizon: int,
) -> dict[str, Any]:
    """Inject a documented local-composition analogue into the taxi panel.

    Taxi trips do not encode a provider mix, so this is intentionally a
    counterfactual intervention rather than a claim about a real taxi field.
    The shock affects one lane's final observed primary value and an aligned
    historical mix feature. A useful structure model should call that lane
    local-or-mix and avoid propagating market alerts to untouched lanes.
    """
    candidates = [
        lane
        for lane in range(primary_history.shape[1])
        if np.isfinite(primary_history[:, lane]).any()
    ]
    if not candidates:
        raise ValueError("controlled mix shock requires at least one observed primary lane")
    lane = candidates[0]
    observed_rows = np.flatnonzero(np.isfinite(primary_history[:, lane]))
    shock_time = int(observed_rows[-1])
    primary = primary_history.copy()
    primary[shock_time, lane] *= 3.0
    mix = np.zeros((primary.shape[0], primary.shape[1], 1), dtype=float)
    mix[shock_time, lane, 0] = 1.0
    model = MarketStructureForecaster().fit(
        MarketPanelFrame(
            lane_ids=lane_ids,
            timestamps=list(range(primary.shape[0])),
            target_names=["taxi_effective_fare_per_mile", "taxi_trip_count"],
            primary=primary,
            secondary=secondary_history,
            origin_ids=origin_ids,
            destination_ids=destination_ids,
            coordinates=coordinates,
            hierarchy_groups=hierarchy_groups,
            mix=mix,
            horizon=horizon,
            frequency="daily",
        )
    )
    explanations = model.nowcast()
    affected = next(row for row in explanations if row["lane_id"] == lane_ids[lane])
    other_market_alerts = sum(
        row["shift"] == "market" for row in explanations if row["lane_id"] != lane_ids[lane]
    )
    return {
        "kind": "synthetic_local_composition_shock",
        "lane_id": lane_ids[lane],
        "timestamp": shock_time,
        "multiplier": 3.0,
        "affected_shift": affected["shift"],
        "false_market_alerts_on_untouched_lanes": int(other_market_alerts),
        "passes": affected["shift"] == "local_or_mix" and other_market_alerts == 0,
    }


def evaluate_edge_stability(
    *,
    lane_ids: list[str],
    primary_history: np.ndarray,
    secondary_history: np.ndarray,
    origin_ids: list[str],
    destination_ids: list[str],
    coordinates: np.ndarray,
    hierarchy_groups: list[list[str]],
    horizon: int,
) -> dict[str, Any]:
    """Measure learned-edge overlap across train-only rolling cutoffs."""
    minimum = horizon + 1
    cutoffs = sorted(
        {
            primary_history.shape[0] - multiplier * horizon
            for multiplier in (3, 2, 1)
            if primary_history.shape[0] - multiplier * horizon >= minimum
        }
    )
    edge_sets: list[set[tuple[str, str]]] = []
    for cutoff in cutoffs:
        model = MarketStructureForecaster(calibrate_intervals=False).fit(
            MarketPanelFrame(
                lane_ids=lane_ids,
                timestamps=list(range(cutoff)),
                target_names=["taxi_effective_fare_per_mile", "taxi_trip_count"],
                primary=primary_history[:cutoff],
                secondary=secondary_history[:cutoff],
                origin_ids=origin_ids,
                destination_ids=destination_ids,
                coordinates=coordinates,
                hierarchy_groups=hierarchy_groups,
                horizon=horizon,
                frequency="daily",
            )
        )
        edge_sets.append(
            {(edge["source_lane_id"], edge["target_lane_id"]) for edge in model.relationships()}
        )
    overlaps = []
    for left, right in zip(edge_sets, edge_sets[1:], strict=False):
        union = left | right
        overlaps.append(float(len(left & right) / len(union)) if union else 1.0)
    return {
        "cutoffs": cutoffs,
        "edge_counts": [len(edges) for edges in edge_sets],
        "consecutive_jaccard": overlaps,
        "mean_consecutive_jaccard": float(np.mean(overlaps)) if overlaps else float("nan"),
    }


def evaluate_market_rolling_origins(
    *,
    lane_ids: list[str],
    primary: np.ndarray,
    secondary: np.ndarray,
    origin_ids: list[str],
    destination_ids: list[str],
    coordinates: np.ndarray,
    hierarchy_groups: list[list[str]],
    horizon: int,
    folds: int,
) -> dict[str, Any]:
    """Leakage-safe daily rolling origins for the native market forecaster."""
    folds = max(1, folds)
    minimum_train = horizon + 1
    cutoffs = [
        primary.shape[0] - horizon * offset
        for offset in range(folds, 0, -1)
        if primary.shape[0] - horizon * offset >= minimum_train
    ]
    if not cutoffs:
        raise ValueError("market rolling-origin evaluation requires more training history")
    rows = []
    for cutoff in cutoffs:
        model = MarketStructureForecaster().fit(
            MarketPanelFrame(
                lane_ids=lane_ids,
                timestamps=list(range(cutoff)),
                target_names=["taxi_effective_fare_per_mile", "taxi_trip_count"],
                primary=primary[:cutoff],
                secondary=secondary[:cutoff],
                origin_ids=origin_ids,
                destination_ids=destination_ids,
                coordinates=coordinates,
                hierarchy_groups=hierarchy_groups,
                horizon=horizon,
                frequency="daily",
            )
        )
        predictions = model.predict(horizon)
        predicted_primary = np.asarray(
            [row["primary"] for row in predictions], dtype=float
        ).reshape(horizon, len(lane_ids))
        predicted_secondary = np.asarray(
            [row["secondary"] for row in predictions], dtype=float
        ).reshape(horizon, len(lane_ids))
        rows.append(
            {
                "cutoff_day": cutoff,
                "primary": market_metric_set(primary[cutoff : cutoff + horizon], predicted_primary),
                "secondary": market_metric_set(
                    secondary[cutoff : cutoff + horizon], predicted_secondary
                ),
            }
        )
    return {
        "split_type": "rolling_origin_daily",
        "fold_count": len(rows),
        "folds": rows,
        "mean_primary_mae": float(np.mean([row["primary"]["mae"] for row in rows])),
        "mean_secondary_mae": float(np.mean([row["secondary"]["mae"] for row in rows])),
    }


def inverse_distance_last_value(
    history: np.ndarray, coordinates: np.ndarray, horizon: int
) -> np.ndarray:
    last = last_observed_panel_values(history)
    prediction = np.empty_like(last)
    for target in range(len(last)):
        deltas = coordinates - coordinates[target]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        mask = distances > 0.0
        if not np.any(mask):
            prediction[target] = last[target]
            continue
        weights = 1.0 / distances[mask]
        prediction[target] = float(np.dot(weights, last[mask]) / weights.sum())
    return np.repeat(prediction[None, :], horizon, axis=0)


def last_observed_panel_values(history: np.ndarray) -> np.ndarray:
    """Return the final recorded state per lane without imputing the panel."""
    values = np.empty(history.shape[1], dtype=float)
    missing: list[int] = []
    for lane in range(history.shape[1]):
        observed = np.flatnonzero(np.isfinite(history[:, lane]))
        if observed.size == 0:
            missing.append(lane)
        else:
            values[lane] = history[observed[-1], lane]
    if missing:
        raise ValueError(
            "last-observed baseline requires one recorded primary value per lane; "
            f"missing lane columns: {missing[:10]}"
        )
    return values


def market_panel_baseline_forecasts(
    history: Any,
    target_column: str,
    holdout_dates: list[Any],
    horizon: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
) -> dict[str, np.ndarray]:
    pl = require_polars()
    lane_ids = sorted(history["lane_id"].unique())
    train = pl.from_pandas(
        history[["lane_id", "date", target_column]].rename(columns={target_column: "loads"})
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    forecasts, _timing = forecast_model_roster(
        train,
        horizon,
        season_length=7,
        cartoboost_config=cartoboost_config,
        model_names=model_names,
        source="market",
        neural_panel_epochs=24,
    )
    results: dict[str, np.ndarray] = {}
    for name in model_names:
        pivot = (
            forecasts.select("series_id", "timestamp", name)
            .to_pandas()
            .pivot(index="timestamp", columns="series_id", values=name)
            .reindex(index=holdout_dates, columns=lane_ids)
        )
        values = pivot.to_numpy(dtype=float)
        if values.shape != (horizon, len(lane_ids)) or not np.isfinite(values).all():
            raise ValueError(f"{name} produced incomplete market-panel predictions")
        results[name] = values
    return results


def distance_csr_adjacency(
    coordinates: np.ndarray, *, max_neighbors: int = 8
) -> tuple[list[int], list[int], list[float]]:
    if max_neighbors <= 0:
        raise ValueError("fixed graph max_neighbors must be positive")
    indptr = [0]
    indices: list[int] = []
    weights: list[float] = []
    for source in range(len(coordinates)):
        deltas = coordinates - coordinates[source]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        candidates = sorted(
            (
                (float(distance), target)
                for target, distance in enumerate(distances)
                if source != target and distance > 0.0
            ),
            key=lambda pair: pair[0],
        )[:max_neighbors]
        for distance, target in candidates:
            indices.append(target)
            weights.append(1.0 / distance)
        indptr.append(len(indices))
    if not indices:
        raise ValueError("fixed graph baseline requires distinct lane coordinates")
    return indptr, indices, weights


def fixed_graph_market_forecast(
    lane_ids: list[str],
    values: np.ndarray,
    indptr: list[int],
    indices: list[int],
    weights: list[float],
    horizon: int,
) -> np.ndarray:
    frame = GraphTemporalFrame(
        node_ids=lane_ids,
        timestamps=list(range(values.shape[0])),
        target=values,
        indptr=indptr,
        indices=indices,
        data=weights,
        horizon=horizon,
        frequency="daily",
    )
    return (
        DCRNNForecaster(epochs=80, hidden_size=16, learning_rate=0.02).fit(frame).predict(horizon)
    )


def fixed_graph_last_observed_forecast(
    history: np.ndarray,
    indptr: list[int],
    indices: list[int],
    weights: list[float],
    horizon: int,
) -> np.ndarray:
    """Fixed-adjacency forecast for sparse panels using only observed states.

    This is intentionally a distinct baseline from DCRNN: it starts from the
    most recent recorded value of each lane and diffuses that state through a
    fixed graph. It neither interpolates nor treats missing daily values as
    zero.
    """
    state = last_observed_panel_values(history)
    forecast = np.empty((horizon, len(state)), dtype=float)
    for step in range(horizon):
        next_state = np.empty_like(state)
        for lane in range(len(state)):
            start, end = indptr[lane], indptr[lane + 1]
            neighbor_indices = indices[start:end]
            neighbor_weights = np.asarray(weights[start:end], dtype=float)
            if neighbor_weights.size == 0:
                next_state[lane] = state[lane]
            else:
                neighbor_value = float(
                    np.dot(neighbor_weights, state[neighbor_indices]) / neighbor_weights.sum()
                )
                next_state[lane] = 0.5 * state[lane] + 0.5 * neighbor_value
        forecast[step] = next_state
        state = next_state
    return forecast


def market_metric_set(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask):
        raise ValueError(
            "market metrics require at least one observed actual and finite prediction"
        )
    actual = actual[mask]
    predicted = predicted[mask]
    error = actual - predicted
    denominator = float(np.abs(actual).sum())
    result: dict[str, float] = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "wape": float(np.abs(error).sum() / denominator) if denominator > 0.0 else float("nan"),
        "observations": int(mask.sum()),
    }
    if lower is not None or upper is not None:
        if lower is None or upper is None or lower.shape != mask.shape or upper.shape != mask.shape:
            raise ValueError(
                "market interval metrics require lower and upper arrays matching actual"
            )
        interval_mask = mask & np.isfinite(lower) & np.isfinite(upper)
        if not np.any(interval_mask):
            raise ValueError("market interval metrics require finite interval bounds")
        result["interval_coverage"] = float(
            np.mean(
                (actual[interval_mask[mask]] >= lower[interval_mask])
                & (actual[interval_mask[mask]] <= upper[interval_mask])
            )
        )
        result["interval_mean_width"] = float(np.mean(upper[interval_mask] - lower[interval_mask]))
    return result


def neural_panel_split_frames(table: Any, *, horizon: int, folds: int) -> dict[str, Any]:
    pl = require_polars()
    cutoffs = rolling_origin_cutoffs(table, horizon=horizon, folds=folds)
    cutoff = cutoffs[-1]
    train, test, cutoff = train_test_split_for_cutoff(table, horizon=horizon, cutoff=cutoff)
    lanes = table.select(pl.col("lane_id").unique().sort()).to_series().to_list()
    cold_lane_count = max(1, len(lanes) // 5)
    cold_lanes = set(lanes[-cold_lane_count:])
    origins = sorted({split_lane_id(str(lane))[0] for lane in lanes})
    cold_origin_count = max(1, len(origins) // 5)
    cold_origins = set(origins[-cold_origin_count:])
    sparse_tail_lanes = set(lanes[-cold_lane_count:])
    sparse_train = sparsify_tail_history(
        train,
        sparse_tail_lanes=sparse_tail_lanes,
        min_history=max(horizon + 2, 8),
    )
    validation_timestamps = test.select(pl.col("date").unique().sort()).to_series().to_list()
    return {
        "rolling_origin": {
            "split_type": "rolling_origin",
            "cutoff": cutoff,
            "train": train,
            "test": test,
            "heldout_lanes": [],
            "heldout_origins": [],
            "sparse_tail_lanes": [],
            "fallback": "none",
        },
        "cold_lane": {
            "split_type": "cold_lane",
            "cutoff": cutoff,
            "train": train.filter(~pl.col("lane_id").is_in(cold_lanes)),
            "test": table.filter(
                pl.col("lane_id").is_in(cold_lanes) & pl.col("date").is_in(validation_timestamps)
            ),
            "heldout_lanes": sorted(cold_lanes),
            "heldout_origins": [],
            "sparse_tail_lanes": [],
            "fallback": "exact_pair_to_origin_to_destination_to_global_by_horizon",
        },
        "cold_origin": {
            "split_type": "cold_origin",
            "cutoff": cutoff,
            "train": train.filter(~lane_origin_expr().is_in(cold_origins)),
            "test": table.filter(
                lane_origin_expr().is_in(cold_origins) & pl.col("date").is_in(validation_timestamps)
            ),
            "heldout_lanes": [],
            "heldout_origins": sorted(cold_origins),
            "sparse_tail_lanes": [],
            "fallback": "exact_pair_to_origin_to_destination_to_global_by_horizon",
        },
        "sparse_tail": {
            "split_type": "sparse_tail",
            "cutoff": cutoff,
            "train": sparse_train,
            "test": test.filter(pl.col("lane_id").is_in(sparse_tail_lanes)),
            "heldout_lanes": [],
            "heldout_origins": [],
            "sparse_tail_lanes": sorted(sparse_tail_lanes),
            "fallback": "none",
        },
    }


def score_neural_panel_split(
    train: Any,
    test: Any,
    *,
    horizon: int,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    fallback: str,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any], Any]:
    pl = require_polars()
    if train.is_empty() or test.is_empty():
        raise ValueError("NeuralPanel split produced empty train or test data")
    actual = (
        test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    predictions, timing = forecast_model_roster(
        train,
        horizon,
        season_length=season_length,
        cartoboost_config=cartoboost_config,
        model_names=model_names,
        source="synthetic",
        known_future=test,
        skip_non_m_raw_auto_candidate=True,
    )
    if fallback != "none":
        predictions = expand_forecasts_with_lane_fallback(predictions, actual, model_names)
    scored = actual.join(predictions, on=["series_id", "timestamp", "horizon"], how="inner")
    if scored.height != actual.height:
        raise RuntimeError("NeuralPanel split forecast alignment dropped rows")
    metrics = {
        model: evaluate_metrics(scored, model, train, season_length=season_length)
        for model in model_names
    }
    quality = quality_summary(metrics, model_names=model_names)
    return metrics, quality, timing, scored


def expand_forecasts_with_lane_fallback(
    predictions: Any, actual: Any, model_names: list[str]
) -> Any:
    pl = require_polars()
    prediction_rows = predictions.iter_rows(named=True)
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    by_origin: dict[tuple[str, int], list[dict[str, float]]] = {}
    by_destination: dict[tuple[str, int], list[dict[str, float]]] = {}
    by_horizon: dict[int, list[dict[str, float]]] = {}
    for row in prediction_rows:
        series_id = str(row["series_id"])
        horizon = int(row["horizon"])
        origin, destination = split_lane_id(series_id)
        values = {model: float(row[model]) for model in model_names}
        lookup[(series_id, horizon)] = values
        by_origin.setdefault((origin, horizon), []).append(values)
        by_destination.setdefault((destination, horizon), []).append(values)
        by_horizon.setdefault(horizon, []).append(values)
    expanded = []
    for row in actual.iter_rows(named=True):
        series_id = str(row["series_id"])
        horizon = int(row["horizon"])
        origin, destination = split_lane_id(series_id)
        values = lookup.get((series_id, horizon))
        if values is None:
            candidates = (
                by_origin.get((origin, horizon))
                or by_destination.get((destination, horizon))
                or by_horizon.get(horizon)
            )
            if not candidates:
                raise RuntimeError(
                    f"no fallback forecast available for {series_id} horizon {horizon}"
                )
            values = {
                model: float(np.mean([candidate[model] for candidate in candidates]))
                for model in model_names
            }
        expanded.append(
            {
                "series_id": series_id,
                "timestamp": row["timestamp"],
                "horizon": horizon,
                **values,
            }
        )
    return pl.DataFrame(expanded).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


def sparsify_tail_history(table: Any, *, sparse_tail_lanes: set[Any], min_history: int) -> Any:
    pl = require_polars()
    sparse = table.filter(pl.col("lane_id").is_in(sparse_tail_lanes)).with_columns(
        pl.int_range(pl.len()).over("lane_id").alias("__row_nr"),
        pl.len().over("lane_id").alias("__row_count"),
    )
    sparse = sparse.filter(pl.col("__row_nr") >= pl.col("__row_count") - min_history).drop(
        "__row_nr",
        "__row_count",
    )
    dense = table.filter(~pl.col("lane_id").is_in(sparse_tail_lanes))
    return pl.concat([dense, sparse], how="vertical").sort(["lane_id", "date"])


def split_lane_id(series_id: str) -> tuple[str, str]:
    if "->" in series_id:
        origin, destination = series_id.split("->", 1)
        return origin, destination
    if ":" in series_id:
        origin, destination = series_id.split(":", 1)
        return origin, destination
    return series_id, series_id


def lane_origin_expr() -> Any:
    pl = require_polars()
    return (
        pl.when(pl.col("lane_id").str.contains("->"))
        .then(pl.col("lane_id").str.split("->").list.get(0))
        .otherwise(pl.col("lane_id").str.split(":").list.get(0))
    )


def aggregate_split_metrics(split_results: dict[str, Any]) -> dict[str, dict[str, float]]:
    first_split = next(iter(split_results.values()))
    model_names = list(first_split["metrics"])
    aggregate: dict[str, dict[str, float]] = {}
    metric_names = ["mae", "rmse", "mase", "wape", "smape", "bias"]
    for model in model_names:
        aggregate[model] = {
            metric: float(
                np.mean(
                    [
                        split["metrics"][model][metric]
                        for split in split_results.values()
                        if metric in split["metrics"][model]
                    ]
                )
            )
            for metric in metric_names
        }
    return aggregate


def combine_forecast_frames(frames: list[Any]) -> Any:
    normalized = [normalize_forecast_frame(frame) for frame in frames]
    combined = normalized[0]
    for frame in normalized[1:]:
        combined = combined.join(frame, on=["series_id", "timestamp", "horizon"], how="inner")
    return combined


def replace_forecast_column(base: Any, replacement: Any, column: str) -> Any:
    columns = [name for name in base.columns if name != column]
    return base.select(*columns).join(
        normalize_forecast_frame(replacement).select("series_id", "timestamp", "horizon", column),
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )


def normalize_forecast_frame(frame: Any) -> Any:
    pl = require_polars()
    return frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


def quality_summary(
    metrics: dict[str, dict[str, float]],
    *,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    if model_names is None:
        model_names = list(metrics)
    cartoboost_models = [name for name in model_names if MODEL_LIBRARIES.get(name) == "cartoboost"]
    library_models = [name for name in model_names if name not in cartoboost_models]
    best_rmse = min(row["rmse"] for row in metrics.values())
    tied_best_models = [
        name for name, row in metrics.items() if np.isclose(row["rmse"], best_rmse, rtol=1e-12)
    ]
    summary: dict[str, Any] = {
        "winner": "tie" if len(tied_best_models) > 1 else tied_best_models[0],
        "comparison_libraries": [
            library
            for library, names in FORECASTING_LIBRARY_MODELS.items()
            if any(name in model_names for name in names)
        ],
        "forecasting_library_models": {
            library: [name for name in names if name in model_names]
            for library, names in FORECASTING_LIBRARY_MODELS.items()
            if any(name in model_names for name in names)
        },
        "model_libraries": MODEL_LIBRARIES,
        "best_rmse": best_rmse,
        "tied_best_models": tied_best_models,
        "rmse_ranking": sorted(metrics, key=lambda name: metrics[name]["rmse"]),
        "mae_ranking": sorted(metrics, key=lambda name: metrics[name]["mae"]),
        "wape_ranking": sorted(metrics, key=lambda name: metrics[name]["wape"]),
    }
    if cartoboost_models:
        best_cartoboost_model = min(cartoboost_models, key=lambda name: metrics[name]["rmse"])
        summary.update(
            {
                "best_cartoboost_model": best_cartoboost_model,
                "cartoboost_rmse": metrics[best_cartoboost_model]["rmse"],
                "cartoboost_mae": metrics[best_cartoboost_model]["mae"],
                "cartoboost_wape": metrics[best_cartoboost_model]["wape"],
            }
        )
    else:
        summary.update(
            {
                "best_cartoboost_model": None,
                "cartoboost_rmse": None,
                "cartoboost_mae": None,
                "cartoboost_wape": None,
            }
        )
    if library_models:
        best_library_model = min(library_models, key=lambda name: metrics[name]["rmse"])
        library_rmse = metrics[best_library_model]["rmse"]
        summary.update(
            {
                "best_forecasting_library": MODEL_LIBRARIES[best_library_model],
                "best_forecasting_library_model": best_library_model,
                "best_forecasting_library_rmse": library_rmse,
                "best_forecasting_library_mae": metrics[best_library_model]["mae"],
                "best_forecasting_library_wape": metrics[best_library_model]["wape"],
            }
        )
        if cartoboost_models:
            cartoboost_rmse = metrics[best_cartoboost_model]["rmse"]
            summary.update(
                {
                    "rmse_delta_vs_best_forecasting_library": cartoboost_rmse - library_rmse,
                    "rmse_ratio_vs_best_forecasting_library": cartoboost_rmse / library_rmse,
                    "rmse_reduction_vs_best_forecasting_library": 1.0
                    - cartoboost_rmse / library_rmse,
                    "mae_delta_vs_best_forecasting_library": metrics[best_cartoboost_model]["mae"]
                    - metrics[best_library_model]["mae"],
                    "mae_reduction_vs_best_forecasting_library": 1.0
                    - metrics[best_cartoboost_model]["mae"] / metrics[best_library_model]["mae"],
                    "wape_reduction_vs_best_forecasting_library": 1.0
                    - metrics[best_cartoboost_model]["wape"] / metrics[best_library_model]["wape"],
                }
            )
        else:
            summary.update(
                {
                    "best_forecasting_library": MODEL_LIBRARIES[best_library_model],
                    "best_forecasting_library_model": best_library_model,
                    "best_forecasting_library_rmse": library_rmse,
                    "best_forecasting_library_mae": metrics[best_library_model]["mae"],
                    "best_forecasting_library_wape": metrics[best_library_model]["wape"],
                    "rmse_delta_vs_best_forecasting_library": None,
                    "rmse_ratio_vs_best_forecasting_library": None,
                    "rmse_reduction_vs_best_forecasting_library": None,
                    "mae_delta_vs_best_forecasting_library": None,
                    "mae_reduction_vs_best_forecasting_library": None,
                    "wape_reduction_vs_best_forecasting_library": None,
                }
            )
    for library, library_model_names in FORECASTING_LIBRARY_MODELS.items():
        available_models = [name for name in library_model_names if name in metrics]
        if not available_models:
            continue
        best_model = min(available_models, key=lambda name: metrics[name]["rmse"])
        summary[f"best_{library}_method"] = best_model
        summary[f"best_{library}_rmse"] = metrics[best_model]["rmse"]
    # The v0.3 lane-demand acceptance gate compares CartoBoost with the
    # strongest completed external learned baseline, rather than a seasonal
    # naive library baseline. Keep that comparison explicit in the artifact so
    # the gate cannot be inferred from a mixed roster or stale prose.
    external_models = [
        name
        for name in model_names
        if MODEL_LIBRARIES.get(name) == "external_trees" and name in metrics
    ]
    if cartoboost_models and external_models:
        best_external_model = min(external_models, key=lambda name: metrics[name]["rmse"])
        external_rmse = metrics[best_external_model]["rmse"]
        cartoboost_rmse = metrics[best_cartoboost_model]["rmse"]
        summary.update(
            {
                "best_external_baseline": best_external_model,
                "best_external_baseline_rmse": external_rmse,
                "best_external_baseline_mae": metrics[best_external_model]["mae"],
                "best_external_baseline_wape": metrics[best_external_model]["wape"],
                "rmse_delta_vs_best_external_baseline": cartoboost_rmse - external_rmse,
                "rmse_ratio_vs_best_external_baseline": cartoboost_rmse / external_rmse,
                "external_baseline_rmse_gate_limit": 1.05,
                "external_baseline_rmse_gate_passed": cartoboost_rmse <= external_rmse * 1.05,
            }
        )
    else:
        summary.update(
            {
                "best_external_baseline": None,
                "best_external_baseline_rmse": None,
                "best_external_baseline_mae": None,
                "best_external_baseline_wape": None,
                "rmse_delta_vs_best_external_baseline": None,
                "rmse_ratio_vs_best_external_baseline": None,
                "external_baseline_rmse_gate_limit": 1.05,
                "external_baseline_rmse_gate_passed": False,
            }
        )
    return summary


def benchmark_objective_artifacts(
    source: str,
    *,
    train_table: Any,
    scored: Any,
    model_names: list[str],
    season_length: int,
    cartoboost_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source in {"m1", "m3", "m4"}:
        label = source.upper()
        return {
            "primary_metric": "owa_proxy",
            "objective": f"{label}-style sMAPE/MASE/OWA proxy against seasonal naive",
            source: m_series_owa_artifact(
                train_table,
                scored,
                source=source,
                model_names=model_names,
                season_length=season_length,
            ),
        }
    if source == "m5":
        return {
            "primary_metric": "wrmsse",
            "objective": "M5 Forecasting Accuracy level-aware weighted RMSSE",
            "m5": m5_wrmsse_artifact(
                train_table,
                scored,
                model_names=model_names,
                seasonal_period=1,
            ),
        }
    if source == "m6":
        calibration_scored = m6_validation_scored_frame(
            train_table,
            scored,
            model_names=model_names,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
        )
        return {
            "primary_metric": "investment_decision_return",
            "objective": "M6-style deterministic decision return with RPS audit reporting",
            "m6": m6_rps_artifact(
                scored,
                model_names=model_names,
                calibration_scored=calibration_scored,
            ),
        }
    return {}


def aggregate_m_series_suite_official_metrics(
    source: str,
    group_order: list[str],
    results: dict[str, Any],
) -> dict[str, Any]:
    label = source.upper()
    group_artifacts = {
        group: results[group]["official_metrics"][source]
        for group in group_order
        if group in results and source in results[group].get("official_metrics", {})
    }
    if not group_artifacts:
        return {}
    model_names = sorted(
        {model for artifact in group_artifacts.values() for model in artifact.get("models", {})}
    )
    models: dict[str, Any] = {}
    expected_group_count = len(group_artifacts)
    for model in model_names:
        rows = [
            (group, artifact["models"][model])
            for group, artifact in group_artifacts.items()
            if model in artifact.get("models", {})
        ]
        if not rows:
            continue
        scored_groups = {group for group, _row in rows}
        missing_groups = [group for group in group_artifacts if group not in scored_groups]
        models[model] = {
            "group_count": len(rows),
            "complete_group_coverage": len(rows) == expected_group_count,
            "missing_groups": missing_groups,
            "mean_smape": float(np.mean([row["smape"] for _group, row in rows])),
            "mean_mase": float(np.mean([row["mase"] for _group, row in rows])),
            "mean_smape_ratio_to_seasonal_naive": float(
                np.mean([row["smape_ratio_to_seasonal_naive"] for _group, row in rows])
            ),
            "mean_mase_ratio_to_seasonal_naive": float(
                np.mean([row["mase_ratio_to_seasonal_naive"] for _group, row in rows])
            ),
            "mean_owa_proxy": float(np.mean([row["owa_proxy"] for _group, row in rows])),
            "group_owa_proxy": {group: float(row["owa_proxy"]) for group, row in rows},
        }
    complete_models = [name for name, row in models.items() if row["complete_group_coverage"]]
    incomplete_models = {
        name: row["missing_groups"]
        for name, row in models.items()
        if not row["complete_group_coverage"]
    }
    ranking = sorted(complete_models, key=lambda name: (models[name]["mean_owa_proxy"], name))
    incomplete_ranking = sorted(
        incomplete_models,
        key=lambda name: (-models[name]["group_count"], models[name]["mean_owa_proxy"], name),
    )
    return {
        "primary_metric": "mean_owa_proxy",
        "objective": f"{label}-style suite mean of per-group sMAPE/MASE/OWA proxy scores",
        source: {
            "group_order": list(group_order),
            "group_count": len(group_artifacts),
            "models": models,
            "ranking": ranking,
            "ranking_scope": "complete_group_coverage",
            "incomplete_models": incomplete_models,
            "incomplete_ranking": incomplete_ranking,
            "notes": [
                f"Suite score is the unweighted mean of per-group {label}-style OWA proxy scores.",
                "Primary suite ranking includes only models scored on every available group; "
                "partial-coverage models are listed separately.",
                "Each group OWA proxy is computed against the local seasonal-naive baseline "
                "on the benchmark holdout tail.",
            ],
        },
    }


def m_series_owa_artifact(
    table: Any,
    scored: Any,
    *,
    source: str,
    model_names: list[str],
    season_length: int,
) -> dict[str, Any]:
    pl = require_polars()
    cutoff = scored.select(pl.col("timestamp").min()).item()
    train = table.filter(pl.col("date") < cutoff)
    if train.is_empty():
        raise ValueError(f"{source.upper()} OWA proxy artifact requires pre-origin training rows")
    horizon = int(scored.select(pl.col("horizon").max()).item())
    naive = seasonal_naive_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="m4_seasonal_naive2",
    )
    scored_with_naive = scored.join(
        naive,
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )
    if scored_with_naive.height != scored.height:
        raise RuntimeError(f"{source.upper()} OWA proxy seasonal-naive alignment dropped rows")
    ordered_scored = scored_with_naive.sort(["series_id", "timestamp", "horizon"])
    training_series = [
        row["loads"]
        for row in train.sort(["lane_id", "date"])
        .group_by("lane_id", maintain_order=True)
        .agg(pl.col("loads"))
        .iter_rows(named=True)
    ]
    actual_values = ordered_scored["actual"].to_list()
    baseline = competition_forecast_metrics(
        training_series,
        actual_values,
        ordered_scored["m4_seasonal_naive2"].to_list(),
        seasonality=season_length,
    )
    baseline_smape = max(float(baseline["smape"]), 1.0e-12)
    baseline_mase = max(float(baseline["mase"]), 1.0e-12)
    models: dict[str, Any] = {}
    for model in model_names:
        metrics = competition_forecast_metrics(
            training_series,
            actual_values,
            ordered_scored[model].to_list(),
            seasonality=season_length,
            baseline_smape=baseline_smape,
            baseline_mase=baseline_mase,
        )
        smape_ratio = float(metrics["smape_ratio_to_baseline"])
        mase_ratio = float(metrics["mase_ratio_to_baseline"])
        models[model] = {
            "smape": float(metrics["smape"]),
            "mase": float(metrics["mase"]),
            "smape_ratio_to_seasonal_naive": smape_ratio,
            "mase_ratio_to_seasonal_naive": mase_ratio,
            "owa_proxy": float(metrics["owa"]),
        }
    ranking = sorted(model_names, key=lambda name: (models[name]["owa_proxy"], name))
    return {
        "season_length": int(season_length),
        "horizon": horizon,
        "baseline_model": "seasonal_naive2_proxy",
        "baseline": {
            "smape": float(baseline["smape"]),
            "mase": float(baseline["mase"]),
        },
        "models": models,
        "ranking": ranking,
        "notes": [
            f"This is a {source.upper()}-style local proxy over the benchmark holdout because "
            f"the harness scores withheld tails from the training panel, not the official "
            f"{source.upper()} test file.",
            "OWA proxy is 0.5 * (model sMAPE / seasonal-naive sMAPE + "
            "model MASE / seasonal-naive MASE).",
        ],
    }


def m5_wrmsse_artifact(
    table: Any,
    scored: Any,
    *,
    model_names: list[str],
    seasonal_period: int,
) -> dict[str, Any]:
    pl = require_polars()
    cutoff = scored.select(pl.col("timestamp").min()).item()
    train = table.filter(pl.col("date") < cutoff)
    if train.is_empty():
        raise ValueError("M5 WRMSSE artifact requires pre-origin training rows")

    metadata_columns = [*STATIC_COVARIATES, *available_m5_hierarchy_covariates(train)]
    metadata = train.select("lane_id", *metadata_columns).unique(subset=["lane_id"])
    actual = scored.select("series_id", "timestamp", "horizon", "actual").unique()
    levels = m5_wrmsse_level_specs(train)
    level_artifacts = {
        level_name: m5_wrmsse_level_artifact(
            train,
            actual,
            scored,
            metadata,
            level_name=level_name,
            group_column=group_column,
            model_names=model_names,
            seasonal_period=seasonal_period,
        )
        for level_name, group_column in levels
    }
    model_scores: dict[str, float | None] = {}
    model_level_contributions: dict[str, Any] = {}
    for model in model_names:
        available_scores = [
            (level_name, level["models"][model]["wrmsse"])
            for level_name, level in level_artifacts.items()
            if model in level["models"] and level["models"][model]["wrmsse"] is not None
        ]
        if available_scores:
            aggregate = aggregate_equal_level_wrmsse(available_scores, return_breakdown=True)
            model_scores[model] = float(aggregate["wrmsse"])
            model_level_contributions[model] = aggregate["levels"]
        else:
            model_scores[model] = None
            model_level_contributions[model] = []
    ranking = sorted(
        model_scores,
        key=lambda name: (
            math.inf if model_scores[name] is None else model_scores[name],
            name,
        ),
    )
    return {
        "seasonal_period": int(seasonal_period),
        "level_order": list(level_artifacts),
        "levels": level_artifacts,
        "model_scores": model_scores,
        "model_level_contributions": model_level_contributions,
        "ranking": ranking,
        "notes": [
            "Top-level M5 WRMSSE uses equal weight across available hierarchy levels; "
            "within each level, value weights are normalized across that level's series.",
            "Weights use recent dollar sales when sell_prices.csv is available in the M5 frame; "
            "older artifacts without weight_value used recent unit-sales volume.",
            "Flat zero-scale series are reported under skipped_zero_scale_series rather than "
            "assigned an artificial denominator.",
        ],
    }


def m5_wrmsse_level_artifact(
    train: Any,
    actual: Any,
    scored: Any,
    metadata: Any,
    *,
    level_name: str,
    group_column: str | list[str] | None,
    model_names: list[str],
    seasonal_period: int,
) -> dict[str, Any]:
    pl = require_polars()
    train_level = m5_level_frame(train, level_name=level_name, group_column=group_column)
    actual_level = m5_level_scored_frame(
        actual,
        metadata,
        level_name=level_name,
        group_column=group_column,
        value_column="actual",
    )
    ids = sorted(actual_level.select("level_id").unique().to_series().to_list())
    train_rows = {
        str(row["level_id"]): row["loads"]
        for row in train_level.sort(["level_id", "date"])
        .group_by("level_id", maintain_order=True)
        .agg(pl.col("loads"))
        .iter_rows(named=True)
    }
    actual_rows = {
        str(row["level_id"]): row["actual"]
        for row in actual_level.sort(["level_id", "horizon"])
        .group_by("level_id", maintain_order=True)
        .agg(pl.col("actual"))
        .iter_rows(named=True)
    }
    weights = m5_level_weights(train_level, ids)
    valid_ids: list[str] = []
    skipped: list[str] = []
    for level_id in ids:
        try:
            rmsse_scale(train_rows[str(level_id)], seasonal_period=seasonal_period)
        except ValueError:
            skipped.append(str(level_id))
        else:
            valid_ids.append(str(level_id))

    model_artifacts: dict[str, Any] = {}
    for model in model_names:
        pred_level = m5_level_scored_frame(
            scored.select("series_id", "timestamp", "horizon", pl.col(model).alias("forecast")),
            metadata,
            level_name=level_name,
            group_column=group_column,
            value_column="forecast",
        )
        pred_rows = {
            str(row["level_id"]): row["forecast"]
            for row in pred_level.sort(["level_id", "horizon"])
            .group_by("level_id", maintain_order=True)
            .agg(pl.col("forecast"))
            .iter_rows(named=True)
        }
        if valid_ids:
            result = wrmsse(
                [train_rows[level_id] for level_id in valid_ids],
                [actual_rows[level_id] for level_id in valid_ids],
                [pred_rows[level_id] for level_id in valid_ids],
                [weights[level_id] for level_id in valid_ids],
                seasonal_period=seasonal_period,
                series_ids=valid_ids,
                return_breakdown=True,
            )
            model_artifacts[model] = {
                "wrmsse": float(result["wrmsse"]),
                "series_count": len(valid_ids),
                "series": result["series"],
            }
        else:
            model_artifacts[model] = {
                "wrmsse": None,
                "series_count": 0,
                "series": [],
            }
    return {
        "series_count": len(ids),
        "scored_series_count": len(valid_ids),
        "skipped_zero_scale_series": skipped,
        "models": model_artifacts,
    }


def m5_wrmsse_level_specs(train: Any) -> list[tuple[str, list[str] | None]]:
    if all(column in train.columns for column in M5_HIERARCHY_COVARIATES):
        return [
            ("total", None),
            ("state", ["m5_state_code"]),
            ("store", ["m5_store_code"]),
            ("category", ["m5_cat_code"]),
            ("department", ["m5_dept_code"]),
            ("state_category", ["m5_state_code", "m5_cat_code"]),
            ("state_department", ["m5_state_code", "m5_dept_code"]),
            ("store_category", ["m5_store_code", "m5_cat_code"]),
            ("store_department", ["m5_store_code", "m5_dept_code"]),
            ("item", ["m5_item_code"]),
            ("state_item", ["m5_state_code", "m5_item_code"]),
            ("item_store", None),
        ]
    return [
        ("total", None),
        ("state", ["pickup_borough_code"]),
        ("store", ["pickup_zone"]),
        ("item", ["dropoff_zone"]),
        ("item_store", None),
    ]


def available_m5_hierarchy_covariates(frame: Any) -> list[str]:
    return [column for column in M5_HIERARCHY_COVARIATES if column in frame.columns]


def m5_level_id_expr(group_column: str | list[str]) -> Any:
    pl = require_polars()
    columns = [group_column] if isinstance(group_column, str) else list(group_column)
    if len(columns) == 1:
        return pl.col(columns[0]).cast(pl.Utf8).alias("level_id")
    return pl.concat_str([pl.col(column).cast(pl.Utf8) for column in columns], separator="/").alias(
        "level_id"
    )


def m5_level_frame(train: Any, *, level_name: str, group_column: str | list[str] | None) -> Any:
    pl = require_polars()
    value_columns = ["loads"]
    aggregations = [pl.col("loads").sum()]
    if "weight_value" in train.columns:
        value_columns.append("weight_value")
        aggregations.append(pl.col("weight_value").sum())
    if level_name == "total":
        return (
            train.with_columns(pl.lit("total").alias("level_id"))
            .group_by(["level_id", "date"], maintain_order=True)
            .agg(aggregations)
        )
    if level_name == "item_store":
        return train.with_columns(pl.col("lane_id").cast(pl.Utf8).alias("level_id")).select(
            "level_id", "date", *value_columns
        )
    if group_column is None:
        raise ValueError(f"M5 level {level_name!r} requires a group column")
    return (
        train.with_columns(m5_level_id_expr(group_column))
        .group_by(["level_id", "date"], maintain_order=True)
        .agg(aggregations)
    )


def m5_level_scored_frame(
    frame: Any,
    metadata: Any,
    *,
    level_name: str,
    group_column: str | list[str] | None,
    value_column: str,
) -> Any:
    pl = require_polars()
    if level_name == "total":
        return (
            frame.with_columns(pl.lit("total").alias("level_id"))
            .group_by(["level_id", "timestamp", "horizon"], maintain_order=True)
            .agg(pl.col(value_column).sum())
        )
    joined = frame.join(metadata, left_on="series_id", right_on="lane_id", how="left")
    if level_name == "item_store":
        return joined.with_columns(pl.col("series_id").cast(pl.Utf8).alias("level_id")).select(
            "level_id", "timestamp", "horizon", value_column
        )
    if group_column is None:
        raise ValueError(f"M5 level {level_name!r} requires a group column")
    return (
        joined.with_columns(m5_level_id_expr(group_column))
        .group_by(["level_id", "timestamp", "horizon"], maintain_order=True)
        .agg(pl.col(value_column).sum())
    )


def m5_level_weights(train_level: Any, ids: list[Any]) -> dict[str, float]:
    pl = require_polars()
    weight_column = "weight_value" if "weight_value" in train_level.columns else "loads"
    weight_rows = (
        train_level.sort(["level_id", "date"])
        .group_by("level_id", maintain_order=True)
        .agg(pl.col(weight_column).tail(28).sum().alias("weight"))
    )
    return {
        str(level_id): float(weight)
        for level_id, weight in _native.ordered_nonnegative_weights_value(
            [str(level_id) for level_id in ids],
            [
                (str(row["level_id"]), float(row["weight"]))
                for row in weight_rows.iter_rows(named=True)
            ],
        ).items()
    }


def m6_validation_scored_frame(
    table: Any,
    scored: Any,
    *,
    model_names: list[str],
    season_length: int,
    cartoboost_config: dict[str, Any] | None,
) -> Any | None:
    if cartoboost_config is None:
        return None
    pl = require_polars()
    cutoff = scored.select(pl.col("timestamp").min()).item()
    horizon = int(scored.select(pl.col("horizon").max()).item())
    pre_holdout = table.filter(pl.col("date") < cutoff)
    timestamps = pre_holdout.select(pl.col("date").unique().sort()).to_series().to_list()
    if len(timestamps) < horizon * 2:
        return None

    validation_timestamps = timestamps[-horizon:]
    validation_start = validation_timestamps[0]
    validation_train = pre_holdout.filter(pl.col("date") < validation_start)
    validation_test = pre_holdout.filter(pl.col("date").is_in(validation_timestamps))
    if validation_train.is_empty() or validation_test.is_empty():
        return None
    validation_raw, _timing = forecast_model_roster(
        validation_train,
        horizon,
        season_length=season_length,
        cartoboost_config=cartoboost_config,
        model_names=model_names,
        source="m6",
    )
    validation_predictions, _selection_timing = apply_shared_candidate_selection(
        validation_train,
        horizon,
        season_length=season_length,
        source="m6",
        raw_predictions=validation_raw,
        cartoboost_config=cartoboost_config,
        model_names=model_names,
    )

    actual = (
        validation_test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    validation_scored = actual.join(
        validation_predictions,
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )
    return None if validation_scored.is_empty() else validation_scored


def m6_rps_artifact(
    scored: Any,
    *,
    model_names: list[str],
    calibration_scored: Any | None = None,
) -> dict[str, Any]:
    summaries = {
        model: m6_model_rps_summary(
            scored,
            prediction_col=model,
            calibration_scored=calibration_scored,
        )
        for model in model_names
    }
    ranking = sorted(model_names, key=lambda name: (summaries[name]["mean_rps"], name))
    investment_ranking = sorted(
        model_names,
        key=lambda name: (-summaries[name]["decision_return"], name),
    )
    return {
        "rank_bucket_count": 5,
        "models": summaries,
        "ranking": investment_ranking,
        "primary_ranking": investment_ranking,
        "rps_ranking": ranking,
        "investment_ranking": investment_ranking,
        "notes": [
            "Rank probabilities are deterministic five-bucket distributions derived from "
            "predicted cumulative holdout returns.",
            "Decision rows are deterministic long/short selections for auditability; they "
            "are not an official M6 submission file.",
        ],
    }


def m6_model_rps_summary(
    scored: Any,
    *,
    prediction_col: str,
    calibration_scored: Any | None = None,
) -> dict[str, Any]:
    pl = require_polars()
    returns = (
        scored.group_by("series_id", maintain_order=True)
        .agg(
            pl.col("actual").sum().alias("actual_return"),
            pl.col(prediction_col).sum().alias("predicted_return"),
        )
        .sort("series_id")
    )
    calibration = m6_calibration_from_scored(
        calibration_scored,
        prediction_col=prediction_col,
        bucket_count=5,
    )
    summary = json.loads(
        _native.rank_portfolio_summary_value(
            [
                (
                    str(row["series_id"]),
                    float(row["actual_return"]),
                    float(row["predicted_return"]),
                )
                for row in returns.iter_rows(named=True)
            ],
            5,
            calibration["probabilities"],
            float(calibration["shrinkage"]),
        )
    )
    summary["rank_probability_calibration"] = calibration["metadata"]
    return summary


def m6_investment_decision_loss(
    scored: Any,
    *,
    prediction_col: str,
    rps_tiebreak_weight: float,
) -> float:
    pl = require_polars()
    returns = (
        scored.group_by("series_id", maintain_order=True)
        .agg(
            pl.col("actual").sum().alias("actual_return"),
            pl.col(prediction_col).sum().alias("predicted_return"),
        )
        .sort("series_id")
    )
    calibration = rank_probability_calibration(
        [],
        [],
        bucket_count=5,
        validation_support=0,
    )
    return float(
        _native.rank_portfolio_decision_loss_value(
            [
                (
                    str(row["series_id"]),
                    float(row["actual_return"]),
                    float(row["predicted_return"]),
                )
                for row in returns.iter_rows(named=True)
            ],
            5,
            calibration["probabilities"],
            float(calibration["shrinkage"]),
            float(rps_tiebreak_weight),
        )
    )


def m6_calibration_from_scored(
    calibration_scored: Any | None,
    *,
    prediction_col: str,
    bucket_count: int,
) -> dict[str, Any]:
    if calibration_scored is None or prediction_col not in calibration_scored.columns:
        return rank_probability_calibration(
            [],
            [],
            bucket_count=bucket_count,
            validation_support=0,
        )
    pl = require_polars()
    returns = (
        calibration_scored.group_by("series_id", maintain_order=True)
        .agg(
            pl.col("actual").sum().alias("actual_return"),
            pl.col(prediction_col).sum().alias("predicted_return"),
        )
        .sort("series_id")
    )
    if returns.is_empty():
        return rank_probability_calibration(
            [],
            [],
            bucket_count=bucket_count,
            validation_support=0,
        )
    actual_buckets = rank_buckets(returns["actual_return"].to_list(), bucket_count=bucket_count)
    predicted_buckets = rank_buckets(
        returns["predicted_return"].to_list(),
        bucket_count=bucket_count,
    )
    return rank_probability_calibration(
        actual_buckets,
        predicted_buckets,
        bucket_count=bucket_count,
        validation_support=len(actual_buckets),
    )


def rank_buckets(values: list[float], *, bucket_count: int) -> list[int]:
    return [
        int(bucket)
        for bucket in _native.rank_buckets_value(
            [float(value) for value in values],
            int(bucket_count),
        )
    ]


def rank_probability_calibration(
    actual_buckets: list[int],
    predicted_buckets: list[int],
    *,
    bucket_count: int,
    validation_support: int,
) -> dict[str, Any]:
    return native_rank_probability_calibration(
        actual_buckets,
        predicted_buckets,
        bucket_count=bucket_count,
        validation_support=validation_support,
    )


def m6_decision_rows(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "series_id": str(series_id),
            "side": str(side),
            "weight": float(weight),
            "actual_return": float(actual_return),
            "predicted_return": float(predicted_return),
        }
        for series_id, side, weight, actual_return, predicted_return in (
            _native.extreme_portfolio_decisions_value(
                [
                    (
                        str(row["series_id"]),
                        float(row["actual_return"]),
                        float(row["predicted_return"]),
                    )
                    for row in asset_rows
                ]
            )
        )
    ]


def portfolio_summary(decisions: list[dict[str, Any]]) -> dict[str, float | int]:
    return native_portfolio_summary(decisions)


def rank_hit_rates(asset_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return native_rank_hit_rates(asset_rows, bucket_count=5)


def forecast_model_roster(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    source: str = "synthetic",
    known_future: Any | None = None,
    skip_m5_raw_auto_candidate: bool = False,
    skip_m6_raw_auto_candidate: bool = False,
    skip_non_m_raw_auto_candidate: bool = False,
    neural_panel_epochs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    forecast_frames = []
    model_timing: dict[str, Any] = {}
    native_config = cartoboost_source_config(cartoboost_config, source=source)
    if SEASONAL_NAIVE_BENCHMARK_MODEL in model_names:
        seasonal_predictions = seasonal_naive_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col=SEASONAL_NAIVE_BENCHMARK_MODEL,
        )
        forecast_frames.append(seasonal_predictions)
        model_timing[SEASONAL_NAIVE_BENCHMARK_MODEL] = {
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "fit_predict_seconds": 0.0,
            "total_seconds": 0.0,
            "baseline": True,
        }
    if "cartoboost_lag" in model_names:
        cartoboost_predictions, cartoboost_timing = cartoboost_raw_forecast(
            train,
            horizon,
            season_length=season_length,
            config=native_config,
            prediction_col="cartoboost_lag",
            known_future=known_future,
        )
        forecast_frames.append(cartoboost_predictions)
        model_timing["cartoboost_lag"] = cartoboost_timing
    if any(model in model_names for model in INTERMITTENT_BENCHMARK_MODELS):
        intermittent_predictions, intermittent_timing = intermittent_forecasts(
            train,
            horizon,
            model_names=model_names,
        )
        forecast_frames.append(
            intermittent_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in INTERMITTENT_BENCHMARK_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in intermittent_timing.items() if model in model_names}
        )
    if "cartoboost_auto_forecast" in model_names:
        if source == "m5" and skip_m5_raw_auto_candidate:
            pl = require_polars()
            if forecast_frames:
                auto_predictions = forecast_frames[0].select("series_id", "timestamp", "horizon")
            else:
                auto_predictions = seasonal_naive_forecast_frame(
                    train,
                    horizon,
                    season_length=season_length,
                    prediction_col="__m5_outer_anchor",
                ).select("series_id", "timestamp", "horizon")
            auto_predictions = auto_predictions.with_columns(
                pl.lit(0.0).alias("cartoboost_auto_forecast")
            )
            auto_timing = {
                "selector_mode": "m5_outer_maps_auto_to_autostats_candidate",
                "fit_predict_seconds": 0.0,
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
                "total_seconds": 0.0,
            }
        elif source == "m6" and skip_m6_raw_auto_candidate:
            pl = require_polars()
            if forecast_frames:
                auto_predictions = forecast_frames[0].select("series_id", "timestamp", "horizon")
            else:
                auto_predictions = seasonal_naive_forecast_frame(
                    train,
                    horizon,
                    season_length=season_length,
                    prediction_col="__m6_outer_anchor",
                ).select("series_id", "timestamp", "horizon")
            auto_predictions = auto_predictions.with_columns(
                pl.lit(0.0).alias("cartoboost_auto_forecast")
            )
            auto_timing = {
                "selector_mode": "m6_outer_skips_raw_auto_candidate",
                "fit_predict_seconds": 0.0,
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
                "total_seconds": 0.0,
            }
        elif source not in {"m4", "m5", "m6"} and skip_non_m_raw_auto_candidate:
            pl = require_polars()
            if forecast_frames:
                auto_predictions = forecast_frames[0].select("series_id", "timestamp", "horizon")
            else:
                auto_predictions = seasonal_naive_forecast_frame(
                    train,
                    horizon,
                    season_length=season_length,
                    prediction_col="__non_m_outer_anchor",
                ).select("series_id", "timestamp", "horizon")
            auto_predictions = auto_predictions.with_columns(
                pl.lit(0.0).alias("cartoboost_auto_forecast")
            )
            auto_timing = {
                "selector_mode": "non_m_outer_lazy_raw_auto_candidate",
                "fit_predict_seconds": 0.0,
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
                "total_seconds": 0.0,
            }
        elif source == "m6":
            pl = require_polars()
            auto_config = cartoboost_auto_config(
                native_config,
                season_length=season_length,
                horizon=horizon,
            )
            auto_predictions, auto_timing = cartoboost_raw_forecast(
                train,
                horizon,
                season_length=season_length,
                config=auto_config,
                prediction_col="cartoboost_auto_forecast",
                known_future=known_future,
            )
            auto_predictions = auto_predictions.with_columns(
                pl.col("cartoboost_auto_forecast").alias("cartoboost_point_auto")
            )
            auto_timing = {
                **auto_timing,
                "selector_mode": "m6_raw_auto_outer_no_nested_calibration",
            }
        else:
            auto_predictions, auto_timing = cartoboost_forecast(
                train,
                horizon,
                season_length=season_length,
                config=native_config,
                prediction_col="cartoboost_auto_forecast",
                known_future=known_future,
            )
        if include_autostats_candidate(source=source, season_length=season_length, horizon=horizon):
            autostats_predictions, autostats_timing = cartoboost_autostats_forecast(
                train,
                horizon,
                season_length=season_length,
                prediction_col="cartoboost_autostats_bank",
                validation_objective=autostats_validation_objective(source),
            )
            auto_predictions = auto_predictions.join(
                autostats_predictions,
                on=["series_id", "timestamp", "horizon"],
                how="inner",
            )
            if source == "m5" and skip_m5_raw_auto_candidate:
                auto_predictions = auto_predictions.with_columns(
                    pl.col("cartoboost_autostats_bank").alias("cartoboost_auto_forecast")
                )
                auto_timing["selector_mode"] = "m5_outer_maps_auto_to_autostats_candidate"
            auto_timing["autostats_candidate"] = autostats_timing
        forecast_frames.append(auto_predictions)
        model_timing["cartoboost_auto_forecast"] = auto_timing
    if "cartoboost_piecewise_linear_seasonal" in model_names:
        piecewise_predictions, piecewise_timing = cartoboost_piecewise_linear_forecast(
            train,
            horizon,
            season_length=season_length,
        )
        forecast_frames.append(piecewise_predictions)
        model_timing["cartoboost_piecewise_linear_seasonal"] = piecewise_timing
    if NEURAL_PANEL_BENCHMARK_MODEL in model_names:
        neural_predictions, neural_timing = cartoboost_neural_panel_forecast(
            train,
            horizon,
            season_length=season_length,
            known_future=known_future,
            epochs=neural_panel_epochs or 80,
        )
        forecast_frames.append(neural_predictions)
        model_timing[NEURAL_PANEL_BENCHMARK_MODEL] = neural_timing
    if any(model in model_names for model in FUNCTIME_MODELS):
        functime_predictions, functime_timing = functime_forecasts(
            train,
            horizon,
            season_length=season_length,
            lightgbm_config=cartoboost_config,
        )
        forecast_frames.append(
            functime_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in FUNCTIME_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in functime_timing.items() if model in model_names}
        )
    if any(model in model_names for model in STATSFORECAST_MODELS):
        statsforecast_predictions, statsforecast_timing = statsforecast_forecasts(
            train,
            horizon,
            season_length=season_length,
        )
        forecast_frames.append(
            statsforecast_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in STATSFORECAST_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {
                model: timing
                for model, timing in statsforecast_timing.items()
                if model in model_names
            }
        )
    if any(model in model_names for model in PROPHET_MODELS):
        prophet_predictions, prophet_timing = prophet_forecasts(
            train,
            horizon,
            season_length=season_length,
        )
        forecast_frames.append(
            prophet_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in PROPHET_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in prophet_timing.items() if model in model_names}
        )
    if any(model in model_names for model in EXTERNAL_TREE_MODELS):
        tree_predictions, tree_timing = external_tree_lag_forecasts(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            known_future=known_future,
        )
        forecast_frames.append(
            tree_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in EXTERNAL_TREE_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in tree_timing.items() if model in model_names}
        )
    predictions = combine_forecast_frames(forecast_frames)
    timing = {"models": model_timing}
    return predictions, timing


def apply_shared_candidate_selection(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    source: str,
    raw_predictions: Any,
    model_timing: dict[str, Any] | None = None,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    validation_cache: dict[Any, dict[str, float]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    started = perf_counter()
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    if len(timestamps) <= horizon + 2:
        selected_predictions, selected = validation_unavailable_selected_predictions(
            raw_predictions,
            model_names=model_names,
            source=source,
        )
        return selected_predictions, {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 0.0,
            "selected_candidates": selected,
            "selection_error": "not enough training timestamps for inner validation",
        }

    selector_model_names = candidate_selection_model_names(model_names)
    if not selector_model_names:
        return raw_predictions, {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 0.0,
            "selected_candidates": {model: model for model in model_names},
            "selection_error": "no candidate-selectable models in roster",
        }

    inner_scores = shared_candidate_validation_scores(
        train,
        horizon,
        season_length=season_length,
        source=source,
        cartoboost_config=cartoboost_config,
        model_names=selector_model_names,
        validation_cache=validation_cache,
    )
    if not inner_scores:
        selected_predictions, selected = validation_unavailable_selected_predictions(
            raw_predictions,
            model_names=model_names,
            source=source,
        )
        return selected_predictions, {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 0.0,
            "selected_candidates": selected,
            "selection_error": "inner forecast roster failed or produced no aligned rows",
        }

    selected: dict[str, str] = {}
    inner_losses: dict[str, dict[str, float | None]] = {}
    candidate_losses_by_model: dict[str, dict[str, float]] = {}
    origin_consistency_guarded: dict[str, dict[str, Any]] = {}
    low_support_guarded: dict[str, dict[str, Any]] = {}
    raw_auto_override_guarded: dict[str, dict[str, Any]] = {}
    m6_raw_auto_guarded: dict[str, dict[str, Any]] = {}
    validation_cache_hits = count_validation_cache_hits(validation_cache)
    validation_cache_misses = count_validation_cache_misses(validation_cache)
    objective = auto_selection_objective(source)
    keep_native_raw_auto = native_auto_raw_candidate_is_confident(model_timing or {})
    inner_origin_count = max(len(losses) for losses in inner_scores.values())
    for model in model_names:
        if model != "cartoboost_auto_forecast":
            selected[model] = model
            if model in inner_scores:
                raw_loss = float(np.mean(inner_scores[model]))
                inner_losses[model] = {
                    "raw": raw_loss,
                    "selected": raw_loss,
                }
                candidate_losses_by_model[model] = {model: raw_loss}
            continue
        eligible_candidates = selectable_candidate_names(model, source=source)
        candidate_scores = {
            candidate: float(np.mean(losses))
            for candidate, losses in inner_scores.items()
            if candidate == model or candidate in eligible_candidates
        }
        base_loss = candidate_scores.get(model, math.inf)
        best_candidate = candidate_choice_for_source(
            candidate_scores,
            source=source,
            inner_origin_count=max(len(losses) for losses in inner_scores.values()),
        )
        consistency_guard = lag_origin_consistency_guard(
            best_candidate,
            source=native_selection_profile(source),
            inner_scores=inner_scores,
        )
        if consistency_guard is not None:
            origin_consistency_guarded[model] = consistency_guard
            best_candidate = "cartoboost_lag"
        lag_loss = candidate_scores.get("cartoboost_lag", math.inf)
        if (
            source not in {"m4", "m5", "m6"}
            and inner_origin_count < 2
            and best_candidate != "cartoboost_lag"
            and math.isfinite(lag_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and lag_loss > 0.0
        ):
            relative_gain = 1.0 - candidate_scores[best_candidate] / lag_loss
            if relative_gain < LOW_SUPPORT_AUTO_MIN_RELATIVE_GAIN:
                low_support_guarded[model] = {
                    "candidate": best_candidate,
                    "reason": "low_validation_support_requires_stronger_gain_vs_lag",
                    "inner_origin_count": inner_origin_count,
                    "relative_gain_vs_lag": relative_gain,
                    "required_relative_gain": LOW_SUPPORT_AUTO_MIN_RELATIVE_GAIN,
                }
                best_candidate = "cartoboost_lag"
        if (
            source not in {"m4", "m5", "m6"}
            and best_candidate == model
            and math.isfinite(lag_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and lag_loss > 0.0
        ):
            relative_gain = 1.0 - candidate_scores[best_candidate] / lag_loss
            if relative_gain < RAW_AUTO_OVERRIDE_MIN_RELATIVE_GAIN:
                raw_auto_override_guarded[model] = {
                    "candidate": best_candidate,
                    "reason": "raw_auto_requires_stronger_gain_vs_lag",
                    "relative_gain_vs_lag": relative_gain,
                    "required_relative_gain": RAW_AUTO_OVERRIDE_MIN_RELATIVE_GAIN,
                }
                best_candidate = "cartoboost_lag"
        if (
            best_candidate != "cartoboost_lag"
            and math.isfinite(lag_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and lag_loss > 0.0
            and 1.0 - candidate_scores[best_candidate] / lag_loss < AUTO_SELECTION_MIN_RELATIVE_GAIN
        ):
            best_candidate = "cartoboost_lag"
        if (
            requires_lag_spine(source=source, season_length=season_length, horizon=horizon)
            and math.isfinite(lag_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and lag_loss <= candidate_scores[best_candidate] * 1.25
        ):
            best_candidate = "cartoboost_lag"
        if keep_native_raw_auto and model in candidate_scores:
            best_candidate = model
            consistency_guard = lag_origin_consistency_guard(
                best_candidate,
                source=native_selection_profile(source),
                inner_scores=inner_scores,
            )
            if consistency_guard is not None:
                origin_consistency_guarded[model] = consistency_guard
                best_candidate = "cartoboost_lag"
            elif (
                source not in {"m4", "m5", "m6"}
                and inner_origin_count < 2
                and math.isfinite(lag_loss)
                and math.isfinite(candidate_scores[best_candidate])
                and lag_loss > 0.0
            ):
                relative_gain = 1.0 - candidate_scores[best_candidate] / lag_loss
                if relative_gain < LOW_SUPPORT_AUTO_MIN_RELATIVE_GAIN:
                    low_support_guarded[model] = {
                        "candidate": best_candidate,
                        "reason": "low_validation_support_requires_stronger_gain_vs_lag",
                        "inner_origin_count": inner_origin_count,
                        "relative_gain_vs_lag": relative_gain,
                        "required_relative_gain": LOW_SUPPORT_AUTO_MIN_RELATIVE_GAIN,
                    }
                    best_candidate = "cartoboost_lag"
        if (
            best_candidate != "cartoboost_lag"
            and math.isfinite(lag_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and lag_loss > 0.0
            and 1.0 - candidate_scores[best_candidate] / lag_loss < AUTO_SELECTION_MIN_RELATIVE_GAIN
        ):
            best_candidate = "cartoboost_lag"
        if (
            source == "m6"
            and best_candidate != model
            and math.isfinite(base_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and base_loss > 0.0
        ):
            relative_gain = 1.0 - candidate_scores[best_candidate] / base_loss
            if not _native.forecast_relative_loss_displacement_allowed_value(
                base_loss,
                candidate_scores[best_candidate],
                M6_RAW_AUTO_DISPLACEMENT_MIN_RELATIVE_GAIN,
            ):
                m6_raw_auto_guarded[model] = {
                    "candidate": best_candidate,
                    "reason": "m6_rank_selection_requires_material_rps_gain_vs_raw_auto",
                    "relative_gain_vs_raw_auto": relative_gain,
                    "required_relative_gain": M6_RAW_AUTO_DISPLACEMENT_MIN_RELATIVE_GAIN,
                }
                best_candidate = model
        best_loss = candidate_scores[best_candidate]
        if (
            best_candidate != model
            and source not in {"m5", "m6"}
            and best_candidate != "cartoboost_lag"
            and base_loss > 0.0
            and best_loss < base_loss
            and 1.0 - best_loss / base_loss < 0.01
        ):
            best_candidate = model
        selected[model] = best_candidate
        inner_losses[model] = {
            "raw": finite_loss_or_none(base_loss),
            "selected": finite_loss_or_none(candidate_scores[best_candidate]),
        }
        candidate_losses_by_model[model] = dict(sorted(candidate_scores.items()))

    outer_candidates = add_shared_candidate_columns(
        train,
        horizon,
        season_length=season_length,
        predictions=raw_predictions,
        source=source,
        cartoboost_config=cartoboost_config,
        required_columns=set(selected.values()),
    )
    magnitude_guarded: dict[str, dict[str, Any]] = {}
    for model, selected_candidate in list(selected.items()):
        if source != "m5" or model != "cartoboost_auto_forecast":
            continue
        if (
            selected_candidate == "cartoboost_lag"
            or selected_candidate not in outer_candidates.columns
        ):
            continue
        replacement = stable_hierarchy_candidate_choice(
            train,
            outer_candidates,
            candidate_scores_by_model=candidate_losses_by_model.get(model, {}),
            selected_candidate=selected_candidate,
            inner_origin_count=inner_origin_count,
        )
        if replacement != selected_candidate:
            magnitude_guarded[model] = {
                "candidate": selected_candidate,
                "replacement": replacement,
                "reason": "forecast_magnitude_exceeds_training_scale_guard",
            }
            selected[model] = replacement
    selected_columns = [
        pl.col(selected[model]).alias(model)
        if selected[model] in selectable_candidate_names(model, source=source)
        else pl.col(model)
        for model in model_names
    ]
    selected_predictions = outer_candidates.select(
        "series_id",
        "timestamp",
        "horizon",
        *selected_columns,
    )
    return selected_predictions, {
        "calibration_seconds": perf_counter() - started,
        "inner_origin_count": float(max(len(losses) for losses in inner_scores.values())),
        "objective": objective,
        "selected_candidates": selected,
        "native_auto_raw_keep": keep_native_raw_auto,
        "origin_consistency_guarded": origin_consistency_guarded,
        "low_support_guarded": low_support_guarded,
        "raw_auto_override_guarded": raw_auto_override_guarded,
        "m6_raw_auto_guarded": m6_raw_auto_guarded,
        "magnitude_guarded": magnitude_guarded,
        "inner_losses": inner_losses,
        "inner_candidate_losses": candidate_losses_by_model,
        "inner_rmse": inner_losses if objective == "rmse" else {},
        "validation_cache_hits": validation_cache_hits,
        "validation_cache_misses": validation_cache_misses,
    }


def validation_unavailable_selected_predictions(
    raw_predictions: Any,
    *,
    model_names: list[str],
    source: str,
) -> tuple[Any, dict[str, str]]:
    pl = require_polars()
    validation_profile = native_validation_profile(source) or "robust"
    available_candidates = [model for model in model_names if model in raw_predictions.columns]
    selected = {
        model: _native.forecast_validation_unavailable_candidate_choice_value(
            model,
            validation_profile,
            available_candidates,
        )
        for model in model_names
    }
    selected_columns = [
        pl.col(selected[model]).alias(model)
        if selected[model] in raw_predictions.columns
        else pl.col(model)
        for model in model_names
    ]
    return (
        raw_predictions.select("series_id", "timestamp", "horizon", *selected_columns),
        selected,
    )


def finite_loss_or_none(loss: float) -> float | None:
    return float(loss) if math.isfinite(loss) else None


def stable_hierarchy_candidate_choice(
    train: Any,
    outer_candidates: Any,
    *,
    candidate_scores_by_model: dict[str, float],
    selected_candidate: str,
    inner_origin_count: int,
) -> str:
    pl = require_polars()
    training_max_abs = float(train.select(pl.col("loads").abs().max()).item() or 0.0)
    forecast_max_abs_by_candidate: dict[str, float] = {}
    for candidate, loss in candidate_scores_by_model.items():
        if candidate not in outer_candidates.columns or not math.isfinite(loss):
            continue
        forecast_max_abs = float(
            outer_candidates.select(pl.col(candidate).abs().max()).item() or 0.0
        )
        forecast_max_abs_by_candidate[candidate] = forecast_max_abs
    return str(
        _native.forecast_stable_magnitude_candidate_choice_value(
            selected_candidate,
            {str(candidate): float(loss) for candidate, loss in candidate_scores_by_model.items()},
            forecast_max_abs_by_candidate,
            training_max_abs,
            inner_origin_count,
        )
    )


def increment_validation_cache_stat(cache: dict[Any, dict[str, float]], name: str) -> None:
    stats = cache.setdefault(VALIDATION_CACHE_STATS_KEY, {})
    stats[name] = stats.get(name, 0.0) + 1.0


def count_validation_cache_hits(cache: dict[Any, dict[str, float]] | None) -> float:
    if cache is None:
        return 0.0
    return float(cache.get(VALIDATION_CACHE_STATS_KEY, {}).get("hits", 0.0))


def count_validation_cache_misses(cache: dict[Any, dict[str, float]] | None) -> float:
    if cache is None:
        return 0.0
    return float(cache.get(VALIDATION_CACHE_STATS_KEY, {}).get("misses", 0.0))


def native_auto_raw_candidate_is_confident(model_timing: dict[str, Any]) -> bool:
    auto_timing = model_timing.get("cartoboost_auto_forecast", {})
    try:
        relative_gain = float(auto_timing.get("inner_raw_relative_rmse_gain", 0.0))
    except (TypeError, ValueError):
        relative_gain = None
    return bool(
        _native.forecast_native_auto_raw_candidate_is_confident_value(
            auto_timing.get("selected_candidate"),
            relative_gain,
        )
    )


def lag_origin_consistency_guard(
    candidate: str,
    *,
    source: str,
    inner_scores: dict[str, list[float]],
) -> dict[str, Any] | None:
    lag_scores = inner_scores.get("cartoboost_lag")
    candidate_scores = inner_scores.get(candidate)
    if not lag_scores or not candidate_scores:
        return None
    profile = normalized_selection_profile(source)
    guard = _native.forecast_lag_origin_consistency_guard_value(
        candidate,
        profile,
        [float(loss) for loss in lag_scores],
        [float(loss) for loss in candidate_scores],
    )
    return None if guard is None else json.loads(str(guard))


def normalized_selection_profile(source: str) -> str:
    if source in {
        "classical_competition",
        "classical_competition_full",
        "hierarchical_reconciliation",
        "rank_portfolio",
        "low_frequency_competition",
        "robust",
    }:
        return source
    return native_selection_profile(source)


def candidate_selection_model_names(model_names: list[str]) -> list[str]:
    if "cartoboost_auto_forecast" not in model_names:
        return []
    selector_models = ["cartoboost_auto_forecast"]
    if "cartoboost_lag" in model_names:
        selector_models.insert(0, "cartoboost_lag")
    return selector_models


def shared_candidate_validation_scores(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    source: str,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    validation_cache: dict[Any, dict[str, float]] | None = None,
) -> dict[str, list[float]]:
    pl = require_polars()
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    cutoffs = shared_candidate_validation_cutoffs(timestamps, horizon=horizon, source=source)
    objective = auto_selection_objective(source)
    scores: dict[str, list[float]] = {}
    candidate_names = sorted(
        {
            candidate
            for model in model_names
            for candidate in [model, *selectable_candidate_names(model, source=source)]
        }
    )
    disqualified_raw_auto = False
    for cutoff in cutoffs:
        inner_train = train.filter(pl.col("date") < cutoff)
        validation_timestamps = timestamps[
            timestamps.index(cutoff) : timestamps.index(cutoff) + horizon
        ]
        inner_test = train.filter(pl.col("date").is_in(validation_timestamps))
        if inner_train.is_empty() or inner_test.is_empty():
            continue
        origin_model_names = model_names
        if disqualified_raw_auto and source not in {"m4", "m5", "m6"}:
            origin_model_names = [
                model for model in model_names if model != "cartoboost_auto_forecast"
            ]
        cache_key = (source, cutoff, tuple(origin_model_names))
        if validation_cache is not None and cache_key in validation_cache:
            increment_validation_cache_stat(validation_cache, "hits")
            origin_losses = dict(validation_cache[cache_key])
        else:
            origin_losses = {}
            inner_raw, _inner_timing = candidate_selection_forecast_roster(
                inner_train,
                horizon,
                season_length=season_length,
                cartoboost_config=cartoboost_config,
                model_names=origin_model_names,
                source=source,
                known_future=known_future_covariate_frame(train),
            )
            actual = (
                inner_test.sort(["lane_id", "date"])
                .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
                .select(
                    pl.col("lane_id").alias("series_id"),
                    pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                    "horizon",
                    pl.col("loads").alias("actual"),
                )
            )
            inner_candidates = add_shared_candidate_columns(
                inner_train,
                horizon,
                season_length=season_length,
                predictions=inner_raw,
                source=source,
                cartoboost_config=cartoboost_config,
            )
            scored = actual.join(
                inner_candidates,
                on=["series_id", "timestamp", "horizon"],
                how="inner",
            )
            if scored.is_empty():
                continue
            for candidate in candidate_names:
                if candidate not in scored.columns:
                    continue
                loss = forecast_objective_loss(
                    objective,
                    train=inner_train,
                    scored=scored,
                    prediction_col=candidate,
                    season_length=season_length,
                )
                if math.isfinite(loss):
                    origin_losses[candidate] = loss
            if validation_cache is not None:
                validation_cache[cache_key] = dict(origin_losses)
                increment_validation_cache_stat(validation_cache, "misses")
        for candidate, loss in origin_losses.items():
            if math.isfinite(loss):
                scores.setdefault(candidate, []).append(loss)
        lag_loss = origin_losses.get("cartoboost_lag")
        raw_auto_loss = origin_losses.get("cartoboost_auto_forecast")
        if (
            source not in {"m4", "m5", "m6"}
            and lag_loss is not None
            and raw_auto_loss is not None
            and raw_auto_loss > lag_loss
        ):
            disqualified_raw_auto = True
        elif (
            disqualified_raw_auto
            and source not in {"m4", "m5", "m6"}
            and lag_loss is not None
            and "cartoboost_auto_forecast" in candidate_names
        ):
            scores.setdefault("cartoboost_auto_forecast", []).append(
                lag_loss * (1.0 + AUTO_SELECTION_MIN_RELATIVE_GAIN)
            )
        if non_m_lag_dominates_origin(source=source, origin_losses=origin_losses):
            break
    return scores


def non_m_lag_dominates_origin(*, source: str, origin_losses: dict[str, float]) -> bool:
    if source in {"m4", "m5", "m6"}:
        return False
    lag_loss = origin_losses.get("cartoboost_lag")
    if lag_loss is None or not math.isfinite(lag_loss) or lag_loss <= 0.0:
        return False
    competitors = [
        loss
        for candidate, loss in origin_losses.items()
        if candidate != "cartoboost_lag" and math.isfinite(loss) and loss >= 0.0
    ]
    if not competitors:
        return False
    next_best = min(competitors)
    return 1.0 - lag_loss / next_best >= NON_M_LAG_DOMINANCE_EARLY_STOP_RELATIVE_GAIN


def candidate_selection_forecast_roster(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    source: str,
    known_future: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    if source == "m5" and "cartoboost_auto_forecast" in model_names:
        return m5_candidate_selection_forecast_roster(
            train,
            horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            known_future=known_future,
        )
    if source == "m4" and "cartoboost_auto_forecast" in model_names:
        return m4_candidate_selection_forecast_roster(
            train,
            horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
        )
    if source not in {"m4", "m5", "m6"} and "cartoboost_auto_forecast" in model_names:
        kwargs: dict[str, Any] = {}
        if known_future is not None:
            kwargs["known_future"] = known_future
        return forecast_model_roster(
            train,
            horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=source,
            skip_non_m_raw_auto_candidate=True,
            **kwargs,
        )
    if source != "m6" or "cartoboost_auto_forecast" not in model_names:
        kwargs = {}
        if known_future is not None:
            kwargs["known_future"] = known_future
        return forecast_model_roster(
            train,
            horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=source,
            **kwargs,
        )

    frames = []
    timing: dict[str, Any] = {"models": {}}
    if "cartoboost_lag" in model_names:
        lag_predictions, lag_timing = cartoboost_raw_forecast(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            prediction_col="cartoboost_lag",
            known_future=known_future,
        )
        frames.append(lag_predictions)
        timing["models"]["cartoboost_lag"] = lag_timing
    if "cartoboost_lag" not in model_names:
        anchor = seasonal_naive_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="__m6_validation_anchor",
        )
        frames.append(anchor)
    auto_config = cartoboost_auto_config(
        cartoboost_config,
        season_length=season_length,
        horizon=horizon,
    )
    auto_predictions, auto_timing = cartoboost_raw_forecast(
        train,
        horizon,
        season_length=season_length,
        config=auto_config,
        prediction_col="cartoboost_auto_forecast",
        known_future=known_future,
    )
    auto_predictions = auto_predictions.with_columns(
        require_polars().col("cartoboost_auto_forecast").alias("cartoboost_point_auto")
    )
    frames.append(auto_predictions)
    timing["models"]["cartoboost_auto_forecast"] = {
        **auto_timing,
        "selector_mode": "m6_inner_validation_includes_raw_auto",
    }
    return combine_forecast_frames(frames), timing


def m4_candidate_selection_forecast_roster(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    known_future: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    frames = []
    timing: dict[str, Any] = {"models": {}}
    if "cartoboost_lag" in model_names:
        lag_predictions, lag_timing = cartoboost_raw_forecast(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            prediction_col="cartoboost_lag",
            known_future=known_future,
        )
        frames.append(lag_predictions)
        timing["models"]["cartoboost_lag"] = lag_timing
    if "cartoboost_lag" not in model_names:
        anchor = seasonal_naive_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="__m4_validation_anchor",
        )
        frames.append(anchor)
    timing["models"]["cartoboost_auto_forecast"] = {
        "selector_mode": "m4_inner_validation_skips_raw_auto",
        "fit_predict_seconds": 0.0,
        "total_seconds": 0.0,
    }
    if include_autostats_candidate(source="m4", season_length=season_length, horizon=horizon):
        autostats_predictions, autostats_timing = cartoboost_autostats_forecast(
            train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_autostats_bank",
            validation_objective=autostats_validation_objective("m4"),
        )
        frames.append(autostats_predictions)
        timing["models"]["cartoboost_autostats_bank"] = autostats_timing
    return combine_forecast_frames(frames), timing


def m5_candidate_selection_forecast_roster(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    known_future: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    frames = []
    timing: dict[str, Any] = {"models": {}}
    if "cartoboost_lag" in model_names:
        lag_predictions, lag_timing = cartoboost_raw_forecast(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            prediction_col="cartoboost_lag",
            known_future=known_future,
        )
        frames.append(lag_predictions)
        timing["models"]["cartoboost_lag"] = lag_timing
    timing["models"]["cartoboost_auto_forecast"] = {
        "selector_mode": "m5_inner_validation_skips_raw_auto",
        "fit_predict_seconds": 0.0,
        "total_seconds": 0.0,
    }
    autostats_predictions, autostats_timing = cartoboost_autostats_forecast(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_autostats_bank",
        validation_objective=autostats_validation_objective("m5"),
    )
    frames.append(autostats_predictions)
    if "cartoboost_auto_forecast" in model_names:
        frames.append(
            autostats_predictions.rename({"cartoboost_autostats_bank": "cartoboost_auto_forecast"})
        )
    timing["models"]["cartoboost_autostats_bank"] = autostats_timing
    return combine_forecast_frames(frames), timing


def shared_candidate_validation_cutoffs(
    timestamps: list[Any],
    *,
    horizon: int,
    source: str | None = None,
) -> list[Any]:
    cutoff_indices = _native.forecast_candidate_validation_cutoff_indices_value(
        len(timestamps),
        int(horizon),
        native_validation_profile(source),
    )
    return [timestamps[int(index)] for index in cutoff_indices]


def native_validation_profile(source: str | None) -> str | None:
    if source is None:
        return None
    if source == "m5":
        return "hierarchical_reconciliation"
    if source == "m6":
        return "rank_portfolio"
    if source in {"m1", "m3"}:
        return "classical_competition_full"
    if source == "m4":
        return "classical_competition"
    return "robust"


def selectable_candidate_names(model: str, *, source: str) -> list[str]:
    return [
        str(candidate)
        for candidate in _native.forecast_selectable_candidate_names_value(
            model,
            native_selection_profile(source),
        )
    ]


def robust_candidate_choice(candidate_scores: dict[str, float]) -> str:
    return native_candidate_choice(candidate_scores, source="synthetic")


def candidate_choice_for_source(
    candidate_scores: dict[str, float],
    *,
    source: str,
    inner_origin_count: int | None = None,
) -> str:
    return native_candidate_choice(
        candidate_scores,
        source=source,
        inner_origin_count=inner_origin_count,
    )


def native_candidate_choice(
    candidate_scores: dict[str, float],
    *,
    source: str,
    inner_origin_count: int | None = None,
) -> str:
    return str(
        _native.forecast_candidate_choice_value(
            native_selection_profile(source),
            {str(candidate): float(loss) for candidate, loss in candidate_scores.items()},
            inner_origin_count,
        )
    )


def candidate_complexity_rank(candidate: str) -> int:
    return int(_native.forecast_candidate_complexity_rank_value(candidate))


def include_autostats_candidate(*, source: str, season_length: int, horizon: int) -> bool:
    return bool(
        _native.forecast_include_autostats_candidate_value(
            native_selection_profile(source),
            int(season_length),
            int(horizon),
        )
    )


def m4_requires_lag_spine(*, season_length: int, horizon: int) -> bool:
    return requires_lag_spine(source="m4", season_length=season_length, horizon=horizon)


def requires_lag_spine(*, source: str, season_length: int, horizon: int) -> bool:
    return bool(
        _native.forecast_requires_lag_spine_value(
            native_lag_spine_profile(source),
            int(season_length),
            int(horizon),
        )
    )


def native_selection_profile(source: str) -> str:
    if source in {"m1", "m3"}:
        return "classical_competition_full"
    if source == "m4":
        return "classical_competition"
    if source == "m5":
        return "hierarchical_reconciliation"
    if source == "m6":
        return "rank_portfolio"
    return "robust"


def native_lag_spine_profile(source: str) -> str:
    return "low_frequency_competition" if source == "m4" else "robust"


def shared_candidate_names() -> list[str]:
    return [str(candidate) for candidate in _native.forecast_shared_candidate_names_value()]


def add_shared_candidate_columns(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    predictions: Any,
    source: str,
    cartoboost_config: dict[str, Any] | None = None,
    required_columns: set[str] | None = None,
) -> Any:
    pl = require_polars()
    required = set(required_columns) if required_columns is not None else None

    def needs(column: str) -> bool:
        return required is None or column in required

    def needs_any(columns: set[str]) -> bool:
        return required is None or bool(required.intersection(columns))

    base_dependencies = {
        "shared_elapsed_phase_rank_blend": {
            "shared_calendar_elapsed_phase",
            "shared_seasonal_base",
        },
        "shared_calendar_autostats_blend": {
            "shared_calendar_elapsed_phase",
            "cartoboost_autostats_bank",
        },
        "shared_elapsed_phase_total_reconciled_020": {
            "shared_calendar_elapsed_phase",
            "cartoboost_autostats_bank",
        },
        "shared_elapsed_phase_total_reconciled_035": {
            "shared_calendar_elapsed_phase",
            "cartoboost_autostats_bank",
        },
        "shared_elapsed_phase_total_reconciled_050": {
            "shared_calendar_elapsed_phase",
            "cartoboost_autostats_bank",
        },
        "shared_reconciled_autostats_blend": {
            "shared_calendar_elapsed_phase",
            "cartoboost_autostats_bank",
        },
        "shared_point_autostats_elapsed_phase_blend": {
            "shared_calendar_elapsed_phase",
            "cartoboost_autostats_bank",
        },
        "shared_total_reconciled_auto": {"cartoboost_auto_forecast"},
    }
    expanded_required = set(required) if required is not None else None
    if expanded_required is not None:
        changed = True
        while changed:
            changed = False
            for column, dependencies in base_dependencies.items():
                if column in expanded_required:
                    before = len(expanded_required)
                    expanded_required.update(dependencies)
                    changed = len(expanded_required) != before
        required = expanded_required

    shared_frames = []
    if needs("shared_seasonal_base"):
        shared_frames.append(
            seasonal_naive_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                prediction_col="shared_seasonal_base",
            )
        )
    if needs("shared_calendar_dom"):
        shared_frames.append(
            calendar_profile_forecast_frame(
                train,
                horizon,
                prediction_col="shared_calendar_dom",
                mode="day_of_month",
            )
        )
    if needs("shared_calendar_elapsed_phase"):
        shared_frames.append(
            calendar_profile_forecast_frame(
                train,
                horizon,
                prediction_col="shared_calendar_elapsed_phase",
                mode="elapsed_phase",
                elapsed_phase_period=CALENDAR_PROFILE_ELAPSED_PHASE_PERIOD,
            )
        )
    if needs("shared_drift"):
        shared_frames.append(
            trend_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                prediction_col="shared_drift",
                mode="drift",
            )
        )
    if needs("shared_half_drift"):
        shared_frames.append(
            trend_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                prediction_col="shared_half_drift",
                mode="half_drift",
            )
        )
    if needs("shared_seasonal_drift"):
        shared_frames.append(
            trend_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                prediction_col="shared_seasonal_drift",
                mode="seasonal_drift",
            )
        )
    if needs("shared_seasonal_cycle_drift_050"):
        shared_frames.append(
            trend_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                prediction_col="shared_seasonal_cycle_drift_050",
                mode="seasonal_cycle_drift_050",
            )
        )
    if needs("shared_seasonal_cycle_drift_075"):
        shared_frames.append(
            trend_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                prediction_col="shared_seasonal_cycle_drift_075",
                mode="seasonal_cycle_drift_075",
            )
        )
    combined = predictions
    for frame in shared_frames:
        combined = combined.join(frame, on=["series_id", "timestamp", "horizon"], how="inner")
    if source == "m6" and needs_any(
        {"shared_market_neutral_zero", "shared_elapsed_phase_rank_blend"}
    ):
        expressions = []
        if needs("shared_market_neutral_zero"):
            expressions.append(pl.lit(0.0).alias("shared_market_neutral_zero"))
        if needs("shared_elapsed_phase_rank_blend"):
            expressions.append(
                pl.Series(
                    "shared_elapsed_phase_rank_blend",
                    _native.forecast_weighted_blend_candidate_value(
                        combined["shared_calendar_elapsed_phase"].to_list(),
                        combined["shared_seasonal_base"].to_list(),
                        0.85,
                    ),
                )
            )
        combined = combined.with_columns(
            *expressions,
        )
    m5_columns = {
        "shared_calendar_autostats_blend",
        "shared_elapsed_phase_total_reconciled_020",
        "shared_elapsed_phase_total_reconciled_035",
        "shared_elapsed_phase_total_reconciled_050",
        "shared_reconciled_autostats_blend",
        "shared_point_autostats_elapsed_phase_blend",
        "shared_total_reconciled_auto",
    }
    if source == "m5" and "cartoboost_auto_forecast" in combined.columns and needs_any(m5_columns):
        if "cartoboost_autostats_bank" in combined.columns:
            calendar_blend_columns = {
                "shared_calendar_autostats_blend",
                "shared_elapsed_phase_total_reconciled_020",
                "shared_elapsed_phase_total_reconciled_035",
                "shared_elapsed_phase_total_reconciled_050",
                "shared_reconciled_autostats_blend",
                "shared_point_autostats_elapsed_phase_blend",
            }
            phase_reconciled_columns = {
                "shared_elapsed_phase_total_reconciled_020",
                "shared_elapsed_phase_total_reconciled_035",
                "shared_elapsed_phase_total_reconciled_050",
                "shared_reconciled_autostats_blend",
            }
            if needs_any(calendar_blend_columns):
                combined = combined.with_columns(
                    (
                        0.35 * pl.col("cartoboost_auto_forecast")
                        + 0.50 * pl.col("shared_calendar_elapsed_phase")
                        + 0.15 * pl.col("cartoboost_autostats_bank")
                    ).alias("shared_calendar_autostats_blend")
                )
            if needs_any(phase_reconciled_columns):
                combined = add_hierarchical_elapsed_phase_total_reconciled_candidates(
                    combined,
                    base_col="shared_calendar_autostats_blend",
                    target_col="shared_calendar_elapsed_phase",
                )
            if needs("shared_reconciled_autostats_blend"):
                combined = combined.with_columns(
                    (
                        0.90 * pl.col("shared_elapsed_phase_total_reconciled_050")
                        + 0.10 * pl.col("cartoboost_autostats_bank")
                    ).alias("shared_reconciled_autostats_blend")
                )
            if needs("shared_point_autostats_elapsed_phase_blend"):
                combined = combined.with_columns(
                    (
                        0.70 * pl.col("cartoboost_autostats_bank")
                        + 0.30 * pl.col("shared_calendar_elapsed_phase")
                    ).alias("shared_point_autostats_elapsed_phase_blend")
                )
        reconciled = m5_autostats_reconciled_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            predictions=combined,
            group_column=None,
            base_col="cartoboost_auto_forecast",
            prediction_col="shared_total_reconciled_auto",
        )
        combined = combined.join(
            reconciled,
            on=["series_id", "timestamp", "horizon"],
            how="inner",
        )
    return combined


def add_hierarchical_elapsed_phase_total_reconciled_candidates(
    frame: Any,
    *,
    base_col: str,
    target_col: str,
) -> Any:
    pl = require_polars()
    gammas = [(0.20, "020"), (0.35, "035"), (0.50, "050")]
    indexed = frame.with_row_index("__m5_reconcile_row")
    reconciliation_rows: list[dict[str, float | int]] = []
    grouped = (
        indexed.select("__m5_reconcile_row", "timestamp", "horizon", base_col, target_col)
        .sort(["timestamp", "horizon", "__m5_reconcile_row"])
        .group_by(["timestamp", "horizon"], maintain_order=True)
        .agg(
            pl.col("__m5_reconcile_row"),
            pl.col(base_col),
            pl.col(target_col).sum().alias("__m5_target_total"),
        )
    )
    for row in grouped.iter_rows(named=True):
        row_indices = [int(value) for value in row["__m5_reconcile_row"]]
        base_values = [float(value) for value in row[base_col]]
        target_total = float(row["__m5_target_total"])
        reconciled_by_suffix = {
            suffix: _native.forecast_proportional_total_reconciliation_value(
                base_values,
                target_total,
                gamma,
            )
            for gamma, suffix in gammas
        }
        for offset, row_index in enumerate(row_indices):
            reconciliation_rows.append(
                {
                    "__m5_reconcile_row": row_index,
                    **{
                        f"shared_elapsed_phase_total_reconciled_{suffix}": float(values[offset])
                        for suffix, values in reconciled_by_suffix.items()
                    },
                }
            )
    reconciled = pl.DataFrame(reconciliation_rows)
    return indexed.join(reconciled, on="__m5_reconcile_row", how="inner").drop("__m5_reconcile_row")


def m5_autostats_reconciled_forecast_frame(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    predictions: Any,
    group_column: str | None,
    base_col: str,
    prediction_col: str,
) -> Any:
    pl = require_polars()
    group_key = "__m5_reconcile_group"
    if group_column is None:
        grouped_train = train.with_columns(pl.lit("total").alias(group_key))
    else:
        grouped_train = train.with_columns(pl.col(group_column).cast(pl.Utf8).alias(group_key))
    aggregate_train = (
        grouped_train.group_by([group_key, "date"], maintain_order=True)
        .agg(pl.col("loads").sum())
        .rename({group_key: "lane_id"})
        .sort(["lane_id", "date"])
    )
    aggregate_target, _timing = cartoboost_autostats_forecast(
        aggregate_train,
        horizon,
        season_length=season_length,
        prediction_col="__m5_reconcile_target",
        validation_objective=autostats_validation_objective("m5"),
    )
    aggregate_target = aggregate_target.with_columns(
        pl.col("series_id").cast(pl.Utf8).alias(group_key),
        pl.max_horizontal(pl.col("__m5_reconcile_target"), pl.lit(0.0)).alias(
            "__m5_reconcile_target"
        ),
    ).select(group_key, "timestamp", "horizon", "__m5_reconcile_target")

    metadata = train.select("lane_id", *STATIC_COVARIATES).unique(subset=["lane_id"])
    base = predictions.select("series_id", "timestamp", "horizon", base_col)
    if group_column is None:
        bottom = base.with_columns(pl.lit("total").alias(group_key))
    else:
        bottom = (
            base.join(metadata, left_on="series_id", right_on="lane_id", how="left")
            .with_columns(pl.col(group_column).cast(pl.Utf8).alias(group_key))
            .select("series_id", "timestamp", "horizon", base_col, group_key)
        )
    group_sums = bottom.group_by([group_key, "timestamp", "horizon"], maintain_order=True).agg(
        pl.col(base_col).sum().alias("__m5_reconcile_base_sum")
    )
    return (
        bottom.join(group_sums, on=[group_key, "timestamp", "horizon"], how="left")
        .join(aggregate_target, on=[group_key, "timestamp", "horizon"], how="left")
        .with_columns(
            pl.when(
                pl.col("__m5_reconcile_target").is_not_null()
                & (pl.col("__m5_reconcile_base_sum").abs() > 1.0e-12)
            )
            .then(pl.col("__m5_reconcile_target") / pl.col("__m5_reconcile_base_sum"))
            .otherwise(1.0)
            .alias("__m5_reconcile_scale")
        )
        .with_columns(
            pl.max_horizontal(
                pl.col(base_col) * pl.col("__m5_reconcile_scale"),
                pl.lit(0.0),
            ).alias(prediction_col)
        )
        .select("series_id", "timestamp", "horizon", prediction_col)
    )


def cartoboost_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
    prediction_col: str = "cartoboost_auto_forecast",
    known_future: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    auto_config = cartoboost_auto_config(config, season_length=season_length, horizon=horizon)
    raw_forecast, timing = cartoboost_raw_forecast(
        train,
        horizon,
        season_length=season_length,
        config=auto_config,
        prediction_col="cartoboost_raw",
        known_future=known_future,
    )
    seasonal_base = seasonal_naive_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_base",
    )
    (
        selected_candidate,
        residual_alpha,
        ensemble_weights,
        calibration_timing,
    ) = calibrate_cartoboost_candidate(
        train,
        horizon,
        season_length=season_length,
        config=auto_config,
    )
    calendar_dom = calendar_profile_forecast_frame(
        train,
        horizon,
        prediction_col="cartoboost_calendar_dom",
        mode="day_of_month",
    )
    calendar_elapsed_phase = calendar_profile_forecast_frame(
        train,
        horizon,
        prediction_col="cartoboost_calendar_elapsed_phase",
        mode="elapsed_phase",
        elapsed_phase_period=CALENDAR_PROFILE_ELAPSED_PHASE_PERIOD,
    )
    drift = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_drift",
        mode="drift",
    )
    half_drift = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_half_drift",
        mode="half_drift",
    )
    seasonal_drift = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_drift",
        mode="seasonal_drift",
    )
    seasonal_cycle_drift_050 = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_cycle_drift_050",
        mode="seasonal_cycle_drift_050",
    )
    seasonal_cycle_drift_075 = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_cycle_drift_075",
        mode="seasonal_cycle_drift_075",
    )
    candidates = (
        raw_forecast.join(seasonal_base, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_dom, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_elapsed_phase, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(half_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_050, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_075, on=["series_id", "timestamp", "horizon"], how="inner")
    )
    candidates = candidates.with_columns(
        (
            pl.col("cartoboost_seasonal_base")
            + residual_alpha * (pl.col("cartoboost_raw") - pl.col("cartoboost_seasonal_base"))
        ).alias("cartoboost_residual_blend"),
        (0.5 * pl.col("cartoboost_seasonal_base") + 0.5 * pl.col("cartoboost_calendar_dom")).alias(
            "cartoboost_calendar_dom_blend"
        ),
        (
            0.5 * pl.col("cartoboost_seasonal_base")
            + 0.5 * pl.col("cartoboost_calendar_elapsed_phase")
        ).alias("cartoboost_calendar_elapsed_phase_blend"),
    ).with_columns(weighted_candidate_expr(ensemble_weights).alias(AUTO_ENSEMBLE_CANDIDATE))
    blended = candidates.select(
        "series_id",
        "timestamp",
        "horizon",
        pl.col(selected_candidate).alias(prediction_col),
    )
    timing = {
        **timing,
        **calibration_timing,
        "selected_candidate": selected_candidate,
        "residual_alpha": residual_alpha,
        "ensemble_weights": ensemble_weights,
        "auto_config": auto_config,
        "total_seconds": timing["total_seconds"] + calibration_timing["calibration_seconds"],
    }
    timing["fit_predict_seconds"] = timing["fit_seconds"] + timing["predict_seconds"]
    return blended, timing


def cartoboost_auto_config(
    config: dict[str, Any],
    *,
    season_length: int,
    horizon: int,
) -> dict[str, Any]:
    auto = dict(config)
    if config.get("auto_n_estimators") is None:
        auto["n_estimators"] = max(int(config["n_estimators"]), 360)
    else:
        auto["n_estimators"] = int(config["auto_n_estimators"])
    auto["max_depth"] = max(int(config["max_depth"]), 5)
    auto["min_samples_leaf"] = min(int(config["min_samples_leaf"]), 6)
    if horizon >= 24 or season_length in {4, 7, 12}:
        auto["max_depth"] = max(auto["max_depth"], 6)
        auto["min_samples_leaf"] = min(auto["min_samples_leaf"], 4)
    return auto


def cartoboost_source_config(config: dict[str, Any], *, source: str) -> dict[str, Any]:
    native = dict(config)
    use_route_context = source in {"synthetic", "polars", "duckdb", "nyc-taxi"}
    use_known_future_context = source == "m5"
    use_rolling_stat_context = use_route_context or source == "m6"
    use_elapsed_calendar_context = source == "m3"
    native["use_static_covariates"] = use_route_context or use_known_future_context
    native["use_known_future_covariates"] = use_known_future_context
    native["use_elapsed_calendar_features"] = use_elapsed_calendar_context
    native["use_rich_calendar_features"] = use_route_context
    native["use_native_rolling_stat_features"] = use_rolling_stat_context
    native["use_native_partial_rolling_mean_features"] = source == "nyc-taxi"
    native["use_native_ewm_features"] = False
    native["use_covariate_calendar_interactions"] = use_route_context
    return native


def cartoboost_elapsed_calendar_periods(
    season_length: int,
    config: dict[str, Any],
) -> list[int]:
    if not config.get("use_elapsed_calendar_features", False):
        return []
    if season_length < 2:
        return []
    return [int(season_length)]


def calibrate_cartoboost_candidate(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
) -> tuple[str, float, dict[str, float], dict[str, Any]]:
    pl = require_polars()
    started = perf_counter()
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    if len(timestamps) <= max(horizon + 14, 60):
        raise ValueError(
            "cartoboost candidate calibration requires more than "
            f"{max(horizon + 14, 60)} timestamps; got {len(timestamps)}"
        )
    cutoff = timestamps[-horizon]
    inner_train = train.filter(pl.col("date") < cutoff)
    inner_test = train.filter(pl.col("date") >= cutoff)
    raw, _timing = cartoboost_raw_forecast(
        inner_train,
        horizon,
        season_length=season_length,
        config=config,
        prediction_col="cartoboost_raw",
    )
    base = seasonal_naive_forecast_frame(
        inner_train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_base",
    )
    calendar_dom = calendar_profile_forecast_frame(
        inner_train,
        horizon,
        prediction_col="cartoboost_calendar_dom",
        mode="day_of_month",
    )
    calendar_elapsed_phase = calendar_profile_forecast_frame(
        inner_train,
        horizon,
        prediction_col="cartoboost_calendar_elapsed_phase",
        mode="elapsed_phase",
        elapsed_phase_period=CALENDAR_PROFILE_ELAPSED_PHASE_PERIOD,
    )
    drift = trend_forecast_frame(
        inner_train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_drift",
        mode="drift",
    )
    half_drift = trend_forecast_frame(
        inner_train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_half_drift",
        mode="half_drift",
    )
    seasonal_drift = trend_forecast_frame(
        inner_train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_drift",
        mode="seasonal_drift",
    )
    seasonal_cycle_drift_050 = trend_forecast_frame(
        inner_train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_cycle_drift_050",
        mode="seasonal_cycle_drift_050",
    )
    seasonal_cycle_drift_075 = trend_forecast_frame(
        inner_train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_cycle_drift_075",
        mode="seasonal_cycle_drift_075",
    )
    actual = (
        inner_test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    scored = (
        actual.join(raw, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(base, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_dom, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_elapsed_phase, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(half_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_050, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_075, on=["series_id", "timestamp", "horizon"], how="inner")
    )
    select_columns = [
        "actual",
        "cartoboost_raw",
        "cartoboost_seasonal_base",
        "cartoboost_calendar_dom",
        "cartoboost_calendar_elapsed_phase",
        "cartoboost_calendar_dom_blend",
        "cartoboost_calendar_elapsed_phase_blend",
        "cartoboost_drift",
        "cartoboost_half_drift",
        "cartoboost_seasonal_drift",
        "cartoboost_seasonal_cycle_drift_050",
        "cartoboost_seasonal_cycle_drift_075",
    ]
    scored = scored.with_columns(
        (0.5 * pl.col("cartoboost_seasonal_base") + 0.5 * pl.col("cartoboost_calendar_dom")).alias(
            "cartoboost_calendar_dom_blend"
        ),
        (
            0.5 * pl.col("cartoboost_seasonal_base")
            + 0.5 * pl.col("cartoboost_calendar_elapsed_phase")
        ).alias("cartoboost_calendar_elapsed_phase_blend"),
    ).select(*select_columns)
    if scored.is_empty():
        raise ValueError("cartoboost candidate calibration produced no aligned scored rows")
    base_rmse = rmse_expr(scored, "cartoboost_seasonal_base")
    raw_rmse = rmse_expr(scored, "cartoboost_raw")
    best_alpha = 1.0
    best_rmse = raw_rmse
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        candidate = scored.with_columns(
            (
                pl.col("cartoboost_seasonal_base")
                + alpha * (pl.col("cartoboost_raw") - pl.col("cartoboost_seasonal_base"))
            ).alias("candidate")
        )
        candidate_rmse = rmse_expr(candidate, "candidate")
        if candidate_rmse < best_rmse:
            best_rmse = candidate_rmse
            best_alpha = alpha
    raw_gain = 1.0 - raw_rmse / base_rmse if base_rmse > 0.0 else 0.0
    blended_gain = 1.0 - best_rmse / raw_rmse if raw_rmse > 0.0 else 0.0
    if blended_gain < AUTO_SELECTION_MIN_RELATIVE_GAIN:
        best_alpha = 1.0
        best_rmse = raw_rmse
    scored = scored.with_columns(
        (
            pl.col("cartoboost_seasonal_base")
            + best_alpha * (pl.col("cartoboost_raw") - pl.col("cartoboost_seasonal_base"))
        ).alias("cartoboost_residual_blend")
    )
    candidate_scores = {
        "cartoboost_raw": raw_rmse,
        "cartoboost_seasonal_base": base_rmse,
        "cartoboost_residual_blend": best_rmse,
        "cartoboost_calendar_dom": rmse_expr(scored, "cartoboost_calendar_dom"),
        "cartoboost_calendar_elapsed_phase": rmse_expr(scored, "cartoboost_calendar_elapsed_phase"),
        "cartoboost_calendar_dom_blend": rmse_expr(scored, "cartoboost_calendar_dom_blend"),
        "cartoboost_calendar_elapsed_phase_blend": rmse_expr(
            scored,
            "cartoboost_calendar_elapsed_phase_blend",
        ),
        "cartoboost_drift": rmse_expr(scored, "cartoboost_drift"),
        "cartoboost_half_drift": rmse_expr(scored, "cartoboost_half_drift"),
        "cartoboost_seasonal_drift": rmse_expr(scored, "cartoboost_seasonal_drift"),
        "cartoboost_seasonal_cycle_drift_050": rmse_expr(
            scored,
            "cartoboost_seasonal_cycle_drift_050",
        ),
        "cartoboost_seasonal_cycle_drift_075": rmse_expr(
            scored,
            "cartoboost_seasonal_cycle_drift_075",
        ),
    }
    ensemble_weights = validation_ensemble_weights(candidate_scores)
    scored = scored.with_columns(
        weighted_candidate_expr(ensemble_weights).alias(AUTO_ENSEMBLE_CANDIDATE)
    )
    candidate_scores[AUTO_ENSEMBLE_CANDIDATE] = rmse_expr(scored, AUTO_ENSEMBLE_CANDIDATE)
    selected_candidate = min(candidate_scores, key=candidate_scores.__getitem__)
    selected_gain = 1.0 - candidate_scores[selected_candidate] / raw_rmse if raw_rmse > 0.0 else 0.0
    if selected_candidate != "cartoboost_raw" and selected_gain < AUTO_SELECTION_MIN_RELATIVE_GAIN:
        selected_candidate = "cartoboost_raw"
        ensemble_weights = {"cartoboost_raw": 1.0}
    return (
        selected_candidate,
        best_alpha,
        ensemble_weights,
        {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 1.0,
            "inner_base_rmse": base_rmse,
            "inner_raw_rmse": raw_rmse,
            "inner_blended_rmse": best_rmse,
            "inner_raw_relative_rmse_gain": raw_gain,
            "inner_blended_relative_rmse_gain": blended_gain,
            "inner_selected_relative_rmse_gain": selected_gain,
            "inner_validation_ensemble_rmse": candidate_scores[AUTO_ENSEMBLE_CANDIDATE],
            "inner_calendar_dom_rmse": candidate_scores["cartoboost_calendar_dom"],
            "inner_calendar_elapsed_phase_rmse": candidate_scores[
                "cartoboost_calendar_elapsed_phase"
            ],
            "inner_drift_rmse": candidate_scores["cartoboost_drift"],
            "inner_half_drift_rmse": candidate_scores["cartoboost_half_drift"],
            "inner_seasonal_drift_rmse": candidate_scores["cartoboost_seasonal_drift"],
            "inner_seasonal_cycle_drift_050_rmse": candidate_scores[
                "cartoboost_seasonal_cycle_drift_050"
            ],
            "inner_seasonal_cycle_drift_075_rmse": candidate_scores[
                "cartoboost_seasonal_cycle_drift_075"
            ],
        },
    )


def validation_ensemble_weights(candidate_scores: dict[str, float]) -> dict[str, float]:
    weights = {
        str(name): float(weight)
        for name, weight in _native.forecast_validation_ensemble_weights_value(
            {str(name): float(score) for name, score in candidate_scores.items()}
        ).items()
    }
    return dict(
        sorted(
            weights.items(),
            key=lambda item: (float(candidate_scores.get(item[0], float("inf"))), item[0]),
        )
    )


def weighted_candidate_expr(weights: dict[str, float]) -> Any:
    pl = require_polars()
    expr = None
    for name, weight in sorted(weights.items()):
        term = float(weight) * pl.col(name)
        expr = term if expr is None else expr + term
    return expr if expr is not None else pl.col("cartoboost_raw")


def rmse_expr(frame: Any, prediction_col: str) -> float:
    pl = require_polars()
    return float(
        frame.select(((pl.col(prediction_col) - pl.col("actual")).pow(2).mean()).sqrt()).item()
    )


def cartoboost_raw_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
    prediction_col: str,
    known_future: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    feature_start = perf_counter()
    model_params = cartoboost_native_forecaster_params(
        season_length,
        horizon,
        config,
        train=train,
    )
    covariate_features = cartoboost_native_covariate_features(train, config)
    selected_columns = ["lane_id", "date", "loads", *covariate_features]
    training_frame = train.select(*selected_columns).to_pandas()
    if not isinstance(training_frame, pd.DataFrame):
        raise TypeError("CartoBoost native benchmark training conversion did not return pandas")
    known_future_frame = None
    if known_future is not None and covariate_features:
        known_future_columns = [
            column for column in covariate_features if column in known_future.columns
        ]
        if known_future_columns:
            future = known_future.select(
                "lane_id",
                "date",
                *known_future_columns,
            ).unique(subset=["lane_id", "date"])
            metadata_columns = [
                column
                for column in covariate_features
                if column not in known_future_columns and column in train.columns
            ]
            if metadata_columns:
                metadata = train.select("lane_id", *metadata_columns).unique(subset=["lane_id"])
                future = future.join(metadata, on="lane_id", how="left")
            missing_future_columns = [
                column for column in covariate_features if column not in future.columns
            ]
            if missing_future_columns:
                raise ValueError(
                    "known_future is missing required CartoBoost covariates: "
                    f"{missing_future_columns}"
                )
            known_future_frame = future.select(
                "lane_id",
                "date",
                *covariate_features,
            ).to_pandas()
            if not isinstance(known_future_frame, pd.DataFrame):
                raise TypeError(
                    "CartoBoost native benchmark known-future conversion did not return pandas"
                )
    feature_seconds = perf_counter() - feature_start
    model = CartoBoostLagForecaster(
        time_col="date",
        target_col="loads",
        panel_cols=["lane_id"],
        frequency="D",
        **model_params,
    )
    fit_start = perf_counter()
    model.fit(training_frame)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    result = (
        model.predict(horizon, known_future=known_future_frame)
        if known_future_frame is not None
        else model.predict(horizon)
    )
    predictions = pl.DataFrame(
        result.predictions(),
        schema=["series_id", "timestamp", "horizon", "model", prediction_col],
        orient="row",
    ).select(
        "series_id",
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")).alias("timestamp"),
        "horizon",
        prediction_col,
    )
    predict_seconds = perf_counter() - predict_start
    feature_count = len(model.metadata_.get("feature_names", []))
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "feature_count": float(feature_count),
    }
    return predictions, timing


def cartoboost_piecewise_linear_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    feature_start = perf_counter()
    training_frame = train.select("lane_id", "date", "loads").to_pandas()
    if not isinstance(training_frame, pd.DataFrame):
        raise TypeError("CartoBoost piecewise benchmark training conversion did not return pandas")
    frame = ForecastFrame.from_pandas(
        training_frame,
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
        allow_irregular=True,
    )
    model = PiecewiseLinearSeasonalForecaster(
        **cartoboost_piecewise_linear_params(season_length=season_length)
    )
    feature_seconds = perf_counter() - feature_start

    fit_start = perf_counter()
    model.fit(frame)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    result = model.predict(horizon)
    predictions = pl.DataFrame(
        [
            (series_id, timestamp, step, value)
            for series_id, timestamp, step, _model, value in result.predictions()
        ],
        schema=[
            "series_id",
            "timestamp",
            "horizon",
            "cartoboost_piecewise_linear_seasonal",
        ],
        orient="row",
    ).select(
        "series_id",
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")).alias("timestamp"),
        "horizon",
        "cartoboost_piecewise_linear_seasonal",
    )
    predict_seconds = perf_counter() - predict_start
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
    }
    return predictions, timing


def cartoboost_neural_panel_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    known_future: Any | None = None,
    epochs: int = 80,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    feature_start = perf_counter()
    covariates = [
        column
        for column in ["airport_lane", "distance_miles", "pickup_zone", "dropoff_zone"]
        if column in train.columns
    ]
    training_frame = train.select("lane_id", "date", "loads", *covariates).to_pandas()
    if not isinstance(training_frame, pd.DataFrame):
        raise TypeError("CartoBoost NeuralPanel benchmark conversion did not return pandas")
    known_future_frame = None
    if "airport_lane" in covariates:
        lane_future_rows = []
        lane_metadata = (
            train.sort(["lane_id", "date"])
            .group_by("lane_id")
            .agg(
                pl.col("date").max().alias("last_date"),
                pl.col("airport_lane").last().alias("airport_lane"),
            )
        )
        for row in lane_metadata.iter_rows(named=True):
            for step in range(1, horizon + 1):
                lane_future_rows.append(
                    {
                        "lane_id": row["lane_id"],
                        "date": row["last_date"] + timedelta(days=step),
                        "airport_lane": row["airport_lane"],
                    }
                )
        future = pl.DataFrame(lane_future_rows)
        if known_future is not None and "airport_lane" in known_future.columns:
            future = pl.concat(
                [
                    future,
                    known_future.select("lane_id", "date", "airport_lane"),
                ],
                how="vertical",
            ).unique(subset=["lane_id", "date"], keep="last")
        future_frame = future.to_pandas()
        if not isinstance(future_frame, pd.DataFrame):
            raise TypeError("CartoBoost NeuralPanel known-future conversion did not return pandas")
        future_frame = future_frame.assign(loads=0.0)
        known_future_frame = ForecastFrame.from_pandas(
            future_frame,
            timestamp_col="date",
            target_col="loads",
            series_id_col="lane_id",
            freq="D",
            allow_irregular=True,
            known_future_covariates=["airport_lane"],
        )
    frame = ForecastFrame.from_pandas(
        training_frame,
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
        allow_irregular=True,
        known_future_covariates=[name for name in ["airport_lane"] if name in covariates],
        historical_covariates=[name for name in ["distance_miles"] if name in covariates],
        static_covariates=[name for name in ["pickup_zone", "dropoff_zone"] if name in covariates],
    )
    min_history = int(train.group_by("lane_id").len().select(pl.col("len").min()).item())
    n_lags = max(
        1,
        min(
            28,
            min_history - horizon,
            season_length * 2 if season_length > 1 else 7,
        ),
    )
    model = LaneNeuralPanelForecaster(
        n_lags=n_lags,
        n_forecasts=horizon,
        quantiles=[0.1, 0.5, 0.9],
        weekly_fourier_order=3 if season_length == 7 else 0,
        custom_seasonalities=[
            ("benchmark_cycle", float(season_length), min(5, max(1, season_length // 2)))
        ]
        if season_length > 1 and season_length != 7
        else None,
        future_regressors={"airport_lane": "additive"} if "airport_lane" in covariates else None,
        lagged_regressors={"distance_miles": n_lags} if "distance_miles" in covariates else None,
        ar_layers=[16],
        lagged_reg_layers=[8] if "distance_miles" in covariates else None,
        trend_mode="glocal",
        local_l2=0.1,
        embedding_dim=8,
        epochs=epochs,
        seed=42,
    )
    feature_seconds = perf_counter() - feature_start

    fit_start = perf_counter()
    model.fit(frame)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    result = (
        model.predict(horizon, known_future=known_future_frame)
        if known_future_frame is not None
        else model.predict(horizon)
    )
    predictions = pl.DataFrame(
        result.predictions(),
        schema=[
            "series_id",
            "timestamp",
            "horizon",
            "model",
            NEURAL_PANEL_BENCHMARK_MODEL,
        ],
        orient="row",
    ).select(
        "series_id",
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")).alias("timestamp"),
        "horizon",
        NEURAL_PANEL_BENCHMARK_MODEL,
    )
    predict_seconds = perf_counter() - predict_start
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "n_lags": float(n_lags),
        "feature_count": float(len(covariates)),
    }
    return predictions, timing


def cartoboost_piecewise_linear_params(*, season_length: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "growth": "linear",
        "component_mode": "additive",
        "changepoints": 12,
        "changepoint_range": 0.8,
        "changepoint_l2_regularization": 0.05,
        "changepoint_l1_regularization": 0.0,
        "seasonality_l2_regularization": 0.01,
        "weekly_fourier_order": 3 if season_length == 7 else 0,
        "yearly_fourier_order": 0,
        "daily_fourier_order": 0,
    }
    if season_length > 1 and season_length != 7:
        params["custom_seasonalities"] = [
            {
                "name": "benchmark_cycle",
                "period_days": float(season_length),
                "fourier_order": min(5, max(1, int(season_length) // 2)),
            }
        ]
    return params


def cartoboost_autostats_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    prediction_col: str,
    validation_objective: str = "mean_squared_error",
) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    feature_start = perf_counter()
    training_data = train.select("lane_id", "date", "loads").sort(["lane_id", "date"])
    observed = training_data.group_by("lane_id", maintain_order=True).agg(
        pl.col("date").n_unique().alias("__observed_count"),
        ((pl.col("date").max() - pl.col("date").min()).dt.total_days() + 1).alias(
            "__expected_count"
        ),
    )
    allow_irregular = bool(
        observed.filter(pl.col("__observed_count") != pl.col("__expected_count")).height
    )
    training_frame = training_data.to_pandas()
    if not isinstance(training_frame, pd.DataFrame):
        raise TypeError("CartoBoost native benchmark training conversion did not return pandas")
    training_frame["date"] = pd.to_datetime(training_frame["date"], errors="raise").astype(
        "datetime64[ns]"
    )
    frame = ForecastFrame.from_pandas(
        training_frame,
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
        allow_irregular=allow_irregular,
    )
    validation_window = max(1, min(int(horizon), 8))
    feature_seconds = perf_counter() - feature_start

    model = AutoStatsBank(
        season_length=max(int(season_length), 1),
        validation_window=validation_window,
        validation_objective=validation_objective,
    )
    fit_start = perf_counter()
    model.fit(frame)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    result = model.predict(horizon)
    predictions = pl.DataFrame(
        result.predictions(),
        schema=["series_id", "timestamp", "horizon", "model", prediction_col],
        orient="row",
    ).select(
        "series_id",
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")).alias("timestamp"),
        "horizon",
        prediction_col,
    )
    predict_seconds = perf_counter() - predict_start
    metadata = model.metadata_
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "validation_window": float(validation_window),
        "metadata": metadata,
    }
    return predictions, timing


def external_tree_lag_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
    known_future: Any | None = None,
) -> tuple[Any, dict[str, dict[str, float]]]:
    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "external tree lag baselines require xgboost and lightgbm; run `uv sync --group bench`."
        ) from exc

    tree_params = cartoboost_tree_regularization(season_length, horizon, config)
    model_specs = {
        "xgboost_lag": xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=tree_params["max_depth"],
            min_child_weight=tree_params["min_samples_leaf"],
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            random_state=0,
        ),
        "lightgbm_lag": lgb.LGBMRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=tree_params["max_depth"],
            min_child_samples=tree_params["min_samples_leaf"],
            verbosity=-1,
            random_state=0,
            n_jobs=1,
        ),
    }
    forecasts = []
    timings = {}
    for name, model in model_specs.items():
        forecast, timing = external_tree_lag_forecast(
            train,
            horizon,
            season_length=season_length,
            model=model,
            prediction_col=name,
            known_future=known_future,
        )
        forecasts.append(forecast)
        timings[name] = timing
    return combine_forecast_frames(forecasts), timings


def external_tree_lag_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    model: Any,
    prediction_col: str,
    known_future: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    feature_start = perf_counter()
    history_features = build_history_features(train, season_length=season_length)
    feature_columns = select_cartoboost_feature_columns(
        history_features,
        season_length=season_length,
    )
    feature_frame = history_features.drop_nulls(feature_columns)
    x = feature_frame.select(feature_columns).to_numpy()
    target_mode = cartoboost_target_mode(season_length, horizon)
    if target_mode == "delta_from_last":
        y = (feature_frame["loads"] - feature_frame["loads_lag_1"]).to_numpy()
    else:
        y = feature_frame.select("loads").to_numpy().ravel()
    feature_seconds = perf_counter() - feature_start

    fit_start = perf_counter()
    model.fit(x, y)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        future = next_future_rows(history, known_future=known_future)
        future_features = build_future_features(
            history,
            future,
            season_length=season_length,
        ).drop_nulls(feature_columns)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            raw_predictions = model.predict(future_features.select(feature_columns).to_numpy())
        if target_mode == "delta_from_last":
            predictions = raw_predictions + future_features["loads_lag_1"].to_numpy()
        else:
            predictions = raw_predictions
        step_forecast = future_features.select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").alias("timestamp"),
            pl.lit(step).alias("horizon"),
        ).with_columns(pl.Series(prediction_col, predictions))
        forecast_frames.append(step_forecast)
        predicted_future = future_features.with_columns(pl.Series(prediction_col, predictions))
        append_frame = recursive_history_append_frame(
            predicted_future,
            history_schema,
            prediction_col=prediction_col,
        )
        history = pl.concat([history, append_frame], how="vertical")
    predict_seconds = perf_counter() - predict_start
    return pl.concat(forecast_frames, how="vertical"), {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "feature_count": float(len(feature_columns)),
        "target_mode_delta": float(target_mode == "delta_from_last"),
    }


def recursive_history_append_frame(
    predicted_future: Any,
    history_schema: dict[str, Any],
    *,
    prediction_col: str,
) -> Any:
    pl = require_polars()
    expressions = [
        pl.col("lane_id").cast(history_schema["lane_id"]),
        pl.col("date").cast(history_schema["date"]),
        pl.col(prediction_col).alias("loads").cast(history_schema["loads"]),
    ]
    for column, dtype in history_schema.items():
        if column in {"lane_id", "date", "loads"}:
            continue
        if column in predicted_future.columns:
            expressions.append(pl.col(column).cast(dtype))
        else:
            expressions.append(pl.lit(None).cast(dtype).alias(column))
    return predicted_future.select(*expressions)


def cartoboost_native_forecaster_params(
    season_length: int,
    horizon: int,
    config: dict[str, Any],
    *,
    train: Any,
) -> dict[str, Any]:
    max_lag, max_window = cartoboost_supported_history_limits(train)
    lags = [lag for lag in cartoboost_lag_values(season_length) if lag <= max_lag]
    rolling_windows = [
        window for window in cartoboost_rolling_windows(season_length) if window <= max_window
    ]
    partial_rolling_mean_windows = (
        cartoboost_partial_rolling_mean_windows(rolling_windows)
        if config.get("use_native_partial_rolling_mean_features", False)
        else []
    )
    difference_lags = [
        lag for lag in cartoboost_difference_lags(season_length) if lag <= max_lag - 1
    ]
    rolling_trend_windows = list(rolling_windows)
    rolling_stat_windows = (
        cartoboost_rolling_stat_windows(rolling_windows)
        if config.get("use_native_rolling_stat_features", False)
        else []
    )
    if not lags:
        lags = [1]
    covariate_features = cartoboost_native_covariate_features(train, config)
    return {
        "lags": lags,
        "rolling_windows": rolling_windows,
        "partial_rolling_mean_windows": partial_rolling_mean_windows,
        "rolling_std_windows": rolling_stat_windows,
        "rolling_min_windows": rolling_stat_windows,
        "rolling_max_windows": rolling_stat_windows,
        "ewm_alpha_percents": [90] if config.get("use_native_ewm_features", False) else [],
        "difference_lags": difference_lags,
        "rolling_trend_windows": rolling_trend_windows,
        "calendar_features": True,
        "rich_calendar_features": config.get("use_rich_calendar_features", False),
        "elapsed_calendar_features": config.get("use_elapsed_calendar_features", False),
        "elapsed_calendar_periods": cartoboost_elapsed_calendar_periods(season_length, config),
        "covariate_features": covariate_features,
        "covariate_indicator_values": low_cardinality_covariate_indicators(
            train,
            covariate_features,
            min_unique=3,
            max_unique=8,
            min_calendar_interaction_strength=0.50,
        ),
        "covariate_calendar_interactions": config.get(
            "use_covariate_calendar_interactions",
            False,
        ),
        "target_mode": cartoboost_target_mode(season_length, horizon),
        "n_estimators": config["n_estimators"],
        "learning_rate": config["learning_rate"],
        **cartoboost_tree_regularization(season_length, horizon, config),
        "split_policy": config.get("split_policy", "structured"),
    }


def low_cardinality_covariate_indicators(
    train: Any,
    covariate_features: list[str],
    *,
    min_unique: int,
    max_unique: int,
    min_calendar_interaction_strength: float,
) -> dict[str, list[float]]:
    pl = require_polars()
    if not covariate_features:
        return {}
    indicators: dict[str, list[float]] = {}
    for name in covariate_features:
        if name not in train.columns:
            continue
        values = train.select(pl.col(name).drop_nulls().unique().sort()).to_series().to_list()
        numeric_values = [float(value) for value in values if value is not None]
        if not (min_unique <= len(numeric_values) <= max_unique):
            continue
        if any(not math.isfinite(value) for value in numeric_values):
            continue
        if any(abs(value - round(value)) > 1.0e-9 for value in numeric_values):
            continue
        strength = covariate_calendar_interaction_strength(train, name)
        if strength < min_calendar_interaction_strength:
            continue
        indicators[name] = numeric_values
    return indicators


def covariate_calendar_interaction_strength(train: Any, covariate: str) -> float:
    pl = require_polars()
    if (
        "date" not in train.columns
        or "loads" not in train.columns
        or covariate not in train.columns
    ):
        return 0.0
    target_std = train.select(pl.col("loads").std()).item()
    if target_std is None or not math.isfinite(float(target_std)) or float(target_std) <= 1.0e-12:
        return 0.0
    global_mean = train.select(pl.col("loads").mean()).item()
    if global_mean is None or not math.isfinite(float(global_mean)):
        return 0.0
    scored = train.with_columns(pl.col("date").dt.day().alias("__calendar_day"))
    covariate_means = scored.group_by(covariate).agg(pl.col("loads").mean().alias("__cov_mean"))
    day_means = scored.group_by("__calendar_day").agg(pl.col("loads").mean().alias("__day_mean"))
    interaction = (
        scored.group_by([covariate, "__calendar_day"])
        .agg(pl.col("loads").mean().alias("__joint_mean"))
        .join(covariate_means, on=covariate, how="inner")
        .join(day_means, on="__calendar_day", how="inner")
        .with_columns(
            (
                pl.col("__joint_mean")
                - pl.col("__cov_mean")
                - pl.col("__day_mean")
                + float(global_mean)
            )
            .abs()
            .alias("__interaction")
        )
    )
    max_interaction = interaction.select(pl.col("__interaction").max()).item()
    if max_interaction is None or not math.isfinite(float(max_interaction)):
        return 0.0
    return float(max_interaction) / float(target_std)


def cartoboost_benchmark_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config)
    settings["native_target_mode_policy"] = (
        "delta_from_last when season_length == 12, horizon == 13, or horizon >= 24; level otherwise"
    )
    settings["native_covariate_policy"] = (
        "use route static covariates, training-gated low-cardinality integer-coded covariate "
        "indicators, and covariate-calendar interactions including event-flag interactions only "
        "when enabled by the source-specific benchmark config"
    )
    settings["auto_selector_policy"] = (
        "auto uses the lag spine unless shared validation clears the source-specific gain and "
        "consistency guards; non-M inner validation skips raw auto and non-M outer scoring "
        "materializes raw auto only when selected; non-M inner validation stops after one origin "
        "when lag beats every finite candidate by at least 15%; rolling-origin suites cache "
        "identical inner validation cutoffs by source, cutoff, and candidate roster"
    )
    settings["native_ewm_feature_policy"] = (
        "EWM target features are available but disabled by default after validation showed "
        "route-mix regression without candidate gating"
    )
    settings["native_tree_regularization_policy"] = (
        "use at least max_depth=5 and at most min_samples_leaf=6 when "
        "horizon >= 24 or season_length == 4; otherwise use configured values"
    )
    settings["native_feature_policy"] = (
        "season-aware lags, complete rolling means, optional MLForecast-style partial "
        "rolling means, lag deltas, and rolling trends capped to the shortest training series"
    )
    return settings


def cartoboost_model_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "cartoboost_lag": cartoboost_benchmark_settings(config),
        "cartoboost_auto_forecast": {
            **cartoboost_benchmark_settings(config),
            "auto_n_estimators": (
                int(config["auto_n_estimators"])
                if config.get("auto_n_estimators") is not None
                else None
            ),
            "auto_n_estimators_policy": (
                "explicit --cartoboost-auto-n-estimators override"
                if config.get("auto_n_estimators") is not None
                else "quality floor: max(--cartoboost-n-estimators, 360)"
            ),
        },
        "cartoboost_piecewise_linear_seasonal": {
            **cartoboost_piecewise_linear_params(season_length=7),
            "benchmark_profile": (
                "Rust-native piecewise-linear additive trend with weekly Fourier "
                "seasonality on daily benchmark panels; non-weekly season lengths use one "
                "generic Fourier cycle named benchmark_cycle"
            ),
        },
        NEURAL_PANEL_BENCHMARK_MODEL: {
            "n_forecasts": "benchmark horizon",
            "n_lags": "min(28, minimum train history - benchmark horizon, two seasonal cycles)",
            "quantiles": [0.1, 0.5, 0.9],
            "weekly_fourier_order": 3,
            "future_regressors": ["airport_lane when present"],
            "lagged_regressors": ["distance_miles when present"],
            "ar_layers": [16],
            "lagged_reg_layers": [8],
            "trend_mode": "glocal",
            "local_l2": 0.1,
            "embedding_dim": 8,
            "seed": 42,
            "benchmark_profile": (
                "Rust-native lane neural panel forecaster with directional lane ids, "
                "direct multi-horizon output, quantile metadata, and cold-identity "
                "benchmark fallback expansion."
            ),
        },
    }


def cartoboost_rolling_stat_windows(rolling_windows: list[int]) -> list[int]:
    preferred = [window for window in [7, 28] if window in rolling_windows]
    if preferred:
        return preferred
    if not rolling_windows:
        return []
    return sorted({rolling_windows[0], rolling_windows[-1]})


def cartoboost_partial_rolling_mean_windows(rolling_windows: list[int]) -> list[int]:
    return [window for window in rolling_windows if window in {7, 14, 28}]


def cartoboost_tree_regularization(
    season_length: int,
    horizon: int,
    config: dict[str, Any],
) -> dict[str, int]:
    max_depth = int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    if horizon >= 24 or season_length == 4:
        max_depth = max(max_depth, 5)
        min_samples_leaf = min(min_samples_leaf, 6)
    return {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }


def cartoboost_target_mode(season_length: int, horizon: int) -> str:
    if season_length == 12 or horizon == 13 or horizon >= 24:
        return "delta_from_last"
    return "level"


def cartoboost_supported_history_limits(train: Any) -> tuple[int, int]:
    pl = require_polars()
    min_history = int(train.group_by("lane_id").len().select(pl.col("len").min()).item())
    if min_history < 2:
        raise ValueError("CartoBoost lag benchmark requires at least two rows per series")
    max_lag = max(1, min_history - 1)
    return max_lag, max_lag


def lane_date_value_history(history: Any) -> dict[Any, tuple[list[datetime], list[float]]]:
    pl = require_polars()
    records: dict[Any, tuple[list[datetime], list[float]]] = {}
    grouped = (
        history.sort(["lane_id", "date"])
        .group_by("lane_id", maintain_order=True)
        .agg(pl.col("date"), pl.col("loads"))
    )
    for row in grouped.iter_rows(named=True):
        records[row["lane_id"]] = (
            list(row["date"]),
            [float(value) for value in row["loads"]],
        )
    return records


def seasonal_naive_forecast_frame(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    prediction_col: str,
) -> Any:
    pl = require_polars()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        rows = []
        histories = lane_date_value_history(history)
        for row in next_future_rows(history).iter_rows(named=True):
            _dates, values = histories[row["lane_id"]]
            prediction = float(
                _native.forecast_seasonal_naive_candidate_value(values, int(season_length))
            )
            rows.append({**row, prediction_col: prediction, "horizon": step})
        future = pl.DataFrame(rows)
        forecast_frames.append(
            future.select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                prediction_col,
            )
        )
        append_frame = recursive_history_append_frame(
            future,
            history_schema,
            prediction_col=prediction_col,
        )
        history = pl.concat([history, append_frame], how="vertical")
    return pl.concat(forecast_frames, how="vertical")


def intermittent_forecasts(
    train: Any,
    horizon: int,
    *,
    model_names: list[str],
) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    histories = lane_date_value_history(train)
    records = []
    timing: dict[str, Any] = {}
    for series_id, (dates, values) in histories.items():
        last_date = dates[-1]
        series_predictions = {}
        if "croston" in model_names:
            series_predictions["croston"] = croston_forecast(values, horizon)
        if "sba" in model_names:
            series_predictions["sba"] = sba_forecast(values, horizon)
        if "tsb" in model_names:
            series_predictions["tsb"] = tsb_forecast(values, horizon)
        for step in range(1, horizon + 1):
            timestamp = last_date + timedelta(days=step)
            row = {
                "series_id": series_id,
                "timestamp": timestamp,
                "horizon": step,
            }
            for name, predictions in series_predictions.items():
                row[name] = float(predictions[step - 1])
            records.append(row)
    for name in model_names:
        timing[name] = {
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "fit_predict_seconds": 0.0,
            "total_seconds": 0.0,
            "baseline": True,
        }
    return pl.DataFrame(records), timing


def calendar_profile_forecast_frame(
    train: Any,
    horizon: int,
    *,
    prediction_col: str,
    mode: str,
    elapsed_phase_period: int | None = None,
) -> Any:
    pl = require_polars()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        rows = []
        histories = lane_date_value_history(history)
        for row in next_future_rows(history).iter_rows(named=True):
            dates, values = histories[row["lane_id"]]
            prediction = calendar_profile_prediction(
                dates,
                values,
                row["date"],
                mode=mode,
                elapsed_phase_period=elapsed_phase_period,
            )
            rows.append({**row, prediction_col: prediction, "horizon": step})
        future = pl.DataFrame(rows)
        forecast_frames.append(
            future.select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                prediction_col,
            )
        )
        append_frame = recursive_history_append_frame(
            future,
            history_schema,
            prediction_col=prediction_col,
        )
        history = pl.concat([history, append_frame], how="vertical")
    return pl.concat(forecast_frames, how="vertical")


def trend_forecast_frame(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    prediction_col: str,
    mode: str,
) -> Any:
    pl = require_polars()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        rows = []
        histories = lane_date_value_history(history)
        for row in next_future_rows(history).iter_rows(named=True):
            _dates, values = histories[row["lane_id"]]
            prediction = trend_prediction(
                values,
                step=step,
                season_length=season_length,
                mode=mode,
            )
            rows.append({**row, prediction_col: max(0.0, prediction), "horizon": step})
        future = pl.DataFrame(rows)
        forecast_frames.append(
            future.select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                prediction_col,
            )
        )
        append_frame = recursive_history_append_frame(
            future,
            history_schema,
            prediction_col=prediction_col,
        )
        history = pl.concat([history, append_frame], how="vertical")
    return pl.concat(forecast_frames, how="vertical")


def trend_prediction(
    values: list[float],
    *,
    step: int,
    season_length: int,
    mode: str,
) -> float:
    return float(
        _native.forecast_trend_candidate_value(
            values,
            int(step),
            int(season_length),
            mode,
        )
    )


def calendar_profile_prediction(
    dates: list[datetime],
    values: list[float],
    timestamp: datetime,
    *,
    mode: str,
    elapsed_phase_period: int | None = None,
) -> float:
    return float(
        _native.forecast_calendar_profile_candidate_value(
            values,
            [int(date.day) for date in dates],
            int(timestamp.day),
            mode,
            elapsed_phase_period,
        )
    )


def unique_positive_ints(values: list[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        if value > 0 and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def cartoboost_lag_values(season_length: int) -> list[int]:
    seasonal_lags: list[int] = []
    if season_length > 1:
        seasonal_lags = [
            season_length - 1,
            season_length,
            season_length + 1,
            2 * season_length,
        ]
    return unique_positive_ints([*BASE_CARTOBOOST_LAGS, *seasonal_lags])


def cartoboost_rolling_windows(season_length: int) -> list[int]:
    seasonal_windows: list[int] = []
    if season_length > 1:
        seasonal_windows = [season_length, 2 * season_length]
    return unique_positive_ints([*BASE_CARTOBOOST_ROLLING_WINDOWS, *seasonal_windows])


def cartoboost_difference_lags(season_length: int) -> list[int]:
    return [lag for lag in cartoboost_lag_values(season_length) if lag > 1]


def cartoboost_target_feature_specs(season_length: int) -> list[tuple[str, int]]:
    specs = [(f"loads_lag_{lag}", lag) for lag in cartoboost_lag_values(season_length)]
    specs.extend(
        (f"loads_roll_{window}", window) for window in cartoboost_rolling_windows(season_length)
    )
    specs.extend(
        (f"loads_delta_lag_{lag}", lag) for lag in cartoboost_difference_lags(season_length)
    )
    specs.extend(
        (f"loads_roll_trend_{window}", 2 * window)
        for window in cartoboost_rolling_windows(season_length)
    )
    return specs


def select_cartoboost_feature_columns(feature_frame: Any, *, season_length: int) -> list[str]:
    target_specs = cartoboost_target_feature_specs(season_length)
    ranked_drop_candidates = sorted(target_specs, key=lambda item: item[1], reverse=True)
    for drop_count in range(len(ranked_drop_candidates) + 1):
        dropped = {name for name, _cost in ranked_drop_candidates[:drop_count]}
        target_columns = [name for name, _cost in target_specs if name not in dropped]
        columns = [*target_columns, *benchmark_exogenous_feature_columns(feature_frame)]
        if feature_frame.drop_nulls(columns).height > 0:
            return columns
    raise ValueError("CartoBoost lag benchmark has no complete lag feature rows")


def benchmark_exogenous_feature_columns(frame: Any) -> list[str]:
    return [*EXOGENOUS_FEATURE_COLUMNS, *known_future_covariate_columns(frame)]


def known_future_covariate_columns(frame: Any | None) -> list[str]:
    if frame is None:
        return []
    return [column for column in M5_KNOWN_FUTURE_COVARIATES if column in frame.columns]


def cartoboost_native_covariate_features(frame: Any, config: dict[str, Any]) -> list[str]:
    features = []
    if config.get("use_static_covariates", False):
        features.extend(column for column in STATIC_COVARIATES if column in frame.columns)
        features.extend(available_m5_hierarchy_covariates(frame))
    if config.get("use_known_future_covariates", False):
        features.extend(known_future_covariate_columns(frame))
    return list(dict.fromkeys(features))


def known_future_covariate_frame(frame: Any) -> Any | None:
    columns = known_future_covariate_columns(frame)
    if not columns:
        return None
    return frame.select("lane_id", "date", *columns).unique(subset=["lane_id", "date"])


def build_history_features(frame: Any, *, season_length: int) -> Any:
    pl = require_polars()
    lags = cartoboost_lag_values(season_length)
    rolling_windows = cartoboost_rolling_windows(season_length)
    difference_lags = cartoboost_difference_lags(season_length)
    return frame.sort(["lane_id", "date"]).with_columns(
        *[pl.col("loads").shift(lag).over("lane_id").alias(f"loads_lag_{lag}") for lag in lags],
        *[
            pl.col("loads")
            .shift(1)
            .rolling_mean(window)
            .over("lane_id")
            .alias(f"loads_roll_{window}")
            for window in rolling_windows
        ],
        *[
            (
                pl.col("loads").shift(1).over("lane_id")
                - pl.col("loads").shift(lag).over("lane_id")
            ).alias(f"loads_delta_lag_{lag}")
            for lag in difference_lags
        ],
        *[
            (
                pl.col("loads").shift(1).rolling_mean(window).over("lane_id")
                - pl.col("loads").shift(window + 1).rolling_mean(window).over("lane_id")
            ).alias(f"loads_roll_trend_{window}")
            for window in rolling_windows
        ],
        date_dayofweek=pl.col("date").dt.weekday().cast(pl.Float64),
        date_day=pl.col("date").dt.day().cast(pl.Float64),
        date_dayofyear=pl.col("date").dt.ordinal_day().cast(pl.Float64),
        date_month=pl.col("date").dt.month().cast(pl.Float64),
        date_elapsed_days=pl.int_range(pl.len()).over("lane_id").cast(pl.Float64),
    )


def next_future_rows(history: Any, *, known_future: Any | None = None) -> Any:
    pl = require_polars()
    future = (
        history.sort(["lane_id", "date"])
        .group_by("lane_id", maintain_order=True)
        .tail(1)
        .with_columns((pl.col("date") + pl.duration(days=1)).alias("date"))
        .select("lane_id", "date", *STATIC_COVARIATES)
    )
    known_columns = known_future_covariate_columns(known_future)
    if not known_columns:
        return future
    return future.join(
        known_future.select("lane_id", "date", *known_columns).unique(subset=["lane_id", "date"]),
        on=["lane_id", "date"],
        how="left",
    ).with_columns([pl.col(column).fill_null(0.0) for column in known_columns])


def build_future_features(history: Any, future: Any, *, season_length: int) -> Any:
    pl = require_polars()
    lags = cartoboost_lag_values(season_length)
    rolling_windows = cartoboost_rolling_windows(season_length)
    difference_lags = cartoboost_difference_lags(season_length)
    histories = lane_date_value_history(history)
    pieces = []
    for row in future.iter_rows(named=True):
        values = dict(row)
        _dates, loads = histories[row["lane_id"]]
        for lag in lags:
            values[f"loads_lag_{lag}"] = float(loads[-lag]) if len(loads) >= lag else None
        for window in rolling_windows:
            values[f"loads_roll_{window}"] = (
                float(np.mean(loads[-window:])) if len(loads) >= window else None
            )
        for lag in difference_lags:
            values[f"loads_delta_lag_{lag}"] = (
                float(loads[-1] - loads[-lag]) if len(loads) >= lag else None
            )
        for window in rolling_windows:
            values[f"loads_roll_trend_{window}"] = (
                float(np.mean(loads[-window:]) - np.mean(loads[-2 * window : -window]))
                if len(loads) >= 2 * window
                else None
            )
        timestamp = values["date"]
        values["date_dayofweek"] = float(timestamp.weekday() + 1)
        values["date_day"] = float(timestamp.day)
        values["date_dayofyear"] = float(timestamp.timetuple().tm_yday)
        values["date_month"] = float(timestamp.month)
        values["date_elapsed_days"] = float(len(loads))
        pieces.append(values)
    return pl.DataFrame(pieces)


def external_autoreg_lag_depth(*, season_length: int, min_series_length: int) -> int:
    max_supported = max(1, min_series_length - 1)
    structural_lags = [28]
    if season_length > 1:
        structural_lags.extend([season_length, 2 * season_length])
    return min(max_supported, max(structural_lags))


def functime_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    lightgbm_config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    try:
        from functime.forecasting import lightgbm, ridge, snaive
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires functime; run `uv sync --group bench`."
        ) from exc

    y = train.select(
        pl.col("lane_id").alias("entity"),
        pl.col("date").alias("time"),
        pl.col("loads").alias("target"),
    )
    min_series_length = int(train.group_by("lane_id").len().select(pl.col("len").min()).item())
    autoreg_lags = external_autoreg_lag_depth(
        season_length=season_length,
        min_series_length=min_series_length,
    )
    model_specs = {
        "functime_snaive": snaive(freq="1d", sp=season_length),
        "functime_ridge": ridge(freq="1d", lags=autoreg_lags),
        "functime_lightgbm": lightgbm(
            freq="1d",
            lags=autoreg_lags,
            n_estimators=lightgbm_config["n_estimators"],
            learning_rate=lightgbm_config["learning_rate"],
            max_depth=lightgbm_config["max_depth"],
            min_child_samples=lightgbm_config["min_samples_leaf"],
            verbosity=-1,
        ),
    }
    forecasts = []
    timings = {}
    for name, model in model_specs.items():
        fit_start = perf_counter()
        model.fit(y)
        fit_seconds = perf_counter() - fit_start
        predict_start = perf_counter()
        forecast = (
            model.predict(horizon)
            .rename({"entity": "series_id", "time": "timestamp", "target": name})
            .sort(["series_id", "timestamp"])
            .with_columns((pl.int_range(pl.len()).over("series_id") + 1).alias("horizon"))
            .select("series_id", "timestamp", "horizon", name)
        )
        predict_seconds = perf_counter() - predict_start
        forecasts.append(forecast)
        timings[name] = {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "fit_predict_seconds": fit_seconds + predict_seconds,
        }

    return combine_forecast_frames(forecasts), timings


def configure_prophet_seasonality(model: Any, *, season_length: int) -> None:
    if season_length <= 1 or season_length == 7:
        return
    fourier_order = min(10, max(3, season_length // 2))
    model.add_seasonality(
        name=f"structural_period_{season_length}",
        period=float(season_length),
        fourier_order=fourier_order,
        prior_scale=10.0,
        mode="additive",
    )


def statsforecast_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import (
            AutoARIMA,
            AutoCES,
            AutoETS,
            AutoTBATS,
            AutoTheta,
            DynamicOptimizedTheta,
            SeasonalNaive,
        )
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires statsforecast; run `uv sync --group bench`."
        ) from exc

    y = (
        train.select(
            pl.col("lane_id").alias("unique_id"),
            pl.col("date").alias("ds"),
            pl.col("loads").alias("y"),
        )
        .sort(["unique_id", "ds"])
        .to_pandas()
    )
    model_specs = {
        "statsforecast_seasonal_naive": SeasonalNaive(season_length=season_length),
        "statsforecast_autoets": AutoETS(season_length=season_length, model="ZZZ"),
        "statsforecast_autoarima": AutoARIMA(season_length=season_length),
        "statsforecast_autotheta": AutoTheta(season_length=season_length),
        "statsforecast_autoces": AutoCES(season_length=season_length),
        "statsforecast_dynamic_optimized_theta": DynamicOptimizedTheta(season_length=season_length),
        "statsforecast_autotbats": AutoTBATS(season_length=season_length),
    }
    forecasts = []
    timings = {}
    for name, model in model_specs.items():
        forecast_runner = StatsForecast(models=[model], freq="D", n_jobs=1)
        fit_start = perf_counter()
        forecast = forecast_runner.forecast(df=y, h=horizon)
        fit_predict_seconds = perf_counter() - fit_start
        value_columns = [column for column in forecast.columns if column not in {"unique_id", "ds"}]
        if len(value_columns) != 1:
            raise RuntimeError(
                f"StatsForecast model {name} returned forecast columns {value_columns!r}"
            )
        forecast_frame = (
            pl.from_pandas(
                forecast.rename(
                    columns={
                        "unique_id": "series_id",
                        "ds": "timestamp",
                        value_columns[0]: name,
                    }
                )
            )
            .sort(["series_id", "timestamp"])
            .with_columns((pl.int_range(pl.len()).over("series_id") + 1).alias("horizon"))
            .select("series_id", "timestamp", "horizon", name)
        )
        forecasts.append(forecast_frame)
        timings[name] = {
            "fit_seconds": fit_predict_seconds,
            "predict_seconds": 0.0,
            "fit_predict_seconds": fit_predict_seconds,
        }
    del pd
    return combine_forecast_frames(forecasts), timings


def prophet_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    prophet_class = ensure_prophet_class()

    fit_seconds = 0.0
    predict_seconds = 0.0
    forecast_frames = []
    grouped = train.sort(["lane_id", "date"]).group_by("lane_id", maintain_order=True)
    for series_id, group in grouped:
        series_frame = group.select(
            pl.col("date").alias("ds"),
            pl.col("loads").alias("y"),
        ).to_pandas()
        fit_start = perf_counter()
        model = prophet_class(
            weekly_seasonality=season_length == 7,
            daily_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode="additive",
            stan_backend="CMDSTANPY",
        )
        configure_prophet_seasonality(model, season_length=season_length)
        model.fit(series_frame)
        fit_seconds += perf_counter() - fit_start
        future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        predict_start = perf_counter()
        forecast = model.predict(future)
        predict_seconds += perf_counter() - predict_start
        forecast_frames.append(
            pl.from_pandas(
                pd.DataFrame(
                    {
                        "series_id": [series_id[0] if isinstance(series_id, tuple) else series_id]
                        * horizon,
                        "timestamp": forecast["ds"],
                        "horizon": np.arange(1, horizon + 1, dtype=int),
                        "prophet_additive": forecast["yhat"].to_numpy(dtype=float),
                    }
                )
            )
        )
    timing = {
        "prophet_additive": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "fit_predict_seconds": fit_seconds + predict_seconds,
        }
    }
    return pl.concat(forecast_frames, how="vertical"), timing


def evaluate_metrics(
    scored: Any,
    prediction_col: str,
    train: Any,
    *,
    season_length: int,
) -> dict[str, float]:
    pl = require_polars()
    error_frame = scored.select(
        error=pl.col(prediction_col) - pl.col("actual"),
        abs_error=(pl.col(prediction_col) - pl.col("actual")).abs(),
        actual_abs=pl.col("actual").abs(),
        smape_den=(pl.col(prediction_col).abs() + pl.col("actual").abs()),
    )
    mae = float(error_frame.select(pl.col("abs_error").mean()).item())
    rmse = float(error_frame.select((pl.col("error").pow(2).mean()).sqrt()).item())
    sse = float(error_frame.select(pl.col("error").pow(2).sum()).item())
    actual_mean = float(scored.select(pl.col("actual").mean()).item())
    sst = float(scored.select((pl.col("actual") - actual_mean).pow(2).sum()).item())
    r2 = 1.0 if sse <= 1.0e-12 else 0.0 if sst <= 1.0e-12 else 1.0 - sse / sst
    actual_abs_sum = float(error_frame.select(pl.col("actual_abs").sum()).item())
    abs_error_sum = float(error_frame.select(pl.col("abs_error").sum()).item())
    wape = 0.0 if abs_error_sum <= 1.0e-12 else abs_error_sum / max(actual_abs_sum, 1.0e-12)
    smape_value = (
        error_frame.filter(pl.col("smape_den") > 0)
        .select((2.0 * pl.col("abs_error") / pl.col("smape_den")).mean())
        .item()
    )
    smape = 0.0 if smape_value is None else float(smape_value)
    bias = float(error_frame.select(pl.col("error").mean()).item())
    train_scale = (
        train.sort(["lane_id", "date"])
        .with_columns(
            (pl.col("loads") - pl.col("loads").shift(season_length).over("lane_id"))
            .abs()
            .alias("d")
        )
        .select(pl.col("d").mean())
        .item()
    )
    mase_denom = max(float(train_scale or 0.0), 1.0e-12)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "mase": mae / mase_denom,
        "wape": wape,
        "smape": smape,
        "bias": bias,
    }


def write_forecast_plots(
    scored: Any, plot_dir: Path, *, prefix: str, models: list[str] | None = None
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "forecasting benchmark plotting requires matplotlib; run `uv sync`."
        ) from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    if models is None:
        models = benchmark_model_names("full")
    frame = scored.sort(["series_id", "timestamp"]).to_pandas()
    paths: list[Path] = []

    metric_path = plot_dir / f"{prefix}_tool_metric_comparison.png"
    rmse_values = [
        float(np.sqrt(np.mean((frame[model].to_numpy() - frame["actual"].to_numpy()) ** 2)))
        for model in models
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(models, rmse_values)
    ax.set_ylabel("RMSE")
    ax.set_title("Forecast RMSE by tool")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(metric_path, dpi=160)
    plt.close(fig)
    paths.append(metric_path)

    horizon_path = plot_dir / f"{prefix}_horizon_rmse_by_tool.png"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in models:
        by_horizon = (
            frame.assign(error=(frame[model] - frame["actual"]) ** 2)
            .groupby("horizon")["error"]
            .mean()
        )
        ax.plot(by_horizon.index, np.sqrt(by_horizon.to_numpy()), marker="o", label=model)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(horizon_path, dpi=160)
    plt.close(fig)
    paths.append(horizon_path)

    lines_path = plot_dir / f"{prefix}_forecast_lines.png"
    top_series = (
        frame.groupby("series_id")["actual"].sum().sort_values(ascending=False).head(3).index
    )
    fig, axes = plt.subplots(len(top_series), 1, figsize=(10, 2.8 * len(top_series)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, series_id in zip(axes, top_series, strict=True):
        subset = frame[frame["series_id"] == series_id].sort_values("timestamp")
        ax.plot(subset["timestamp"], subset["actual"], marker="o", label="actual", linewidth=2)
        for model in models[:4]:
            ax.plot(subset["timestamp"], subset[model], marker=".", label=model, alpha=0.8)
        ax.set_title(str(series_id))
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(lines_path, dpi=160)
    plt.close(fig)
    paths.append(lines_path)

    scatter_path = plot_dir / f"{prefix}_actual_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in models[:5]:
        ax.scatter(frame["actual"], frame[model], s=12, alpha=0.45, label=model)
    low = float(min(frame["actual"].min(), *(frame[model].min() for model in models[:5])))
    high = float(max(frame["actual"].max(), *(frame[model].max() for model in models[:5])))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)
    paths.append(scatter_path)

    return [str(path) for path in paths]


def require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires polars; run `uv sync --group bench`."
        ) from exc
    return pl


def require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires duckdb for --source duckdb; run "
            "`uv sync --group bench`."
        ) from exc
    return duckdb


def require_pandas_for_benchmark() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires pandas; run `uv sync --group bench`."
        ) from exc
    return pd


def ensure_prophet_class() -> Any:
    global PROPHET_CLASS
    if PROPHET_CLASS is not None:
        return PROPHET_CLASS
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires prophet; run `uv sync --group bench`."
        ) from exc
    PROPHET_CLASS = Prophet
    return PROPHET_CLASS


if __name__ == "__main__":
    raise SystemExit(main())
