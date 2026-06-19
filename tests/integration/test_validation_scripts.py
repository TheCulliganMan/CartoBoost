from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pytest


def test_full_validation_script_uses_native_estimator_without_backend_toggle():
    repo_root = Path(__file__).resolve().parents[2]

    full_validation = (repo_root / "scripts" / "run_full_validation.py").read_text(encoding="utf-8")
    splitter_metrics = (repo_root / "scripts" / "run_splitter_acceptance_metrics.py").read_text(
        encoding="utf-8"
    )
    lane_metrics = (repo_root / "scripts" / "run_lane_level_acceptance_metrics.py").read_text(
        encoding="utf-8"
    )

    assert "scripts/run_splitter_acceptance_metrics.py" in full_validation
    assert "scripts/run_lane_level_acceptance_metrics.py" in full_validation
    assert "CartoBoostRegressor(" in splitter_metrics
    assert "CartoBoostRegressor(" in lane_metrics
    backend_kwarg = "backend" + "="
    assert backend_kwarg not in splitter_metrics
    assert backend_kwarg not in lane_metrics


def test_ci_installs_native_extension_before_validation_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    install_step = workflow.index("uv run maturin develop")
    validation_step = workflow.index("uv run python scripts/run_full_validation.py")

    assert install_step < validation_step


def test_model_benchmark_suite_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_benchmarks"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "normal",
            "--models",
            "mean,cartoboost",
            "--n-rows",
            "120",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = output_dir / "results.json"
    report = output_dir / "results.md"
    assert results.exists()
    assert report.exists()
    assert "Normal dense" in report.read_text(encoding="utf-8")


def test_model_benchmark_suite_reports_best_cartoboost_vs_lightgbm():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "run_model_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_model_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark_suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark_suite
    spec.loader.exec_module(benchmark_suite)

    payload = {
        "workloads": {
            "normal": {
                "splits": {
                    "random": {
                        "models": {
                            "cartoboost": {
                                "status": "ok",
                                "metrics": {"rmse": 0.42, "r2": 0.91},
                            },
                            "cartoboost_neural": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                            "lightgbm": {
                                "status": "ok",
                                "metrics": {"rmse": 0.50, "r2": 0.88},
                            },
                        },
                    },
                },
            },
        },
    }

    rows = benchmark_suite.lightgbm_comparison(payload)

    assert rows == [
        {
            "workload": "normal",
            "split": "random",
            "best_cartoboost_model": "cartoboost",
            "best_cartoboost_rmse": 0.42,
            "lightgbm_rmse": 0.50,
            "rmse_delta_vs_lightgbm": -0.08000000000000002,
            "best_cartoboost_r2": 0.91,
            "lightgbm_r2": 0.88,
            "r2_delta_vs_lightgbm": 0.030000000000000027,
            "winner": "cartoboost",
        }
    ]


def test_model_benchmark_suite_link_negatives_avoid_positive_edges():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "run_model_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_model_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark_suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark_suite
    spec.loader.exec_module(benchmark_suite)

    positive_set = {(0, 0), (0, 1), (0, 2), (1, 0)}
    candidate = benchmark_suite.verified_negative_pair(
        0,
        0,
        node_count=3,
        positive_set=positive_set,
    )

    assert candidate == (2, 0)
    assert candidate not in positive_set
    assert (
        benchmark_suite.verified_negative_pair(
            0,
            0,
            node_count=1,
            positive_set={(0, 0)},
        )
        is None
    )


def test_forecasting_benchmark_loads_m5_local_competition_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location("forecasting_library_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    data_dir = tmp_path / "m5"
    data_dir.mkdir()
    days = 35
    d_columns = ",".join(f"d_{day}" for day in range(1, days + 1))
    values_a = ",".join(str(day) for day in range(1, days + 1))
    values_b = ",".join(str(day * 2) for day in range(1, days + 1))
    (data_dir / "sales_train_validation.csv").write_text(
        "\n".join(
            [
                f"id,item_id,dept_id,cat_id,store_id,state_id,{d_columns}",
                f"FOODS_1_001_CA_1_validation,FOODS_1_001,FOODS_1,FOODS,CA_1,CA,{values_a}",
                f"HOBBIES_1_001_TX_1_validation,HOBBIES_1_001,HOBBIES_1,HOBBIES,TX_1,TX,{values_b}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "calendar.csv").write_text(
        "d,date\n" + "\n".join(f"d_{day},2020-01-{day:02d}" for day in range(1, days + 1)) + "\n",
        encoding="utf-8",
    )

    table, dataset = benchmark.load_m5_fixture(
        types.SimpleNamespace(
            m5_data_dir=data_dir, m5_series_limit=0, m5_history_days=0, no_download=True
        )
    )

    assert dataset["series"] == 2
    assert dataset["horizon"] == 28
    assert table.height == 70
    assert {"lane_id", "date", "loads", *benchmark.STATIC_COVARIATES} <= set(table.columns)


def test_forecasting_benchmark_m5_requires_real_local_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_missing",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    try:
        benchmark.load_m5_fixture(
            types.SimpleNamespace(
                m5_data_dir=tmp_path,
                m5_series_limit=0,
                m5_history_days=0,
                no_download=True,
            )
        )
    except FileNotFoundError as exc:
        assert "sales_train_evaluation.csv or sales_train_validation.csv" in str(exc)
    else:
        raise AssertionError("missing M5 files should hard-fail")


def test_forecasting_benchmark_allows_bounded_m5_full_roster():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_roster",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    benchmark.validate_args(
        types.SimpleNamespace(
            lanes=1,
            horizon=28,
            source="m5",
            days=120,
            suite=False,
            m4_suite=False,
            m4_series_limit=0,
            m5_series_limit=100,
            m5_history_days=90,
            m6_series_limit=0,
            m6_horizon=28,
            model_roster="full",
            allow_full_m5_roster=False,
            cartoboost_n_estimators=1,
            cartoboost_max_depth=3,
            cartoboost_min_samples_leaf=8,
            suite_folds=1,
        )
    )

    try:
        benchmark.validate_args(
            types.SimpleNamespace(
                lanes=1,
                horizon=28,
                source="m5",
                days=120,
                suite=False,
                m4_suite=False,
                m4_series_limit=0,
                m5_series_limit=0,
                m5_history_days=90,
                m6_series_limit=0,
                m6_horizon=28,
                model_roster="full",
                allow_full_m5_roster=False,
                cartoboost_n_estimators=1,
                cartoboost_max_depth=3,
                cartoboost_min_samples_leaf=8,
                suite_folds=1,
            )
        )
    except ValueError as exc:
        assert "requires a positive --m5-series-limit" in str(exc)
    else:
        raise AssertionError("unbounded M5 full-roster runs should require explicit opt-in")


def test_forecasting_benchmark_auto_config_is_fixed_stronger_policy():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_auto_policy",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    config = {
        "n_estimators": 180,
        "learning_rate": 0.06,
        "max_depth": 4,
        "min_samples_leaf": 8,
    }
    auto = benchmark.cartoboost_auto_config(config, season_length=7, horizon=28)

    assert auto["n_estimators"] == 360
    assert auto["learning_rate"] == 0.06
    assert auto["max_depth"] == 6
    assert auto["min_samples_leaf"] == 4
    assert config["n_estimators"] == 180


def test_forecasting_benchmark_auto_ensemble_weights_are_stable_and_normalized():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_auto_weights",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    weights = benchmark.validation_ensemble_weights(
        {
            "worse": 4.0,
            "best": 1.0,
            "nan": float("nan"),
            "second": 2.0,
            "third": 3.0,
            "fourth": 3.5,
            "fifth": 3.6,
        }
    )

    assert list(weights) == ["best", "second", "third", "fourth"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["best"] > weights["second"] > weights["third"] > weights["fourth"]


def test_forecasting_benchmark_loads_m6_assets_file(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location("forecasting_library_benchmark_m6", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    path = tmp_path / "assets_m6.csv"
    lines = ["symbol,date,price"]
    start = date(2020, 1, 1)
    for day in range(90):
        current = start + timedelta(days=day)
        lines.append(f"AAA,{current.isoformat()},{100.0 + day}")
        lines.append(f"BBB,{current.isoformat()},{200.0 + day * 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table, dataset = benchmark.load_m6_fixture(
        types.SimpleNamespace(
            m6_assets_path=path,
            m6_series_limit=0,
            m6_horizon=28,
            no_download=True,
        )
    )

    assert dataset["series"] == 2
    assert dataset["horizon"] == 28
    assert dataset["days"] == 90
    assert table.height == 180
    assert {"lane_id", "date", "loads", *benchmark.STATIC_COVARIATES} <= set(table.columns)


def test_forecasting_benchmark_emits_m5_wrmsse_artifact():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    rows = []
    start = date(2026, 1, 1)
    for lane, base, pickup_zone in [("PU1->DO1", 10.0, 1.0), ("PU2->DO2", 20.0, 2.0)]:
        for day in range(8):
            rows.append(
                {
                    "lane_id": lane,
                    "date": start + timedelta(days=day),
                    "loads": base + day,
                    "pickup_zone": pickup_zone,
                    "dropoff_zone": pickup_zone + 10.0,
                    "distance_miles": 1.0,
                    "airport_lane": 0.0,
                    "pickup_borough_code": pickup_zone,
                }
            )
    table = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    scored = pl.DataFrame(
        [
            {
                "series_id": lane,
                "timestamp": start + timedelta(days=day),
                "horizon": day - 5,
                "actual": base + day,
                "cartoboost_lag": base + day - 1.0,
                "cartoboost_auto_forecast": base + day,
            }
            for lane, base in [("PU1->DO1", 10.0), ("PU2->DO2", 20.0)]
            for day in range(6, 8)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.benchmark_objective_artifacts(
        "m5",
        train_table=table,
        scored=scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        season_length=1,
    )

    m5 = artifact["m5"]
    assert artifact["primary_metric"] == "wrmsse"
    assert m5["ranking"] == ["cartoboost_auto_forecast", "cartoboost_lag"]
    assert set(m5["levels"]) == {"total", "state", "store", "item", "item_store"}
    total = m5["levels"]["total"]["models"]
    assert total["cartoboost_auto_forecast"]["wrmsse"] == pytest.approx(0.0)
    assert total["cartoboost_lag"]["wrmsse"] > 0.0


def test_forecasting_benchmark_emits_m6_rps_artifact():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    scored = pl.DataFrame(
        [
            {
                "series_id": symbol,
                "timestamp": date(2026, 1, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "cartoboost_lag": lag,
                "cartoboost_auto_forecast": actual,
            }
            for symbol, actual, lag in [
                ("AAA", -0.03, 0.04),
                ("BBB", -0.01, 0.03),
                ("CCC", 0.01, 0.02),
                ("DDD", 0.03, 0.01),
                ("EEE", 0.05, -0.02),
            ]
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.benchmark_objective_artifacts(
        "m6",
        train_table=scored,
        scored=scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        season_length=7,
    )

    m6 = artifact["m6"]
    assert artifact["primary_metric"] == "rank_probability_score"
    assert m6["ranking"][0] == "cartoboost_auto_forecast"
    auto = m6["models"]["cartoboost_auto_forecast"]
    assert auto["asset_count"] == 5
    assert auto["mean_rps"] < m6["models"]["cartoboost_lag"]["mean_rps"]
    assert sum(row["weight"] for row in auto["decisions"]) == pytest.approx(0.0)


def test_model_benchmark_suite_graph_families_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_benchmarks_graph"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "graph",
            "--models",
            (
                "cartoboost_graph_node2vec,cartoboost_graph_graphsage,"
                "cartoboost_graph_hetero_graphsage,cartoboost_graph_hinsage"
            ),
            "--n-rows",
            "96",
            "--graph-dim",
            "2",
            "--graph-epochs",
            "1",
            "--n-estimators",
            "4",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    models = results["workloads"]["graph"]["splits"]["random"]["models"]
    for model_name, family in {
        "cartoboost_graph_node2vec": "node2vec",
        "cartoboost_graph_graphsage": "graphsage",
        "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
        "cartoboost_graph_hinsage": "hinsage",
    }.items():
        assert models[model_name]["status"] == "ok"
        assert models[model_name]["config"]["graph_family"] == family


def test_nyc_taxi_benchmark_graph_family_switch_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    for family in ["node2vec", "graphsage", "hetero_graphsage", "hinsage"]:
        output_dir = tmp_path / f"nyc_taxi_{family}"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"),
                "--synthetic-smoke",
                "--output-dir",
                str(output_dir),
                "--tasks",
                "duration",
                "--models",
                "cartoboost_graph",
                "--graph-family",
                family,
                "--graph-dim",
                "2",
                "--graph-epochs",
                "1",
                "--cartoboost-n-estimators",
                "4",
            ],
            cwd=repo_root,
            check=True,
        )

        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        assert results["model_config"]["graph_family"] == family
        model = results["tasks"]["duration"]["splits"]["random"]["models"]["cartoboost_graph"]
        assert model["status"] == "ok"
        assert model["config"]["graph_family"] == family
