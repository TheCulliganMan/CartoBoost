from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest


def ordinal_word(rank: int) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
    }
    return words.get(rank, f"{rank}th")


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


def test_ci_runs_benchmark_provenance_freshness_gate_after_smoke_benchmark():
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    benchmark_step = workflow.index("scripts/run_model_benchmark_suite.py")
    freshness_step = workflow.index("scripts/check_benchmark_freshness.py")
    assert benchmark_step < freshness_step
    assert "--output-dir target/benchmark-freshness" in workflow
    assert "--artifact target/benchmark-freshness/results.json" in workflow
    assert "scripts/run_scale_performance_gate.py" in workflow


def test_release_workflow_requires_ci_ancestry_and_distribution_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github" / "workflows" / "publish-pypi.yml").read_text(
        encoding="utf-8"
    )

    assert "release-checks" in workflow
    assert "git merge-base --is-ancestor" in workflow
    assert "workflow_id: 'ci.yml'" in workflow
    assert "wheels-${{ matrix.platform }}-py${{ matrix.python-version }}" in workflow
    assert "smoke-sdist" in workflow
    assert "name: pypi" in workflow
    assert "password: ${{ secrets.PYPI_API_TOKEN }}" not in workflow


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
    jsonl = output_dir / "results.jsonl"
    report = output_dir / "results.md"
    assert results.exists()
    assert jsonl.exists()
    assert report.exists()
    assert "Normal dense" in report.read_text(encoding="utf-8")
    first_row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert first_row["track"] == "diagnostic"


def test_scale_performance_gate_structured_workload_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "scale.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_scale_performance_gate.py"),
            "--rows",
            "1000",
            "--threads",
            "2",
            "--estimators",
            "2",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "cartoboost.scale_performance_gate"
    assert payload["workload"]["features"] == 20
    assert payload["workload"]["predict_rows_per_second"] > 0


def test_scale_performance_audit_checks_release_budget(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact = tmp_path / "scale.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "cartoboost.scale_performance_gate",
                "workload": {
                    "rows": 1_000_000,
                    "fit_seconds": 120.0,
                    "peak_rss_mb": 1024.0,
                    "predict_rows_per_second": 1_500_000.0,
                    "thread_speedup": 3.2,
                },
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "check_scale_performance_gate.py"),
            str(artifact),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["passed"] is True


def test_scale_performance_audit_requires_thread_speedup_for_release(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    artifact = tmp_path / "scale-missing-thread-baseline.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "cartoboost.scale_performance_gate",
                "workload": {
                    "rows": 1_000_000,
                    "fit_seconds": 120.0,
                    "peak_rss_mb": 1024.0,
                    "predict_rows_per_second": 1_500_000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "check_scale_performance_gate.py"),
            str(artifact),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert json.loads(result.stdout)["checks"]["thread_speedup"] is False


def test_model_benchmark_suite_public_workloads_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_public_benchmarks"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "diabetes,karate",
            "--models",
            "mean,cartoboost",
            "--n-estimators",
            "4",
            "--graph-dim",
            "2",
            "--graph-epochs",
            "1",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert (
        results["git_commit"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert set(results["workloads"]) == {"diabetes", "karate"}
    assert results["workloads"]["diabetes"]["row_count"] == 442
    assert results["workloads"]["karate"]["row_count"] == 78

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert ("tabular", "diabetes", "random", "cartoboost", "rmse") in {
        (row["track"], row["task_id"], row["split_id"], row["model_family"], row["metric"])
        for row in rows
    }
    assert ("graph", "karate", "random", "cartoboost", "rmse") in {
        (row["track"], row["task_id"], row["split_id"], row["model_family"], row["metric"])
        for row in rows
    }


def test_model_benchmark_suite_reports_cartoboost_vs_external_baseline():
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

    rows = benchmark_suite.external_baseline_comparison(payload)

    assert len(rows) == 1
    row = rows[0]
    assert row["cartoboost_wape"] != row["cartoboost_wape"]
    assert row["best_external_wape"] != row["best_external_wape"]
    assert row | {"cartoboost_wape": None, "best_external_wape": None} == {
        "workload": "normal",
        "split": "random",
        "cartoboost_model": "cartoboost",
        "cartoboost_rmse": 0.42,
        "cartoboost_wape": None,
        "cartoboost_r2": 0.91,
        "best_external_baseline": "lightgbm",
        "best_external_rmse": 0.50,
        "best_external_wape": None,
        "best_external_r2": 0.88,
        "rmse_delta_vs_external": -0.08000000000000002,
        "r2_delta_vs_external": 0.030000000000000027,
        "status": "cartoboost_lower_rmse",
    }


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
        "d,date,wm_yr_wk,event_name_1,event_type_1,snap_CA,snap_TX,snap_WI\n"
        + "\n".join(
            (
                f"d_{day},2020-01-{day:02d},1001,"
                f"{'Promo' if day % 7 == 0 else ''},"
                f"{'Event' if day % 7 == 0 else ''},"
                f"{day % 2},{(day + 1) % 2},0"
            )
            for day in range(1, days + 1)
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "sell_prices.csv").write_text(
        "\n".join(
            [
                "store_id,item_id,wm_yr_wk,sell_price",
                "CA_1,FOODS_1_001,1001,2.0",
                "TX_1,HOBBIES_1_001,1001,3.0",
            ]
        )
        + "\n",
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
    assert {
        "lane_id",
        "date",
        "loads",
        "weight_value",
        "m5_sell_price",
        "m5_event_name_1_code",
        "m5_event_type_1_code",
        "m5_snap_CA",
        "m5_snap_TX",
        "m5_snap_WI",
        *benchmark.M5_HIERARCHY_COVARIATES,
        *benchmark.STATIC_COVARIATES,
    } <= set(table.columns)
    assert dataset["official_style_inputs"]["sell_prices_present"] is True
    assert dataset["known_future_covariates"] == [
        column for column in benchmark.M5_KNOWN_FUTURE_COVARIATES if column in table.columns
    ]
    assert dataset["hierarchy_covariates"] == benchmark.M5_HIERARCHY_COVARIATES


def test_forecasting_benchmark_loads_m1_local_tsf(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location("forecasting_library_benchmark_m1", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    tsf_dir = tmp_path / "m1"
    tsf_dir.mkdir()
    (tsf_dir / "m1_yearly_dataset.tsf").write_text(
        "\n".join(
            [
                "@attribute series_name string",
                "@frequency yearly",
                "@horizon 2",
                "@missing false",
                "@equallength false",
                "@data",
                "Y1:1,2,3,4,5,6,7,8",
                "Y2:2,3,4,5,6,7,8,9",
            ]
        )
        + "\n",
        encoding="cp1252",
    )

    table, dataset = benchmark.load_m1_fixture(
        types.SimpleNamespace(
            cache_dir=tmp_path,
            m1_group="Yearly",
            m1_series_limit=1,
            no_download=True,
        )
    )

    assert dataset["source"] == "m1_zenodo_tsf"
    assert dataset["series"] == 1
    assert dataset["available_series"] == 2
    assert dataset["horizon"] == 2
    assert dataset["season_length"] == 1
    assert table.height == 8
    assert {"lane_id", "date", "loads", *benchmark.STATIC_COVARIATES} <= set(table.columns)


def test_forecasting_benchmark_provenance_helpers_are_stable(tmp_path):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_provenance",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    first = pl.DataFrame(
        [
            {"lane_id": "PU2->DO2", "date": date(2026, 1, 2), "loads": 2.0},
            {"lane_id": "PU1->DO1", "date": date(2026, 1, 1), "loads": 1.0},
        ]
    )
    second = pl.DataFrame(
        [
            {"loads": 1.0, "date": date(2026, 1, 1), "lane_id": "PU1->DO1"},
            {"loads": 2.0, "date": date(2026, 1, 2), "lane_id": "PU2->DO2"},
        ]
    )

    assert benchmark.canonical_dataset_hash(first) == benchmark.canonical_dataset_hash(second)

    source = tmp_path / "assets_m6.csv"
    source.write_text("symbol,date,price\nAAA,2026-01-01,10.0\n", encoding="utf-8")
    hashes = benchmark.source_file_hashes({"assets_file": str(source)})
    assert hashes["assets_file"] == benchmark.file_sha256(source)

    resources = benchmark.resource_usage_snapshot()
    assert resources["process_cpu_seconds"] >= 0.0
    assert resources["peak_rss_mb"] > 0.0


def test_forecasting_benchmark_auto_objectives_are_metric_aware():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_objectives",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    args = types.SimpleNamespace(source="m")
    assert benchmark.normalize_competition_source(args) is args
    assert args.source == "m1"
    assert args.requested_source == "m"
    assert args.source_alias == "m"

    assert benchmark.auto_selection_objective("m") == "owa_proxy"
    assert benchmark.auto_selection_objective("m1") == "owa_proxy"
    assert benchmark.auto_selection_objective("m3") == "owa_proxy"
    assert benchmark.auto_selection_objective("m4") == "owa_proxy"
    assert benchmark.auto_selection_objective("m5") == "wrmsse"
    assert benchmark.auto_selection_objective("m6") == "investment_decision_return_then_rps"
    assert benchmark.auto_selection_objective("nyc-taxi") == "rmse"
    assert benchmark.auto_selection_objective("synthetic") == "rmse"


def test_forecasting_generalization_wrapper_is_non_m_scalable_guardrail():
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = (repo_root / "scripts" / "forecasting_generalization.py").read_text(encoding="utf-8")

    assert "--no-hyperopt is required for benchmark integrity" in wrapper
    assert '"synthetic"' in wrapper
    assert '"scalable"' in wrapper
    assert '"m4"' not in wrapper
    assert '"m5"' not in wrapper
    assert '"m6"' not in wrapper
    assert '"--no-candidate-selection"' in wrapper
    assert "forecasting_generalization_scalable_synthetic.json" in wrapper


def test_forecasting_benchmark_shared_candidate_cutoffs_are_deterministic():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_cutoffs",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamps = list(range(100))
    assert benchmark.shared_candidate_validation_cutoffs(timestamps, horizon=14) == [58, 72, 86]
    assert benchmark.shared_candidate_validation_cutoffs(timestamps, horizon=14, source="m5") == [
        72,
        86,
    ]
    assert benchmark.shared_candidate_validation_cutoffs(timestamps, horizon=14, source="m6") == [
        58,
        72,
        86,
    ]
    assert benchmark.shared_candidate_validation_cutoffs(list(range(62)), horizon=28) == [34]
    assert benchmark.shared_candidate_validation_cutoffs(list(range(20)), horizon=14) == []


def test_forecasting_benchmark_non_m_auto_requires_origin_consistency():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_origin_consistency",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    guarded = benchmark.lag_origin_consistency_guard(
        "cartoboost_auto_forecast",
        source="synthetic",
        inner_scores={
            "cartoboost_lag": [1.0, 1.0, 1.0],
            "cartoboost_auto_forecast": [0.8, 1.1, 0.7],
        },
    )
    assert guarded is not None
    assert guarded["losing_origin_count"] == 1
    assert guarded["min_relative_gain_vs_lag"] == pytest.approx(-0.1)

    assert (
        benchmark.lag_origin_consistency_guard(
            "cartoboost_auto_forecast",
            source="synthetic",
            inner_scores={
                "cartoboost_lag": [1.0, 1.0, 1.0],
                "cartoboost_auto_forecast": [0.8, 0.9, 0.7],
            },
        )
        is None
    )
    assert (
        benchmark.lag_origin_consistency_guard(
            "cartoboost_auto_forecast",
            source="m4",
            inner_scores={
                "cartoboost_lag": [1.0, 1.0, 1.0],
                "cartoboost_auto_forecast": [0.8, 1.1, 0.7],
            },
        )
        is None
    )


def test_forecasting_benchmark_non_m_skips_disqualified_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_raw_auto_short_circuit",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[list[str]] = []
    dates = [datetime(2026, 1, 1) + timedelta(days=day) for day in range(20)]
    train = pl.DataFrame(
        {
            "lane_id": ["A"] * len(dates),
            "date": dates,
            "loads": [1.0] * len(dates),
            "pickup_zone": [1.0] * len(dates),
            "dropoff_zone": [2.0] * len(dates),
            "distance_miles": [3.0] * len(dates),
            "airport_lane": [0.0] * len(dates),
            "pickup_borough_code": [1.0] * len(dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    def fake_roster(inner_train, horizon, *, model_names, **_kwargs):
        calls.append(list(model_names))
        next_timestamp = inner_train.select(pl.col("date").max()).item() + timedelta(days=1)
        columns = {
            "series_id": ["A"],
            "timestamp": [next_timestamp],
            "horizon": [1],
            "cartoboost_lag": [2.0],
        }
        if "cartoboost_auto_forecast" in model_names:
            columns["cartoboost_auto_forecast"] = [2.1]
        return (
            pl.DataFrame(columns).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"models": {}},
        )

    monkeypatch.setattr(benchmark, "candidate_selection_forecast_roster", fake_roster)
    monkeypatch.setattr(
        benchmark,
        "add_shared_candidate_columns",
        lambda _train, _horizon, **kwargs: kwargs["predictions"],
    )

    scores = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert calls[0] == ["cartoboost_lag", "cartoboost_auto_forecast"]
    assert calls[1:] == [["cartoboost_lag"]]
    assert len(scores["cartoboost_lag"]) == 2
    assert len(scores["cartoboost_auto_forecast"]) == 2
    assert scores["cartoboost_auto_forecast"][1:] == pytest.approx([1.03])


def test_forecasting_benchmark_non_m_lag_dominance_stops_inner_validation(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lag_dominance_stop",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.non_m_lag_dominates_origin(
        source="synthetic",
        origin_losses={
            "cartoboost_lag": 1.0,
            "cartoboost_auto_forecast": 1.25,
            "shared_seasonal_base": 1.30,
        },
    )
    assert not benchmark.non_m_lag_dominates_origin(
        source="synthetic",
        origin_losses={
            "cartoboost_lag": 1.0,
            "cartoboost_auto_forecast": 1.10,
            "shared_seasonal_base": 1.30,
        },
    )
    assert not benchmark.non_m_lag_dominates_origin(
        source="m4",
        origin_losses={
            "cartoboost_lag": 1.0,
            "cartoboost_auto_forecast": 1.25,
            "shared_seasonal_base": 1.30,
        },
    )

    calls = 0
    dates = [datetime(2026, 1, 1) + timedelta(days=day) for day in range(20)]
    train = pl.DataFrame(
        {
            "lane_id": ["A"] * len(dates),
            "date": dates,
            "loads": [0.0] * len(dates),
            "pickup_zone": [1.0] * len(dates),
            "dropoff_zone": [2.0] * len(dates),
            "distance_miles": [3.0] * len(dates),
            "airport_lane": [0.0] * len(dates),
            "pickup_borough_code": [1.0] * len(dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    def fake_roster(inner_train, horizon, *, model_names, **_kwargs):
        nonlocal calls
        calls += 1
        next_timestamp = inner_train.select(pl.col("date").max()).item() + timedelta(days=1)
        columns = {
            "series_id": ["A"],
            "timestamp": [next_timestamp],
            "horizon": [1],
            "cartoboost_lag": [1.0],
            "cartoboost_auto_forecast": [1.30],
        }
        return (
            pl.DataFrame(columns).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"models": {}},
        )

    monkeypatch.setattr(benchmark, "candidate_selection_forecast_roster", fake_roster)
    monkeypatch.setattr(
        benchmark,
        "add_shared_candidate_columns",
        lambda _train, _horizon, **kwargs: kwargs["predictions"],
    )

    scores = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert calls == 1
    assert len(scores["cartoboost_lag"]) == 1


def test_forecasting_benchmark_validation_scores_reuse_cutoff_cache(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_validation_cache",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls = 0
    dates = [datetime(2026, 1, 1) + timedelta(days=day) for day in range(20)]
    train = pl.DataFrame(
        {
            "lane_id": ["A"] * len(dates),
            "date": dates,
            "loads": [1.0] * len(dates),
            "pickup_zone": [1.0] * len(dates),
            "dropoff_zone": [2.0] * len(dates),
            "distance_miles": [3.0] * len(dates),
            "airport_lane": [0.0] * len(dates),
            "pickup_borough_code": [1.0] * len(dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    def fake_roster(inner_train, horizon, *, model_names, **_kwargs):
        nonlocal calls
        calls += 1
        next_timestamp = inner_train.select(pl.col("date").max()).item() + timedelta(days=1)
        columns = {
            "series_id": ["A"],
            "timestamp": [next_timestamp],
            "horizon": [1],
            "cartoboost_lag": [1.0],
        }
        if "cartoboost_auto_forecast" in model_names:
            columns["cartoboost_auto_forecast"] = [1.0]
        return (
            pl.DataFrame(columns).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"models": {}},
        )

    monkeypatch.setattr(benchmark, "candidate_selection_forecast_roster", fake_roster)
    cache: dict[object, dict[str, float]] = {}

    first = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        validation_cache=cache,
    )
    first_calls = calls
    second = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        validation_cache=cache,
    )

    assert second == first
    assert calls == first_calls
    assert benchmark.count_validation_cache_hits(cache) == pytest.approx(2.0)
    assert benchmark.count_validation_cache_misses(cache) == pytest.approx(2.0)


def test_forecasting_benchmark_non_m_outer_raw_auto_is_lazy(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lazy_outer_auto",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamp = datetime(2026, 1, 3)
    calls: list[str] = []

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(prediction_col)
        return (
            pl.DataFrame(
                {
                    "series_id": ["A"],
                    "timestamp": [timestamp],
                    "horizon": [1],
                    prediction_col: [1.0],
                }
            ).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"total_seconds": 0.01},
        )

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)

    predictions, timing = benchmark.forecast_model_roster(
        pl.DataFrame(
            {
                "lane_id": ["A", "A"],
                "date": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
                "loads": [1.0, 2.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("us"))),
        1,
        season_length=7,
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="synthetic",
        skip_non_m_raw_auto_candidate=True,
    )

    assert calls == ["cartoboost_lag"]
    assert predictions["cartoboost_auto_forecast"].to_list() == [0.0]
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "non_m_outer_lazy_raw_auto_candidate"
    )


def test_forecasting_benchmark_materializes_lazy_raw_auto_when_selected(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lazy_outer_auto_materialize",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    table = pl.DataFrame(
        {
            "lane_id": ["A", "A", "A"],
            "date": [start, start + timedelta(days=1), start + timedelta(days=2)],
            "loads": [1.0, 1.0, 2.0],
            "pickup_zone": [1.0, 1.0, 1.0],
            "dropoff_zone": [2.0, 2.0, 2.0],
            "distance_miles": [3.0, 3.0, 3.0],
            "airport_lane": [0.0, 0.0, 0.0],
            "pickup_borough_code": [1.0, 1.0, 1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    prediction_timestamp = start + timedelta(days=2)

    def frame(column: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["A"],
                "timestamp": [prediction_timestamp],
                "horizon": [1],
                column: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_roster(*_args, **_kwargs):
        return (
            frame("cartoboost_lag", 1.0).join(
                frame("cartoboost_auto_forecast", 0.0),
                on=["series_id", "timestamp", "horizon"],
            ),
            {
                "models": {
                    "cartoboost_lag": {"total_seconds": 0.01},
                    "cartoboost_auto_forecast": {
                        "selector_mode": "non_m_outer_lazy_raw_auto_candidate",
                        "total_seconds": 0.0,
                    },
                }
            },
        )

    def fake_selection(_train, _horizon, *, raw_predictions, **_kwargs):
        return raw_predictions, {
            "calibration_seconds": 0.0,
            "inner_origin_count": 1.0,
            "selected_candidates": {
                "cartoboost_lag": "cartoboost_lag",
                "cartoboost_auto_forecast": "cartoboost_auto_forecast",
            },
        }

    def fake_auto(*_args, prediction_col: str, **_kwargs):
        return frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "forecast_model_roster", fake_roster)
    monkeypatch.setattr(benchmark, "apply_shared_candidate_selection", fake_selection)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto)

    metrics, _quality, timing, _scored = benchmark.score_models(
        table,
        horizon=1,
        season_length=7,
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="synthetic",
        candidate_selection=True,
    )

    assert metrics["cartoboost_auto_forecast"]["rmse"] == pytest.approx(0.0)
    assert metrics["cartoboost_lag"]["rmse"] == pytest.approx(1.0)
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "non_m_outer_lazy_raw_auto_selected"
    )


def test_forecasting_benchmark_non_m_inner_selection_skips_raw_auto(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lazy_inner_auto",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[dict[str, object]] = []

    def fake_forecast_model_roster(*_args, **kwargs):
        calls.append(kwargs)
        return "predictions", {"models": {}}

    monkeypatch.setattr(benchmark, "forecast_model_roster", fake_forecast_model_roster)

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        "train",
        7,
        season_length=7,
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="synthetic",
    )

    assert predictions == "predictions"
    assert timing == {"models": {}}
    assert calls == [
        {
            "season_length": 7,
            "cartoboost_config": {},
            "model_names": ["cartoboost_lag", "cartoboost_auto_forecast"],
            "source": "synthetic",
            "skip_non_m_raw_auto_candidate": True,
        }
    ]


def test_forecasting_benchmark_m4_suite_keeps_group_order(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_serial_suite",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    groups = ["Hourly", "Daily", "Weekly"]
    monkeypatch.setattr(benchmark, "M4_GROUPS", groups)
    monkeypatch.setattr(
        benchmark,
        "load_m4_fixture",
        lambda group_args: (
            group_args.m4_group,
            {
                "group": group_args.m4_group,
                "horizon": 1,
                "season_length": 1,
            },
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "canonical_dataset_hash",
        lambda table: f"hash-{table}",
    )
    monkeypatch.setattr(
        benchmark,
        "score_models",
        lambda table, **_kwargs: (
            {
                "cartoboost_auto_forecast": {
                    "rmse": 1.0,
                    "mae": 1.0,
                    "wape": 1.0,
                },
                "cartoboost_lag": {
                    "rmse": 2.0,
                    "mae": 2.0,
                    "wape": 2.0,
                },
            },
            {"winner": "cartoboost_auto_forecast"},
            {"models": {}, "candidate_selection": {"calibration_seconds": 0.0}},
            None,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_objective_artifacts",
        lambda source, **_kwargs: {
            "primary_metric": "owa_proxy",
            source: {
                "models": {
                    "cartoboost_auto_forecast": {
                        "mase": 1.0,
                        "mase_ratio_to_seasonal_naive": 1.0,
                        "owa_proxy": 1.0,
                        "smape": 1.0,
                        "smape_ratio_to_seasonal_naive": 1.0,
                    },
                    "cartoboost_lag": {
                        "mase": 2.0,
                        "mase_ratio_to_seasonal_naive": 2.0,
                        "owa_proxy": 2.0,
                        "smape": 2.0,
                        "smape_ratio_to_seasonal_naive": 2.0,
                    },
                },
                "ranking": ["cartoboost_auto_forecast", "cartoboost_lag"],
            },
        },
    )

    output = tmp_path / "m4_suite.json"
    args = types.SimpleNamespace(
        output=output,
        source="m4",
        m4_suite=True,
        m4_group="Hourly",
        m4_series_limit=96,
        model_roster="cartoboost",
        no_hyperopt=True,
        no_candidate_selection=False,
        seed=42,
    )

    assert benchmark.run_m4_suite(args, {"n_estimators": 1}, benchmark.perf_counter()) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"]["groups"] == groups
    assert set(payload["groups"]) == set(groups)
    assert set(payload["timing"]["groups"]) == set(groups)
    assert payload["groups"]["Hourly"]["official_metrics"]["primary_metric"] == "owa_proxy"
    assert payload["official_metrics"]["primary_metric"] == "mean_owa_proxy"
    assert payload["official_metrics"]["m4"]["ranking"] == [
        "cartoboost_auto_forecast",
        "cartoboost_lag",
    ]


def test_forecasting_benchmark_m1_suite_keeps_group_order(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m1_serial_suite",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    groups = ["Yearly", "Monthly"]
    monkeypatch.setattr(benchmark, "M1_GROUPS", groups)
    monkeypatch.setattr(
        benchmark,
        "load_m1_fixture",
        lambda group_args: (
            group_args.m1_group,
            {
                "group": group_args.m1_group,
                "horizon": 1,
                "season_length": 1,
                "tsf_file": str(tmp_path / f"{group_args.m1_group}.tsf"),
            },
        ),
    )
    monkeypatch.setattr(benchmark, "canonical_dataset_hash", lambda table: f"hash-{table}")
    monkeypatch.setattr(benchmark, "source_file_hashes", lambda _dataset: {})
    monkeypatch.setattr(
        benchmark,
        "score_models",
        lambda table, **_kwargs: (
            {
                "cartoboost_auto_forecast": {
                    "rmse": 1.0,
                    "mae": 1.0,
                    "wape": 1.0,
                },
                "cartoboost_lag": {
                    "rmse": 2.0,
                    "mae": 2.0,
                    "wape": 2.0,
                },
            },
            {"winner": "cartoboost_auto_forecast"},
            {"models": {}, "candidate_selection": {"calibration_seconds": 0.0}},
            None,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_objective_artifacts",
        lambda source, **_kwargs: {
            "primary_metric": "owa_proxy",
            source: {
                "models": {
                    "cartoboost_auto_forecast": {
                        "mase": 1.0,
                        "mase_ratio_to_seasonal_naive": 1.0,
                        "owa_proxy": 1.0,
                        "smape": 1.0,
                        "smape_ratio_to_seasonal_naive": 1.0,
                    },
                    "cartoboost_lag": {
                        "mase": 2.0,
                        "mase_ratio_to_seasonal_naive": 2.0,
                        "owa_proxy": 2.0,
                        "smape": 2.0,
                        "smape_ratio_to_seasonal_naive": 2.0,
                    },
                },
                "ranking": ["cartoboost_auto_forecast", "cartoboost_lag"],
            },
        },
    )

    output = tmp_path / "m1_suite.json"
    args = types.SimpleNamespace(
        output=output,
        source="m1",
        m1_suite=True,
        m1_group="Yearly",
        m1_series_limit=96,
        model_roster="cartoboost",
        no_hyperopt=True,
        no_candidate_selection=False,
        seed=42,
    )

    assert benchmark.run_m1_suite(args, {"n_estimators": 1}, benchmark.perf_counter()) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"]["groups"] == groups
    assert set(payload["groups"]) == set(groups)
    assert payload["groups"]["Yearly"]["official_metrics"]["primary_metric"] == "owa_proxy"
    assert payload["official_metrics"]["primary_metric"] == "mean_owa_proxy"
    assert payload["official_metrics"]["m1"]["ranking"] == [
        "cartoboost_auto_forecast",
        "cartoboost_lag",
    ]


def test_forecasting_benchmark_m3_suite_keeps_group_order(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m3_serial_suite",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    groups = ["Yearly", "Monthly"]
    monkeypatch.setattr(benchmark, "M3_GROUPS", groups)
    monkeypatch.setattr(
        benchmark,
        "load_m3_fixture",
        lambda group_args: (
            group_args.m3_group,
            {
                "group": group_args.m3_group,
                "horizon": 1,
                "season_length": 1,
            },
        ),
    )
    monkeypatch.setattr(benchmark, "canonical_dataset_hash", lambda table: f"hash-{table}")
    monkeypatch.setattr(
        benchmark,
        "score_models",
        lambda table, **_kwargs: (
            {
                "cartoboost_auto_forecast": {
                    "rmse": 1.0,
                    "mae": 1.0,
                    "wape": 1.0,
                },
                "cartoboost_lag": {
                    "rmse": 2.0,
                    "mae": 2.0,
                    "wape": 2.0,
                },
            },
            {"winner": "cartoboost_auto_forecast"},
            {"models": {}, "candidate_selection": {"calibration_seconds": 0.0}},
            None,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_objective_artifacts",
        lambda source, **_kwargs: {
            "primary_metric": "owa_proxy",
            source: {
                "models": {
                    "cartoboost_auto_forecast": {
                        "mase": 1.0,
                        "mase_ratio_to_seasonal_naive": 1.0,
                        "owa_proxy": 1.0,
                        "smape": 1.0,
                        "smape_ratio_to_seasonal_naive": 1.0,
                    },
                    "cartoboost_lag": {
                        "mase": 2.0,
                        "mase_ratio_to_seasonal_naive": 2.0,
                        "owa_proxy": 2.0,
                        "smape": 2.0,
                        "smape_ratio_to_seasonal_naive": 2.0,
                    },
                },
                "ranking": ["cartoboost_auto_forecast", "cartoboost_lag"],
            },
        },
    )

    output = tmp_path / "m3_suite.json"
    args = types.SimpleNamespace(
        output=output,
        source="m3",
        m3_suite=True,
        m3_group="Yearly",
        m3_series_limit=96,
        model_roster="cartoboost",
        no_hyperopt=True,
        no_candidate_selection=False,
        seed=42,
    )

    assert benchmark.run_m3_suite(args, {"n_estimators": 1}, benchmark.perf_counter()) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"]["groups"] == groups
    assert set(payload["groups"]) == set(groups)
    assert payload["groups"]["Yearly"]["official_metrics"]["primary_metric"] == "owa_proxy"
    assert payload["official_metrics"]["primary_metric"] == "mean_owa_proxy"
    assert payload["official_metrics"]["m3"]["ranking"] == [
        "cartoboost_auto_forecast",
        "cartoboost_lag",
    ]


def test_forecasting_benchmark_m_series_suite_ranking_requires_group_coverage():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_suite_coverage",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    def row(owa: float) -> dict[str, float]:
        return {
            "mase": owa,
            "mase_ratio_to_seasonal_naive": owa,
            "owa_proxy": owa,
            "smape": owa,
            "smape_ratio_to_seasonal_naive": owa,
        }

    artifact = benchmark.aggregate_m_series_suite_official_metrics(
        "m4",
        ["Yearly", "Monthly"],
        {
            "Yearly": {
                "official_metrics": {
                    "m4": {
                        "models": {
                            "complete_model": row(1.0),
                            "partial_model": row(0.1),
                        }
                    }
                }
            },
            "Monthly": {
                "official_metrics": {
                    "m4": {
                        "models": {
                            "complete_model": row(1.2),
                        }
                    }
                }
            },
        },
    )

    m4 = artifact["m4"]
    assert m4["ranking_scope"] == "complete_group_coverage"
    assert m4["ranking"] == ["complete_model"]
    assert m4["models"]["complete_model"]["complete_group_coverage"]
    assert not m4["models"]["partial_model"]["complete_group_coverage"]
    assert m4["incomplete_models"] == {"partial_model": ["Monthly"]}
    assert m4["incomplete_ranking"] == ["partial_model"]


def test_forecasting_m4_wrapper_runs_committed_suite(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_m4.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_m4_wrapper",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)

    captured: dict[str, list[str]] = {}

    def fake_call(cmd):
        captured["cmd"] = cmd
        return 0

    output = tmp_path / "m4.json"
    monkeypatch.setattr(wrapper.subprocess, "call", fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forecasting_m4.py",
            "--committed",
            "--no-hyperopt",
            "--output",
            str(output),
        ],
    )

    assert wrapper.main() == 0
    cmd = captured["cmd"]
    assert "--m4-suite-workers" not in cmd
    assert "--m4-suite" in cmd
    assert cmd[cmd.index("--output") + 1] == str(output)


def test_forecasting_benchmark_emits_native_backed_m_series_owa_artifact():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m_series_owa_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = date(2026, 1, 1)
    table = pl.DataFrame(
        [
            {
                "lane_id": lane,
                "date": start + timedelta(days=offset),
                "loads": value,
                "pickup_zone": pickup_zone,
                "dropoff_zone": pickup_zone + 10.0,
                "distance_miles": 1.0,
                "airport_lane": 0.0,
                "pickup_borough_code": pickup_zone,
            }
            for lane, (values, pickup_zone) in {
                "series_a": ([10.0, 12.0, 14.0], 1.0),
                "series_b": ([20.0, 23.0, 26.0], 2.0),
            }.items()
            for offset, value in enumerate(values)
        ]
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    scored = pl.DataFrame(
        [
            {
                "series_id": "series_a",
                "timestamp": start + timedelta(days=3),
                "horizon": 1,
                "actual": 16.0,
                "cartoboost_lag": 14.0,
                "cartoboost_auto_forecast": 16.0,
            },
            {
                "series_id": "series_b",
                "timestamp": start + timedelta(days=3),
                "horizon": 1,
                "actual": 29.0,
                "cartoboost_lag": 26.0,
                "cartoboost_auto_forecast": 29.0,
            },
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.benchmark_objective_artifacts(
        "m4",
        train_table=table,
        scored=scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        season_length=1,
    )

    m4 = artifact["m4"]
    assert artifact["primary_metric"] == "owa_proxy"
    assert m4["ranking"] == ["cartoboost_auto_forecast", "cartoboost_lag"]
    assert m4["baseline"]["mase"] == pytest.approx(1.0)
    assert m4["models"]["cartoboost_auto_forecast"]["owa_proxy"] == pytest.approx(0.0)
    assert m4["models"]["cartoboost_lag"]["owa_proxy"] == pytest.approx(1.0)


def test_forecasting_benchmark_metrics_include_r2():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_metrics",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 4,
            "date": [date(2026, 1, day) for day in range(1, 5)],
            "loads": [1.0, 2.0, 3.0, 4.0],
        }
    )
    scored = pl.DataFrame(
        {
            "series_id": ["a", "a", "a"],
            "timestamp": [
                datetime(2026, 1, 5),
                datetime(2026, 1, 6),
                datetime(2026, 1, 7),
            ],
            "horizon": [1, 2, 3],
            "actual": [1.0, 2.0, 3.0],
            "forecast": [1.0, 2.0, 4.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    metrics = benchmark.evaluate_metrics(scored, "forecast", train, season_length=1)

    assert metrics["rmse"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert metrics["r2"] == pytest.approx(0.5)


def test_forecasting_benchmark_r2_constant_target_is_finite():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_constant_r2",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 4,
            "date": [date(2026, 1, day) for day in range(1, 5)],
            "loads": [2.0, 2.0, 2.0, 2.0],
        }
    )
    scored = pl.DataFrame(
        {
            "series_id": ["a", "a"],
            "timestamp": [datetime(2026, 1, 5), datetime(2026, 1, 6)],
            "horizon": [1, 2],
            "actual": [2.0, 2.0],
            "perfect": [2.0, 2.0],
            "miss": [1.0, 3.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    perfect = benchmark.evaluate_metrics(scored, "perfect", train, season_length=1)
    miss = benchmark.evaluate_metrics(scored, "miss", train, season_length=1)

    assert perfect["r2"] == pytest.approx(1.0)
    assert miss["r2"] == pytest.approx(0.0)


def test_forecasting_benchmark_robust_selector_prefers_simple_close_candidate():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_robust_selector",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.candidate_selection_model_names(
        [
            "cartoboost_lag",
            "cartoboost_auto_forecast",
            "statsforecast_autoarima",
        ]
    ) == ["cartoboost_lag", "cartoboost_auto_forecast"]
    assert (
        benchmark.candidate_selection_model_names(["cartoboost_lag", "statsforecast_autoarima"])
        == []
    )

    selected = benchmark.robust_candidate_choice(
        {
            "cartoboost_auto_forecast": 1.00,
            "shared_seasonal_base": 1.03,
            "shared_calendar_dom": 0.99,
        }
    )

    assert selected == "shared_seasonal_base"
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_auto_forecast": 1.00,
                "shared_seasonal_base": 1.03,
                "shared_calendar_dom": 0.99,
            },
            source="m5",
        )
        == "shared_calendar_dom"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_calendar_elapsed_phase": 0.7025,
                "shared_elapsed_phase_total_reconciled_020": 0.7194,
                "shared_elapsed_phase_total_reconciled_035": 0.7142,
                "shared_elapsed_phase_total_reconciled_050": 0.7096,
                "shared_reconciled_autostats_blend": 0.7091,
                "shared_point_autostats_elapsed_phase_blend": 0.7187,
            },
            source="m5",
        )
        == "shared_elapsed_phase_total_reconciled_050"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_lag": 3.7045505738956943,
                "shared_point_autostats_elapsed_phase_blend": 3.640384191677357,
            },
            source="m5",
            inner_origin_count=1,
        )
        == "cartoboost_lag"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_auto_forecast": 0.1996633147694055,
                "shared_calendar_elapsed_phase": 0.19966331430819997,
                "cartoboost_lag": 0.19966331443943974,
                "shared_market_neutral_zero": 0.19966331426201722,
            },
            source="m6",
        )
        == "shared_market_neutral_zero"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_auto_forecast": 1.00,
                "shared_seasonal_base": 1.03,
                "shared_calendar_dom": 0.99,
            },
            source="synthetic",
        )
        == "shared_seasonal_base"
    )
    assert benchmark.native_auto_raw_candidate_is_confident(
        {
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_raw",
                "inner_raw_relative_rmse_gain": 0.50,
            }
        }
    )
    assert not benchmark.native_auto_raw_candidate_is_confident(
        {
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_raw",
                "inner_raw_relative_rmse_gain": 0.49,
            }
        }
    )
    assert not benchmark.native_auto_raw_candidate_is_confident(
        {
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_residual_blend",
                "inner_raw_relative_rmse_gain": 0.90,
            }
        }
    )


def test_forecasting_benchmark_keeps_confident_native_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_native_auto_confidence",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 5,
            "date": [start + timedelta(days=offset) for offset in range(5)],
            "loads": [10.0, 11.0, 12.0, 13.0, 14.0],
            "pickup_zone": [1] * 5,
            "dropoff_zone": [2] * 5,
            "distance_miles": [3.5] * 5,
            "airport_lane": [0] * 5,
            "pickup_borough_code": [4] * 5,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    raw_predictions = pl.DataFrame(
        {
            "series_id": ["a"],
            "timestamp": [start + timedelta(days=5)],
            "horizon": [1],
            "cartoboost_lag": [1.0],
            "cartoboost_auto_forecast": [2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_inner_scores(*_args, **_kwargs):
        return {
            "cartoboost_lag": [0.4, 0.4],
            "cartoboost_auto_forecast": [0.5, 0.5],
        }

    monkeypatch.setattr(benchmark, "shared_candidate_validation_scores", fake_inner_scores)

    selected, timing = benchmark.apply_shared_candidate_selection(
        train,
        1,
        season_length=7,
        source="synthetic",
        raw_predictions=raw_predictions,
        model_timing={
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_raw",
                "inner_raw_relative_rmse_gain": 0.75,
            }
        },
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert timing["native_auto_raw_keep"] is True
    assert timing["selected_candidates"]["cartoboost_auto_forecast"] == "cartoboost_lag"
    assert selected["cartoboost_auto_forecast"].to_list() == [1.0]


def test_forecasting_benchmark_autostats_candidate_targets_m_series_and_m5():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_autostats_gate",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.include_autostats_candidate(source="m1", season_length=12, horizon=18)
    assert benchmark.include_autostats_candidate(source="m3", season_length=4, horizon=8)
    assert benchmark.include_autostats_candidate(source="m4", season_length=12, horizon=18)
    assert benchmark.include_autostats_candidate(source="m4", season_length=4, horizon=8)
    assert not benchmark.include_autostats_candidate(source="m4", season_length=24, horizon=48)
    assert benchmark.requires_lag_spine(source="m4", season_length=1, horizon=14)
    assert benchmark.requires_lag_spine(source="m4", season_length=12, horizon=18)
    assert not benchmark.requires_lag_spine(source="m4", season_length=24, horizon=48)
    assert not benchmark.requires_lag_spine(source="m3", season_length=12, horizon=18)
    assert benchmark.include_autostats_candidate(source="m5", season_length=1, horizon=28)
    assert "cartoboost_autostats_bank" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m1",
    )
    assert "cartoboost_autostats_bank" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m3",
    )
    assert "shared_elapsed_phase_total_reconciled_035" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_point_autostats_elapsed_phase_blend" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_reconciled_autostats_blend" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_total_reconciled_auto" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_state_reconciled_auto" not in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_store_reconciled_auto" not in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_elapsed_phase_total_reconciled_035": 1.0,
                "shared_point_autostats_elapsed_phase_blend": 1.015,
            },
            source="m5",
        )
        == "shared_point_autostats_elapsed_phase_blend"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_elapsed_phase_total_reconciled_035": 1.0,
                "shared_point_autostats_elapsed_phase_blend": 1.016,
            },
            source="m5",
        )
        == "shared_elapsed_phase_total_reconciled_035"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_elapsed_phase_total_reconciled_035": 1.0,
                "shared_reconciled_autostats_blend": 1.0005,
            },
            source="m5",
            inner_origin_count=1,
        )
        == "shared_reconciled_autostats_blend"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_elapsed_phase_total_reconciled_035": 1.0,
                "shared_reconciled_autostats_blend": 1.0005,
            },
            source="m5",
            inner_origin_count=3,
        )
        == "shared_elapsed_phase_total_reconciled_035"
    )

    pl = pytest.importorskip("polars")
    train = pl.DataFrame({"loads": [0.0, 1.0, 10.0]})
    candidates = pl.DataFrame(
        {
            "cartoboost_lag": [1.0, 2.0],
            "shared_calendar_elapsed_phase": [3.0, 4.0],
            "shared_elapsed_phase_total_reconciled_050": [2.0e9, 4.0],
        }
    )
    assert (
        benchmark.stable_hierarchy_candidate_choice(
            train,
            candidates,
            candidate_scores_by_model={
                "shared_elapsed_phase_total_reconciled_050": 0.60,
                "shared_calendar_elapsed_phase": 0.70,
                "cartoboost_lag": 0.80,
            },
            selected_candidate="shared_elapsed_phase_total_reconciled_050",
            inner_origin_count=2,
        )
        == "shared_calendar_elapsed_phase"
    )


def test_forecasting_benchmark_shared_candidate_helpers_preserve_outputs():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_shared_candidate_helpers",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(8)]
    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 8 + ["b"] * 8,
            "date": [*dates, *dates],
            "loads": [*range(1, 9), *range(11, 19)],
            "pickup_zone": [1] * 16,
            "dropoff_zone": [2] * 16,
            "distance_miles": [3.5] * 16,
            "airport_lane": [0] * 16,
            "pickup_borough_code": [4] * 16,
            "weight_value": [10.0] * 16,
            "m5_sell_price": [1.25] * 16,
            "m5_event_name_1_code": [0.0] * 16,
            "m5_snap_CA": [1.0] * 16,
            "m5_state_code": [0.0] * 16,
            "m5_store_code": [0.0] * 16,
            "m5_cat_code": [0.0] * 16,
            "m5_dept_code": [0.0] * 16,
            "m5_item_code": [0.0] * 16,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    seasonal = benchmark.seasonal_naive_forecast_frame(
        train,
        2,
        season_length=7,
        prediction_col="prediction",
    )
    calendar = benchmark.calendar_profile_forecast_frame(
        train,
        2,
        prediction_col="prediction",
        mode="elapsed_phase",
        elapsed_phase_period=14,
    )
    trend = benchmark.trend_forecast_frame(
        train,
        2,
        season_length=7,
        prediction_col="prediction",
        mode="half_drift",
    )

    assert list(seasonal.select("series_id", "horizon", "prediction").iter_rows()) == [
        ("a", 1, 2.0),
        ("b", 1, 12.0),
        ("a", 2, 3.0),
        ("b", 2, 13.0),
    ]
    assert list(calendar.select("series_id", "horizon", "prediction").iter_rows()) == [
        ("a", 1, 2.0),
        ("b", 1, 12.0),
        ("a", 2, 3.0),
        ("b", 2, 13.0),
    ]
    assert list(trend.select("series_id", "horizon", "prediction").iter_rows()) == [
        ("a", 1, pytest.approx(8.5)),
        ("b", 1, pytest.approx(18.5)),
        ("a", 2, pytest.approx(8.875)),
        ("b", 2, pytest.approx(18.875)),
    ]

    candidate_frame = pl.DataFrame(
        {
            "series_id": ["a", "b"],
            "timestamp": [start + timedelta(days=8), start + timedelta(days=8)],
            "horizon": [1, 1],
            "base": [2.0, 3.0],
            "target": [6.0, 4.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    reconciled = benchmark.add_hierarchical_elapsed_phase_total_reconciled_candidates(
        candidate_frame,
        base_col="base",
        target_col="target",
    ).sort("series_id")

    assert reconciled["shared_elapsed_phase_total_reconciled_020"].to_list() == pytest.approx(
        [2.4, 3.6]
    )
    assert reconciled["shared_elapsed_phase_total_reconciled_050"].to_list() == pytest.approx(
        [3.0, 4.5]
    )

    m6_anchor = seasonal.select("series_id", "timestamp", "horizon")
    m6_candidates = benchmark.add_shared_candidate_columns(
        train,
        2,
        season_length=7,
        predictions=m6_anchor,
        source="m6",
        required_columns={"shared_elapsed_phase_rank_blend"},
    ).sort(["horizon", "series_id"])

    assert m6_candidates["shared_elapsed_phase_rank_blend"].to_list() == pytest.approx(
        [2.0, 12.0, 3.0, 13.0]
    )


def test_forecasting_benchmark_future_feature_builder_preserves_outputs():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_future_feature_builder",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    history_dates = [start + timedelta(days=offset) for offset in range(8)]
    history = pl.DataFrame(
        {
            "lane_id": ["a"] * 8 + ["b"] * 8,
            "date": [*history_dates, *history_dates],
            "loads": [*range(1, 9), *range(11, 19)],
            "pickup_zone": [1] * 8 + [9] * 8,
            "dropoff_zone": [2] * 8 + [8] * 8,
            "distance_miles": [3.5] * 8 + [7.5] * 8,
            "airport_lane": [0] * 16,
            "pickup_borough_code": [4] * 8 + [6] * 8,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    future = pl.DataFrame(
        {
            "lane_id": ["a", "b"],
            "date": [start + timedelta(days=8), start + timedelta(days=8)],
            "pickup_zone": [1, 9],
            "dropoff_zone": [2, 8],
            "distance_miles": [3.5, 7.5],
            "airport_lane": [0, 0],
            "pickup_borough_code": [4, 6],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    features = benchmark.build_future_features(history, future, season_length=4).sort("lane_id")

    rows = features.select(
        "lane_id",
        "loads_lag_1",
        "loads_lag_3",
        "loads_lag_4",
        "loads_lag_8",
        "loads_roll_7",
        "loads_roll_4",
        "loads_delta_lag_3",
        "loads_delta_lag_4",
        "loads_roll_trend_4",
        "date_dayofweek",
        "date_day",
        "date_dayofyear",
        "date_month",
        "date_elapsed_days",
    ).iter_rows(named=True)
    assert list(rows) == [
        {
            "lane_id": "a",
            "loads_lag_1": 8.0,
            "loads_lag_3": 6.0,
            "loads_lag_4": 5.0,
            "loads_lag_8": 1.0,
            "loads_roll_7": pytest.approx(5.0),
            "loads_roll_4": pytest.approx(6.5),
            "loads_delta_lag_3": 2.0,
            "loads_delta_lag_4": 3.0,
            "loads_roll_trend_4": pytest.approx(4.0),
            "date_dayofweek": 5.0,
            "date_day": 9.0,
            "date_dayofyear": 9.0,
            "date_month": 1.0,
            "date_elapsed_days": 8.0,
        },
        {
            "lane_id": "b",
            "loads_lag_1": 18.0,
            "loads_lag_3": 16.0,
            "loads_lag_4": 15.0,
            "loads_lag_8": 11.0,
            "loads_roll_7": pytest.approx(15.0),
            "loads_roll_4": pytest.approx(16.5),
            "loads_delta_lag_3": 2.0,
            "loads_delta_lag_4": 3.0,
            "loads_roll_trend_4": pytest.approx(4.0),
            "date_dayofweek": 5.0,
            "date_day": 9.0,
            "date_dayofyear": 9.0,
            "date_month": 1.0,
            "date_elapsed_days": 8.0,
        },
    ]


def test_forecasting_benchmark_m5_known_future_features_use_future_calendar():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_known_future_features",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    history = pl.DataFrame(
        {
            "lane_id": ["a"],
            "date": [start],
            "loads": [1.0],
            "pickup_zone": [1.0],
            "dropoff_zone": [2.0],
            "distance_miles": [3.0],
            "airport_lane": [0.0],
            "pickup_borough_code": [4.0],
            "m5_event_name_1_code": [0.0],
            "m5_snap_CA": [0.0],
            "m5_sell_price": [2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    known_future = pl.DataFrame(
        {
            "lane_id": ["a"],
            "date": [start + timedelta(days=1)],
            "m5_event_name_1_code": [3.0],
            "m5_snap_CA": [1.0],
            "m5_sell_price": [2.5],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    future = benchmark.next_future_rows(history, known_future=known_future)
    features = benchmark.build_future_features(history, future, season_length=1)

    assert future.select("m5_event_name_1_code", "m5_snap_CA", "m5_sell_price").row(0) == (
        3.0,
        1.0,
        2.5,
    )
    assert {
        "m5_event_name_1_code",
        "m5_snap_CA",
        "m5_sell_price",
    } <= set(benchmark.benchmark_exogenous_feature_columns(features))


def test_forecasting_benchmark_external_tree_uses_m5_known_future_features():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_known_future_tree",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(8)]
    history = pl.DataFrame(
        {
            "lane_id": ["a"] * 8,
            "date": dates,
            "loads": list(range(1, 9)),
            "pickup_zone": [1.0] * 8,
            "dropoff_zone": [2.0] * 8,
            "distance_miles": [3.0] * 8,
            "airport_lane": [0.0] * 8,
            "pickup_borough_code": [4.0] * 8,
            "m5_event_name_1_code": [0.0] * 8,
            "m5_snap_CA": [0.0] * 8,
            "m5_sell_price": [2.0] * 8,
            "weight_value": [2.0] * 8,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    known_future = pl.DataFrame(
        {
            "lane_id": ["a"],
            "date": [start + timedelta(days=8)],
            "m5_event_name_1_code": [4.0],
            "m5_snap_CA": [1.0],
            "m5_sell_price": [2.75],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    feature_columns = benchmark.select_cartoboost_feature_columns(
        benchmark.build_history_features(history, season_length=1),
        season_length=1,
    )
    captured: dict[str, object] = {}

    class FakeModel:
        def fit(self, x, y):
            captured["fit_shape"] = x.shape

        def predict(self, x):
            captured["predict_row"] = x[0]
            return [9.0]

    forecast, _timing = benchmark.external_tree_lag_forecast(
        history,
        1,
        season_length=1,
        model=FakeModel(),
        prediction_col="fake_tree",
        known_future=known_future,
    )

    predicted_row = captured["predict_row"]
    assert predicted_row[feature_columns.index("m5_event_name_1_code")] == pytest.approx(4.0)
    assert predicted_row[feature_columns.index("m5_snap_CA")] == pytest.approx(1.0)
    assert predicted_row[feature_columns.index("m5_sell_price")] == pytest.approx(2.75)
    assert forecast["fake_tree"].to_list() == [9.0]


def test_forecasting_benchmark_cartoboost_uses_m5_known_future_features(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_known_future_cartoboost",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    history = pl.DataFrame(
        {
            "lane_id": ["a"] * 8,
            "date": [start + timedelta(days=offset) for offset in range(8)],
            "loads": list(range(1, 9)),
            "m5_state_code": [1.0] * 8,
            "m5_store_code": [2.0] * 8,
            "m5_cat_code": [3.0] * 8,
            "m5_dept_code": [4.0] * 8,
            "m5_item_code": [5.0] * 8,
            "m5_event_name_1_code": [0.0] * 8,
            "m5_snap_CA": [0.0] * 8,
            "m5_sell_price": [2.0] * 8,
            "weight_value": [2.0] * 8,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    known_future = pl.DataFrame(
        {
            "lane_id": ["a"],
            "date": [start + timedelta(days=8)],
            "m5_event_name_1_code": [4.0],
            "m5_snap_CA": [1.0],
            "m5_sell_price": [2.75],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    captured: dict[str, object] = {}

    class FakeResult:
        def predictions(self):
            return [("a", "2026-01-09T00:00:00", 1, "fake", 9.0)]

    class FakeCartoBoostLagForecaster:
        metadata_ = {"feature_names": ["target_lag_1", "m5_sell_price"]}

        def __init__(self, **params):
            captured["params"] = params

        def fit(self, frame):
            captured["fit_columns"] = list(frame.columns)
            return self

        def predict(self, horizon, *, known_future=None):
            captured["horizon"] = horizon
            captured["known_future_columns"] = (
                [] if known_future is None else list(known_future.columns)
            )
            captured["known_future_row"] = (
                None if known_future is None else known_future.iloc[0].to_dict()
            )
            return FakeResult()

    monkeypatch.setattr(
        benchmark,
        "CartoBoostLagForecaster",
        FakeCartoBoostLagForecaster,
    )
    config = benchmark.cartoboost_source_config(
        {
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        source="m5",
    )

    forecast, _timing = benchmark.cartoboost_raw_forecast(
        history,
        1,
        season_length=1,
        config=config,
        prediction_col="cartoboost_lag",
        known_future=known_future,
    )

    assert "m5_store_code" in captured["fit_columns"]
    assert "m5_sell_price" in captured["params"]["covariate_features"]
    assert "m5_store_code" in captured["known_future_columns"]
    assert "m5_sell_price" in captured["known_future_columns"]
    assert captured["known_future_row"]["m5_store_code"] == pytest.approx(2.0)
    assert captured["known_future_row"]["m5_sell_price"] == pytest.approx(2.75)
    assert forecast["cartoboost_lag"].to_list() == [9.0]


def test_forecasting_benchmark_m5_selector_uses_raw_auto_without_nested_calibration(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_selector_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["PU1->DO1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 99.0}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 3.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"],
            "date": [timestamp],
            "loads": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m5",
    )

    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_autostats_bank" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m5_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_m4_selector_scores_autostats_candidate(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_selector_autostats",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["PU1->DO1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 99.0}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 3.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"],
            "date": [timestamp],
            "loads": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        8,
        season_length=4,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m4",
    )

    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("raw", "cartoboost_lag") in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_autostats_bank" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m4_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_outer_candidates_only_build_required_columns():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_required_outer_columns",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamp = datetime(2026, 1, 15)
    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"] * 7,
            "date": [datetime(2026, 1, day) for day in range(1, 8)],
            "loads": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    predictions = pl.DataFrame(
        {
            "series_id": ["PU1->DO1"],
            "timestamp": [timestamp],
            "horizon": [1],
            "cartoboost_lag": [3.0],
            "cartoboost_auto_forecast": [2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    selected = benchmark.add_shared_candidate_columns(
        train,
        1,
        season_length=7,
        predictions=predictions,
        source="m5",
        required_columns={"cartoboost_lag"},
    )

    assert selected.columns == predictions.columns


def test_forecasting_benchmark_m5_required_blend_builds_dependencies():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_required_dependencies",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamp = datetime(2026, 1, 15)
    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"] * 14,
            "date": [datetime(2026, 1, day) for day in range(1, 15)],
            "loads": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0] * 2,
            "pickup_zone": [1.0] * 14,
            "dropoff_zone": [2.0] * 14,
            "distance_miles": [3.0] * 14,
            "airport_lane": [0.0] * 14,
            "pickup_borough_code": [4.0] * 14,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    predictions = pl.DataFrame(
        {
            "series_id": ["PU1->DO1"],
            "timestamp": [timestamp],
            "horizon": [1],
            "cartoboost_auto_forecast": [2.0],
            "cartoboost_autostats_bank": [3.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    selected = benchmark.add_shared_candidate_columns(
        train,
        1,
        season_length=7,
        predictions=predictions,
        source="m5",
        required_columns={"shared_calendar_autostats_blend"},
    )

    assert "shared_calendar_elapsed_phase" in selected.columns
    assert "shared_calendar_autostats_blend" in selected.columns
    assert selected.height == 1


def test_forecasting_benchmark_m5_total_reconciled_does_not_require_calendar_blend():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_total_required_dependencies",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamp = datetime(2026, 1, 15)
    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"] * 14,
            "date": [datetime(2026, 1, day) for day in range(1, 15)],
            "loads": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0] * 2,
            "pickup_zone": [1.0] * 14,
            "dropoff_zone": [2.0] * 14,
            "distance_miles": [3.0] * 14,
            "airport_lane": [0.0] * 14,
            "pickup_borough_code": [4.0] * 14,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    predictions = pl.DataFrame(
        {
            "series_id": ["PU1->DO1"],
            "timestamp": [timestamp],
            "horizon": [1],
            "cartoboost_auto_forecast": [2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    selected = benchmark.add_shared_candidate_columns(
        train,
        1,
        season_length=7,
        predictions=predictions,
        source="m5",
        required_columns={"shared_total_reconciled_auto"},
    )

    assert "shared_calendar_elapsed_phase" not in selected.columns
    assert "shared_total_reconciled_auto" in selected.columns
    assert selected.height == 1


def test_forecasting_benchmark_m6_point_candidate_uses_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_point_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["AAA"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["AAA"],
            "date": [timestamp],
            "loads": [0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.forecast_model_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m6",
    )

    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("raw", "cartoboost_auto_forecast") in calls
    assert ("auto", "cartoboost_point_auto") not in calls
    assert ("raw", "cartoboost_point_auto") not in calls
    assert "cartoboost_point_auto" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m6_raw_auto_outer_no_nested_calibration"
    )


def test_forecasting_benchmark_m5_outer_selection_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_outer_selection_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["FOODS_1_CA_1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 99.0}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 3.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["FOODS_1_CA_1"],
            "date": [timestamp],
            "loads": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.forecast_model_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m5",
        skip_m5_raw_auto_candidate=True,
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_autostats_bank" in predictions.columns
    assert predictions["cartoboost_auto_forecast"].to_list() == [3.0]
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m5_outer_maps_auto_to_autostats_candidate"
    )


def test_forecasting_benchmark_m6_outer_selection_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_outer_selection_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["AAA"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["AAA"],
            "date": [timestamp],
            "loads": [0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.forecast_model_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m6",
        skip_m6_raw_auto_candidate=True,
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_point_auto" not in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m6_outer_skips_raw_auto_candidate"
    )


def test_forecasting_benchmark_m4_inner_validation_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_inner_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["M4_Monthly_1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["M4_Monthly_1"],
            "date": [timestamp],
            "loads": [100.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=12,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m4",
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_lag" in predictions.columns
    assert "cartoboost_autostats_bank" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m4_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_m6_inner_validation_includes_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_inner_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["AAA"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["AAA"],
            "date": [timestamp],
            "loads": [0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m6",
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") in calls
    assert "cartoboost_lag" in predictions.columns
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_point_auto" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m6_inner_validation_includes_raw_auto"
    )


def test_forecasting_benchmark_m6_selection_keeps_raw_auto_on_tiny_rps_gain(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_raw_auto_guard",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    train = pl.DataFrame(
        {
            "lane_id": ["AAA"] * 5,
            "date": [start + timedelta(days=offset) for offset in range(5)],
            "loads": [0.01, -0.01, 0.02, -0.02, 0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    raw_predictions = pl.DataFrame(
        {
            "series_id": ["AAA"],
            "timestamp": [start + timedelta(days=5)],
            "horizon": [1],
            "cartoboost_lag": [0.02],
            "cartoboost_auto_forecast": [0.03],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_scores(*_args, **_kwargs):
        return {
            "cartoboost_auto_forecast": [0.2000000000],
            "cartoboost_lag": [0.1999999995],
        }

    monkeypatch.setattr(benchmark, "shared_candidate_validation_scores", fake_scores)

    selected, timing = benchmark.apply_shared_candidate_selection(
        train,
        1,
        season_length=1,
        source="m6",
        raw_predictions=raw_predictions,
        cartoboost_config={"n_estimators": 1},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert timing["selected_candidates"]["cartoboost_auto_forecast"] == "cartoboost_auto_forecast"
    assert timing["m6_raw_auto_guarded"]["cartoboost_auto_forecast"]["candidate"] == (
        "cartoboost_lag"
    )
    assert selected["cartoboost_auto_forecast"].to_list() == [pytest.approx(0.03)]


def test_forecasting_benchmark_m5_inner_validation_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_inner_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["FOODS_1_CA_1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["FOODS_1_CA_1"],
            "date": [timestamp],
            "loads": [1.0],
            "pickup_zone": [1.0],
            "dropoff_zone": [1.0],
            "distance_miles": [1.0],
            "airport_lane": [0.0],
            "pickup_borough_code": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m5",
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_autostats_bank" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m5_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_static_covariates_are_source_gated():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_covariate_gate",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    base = {
        "n_estimators": 1,
        "learning_rate": 0.06,
        "max_depth": 2,
        "min_samples_leaf": 1,
    }

    synthetic = benchmark.cartoboost_source_config(base, source="synthetic")
    taxi = benchmark.cartoboost_source_config(base, source="nyc-taxi")
    m1 = benchmark.cartoboost_source_config(base, source="m1")
    m3 = benchmark.cartoboost_source_config(base, source="m3")
    m4 = benchmark.cartoboost_source_config(base, source="m4")
    m5 = benchmark.cartoboost_source_config(base, source="m5")
    m6 = benchmark.cartoboost_source_config(base, source="m6")

    assert synthetic["use_static_covariates"]
    assert synthetic["use_rich_calendar_features"]
    assert synthetic["use_native_rolling_stat_features"]
    assert not synthetic["use_native_partial_rolling_mean_features"]
    assert not synthetic["use_native_ewm_features"]
    assert synthetic["use_covariate_calendar_interactions"]
    assert taxi["use_static_covariates"]
    assert taxi["use_rich_calendar_features"]
    assert taxi["use_native_rolling_stat_features"]
    assert taxi["use_native_partial_rolling_mean_features"]
    assert not taxi["use_native_ewm_features"]
    assert taxi["use_covariate_calendar_interactions"]
    assert not m1["use_elapsed_calendar_features"]
    assert not m1["use_static_covariates"]
    assert not m1["use_known_future_covariates"]
    assert m3["use_elapsed_calendar_features"]
    assert not m3["use_static_covariates"]
    assert not m3["use_known_future_covariates"]
    assert not m4["use_elapsed_calendar_features"]
    assert not m4["use_static_covariates"]
    assert not m4["use_known_future_covariates"]
    assert not m4["use_rich_calendar_features"]
    assert not m4["use_native_rolling_stat_features"]
    assert not m4["use_native_partial_rolling_mean_features"]
    assert not m4["use_native_ewm_features"]
    assert not m4["use_covariate_calendar_interactions"]
    assert m5["use_static_covariates"]
    assert m5["use_known_future_covariates"]
    assert not m5["use_elapsed_calendar_features"]
    assert not m5["use_rich_calendar_features"]
    assert not m5["use_native_rolling_stat_features"]
    assert not m5["use_native_partial_rolling_mean_features"]
    assert not m5["use_native_ewm_features"]
    assert not m5["use_covariate_calendar_interactions"]
    assert not m6["use_static_covariates"]
    assert not m6["use_known_future_covariates"]
    assert not m6["use_elapsed_calendar_features"]
    assert not m6["use_rich_calendar_features"]
    assert m6["use_native_rolling_stat_features"]
    assert not m6["use_native_partial_rolling_mean_features"]
    assert not m6["use_native_ewm_features"]
    assert not m6["use_covariate_calendar_interactions"]


def test_forecasting_benchmark_native_params_respect_elapsed_calendar_feature_gate():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_elapsed_calendar_gate",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    frame = pl.DataFrame(
        {
            "lane_id": ["a", "a", "a", "a"],
            "date": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                datetime(2026, 1, 3),
                datetime(2026, 1, 4),
            ],
            "loads": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    base = {
        "n_estimators": 1,
        "learning_rate": 0.06,
        "max_depth": 2,
        "min_samples_leaf": 1,
    }

    m3_config = benchmark.cartoboost_source_config(base, source="m3")
    m4_config = benchmark.cartoboost_source_config(base, source="m4")

    m3_params = benchmark.cartoboost_native_forecaster_params(
        12,
        1,
        m3_config,
        train=frame,
    )
    assert m3_params["elapsed_calendar_features"]
    assert m3_params["elapsed_calendar_periods"] == [12]
    assert (
        benchmark.cartoboost_native_forecaster_params(
            1,
            1,
            m3_config,
            train=frame,
        )["elapsed_calendar_periods"]
        == []
    )
    m4_params = benchmark.cartoboost_native_forecaster_params(
        1,
        1,
        m4_config,
        train=frame,
    )
    assert not m4_params["elapsed_calendar_features"]
    assert m4_params["elapsed_calendar_periods"] == []


def test_forecasting_benchmark_validation_unavailable_fallback_prefers_lag_for_classical_auto():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_validation_unavailable_fallback",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    raw = pl.DataFrame(
        {
            "series_id": ["a", "a"],
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "horizon": [1, 2],
            "cartoboost_lag": [10.0, 12.0],
            "cartoboost_auto_forecast": [0.0, 0.0],
        }
    )

    selected, choices = benchmark.validation_unavailable_selected_predictions(
        raw,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m3",
    )

    assert choices["cartoboost_auto_forecast"] == "cartoboost_lag"
    assert selected["cartoboost_auto_forecast"].to_list() == [10.0, 12.0]


def test_forecasting_benchmark_m4_lag_spine_targets_high_frequency_risk():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_lag_spine",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert not benchmark.m4_requires_lag_spine(season_length=24, horizon=48)
    assert benchmark.m4_requires_lag_spine(season_length=12, horizon=18)
    assert benchmark.m4_requires_lag_spine(season_length=1, horizon=13)
    assert not benchmark.m4_requires_lag_spine(season_length=1, horizon=6)
    assert not benchmark.m4_requires_lag_spine(season_length=4, horizon=8)
    assert not benchmark.requires_lag_spine(source="synthetic", season_length=7, horizon=14)
    assert not benchmark.requires_lag_spine(source="m4", season_length=24, horizon=48)
    assert not benchmark.requires_lag_spine(source="m5", season_length=1, horizon=28)
    assert not benchmark.requires_lag_spine(source="m6", season_length=1, horizon=28)
    script = module_path.read_text(encoding="utf-8")
    assert 'source in {"synthetic", "m4"} and best_candidate == "cartoboost_lag"' not in script
    assert 'and best_candidate != "cartoboost_lag"' in script


def test_forecasting_benchmark_docs_match_committed_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    docs = (repo_root / "docs" / "benchmarks" / "forecasting.md").read_text(encoding="utf-8")
    artifacts = repo_root / "docs" / "assets" / "nyc_taxi_benchmarks"

    assert "badge of accuracy" not in docs
    assert "WAPE is diagnostic only" not in docs
    assert "guarded by the lag spine" not in docs
    assert "provenance" not in docs

    taxi = json.loads((artifacts / "forecasting_library_benchmark_real.json").read_text())
    assert taxi["comparability_audit"]["same_forecast_rows"] is True
    assert taxi["comparability_audit"]["selection_uses_outer_test_labels"] is False
    assert taxi["comparability_audit"]["completed_cartoboost_models"]
    taxi_auto = taxi["metrics"]["cartoboost_auto_forecast"]
    taxi_lag = taxi["metrics"]["cartoboost_lag"]
    assert (
        f"| 2 | `cartoboost_auto_forecast` | {taxi_auto['rmse']:.6f} | "
        f"{taxi_auto['mae']:.6f} | {taxi_auto['wape']:.6f} |"
    ) in docs
    assert (
        f"| 3 | `cartoboost_lag` | {taxi_lag['rmse']:.6f} | "
        f"{taxi_lag['mae']:.6f} | {taxi_lag['wape']:.6f} |"
    ) in docs
    assert taxi_auto["rmse"] < taxi_lag["rmse"]

    synthetic = json.loads((artifacts / "forecasting_overhaul_committed_suite.json").read_text())
    synthetic_quality = synthetic["aggregate_quality"]
    synthetic_lag = synthetic_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_lag"]
    synthetic_auto = synthetic_quality["mean_rmse_ratio_to_problem_best"][
        "cartoboost_auto_forecast"
    ]
    assert f"| CartoBoost sample | 1 | `cartoboost_auto_forecast` | {synthetic_auto:.6f} |" in docs
    assert f"| CartoBoost sample | 1 | `cartoboost_lag` | {synthetic_lag:.6f} |" not in docs
    assert synthetic_auto == pytest.approx(synthetic_lag)
    assert synthetic_auto <= synthetic_lag

    m4 = json.loads((artifacts / "forecasting_overhaul_m4_committed.json").read_text())
    m4_quality = m4["aggregate_quality"]
    m4_lag = m4_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_lag"]
    m4_auto = m4_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_auto_forecast"]
    assert f"| 1 | `cartoboost_auto_forecast` | {m4_auto:.6f} | 6 | 6 |" in docs
    assert f"| 2 | `cartoboost_lag` | {m4_lag:.6f} | 3 | 6 |" in docs
    assert m4_auto < m4_lag

    m5 = json.loads((artifacts / "forecasting_overhaul_m5_committed.json").read_text())
    m5_auto = m5["metrics"]["cartoboost_auto_forecast"]
    m5_lag = m5["metrics"]["cartoboost_lag"]
    m5_wrmsse = m5["official_metrics"]["m5"]["model_scores"]["cartoboost_auto_forecast"]
    m5_auto_wrmsse = m5["official_metrics"]["m5"]["model_scores"]["cartoboost_auto_forecast"]
    assert (
        f"| Sample | 1 | `cartoboost_auto_forecast` | {m5_auto['rmse']:.6f} | "
        f"{m5_auto['mae']:.6f} | {m5_auto['wape']:.6f} | {m5_auto_wrmsse:.6f} |"
    ) in docs
    assert (
        f"| Sample | 2 | `cartoboost_lag` | {m5_lag['rmse']:.6f} | "
        f"{m5_lag['mae']:.6f} | {m5_lag['wape']:.6f} | "
        f"{m5['official_metrics']['m5']['model_scores']['cartoboost_lag']:.6f} |"
    ) in docs
    assert m5_auto["rmse"] < m5_lag["rmse"]
    assert m5_auto_wrmsse < m5["official_metrics"]["m5"]["model_scores"]["cartoboost_lag"]

    m6 = json.loads((artifacts / "forecasting_overhaul_m6_committed.json").read_text())
    m6_auto = m6["metrics"]["cartoboost_auto_forecast"]
    m6_lag = m6["metrics"]["cartoboost_lag"]
    m6_rps = m6["official_metrics"]["m6"]["models"]
    assert (
        f"| Sample | 1 | `cartoboost_auto_forecast` | {m6_auto['rmse']:.6f} | "
        f"{m6_auto['mae']:.6f} | {m6_auto['wape']:.6f} | "
        f"{m6_rps['cartoboost_auto_forecast']['mean_rps']:.6f} |"
    ) in docs
    assert (
        f"| Sample | 2 | `cartoboost_lag` | {m6_lag['rmse']:.6f} | "
        f"{m6_lag['mae']:.6f} | {m6_lag['wape']:.6f} | "
        f"{m6_rps['cartoboost_lag']['mean_rps']:.6f} |"
    ) in docs
    assert m6_auto["rmse"] < m6_lag["rmse"]

    generalization = json.loads(
        (artifacts / "forecasting_generalization_scalable_synthetic.json").read_text()
    )
    generalization_quality = generalization["aggregate_quality"]
    generalization_ratios = generalization_quality["mean_rmse_ratio_to_problem_best"]
    generalization_auto_ratio = generalization_ratios["cartoboost_auto_forecast"]
    generalization_lag_ratio = generalization_ratios["cartoboost_lag"]
    generalization_lightgbm_ratio = generalization_ratios["lightgbm_lag"]
    generalization_xgboost_ratio = generalization_ratios["xgboost_lag"]
    assert (
        f"| Generalization guardrail | 1 | `cartoboost_auto_forecast` | "
        f"{generalization_auto_ratio:.6f} |"
    ) in docs
    assert (
        f"| Generalization guardrail | 1 | `cartoboost_lag` | {generalization_lag_ratio:.6f} |"
    ) not in docs
    assert (
        f"| Generalization guardrail | 3 | `lightgbm_lag` | {generalization_lightgbm_ratio:.6f} |"
    ) in docs
    assert (
        f"| Generalization guardrail | 4 | `xgboost_lag` | {generalization_xgboost_ratio:.6f} |"
    ) in docs
    assert generalization_auto_ratio == pytest.approx(generalization_lag_ratio)
    assert generalization_auto_ratio < generalization_lightgbm_ratio
    assert generalization_auto_ratio < generalization_xgboost_ratio
    assert generalization["timing"]["total_seconds"] > 0.0
    assert generalization["resource_usage"]["peak_rss_mb"] > 0.0

    m5_sample = json.loads((artifacts / "forecasting_m5_full_roster_sample.json").read_text())
    m5_sample_winner = m5_sample["quality"]["winner"]
    m5_sample_cartoboost_rmse = m5_sample["metrics"]["cartoboost_auto_forecast"]["rmse"]
    m5_sample_rmse_ranking = sorted(
        m5_sample["metrics"],
        key=lambda model: m5_sample["metrics"][model]["rmse"],
    )
    m5_sample_second_rmse_model = m5_sample_rmse_ranking[1]
    m5_sample_second_rmse = m5_sample["metrics"][m5_sample_second_rmse_model]["rmse"]
    m5_wrmsse = m5_sample["official_metrics"]["m5"]
    m5_wrmsse_winner = m5_wrmsse["ranking"][0]
    m5_wrmsse_winner_score = m5_wrmsse["model_scores"][m5_wrmsse_winner]
    m5_cartoboost_wrmsse = m5_wrmsse["model_scores"]["cartoboost_auto_forecast"]
    assert (
        f"| 100-series comparison | 1 | `cartoboost_auto_forecast` | "
        f"{m5_sample_cartoboost_rmse:.6f} |"
    ) in docs
    assert f"| 100-series comparison | 4 | `{m5_wrmsse_winner}` |" in docs
    assert f"{m5_wrmsse_winner_score:.6f}" in docs
    assert m5_sample_second_rmse < float("inf")
    assert m5_cartoboost_wrmsse > m5_wrmsse_winner_score
    assert m5_sample_winner == "cartoboost_auto_forecast"

    m6_full = json.loads((artifacts / "forecasting_m6_full.json").read_text())
    m6_full_winner = m6_full["quality"]["winner"]
    m6_full_auto = m6_full["metrics"]["cartoboost_auto_forecast"]
    m6_full_cartoboost_rmse = m6_full_auto["rmse"]
    m6_full_rmse_ranking = sorted(
        m6_full["metrics"],
        key=lambda model: m6_full["metrics"][model]["rmse"],
    )
    m6_full_second_rmse_model = m6_full_rmse_ranking[1]
    m6_full_second_rmse = m6_full["metrics"][m6_full_second_rmse_model]["rmse"]
    m6_full_rps = m6_full["official_metrics"]["m6"]
    m6_full_rps_winner = m6_full_rps["rps_ranking"][0]
    m6_full_rps_winner_score = m6_full_rps["models"][m6_full_rps_winner]["mean_rps"]
    m6_full_auto_rps = m6_full_rps["models"]["cartoboost_auto_forecast"]["mean_rps"]
    assert (
        f"| 100-symbol comparison | 1 | `cartoboost_auto_forecast` | "
        f"{m6_full_cartoboost_rmse:.6f} |"
    ) in docs
    assert f"{m6_full_rps_winner_score:.6f}" in docs
    assert m6_full_second_rmse > m6_full_cartoboost_rmse
    assert m6_full_auto_rps > m6_full_rps_winner_score
    assert m6_full_winner == "cartoboost_auto_forecast"
    assert m6_full_rps_winner != "cartoboost_auto_forecast"


def test_committed_forecasting_artifacts_include_provenance_fields():
    repo_root = Path(__file__).resolve().parents[2]
    artifacts = repo_root / "docs" / "assets" / "nyc_taxi_benchmarks"
    names = [
        "forecasting_overhaul_committed_suite.json",
        "forecasting_overhaul_m4_committed.json",
        "forecasting_overhaul_m5_committed.json",
        "forecasting_overhaul_m6_committed.json",
    ]
    for name in names:
        artifact = json.loads((artifacts / name).read_text())
        assert artifact["git_commit"]
        assert artifact["dataset_hash"]
        assert "source_file_hashes" in artifact
        assert artifact["benchmark_integrity"]["no_hyperopt"] is True
        assert artifact["benchmark_integrity"]["seed"] == 42
        assert artifact["resource_usage"]["peak_rss_mb"] > 0.0
        assert artifact["resource_usage"]["process_cpu_seconds"] >= 0.0


def test_maintained_forecasting_artifacts_include_comparability_audits():
    repo_root = Path(__file__).resolve().parents[2]
    artifacts = repo_root / "docs" / "assets" / "nyc_taxi_benchmarks"
    missing = []
    for path in sorted(artifacts.glob("forecasting*.json")):
        artifact = json.loads(path.read_text())
        audit = artifact.get("comparability_audit")
        if not audit:
            missing.append(path.name)
            continue
        assert audit["same_forecast_rows"] is True
        assert audit["same_horizon"] is True
        assert audit["selection_uses_outer_test_labels"] is False
        assert audit["completed_models"], path.name
        assert set(audit["completed_models"]) <= set(artifact["models"])

    assert missing == []


def test_forecasting_benchmark_comparability_audit_tracks_completed_models():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_comparability",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    args = argparse.Namespace(model_roster="full", no_candidate_selection=False)
    model_names = [
        "cartoboost_lag",
        "cartoboost_auto_forecast",
        "seasonal_naive",
        "xgboost_lag",
    ]
    audit = benchmark.forecasting_comparability_audit(
        args=args,
        model_names=model_names,
        metrics={
            "cartoboost_lag": {"rmse": 1.0},
            "cartoboost_auto_forecast": {"rmse": 0.8},
            "seasonal_naive": {"rmse": 1.3},
        },
    )

    assert audit["same_forecast_rows"] is True
    assert audit["selection_uses_outer_test_labels"] is False
    assert audit["candidate_selection"] is True
    assert audit["completed_cartoboost_models"] == [
        "cartoboost_lag",
        "cartoboost_auto_forecast",
    ]
    assert audit["completed_forecasting_library_models"] == ["seasonal_naive"]
    assert audit["skipped_requested_models"] == ["xgboost_lag"]
    assert audit["best_cartoboost_compared_to_best_library"] is True


def test_forecasting_benchmark_invocation_metadata_quotes_argv(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_invocation",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["scripts/forecasting_library_benchmark.py", "--output", "target/path with spaces.json"],
    )

    metadata = benchmark.invocation_metadata()

    assert metadata["argv"] == [
        "scripts/forecasting_library_benchmark.py",
        "--output",
        "target/path with spaces.json",
    ]
    assert metadata["command"] == (
        "scripts/forecasting_library_benchmark.py --output 'target/path with spaces.json'"
    )


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
        "auto_n_estimators": None,
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
    assert config["auto_n_estimators"] is None

    override = benchmark.cartoboost_auto_config(
        {
            **config,
            "n_estimators": 60,
            "auto_n_estimators": 72,
        },
        season_length=7,
        horizon=28,
    )
    assert override["n_estimators"] == 72


def test_forecasting_benchmark_model_settings_include_auto_policy():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_model_settings_policy",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    settings = benchmark.cartoboost_model_settings(
        {
            "n_estimators": 60,
            "auto_n_estimators": 72,
            "learning_rate": 0.06,
            "max_depth": 4,
            "min_samples_leaf": 8,
        }
    )

    assert set(settings) == {
        "cartoboost_lag",
        "cartoboost_auto_forecast",
        "cartoboost_neural_panel",
        "cartoboost_piecewise_linear_seasonal",
    }
    assert settings["cartoboost_lag"]["n_estimators"] == 60
    assert settings["cartoboost_auto_forecast"]["auto_n_estimators"] == 72
    assert settings["cartoboost_neural_panel"]["weekly_fourier_order"] == 3
    assert settings["cartoboost_neural_panel"]["trend_mode"] == "glocal"
    assert settings["cartoboost_piecewise_linear_seasonal"]["weekly_fourier_order"] == 3
    assert (
        "piecewise-linear" in settings["cartoboost_piecewise_linear_seasonal"]["benchmark_profile"]
    )
    assert (
        settings["cartoboost_auto_forecast"]["auto_n_estimators_policy"]
        == "explicit --cartoboost-auto-n-estimators override"
    )
    assert (
        "covariate-calendar interactions" in settings["cartoboost_lag"]["native_covariate_policy"]
    )
    assert (
        "non-M inner validation skips raw auto"
        in settings["cartoboost_auto_forecast"]["auto_selector_policy"]
    )
    assert benchmark.benchmark_model_names("cartoboost") == [
        "cartoboost_lag",
        "cartoboost_auto_forecast",
    ]
    assert benchmark.benchmark_model_names("piecewise") == [
        "cartoboost_piecewise_linear_seasonal",
    ]
    assert benchmark.forecasting_library_models_for_roster("piecewise") == {}
    assert benchmark.benchmark_model_names("prophet-comparison") == [
        "cartoboost_piecewise_linear_seasonal",
        "prophet_additive",
    ]
    assert benchmark.forecasting_library_models_for_roster("prophet-comparison") == {
        "prophet": ["prophet_additive"]
    }
    assert (
        "cache identical inner validation cutoffs"
        in settings["cartoboost_auto_forecast"]["auto_selector_policy"]
    )


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
            "second": 2.0,
            "third": 3.0,
            "fourth": 3.5,
            "fifth": 3.6,
        }
    )

    assert list(weights) == ["best", "second", "third", "fourth"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["best"] > weights["second"] > weights["third"] > weights["fourth"]

    with pytest.raises(ValueError, match="must be finite and non-negative"):
        benchmark.validation_ensemble_weights({"best": 1.0, "invalid": float("nan")})


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
    assert "shared_elapsed_phase_rank_blend" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m6",
    )
    assert "shared_market_neutral_zero" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m6",
    )
    assert "shared_elapsed_phase_rank_blend" not in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )


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
                    "m5_state_code": pickup_zone,
                    "m5_store_code": pickup_zone,
                    "m5_cat_code": 1.0,
                    "m5_dept_code": pickup_zone,
                    "m5_item_code": pickup_zone + 10.0,
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
    assert m5["model_level_contributions"]["cartoboost_auto_forecast"][0] == {
        "level": "total",
        "wrmsse": pytest.approx(0.0),
        "level_weight": pytest.approx(1.0 / 12.0),
        "contribution": pytest.approx(0.0),
    }
    assert set(m5["levels"]) == {
        "total",
        "state",
        "store",
        "category",
        "department",
        "state_category",
        "state_department",
        "store_category",
        "store_department",
        "item",
        "state_item",
        "item_store",
    }
    total = m5["levels"]["total"]["models"]
    assert total["cartoboost_auto_forecast"]["wrmsse"] == pytest.approx(0.0)
    assert total["cartoboost_lag"]["wrmsse"] > 0.0

    state_item_ids = {
        series["series_id"]
        for series in m5["levels"]["state_item"]["models"]["cartoboost_auto_forecast"]["series"]
    }
    assert state_item_ids == {"1.0/11.0", "2.0/12.0"}


def test_forecasting_benchmark_exposes_lag_as_traceable_auto_candidate():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_no_auto_protection",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert not hasattr(benchmark, "validation_auto_lag_protection")
    assert "cartoboost_lag" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m4",
    )
    assert benchmark.candidate_complexity_rank("cartoboost_lag") < (
        benchmark.candidate_complexity_rank("cartoboost_auto_forecast")
    )


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
    assert artifact["primary_metric"] == "investment_decision_return"
    assert m6["ranking"] == m6["investment_ranking"]
    assert "rps_ranking" in m6
    assert m6["investment_ranking"][0] == "cartoboost_auto_forecast"
    auto = m6["models"]["cartoboost_auto_forecast"]
    assert auto["asset_count"] == 5
    assert auto["rank_probability_calibration"]["fallback"] == "uniform_when_no_validation_support"
    assert auto["mean_rps"] == pytest.approx(m6["models"]["cartoboost_lag"]["mean_rps"])
    assert sum(row["weight"] for row in auto["decisions"]) == pytest.approx(0.0)
    assert auto["portfolio"]["gross_exposure"] == pytest.approx(1.0)
    assert auto["portfolio"]["net_exposure"] == pytest.approx(0.0)
    assert auto["decision_return"] == pytest.approx(auto["portfolio"]["net_return"])
    assert auto["portfolio"]["long_count"] == 1
    assert auto["portfolio"]["short_count"] == 1
    assert auto["rank_hit_rates"]["exact_bucket_rate"] == pytest.approx(1.0)
    assert auto["rank_hit_rates"]["within_one_bucket_rate"] == pytest.approx(1.0)
    assert auto["rank_hit_rates"]["directional_extreme_count"] == 2


def test_forecasting_benchmark_m6_rps_uses_validation_calibration():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_calibrated_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    rows = [
        ("AAA", -0.05),
        ("BBB", -0.02),
        ("CCC", 0.00),
        ("DDD", 0.02),
        ("EEE", 0.05),
    ]
    scored = pl.DataFrame(
        [
            {
                "series_id": symbol,
                "timestamp": date(2026, 2, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "cartoboost_auto_forecast": actual,
                "cartoboost_lag": actual,
            }
            for symbol, actual in rows
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    calibration_scored = pl.DataFrame(
        [
            {
                "series_id": symbol,
                "timestamp": date(2026, 1, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "cartoboost_auto_forecast": actual,
                "cartoboost_lag": -actual,
            }
            for symbol, actual in rows
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.m6_rps_artifact(
        scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        calibration_scored=calibration_scored,
    )

    auto = artifact["models"]["cartoboost_auto_forecast"]
    lag = artifact["models"]["cartoboost_lag"]
    assert auto["rank_probability_calibration"]["validation_support"] == 5
    assert auto["rank_probability_calibration"]["shrinkage_to_confusion"] > 0.0
    assert auto["mean_rps"] < lag["mean_rps"]
    assert artifact["rps_ranking"][0] == "cartoboost_auto_forecast"


def test_forecasting_benchmark_m6_objective_prefers_decision_return():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_decision_objective",
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
                "timestamp": date(2026, 3, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "profitable_rank": actual,
                "bad_rank": -actual,
            }
            for symbol, actual in [
                ("AAA", -0.05),
                ("BBB", -0.02),
                ("CCC", 0.00),
                ("DDD", 0.02),
                ("EEE", 0.05),
            ]
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    objective = benchmark.auto_selection_objective("m6")
    profitable_loss = benchmark.forecast_objective_loss(
        objective,
        train=scored,
        scored=scored,
        prediction_col="profitable_rank",
        season_length=1,
    )
    bad_loss = benchmark.forecast_objective_loss(
        objective,
        train=scored,
        scored=scored,
        prediction_col="bad_rank",
        season_length=1,
    )

    assert profitable_loss < bad_loss
    assert profitable_loss < 0.0


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
