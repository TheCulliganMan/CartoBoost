from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_nyc_taxi_quality_benchmarks import (  # noqa: E402
    BenchmarkTask,
    ZoneContext,
    benchmark_needs_zone_centroids,
    build_real_tasks,
    calibration_report_fields,
    cartoboost_schema,
    clean_tlc_frame,
    external_baseline_comparison,
    graph_augmented_split_features,
    pickup_demand_cold_zone_fraction,
    pickup_zone_diagnostics,
    required_benchmark_dependencies,
    sample_tlc_frame,
    split_indices,
    split_modes_for_task,
    validate_requested_benchmark_dependencies,
)
from scripts.run_nyc_taxi_quality_benchmarks import (  # noqa: E402
    output_artifact_manifest as nyc_output_artifact_manifest,
)
from scripts.run_repeated_nyc_taxi_benchmarks import (  # noqa: E402
    collect_quality,
    repeated_comparability_audit,
)
from scripts.run_repeated_nyc_taxi_benchmarks import (  # noqa: E402
    output_artifact_manifest as repeated_output_artifact_manifest,
)


def test_nyc_taxi_quality_benchmark_synthetic_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "nyc_taxi_benchmarks"
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--models",
            "mean",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    jsonl_rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert results["artifact_version"] == 1
    assert results["dataset"]["source"] == "synthetic_smoke"
    assert len(results["dataset_hash"]) == 64
    assert results["dataset"]["dataset_hash"] == results["dataset_hash"]
    assert results["source_file_hashes"] == {}
    assert results["git_commit"] is None or len(results["git_commit"]) == 40
    assert results["benchmark_integrity"]["hpo"] == "fixed_settings_no_hpo"
    assert results["benchmark_integrity"]["split_modes"] == [
        "random",
        "spatial_holdout",
        "out_of_time",
    ]
    assert results["benchmark_integrity"]["split_modes_by_task"]["pickup_demand"] == [
        "random",
        "spatial_holdout",
    ]
    selection = results["benchmark_integrity"]["selection_policy"]
    assert "test labels" in selection["global_hyperparameters"]
    assert "training split only" in selection["graph_feature_gate"]
    assert "excluded from fitting" in selection["segment_diagnostics"]
    assert results["feature_access_policy"]["baseline_feature_access"]
    assert set(results["split_definitions"]) == {
        "random",
        "spatial_holdout",
        "out_of_time",
    }
    assert results["model_roster"] == ["mean"]
    assert results["model_status_summary"]["completed_external_baselines"] == ["mean"]
    assert results["comparability_audit"]["same_outer_splits"] is True
    assert results["comparability_audit"]["selection_mode"] == "fixed_settings_no_hpo"
    assert results["comparability_audit"]["selection_uses_outer_test_labels"] is False
    assert results["comparability_audit"]["completed_external_baselines"] == ["mean"]
    assert results["comparability_audit"]["skipped_requested_external_baselines"] == []
    assert results["resource_usage"]["python"]
    assert results["baseline_environment"]["sklearn"]["package"] == "scikit-learn"
    assert "module_importable" in results["baseline_environment"]["sklearn"]
    assert "required_class_available" in results["baseline_environment"]["xgboost"]
    assert results["output_artifacts"]["results.json"]["size_bytes"] > 0
    assert results["output_artifacts"]["results.md"]["size_bytes"] > 0
    assert results["output_artifacts"]["results.jsonl"]["size_bytes"] > 0
    assert nyc_output_artifact_manifest(output_dir)["results.json"]["size_bytes"] > 0
    assert set(results["tasks"]) == {"duration", "fare", "pickup_demand"}
    assert jsonl_rows
    assert {"task_id", "split_id", "model_family", "metric", "value"} <= set(jsonl_rows[0])
    assert {row["model_family"] for row in jsonl_rows} == {"mean"}

    aggregate_path = output_dir / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "runners" / "aggregate_results.py"),
            "--input",
            str(output_dir / "results.jsonl"),
            "--output",
            str(aggregate_path),
        ],
        check=True,
        cwd=repo_root,
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_keys = {
        (row["track"], row["task_id"], row["split_id"], row["model_family"], row["metric"])
        for row in aggregate["metrics"]
    }
    assert ("spatial", "duration", "random", "mean", "rmse") in aggregate_keys
    assert ("spatial", "duration", "spatial_holdout", "mean", "rmse") in aggregate_keys
    assert ("temporal", "duration", "out_of_time", "mean", "rmse") in aggregate_keys

    for task_name, task in results["tasks"].items():
        expected_splits = {"random", "spatial_holdout"}
        if task_name in {"duration", "fare"}:
            expected_splits.add("out_of_time")
        assert set(task["splits"]) == expected_splits
        for split in task["splits"].values():
            model = split["models"]["mean"]
            assert model["status"] == "ok"
            assert np.isfinite(model["metrics"]["rmse"])
            assert np.isfinite(model["metrics"]["mae"])
            assert np.isfinite(model["metrics"]["r2"])
            assert np.isfinite(model["metrics"]["wape"])
            calibration = model["calibration_report_fields"]
            assert calibration["status"] == "not_computed"
            assert set(calibration) == {
                "status",
                "coverage_by_horizon",
                "coverage_by_spatial_block",
                "width_by_horizon",
                "residual_morans_i_after_calibration",
            }
            diagnostics = model["segment_diagnostics"]["pickup_zone"]
            assert diagnostics["segment_key"] == "pickup_zone"
            assert diagnostics["segment_count"] > 0
            assert diagnostics["row_count_min"] > 0
            assert diagnostics["row_count_max"] >= diagnostics["row_count_min"]
            assert np.isfinite(diagnostics["rmse_quantiles"]["p50"])
            assert diagnostics["worst_rmse_segments"]
            assert model["timing"]["train_seconds"] >= 0.0
            assert model["timing"]["predict_seconds"] >= 0.0
            assert model["timing"]["fit_predict_seconds"] >= 0.0
            assert model["timing"]["prediction_rows"] > 0.0
            assert model["timing"]["predict_rows_per_second"] > 0.0
        if "out_of_time" in task["splits"]:
            manifest = task["splits"]["out_of_time"]["split_manifest"]
            assert manifest["split_kind"] == "chronological_timestamp_holdout"
            assert manifest["leakage_safe"] is True
            assert manifest["timestamp_tie_policy"] == ("grouped_at_boundary_no_timestamp_overlap")
            assert manifest["train_timestamp_max"] < manifest["test_timestamp_min"]

    markdown = (output_dir / "results.md").read_text(encoding="utf-8")
    assert "NYC Taxi Model Quality Benchmarks" in markdown
    assert "## Comparability Audit" in markdown
    assert "Trip duration" in markdown
    assert "Pickup-zone demand" in markdown

    summary = mpimg.imread(output_dir / "metric_summary.png")
    assert summary.shape[0] >= 300
    assert summary.shape[1] >= 500
    assert float(np.std(summary[..., :3])) > 0.01

    speed_summary = mpimg.imread(output_dir / "speed_summary.png")
    assert speed_summary.shape[0] >= 300
    assert speed_summary.shape[1] >= 500
    assert float(np.std(speed_summary[..., :3])) > 0.01

    throughput = mpimg.imread(output_dir / "prediction_throughput.png")
    assert throughput.shape[0] >= 300
    assert throughput.shape[1] >= 500
    assert float(np.std(throughput[..., :3])) > 0.01


def test_nyc_taxi_quality_benchmark_rejects_removed_autogeo_selector(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "nyc_taxi_autogeo"
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--tasks",
            "duration",
            "--models",
            "autogeo",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
        check=False,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown models: autogeo" in (result.stderr + result.stdout)


def test_nyc_taxi_quality_benchmark_requires_requested_baseline_dependencies(monkeypatch):
    assert required_benchmark_dependencies(
        ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting", "mean"]
    ) == {
        "lightgbm": "LGBMRegressor",
        "xgboost": "XGBRegressor",
        "catboost": "CatBoostRegressor",
        "sklearn": None,
    }

    from scripts import run_nyc_taxi_quality_benchmarks as module

    def fake_dependency_status(import_name, class_name=None, *, distribution_name=None):
        del class_name, distribution_name
        return {
            "package": import_name,
            "import_name": import_name,
            "version": None,
            "module_importable": import_name != "lightgbm",
            "required_class_available": import_name != "lightgbm",
        }

    monkeypatch.setattr(module, "dependency_status", fake_dependency_status)

    with pytest.raises(RuntimeError, match=r"lightgbm.*uv sync --group bench --group dev"):
        validate_requested_benchmark_dependencies(["lightgbm"])


def test_nyc_taxi_public_doc_matches_maintained_artifact_rows():
    results_path = REPO_ROOT / "docs" / "assets" / "nyc_taxi_benchmarks" / "results.json"
    doc_path = REPO_ROOT / "docs" / "benchmarks" / "nyc-taxi.md"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    doc = doc_path.read_text(encoding="utf-8")

    for task_name, split_name, model_name in [
        ("duration", "random", "cartoboost"),
        ("fare", "spatial_holdout", "cartoboost"),
        ("pickup_demand", "random", "cartoboost"),
    ]:
        result = results["tasks"][task_name]["splits"][split_name]["models"][model_name]
        metrics = result["metrics"]
        expected_row = (
            f"| {task_name.replace('_', ' ').capitalize()} / "
            f"{split_name.replace('_', ' ')} | {metrics['rmse']:.6f} | "
            f"{metrics['wape']:.6f} |"
        )
        assert expected_row in doc

    assert "The pickup-demand spatial holdout skips learned models because held-out pickup" in doc
    assert "The comparisons are fair at the benchmark level" in doc


def test_nyc_taxi_maintained_artifacts_are_complete():
    artifact_dir = REPO_ROOT / "docs" / "assets" / "nyc_taxi_benchmarks"
    paths = [
        artifact_dir / "results.json",
        artifact_dir / "results.jsonl",
        artifact_dir / "results.md",
        artifact_dir / "repeated_results.json",
        artifact_dir / "repeated_results.md",
    ]
    for path in paths:
        assert path.exists(), f"missing maintained NYC benchmark artifact: {path}"
        assert path.stat().st_size > 0

    results = json.loads((artifact_dir / "results.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (artifact_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(results["external_baseline_comparison"]) == 7
    assert results["comparability_audit"]["same_outer_splits"] is True
    assert results["comparability_audit"]["selection_uses_outer_test_labels"] is False
    assert results["comparability_audit"]["same_feature_access_policy"] is True
    assert results["comparability_audit"]["completed_external_baselines"]
    assert results["model_status_summary"]["completed_external_baselines"]
    assert results["output_artifacts"]["results.jsonl"]["size_bytes"] == len(
        (artifact_dir / "results.jsonl").read_bytes().replace(b"\r\n", b"\n")
    )
    assert {
        (row["track"], row["task_id"], row["split_id"], row["model_family"], row["metric"])
        for row in rows
    } >= {
        ("spatial", "duration", "random", "cartoboost", "rmse"),
        ("spatial", "fare", "spatial_holdout", "hist_gradient_boosting", "wape"),
        ("spatial", "pickup_demand", "random", "hist_gradient_boosting", "r2"),
    }

    repeated = json.loads((artifact_dir / "repeated_results.json").read_text(encoding="utf-8"))
    assert repeated["seeds"] == [11, 29, 47]
    assert repeated["comparability_audit"]["same_outer_splits"] is True
    assert repeated["comparability_audit"]["selection_uses_outer_test_labels"] is False
    assert repeated["comparability_audit"]["completed_external_baselines_all_runs"]
    assert repeated["output_artifacts"]["docs/assets/nyc_taxi_benchmarks/repeated_results.json"][
        "size_bytes"
    ] == len((artifact_dir / "repeated_results.json").read_bytes().replace(b"\r\n", b"\n"))
    assert repeated["output_artifacts"]["docs/assets/nyc_taxi_benchmarks/repeated_results.md"][
        "size_bytes"
    ] == len((artifact_dir / "repeated_results.md").read_bytes().replace(b"\r\n", b"\n"))
    assert repeated["quality"]


def test_repeated_nyc_comparability_audit_intersects_child_runs():
    audit = repeated_comparability_audit(
        [
            {
                "comparability_audit": {
                    "same_outer_splits": True,
                    "selection_uses_outer_test_labels": False,
                    "same_feature_access_policy": True,
                    "completed_external_baselines": ["mean", "xgboost"],
                    "skipped_requested_external_baselines": [],
                    "completed_cartoboost_family_models": ["cartoboost"],
                    "cartoboost_external_comparison_count": 2,
                }
            },
            {
                "comparability_audit": {
                    "same_outer_splits": True,
                    "selection_uses_outer_test_labels": False,
                    "same_feature_access_policy": True,
                    "completed_external_baselines": ["mean", "ridge"],
                    "skipped_requested_external_baselines": ["lightgbm"],
                    "completed_cartoboost_family_models": ["cartoboost"],
                    "cartoboost_external_comparison_count": 1,
                }
            },
        ]
    )

    assert audit["run_count"] == 2
    assert audit["same_outer_splits"] is True
    assert audit["selection_uses_outer_test_labels"] is False
    assert audit["completed_external_baselines_all_runs"] == ["mean"]
    assert audit["skipped_requested_external_baselines_any_run"] == ["lightgbm"]
    assert audit["completed_cartoboost_family_models_all_runs"] == ["cartoboost"]
    assert audit["cartoboost_external_comparison_rows"] == 3


def test_nyc_taxi_quality_benchmark_skips_optional_model_failures():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"
    source = script.read_text(encoding="utf-8")

    assert '"status": "skipped"' in source
    assert "pickup_demand cold-zone spatial holdout" in source


def test_nyc_taxi_quality_benchmark_reports_best_cartoboost_vs_external_baseline():
    payload = {
        "tasks": {
            "fare": {
                "splits": {
                    "random": {
                        "models": {
                            "cartoboost": {
                                "status": "ok",
                                "metrics": {"rmse": 0.15, "r2": 0.86, "wape": 0.11},
                            },
                            "cartoboost_graph_node2vec": {
                                "status": "ok",
                                "metrics": {"rmse": 0.13, "r2": 0.89, "wape": 0.10},
                            },
                            "lightgbm": {
                                "status": "ok",
                                "metrics": {"rmse": 0.17, "r2": 0.84, "wape": 0.12},
                            },
                            "ridge": {
                                "status": "ok",
                                "metrics": {"rmse": 0.14, "r2": 0.88, "wape": 0.09},
                            },
                        },
                    },
                    "spatial_holdout": {
                        "models": {
                            "cartoboost": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                            "lightgbm": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                        },
                    },
                },
            },
        },
    }

    assert external_baseline_comparison(payload) == [
        {
            "task": "fare",
            "split": "random",
            "cartoboost_model": "cartoboost_graph_node2vec",
            "cartoboost_rmse": 0.13,
            "cartoboost_wape": 0.10,
            "cartoboost_r2": 0.89,
            "best_external_baseline": "ridge",
            "best_external_rmse": 0.14,
            "best_external_wape": 0.09,
            "best_external_r2": 0.88,
            "rmse_delta_vs_external": -0.010000000000000009,
            "r2_delta_vs_external": 0.010000000000000009,
            "status": "cartoboost_lower_rmse",
        }
    ]


def test_pickup_zone_diagnostics_reports_quantiles_and_worst_segments():
    diagnostics = pickup_zone_diagnostics(
        np.asarray([1.0, 2.0, 4.0, 8.0]),
        np.asarray([1.0, 3.0, 3.0, 4.0]),
        np.asarray([10, 10, 11, 12]),
        top_n=2,
    )

    assert diagnostics["segment_key"] == "pickup_zone"
    assert diagnostics["segment_count"] == 3
    assert diagnostics["row_count_min"] == 1
    assert diagnostics["row_count_max"] == 2
    assert diagnostics["rmse_quantiles"]["max"] == pytest.approx(4.0)
    assert diagnostics["worst_rmse_segments"][0]["pickup_zone"] == 12
    assert diagnostics["largest_abs_bias_segments"][0]["pickup_zone"] == 12


def test_calibration_report_fields_groups_interval_metrics():
    fields = calibration_report_fields(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0.0, 1.0, 4.0, 2.0]),
        np.asarray([2.0, 3.0, 5.0, 3.0]),
        horizons=np.asarray([1, 1, 2, 2]),
        spatial_blocks=np.asarray(["a", "a", "b", "b"]),
        residual_morans_i_after_calibration=0.04,
    )

    assert fields["status"] == "computed"
    assert fields["coverage_by_horizon"] == {"1": 1.0, "2": 0.0}
    assert fields["coverage_by_spatial_block"] == {"a": 1.0, "b": 0.0}
    assert fields["width_by_horizon"] == {"1": 2.0, "2": 1.0}
    assert fields["residual_morans_i_after_calibration"] == 0.04


def test_repeated_nyc_quality_summary_reports_cis_and_paired_deltas():
    results = [
        {
            "tasks": {
                "fare": {
                    "splits": {
                        "random": {
                            "models": {
                                "cartoboost": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.0, "mae": 0.7, "r2": 0.8},
                                    "timing": {
                                        "train_seconds": 2.0,
                                        "predict_seconds": 0.2,
                                        "predict_rows_per_second": 100.0,
                                    },
                                },
                                "lightgbm": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.2, "mae": 0.8, "r2": 0.7},
                                    "timing": {
                                        "train_seconds": 1.0,
                                        "predict_seconds": 0.1,
                                        "predict_rows_per_second": 200.0,
                                    },
                                },
                            },
                        }
                    },
                }
            }
        },
        {
            "tasks": {
                "fare": {
                    "splits": {
                        "random": {
                            "models": {
                                "cartoboost": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.1, "mae": 0.75, "r2": 0.78},
                                    "timing": {
                                        "train_seconds": 3.0,
                                        "predict_seconds": 0.3,
                                        "predict_rows_per_second": 90.0,
                                    },
                                },
                                "lightgbm": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.3, "mae": 0.85, "r2": 0.68},
                                    "timing": {
                                        "train_seconds": 1.1,
                                        "predict_seconds": 0.1,
                                        "predict_rows_per_second": 190.0,
                                    },
                                },
                            },
                        }
                    },
                }
            }
        },
    ]

    summary = collect_quality(results)
    cartoboost = summary["fare/random"]["models"]["cartoboost"]
    assert cartoboost["metrics"]["rmse"]["n"] == 2
    assert cartoboost["metrics"]["rmse"]["mean"] == pytest.approx(1.05)
    assert cartoboost["rmse_wins_or_ties"] == 2

    primary = summary["fare/random"]["primary_vs_best_external"]
    assert primary["n"] == 2
    assert primary["best_external_model_counts"] == {"lightgbm": 2}
    assert primary["rmse_delta_mean"] == pytest.approx(-0.2)
    assert primary["r2_delta_mean"] == pytest.approx(0.1)

    delta = summary["fare/random"]["paired_deltas"]["cartoboost_vs_lightgbm"]
    assert delta["rmse_delta_mean"] == pytest.approx(-0.2)
    assert delta["r2_delta_mean"] == pytest.approx(0.1)


def test_repeated_nyc_output_artifact_manifest_reports_summary_sizes(tmp_path: Path):
    summary_json = tmp_path / "repeated_results.json"
    summary_md = tmp_path / "repeated_results.md"
    summary_json.write_text("{}\n", encoding="utf-8")
    summary_md.write_text("# report\n", encoding="utf-8")

    manifest = repeated_output_artifact_manifest(summary_json, summary_md)

    assert manifest[str(summary_json)]["size_bytes"] == summary_json.stat().st_size
    assert manifest[str(summary_md)]["size_bytes"] == summary_md.stat().st_size


def test_nyc_taxi_centroids_are_only_required_for_cartoboost_geometry():
    class Args:
        models = "lightgbm,mean"
        zone_treatment = "target_mean"
        cartoboost_splitters = "axis_histogram:512,diagonal_2d,gaussian_2d"

    assert not benchmark_needs_zone_centroids(Args())

    Args.models = "cartoboost"
    assert benchmark_needs_zone_centroids(Args())

    Args.cartoboost_splitters = "axis_histogram:512,periodic:24,sparse_set"
    assert not benchmark_needs_zone_centroids(Args())

    Args.zone_treatment = "raw"
    assert benchmark_needs_zone_centroids(Args())


def test_real_pickup_demand_aggregates_full_cleaned_frame_when_rows_are_sampled():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "tpep_pickup_datetime": pandas.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 08:15:00",
                    "2024-01-01 08:30:00",
                    "2024-01-02 09:00:00",
                ]
            ),
            "tpep_dropoff_datetime": pandas.to_datetime(
                [
                    "2024-01-01 08:10:00",
                    "2024-01-01 08:25:00",
                    "2024-01-01 08:40:00",
                    "2024-01-02 09:12:00",
                ]
            ),
            "passenger_count": [1.0, 1.0, 1.0, 1.0],
            "trip_distance": [1.0, 1.2, 1.1, 2.0],
            "fare_amount": [8.0, 9.0, 8.5, 12.0],
            "total_amount": [10.0, 11.0, 10.5, 14.0],
            "PULocationID": [10, 10, 10, 20],
            "DOLocationID": [20, 20, 20, 10],
        }
    )
    cleaned = clean_tlc_frame(frame)
    sampled_rows = sample_tlc_frame(cleaned, sample_size=2, seed=0)
    tasks = build_real_tasks(
        sampled_rows,
        {
            10: ZoneContext(borough_code=1, service_zone_code=4),
            20: ZoneContext(borough_code=2, service_zone_code=2),
        },
        demand_frame=cleaned,
    )

    duration = next(task for task in tasks if task.name == "duration")
    demand = next(task for task in tasks if task.name == "pickup_demand")
    assert len(duration.target) == 2
    assert duration.timestamp_column == "tpep_pickup_datetime"
    assert duration.timestamps is not None
    assert duration.timestamps.dtype == np.dtype("datetime64[ns]")
    assert len(duration.timestamps) == len(duration.target)
    assert demand.timestamps is None
    demand_by_zone = {
        int(features[0]): float(target)
        for features, target in zip(demand.features, demand.target, strict=True)
    }
    assert demand_by_zone[10] == np.log1p(3.0)
    assert demand_by_zone[20] == np.log1p(1.0)


def test_cartoboost_schema_marks_zone_geometry_spatial_without_dense_sparse_schema():
    task = BenchmarkTask(
        name="fare",
        display_name="Fare",
        description="fixture",
        features=np.asarray([[1.0, 2.0]], dtype=float),
        target=np.asarray([1.0], dtype=float),
        pickup_zones=np.asarray([1], dtype=int),
        feature_names=["PULocationID", "hour"],
        sparse_sets={"pickup_zone": [[1]]},
    )

    schema = cartoboost_schema(
        task,
        feature_names=[
            "PULocationID",
            "hour",
            "PULocationID_centroid_x",
            "PULocationID_centroid_y",
            "od_centroid_distance",
        ],
        dense_id_sets=True,
        include_sparse_sets=False,
    )

    assert schema == {
        "dense": [
            {"name": "PULocationID", "kind": "numeric"},
            {"name": "hour", "kind": "periodic", "period": 24},
            {"name": "PULocationID_centroid_x", "kind": "spatial"},
            {"name": "PULocationID_centroid_y", "kind": "spatial"},
            {"name": "od_centroid_distance", "kind": "numeric"},
        ],
        "sparse_sets": [],
    }


def test_nyc_taxi_quality_benchmark_runs_neural_and_graph_models(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "nyc_taxi_neural_graph"
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--tasks",
            "duration",
            "--models",
            (
                "cartoboost_neural,cartoboost_graph_node2vec,cartoboost_graph_graphsage,"
                "cartoboost_graph_hetero_graphsage,cartoboost_graph_hinsage"
            ),
            "--graph-dim",
            "2",
            "--graph-epochs",
            "1",
            "--neural-dim",
            "2",
            "--cartoboost-n-estimators",
            "4",
            "--model-workers",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    task = results["tasks"]["duration"]
    for split in task["splits"].values():
        assert split["models"]["cartoboost_neural"]["status"] == "ok"
        assert split["models"]["cartoboost_neural"]["config"]["neural_dim"] > 0
        for model_name, family in {
            "cartoboost_graph_node2vec": "node2vec",
            "cartoboost_graph_graphsage": "graphsage",
            "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
            "cartoboost_graph_hinsage": "hinsage",
        }.items():
            assert split["models"][model_name]["status"] == "ok"
            assert split["models"][model_name]["config"]["graph_family"] == family
            assert split["models"][model_name]["config"]["graph_edges"] > 0


def test_pickup_demand_graph_uses_zone_context_topology_for_cold_holdout():
    task = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="fixture",
        features=np.asarray(
            [
                [1.0, 8.0, 0.0],
                [1.0, 9.0, 0.0],
                [2.0, 8.0, 0.0],
                [3.0, 8.0, 0.0],
                [4.0, 8.0, 0.0],
            ],
            dtype=float,
        ),
        target=np.asarray([1.0, 1.2, 1.4, 1.6, 1.8], dtype=float),
        pickup_zones=np.asarray([1, 1, 2, 3, 4], dtype=int),
        feature_names=["PULocationID", "hour", "dayofweek"],
        sparse_sets={
            "pickup_zone": [[1], [1], [2], [3], [4]],
            "pickup_borough": [[1], [1], [1], [2], [2]],
            "pickup_service_zone": [[4], [4], [4], [2], [2]],
        },
        zone_adjacency={1: [2], 2: [1, 3], 3: [2, 4], 4: [3]},
    )
    train_indices = np.asarray([0, 1, 2], dtype=int)
    test_indices = np.asarray([3, 4], dtype=int)

    class Args:
        graph_dim = 2
        graph_epochs = 1
        seed = 7

    train_augmented, test_augmented, config = graph_augmented_split_features(
        task,
        train_indices,
        test_indices,
        task.features[train_indices],
        task.features[test_indices],
        Args(),
        graph_family="graphsage",
    )

    assert config["graph_topology"] == "zone_adjacency_borough_service"
    assert config["adjacency_edges"] > 0
    assert config["borough_edges"] > 0
    assert config["service_zone_edges"] > 0
    assert config["context_hub_nodes"] > 0
    assert train_augmented.shape[1] == test_augmented.shape[1]
    assert config["graph_feature_count"] == train_augmented.shape[1] - task.features.shape[1]
    assert float(np.std(test_augmented[:, task.features.shape[1] :])) > 0.0


def test_pickup_demand_cold_zone_fraction_detects_spatial_holdout():
    task = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="fixture",
        features=np.asarray(
            [
                [1.0, 8.0, 0.0],
                [1.0, 9.0, 0.0],
                [2.0, 8.0, 0.0],
                [3.0, 8.0, 0.0],
            ],
            dtype=float,
        ),
        target=np.asarray([1.0, 1.2, 1.4, 1.6], dtype=float),
        pickup_zones=np.asarray([1, 1, 2, 3], dtype=int),
        feature_names=["PULocationID", "hour", "dayofweek"],
        sparse_sets={"pickup_zone": [[1], [1], [2], [3]]},
    )

    assert (
        pickup_demand_cold_zone_fraction(
            task,
            np.asarray([0, 1, 2], dtype=int),
            np.asarray([3], dtype=int),
        )
        == 1.0
    )
    assert (
        pickup_demand_cold_zone_fraction(
            task,
            np.asarray([0, 2, 3], dtype=int),
            np.asarray([1], dtype=int),
        )
        == 0.0
    )


def test_out_of_time_split_keeps_equal_timestamps_on_one_side():
    timestamps = np.asarray(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-04T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    task = BenchmarkTask(
        name="duration",
        display_name="Trip duration",
        description="fixture",
        features=np.arange(5.0).reshape(-1, 1),
        target=np.arange(5.0),
        pickup_zones=np.ones(5, dtype=int),
        feature_names=["value"],
        sparse_sets={},
        timestamps=timestamps,
        timestamp_column="tpep_pickup_datetime",
    )

    train, test = split_indices(task, mode="out_of_time", seed=42)

    assert train.tolist() == [0, 1, 2, 3]
    assert test.tolist() == [4]
    assert np.intersect1d(timestamps[train], timestamps[test]).size == 0
    assert timestamps[train].max() < timestamps[test].min()


def test_out_of_time_split_requires_timestamps_and_excludes_demand_tasks():
    task = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup demand",
        description="fixture",
        features=np.ones((3, 1)),
        target=np.ones(3),
        pickup_zones=np.ones(3, dtype=int),
        feature_names=["value"],
        sparse_sets={},
    )

    assert split_modes_for_task(task) == ["random", "spatial_holdout"]
    with pytest.raises(ValueError, match="requires source row timestamps"):
        split_indices(task, mode="out_of_time", seed=42)
