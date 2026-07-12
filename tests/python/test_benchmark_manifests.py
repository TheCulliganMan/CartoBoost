from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

aggregate_module = importlib.import_module("benchmarks.runners.aggregate_results")
manifest_module = importlib.import_module("benchmarks.runners.manifest")
model_suite_module = importlib.import_module("scripts.run_model_benchmark_suite")
forecasting_benchmark_module = importlib.import_module("scripts.forecasting_library_benchmark")
autogeo_gate_module = importlib.import_module("scripts.run_autogeo_benchmark_gate")
artifact_gate_module = importlib.import_module("scripts.check_artifact_compatibility")
forecasting_quality_gate_module = importlib.import_module("scripts.check_forecasting_quality_gate")
nyc_row_quality_gate_module = importlib.import_module("scripts.check_nyc_row_quality_gate")
official_geo_evidence_module = importlib.import_module("scripts.check_official_geo_evidence")
performance_gate_module = importlib.import_module("scripts.check_performance_thresholds")
freshness_gate_module = importlib.import_module("scripts.check_benchmark_freshness")
release_gate_module = importlib.import_module("scripts.check_release_gates")
significance_module = importlib.import_module("benchmarks.runners.significance")

aggregate = aggregate_module.aggregate
check_artifact_compatibility = artifact_gate_module.check_artifact_compatibility
check_forecasting_quality_gate = forecasting_quality_gate_module.check_forecasting_quality_gate
check_nyc_row_quality_gate = nyc_row_quality_gate_module.check_nyc_row_quality_gate
official_geo_evidence_report = official_geo_evidence_module.official_geo_evidence_report
check_performance_thresholds = performance_gate_module.check_performance_thresholds
check_benchmark_freshness = freshness_gate_module.check_benchmark_freshness
read_jsonl = aggregate_module.read_jsonl
load_all_tracks = manifest_module.load_all_tracks
load_config = manifest_module.load_config
validate_official_geo_benchmark_suite = manifest_module.validate_official_geo_benchmark_suite
validate_configs = manifest_module.validate_configs
failed_validation_search_reason = model_suite_module.failed_validation_search_reason
repeated_external_comparison_summary = model_suite_module.repeated_external_comparison_summary
required_benchmark_dependencies = model_suite_module.required_benchmark_dependencies
validate_requested_benchmark_dependencies = (
    model_suite_module.validate_requested_benchmark_dependencies
)
check_benchmark_dependencies = release_gate_module.check_benchmark_dependencies
check_ci_release_gates = release_gate_module.check_ci_release_gates
check_external_baseline_install_metadata = (
    release_gate_module.check_external_baseline_install_metadata
)
check_publish_artifact_attestation = release_gate_module.check_publish_artifact_attestation
average_ranks = significance_module.average_ranks
paired_bootstrap_ci = significance_module.paired_bootstrap_ci


def test_public_benchmark_manifests_are_valid() -> None:
    validate_configs()
    specs = load_all_tracks()

    assert {spec.name for spec in specs} == {"forecasting", "graph", "spatial", "tabular"}


def test_nyc_row_quality_gate_reports_current_evidence_without_overclaiming() -> None:
    report = check_nyc_row_quality_gate(ROOT / "docs/assets/nyc_taxi_benchmarks/results.json")
    assert report["dataset_source"] == "nyc_tlc_trip_records"
    assert len(report["comparisons"]) == 4
    assert report["passed"] is True
    assert report["wins"] >= report["minimum_wins"]


def test_sklearn_dependency_is_optional_extra() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies_block = re.search(r"dependencies = \[(.*?)\]", text, re.S)
    assert dependencies_block is not None
    assert "scikit-learn" not in dependencies_block.group(1)
    assert re.search(
        r"\[project\.optional-dependencies\]\s+sklearn = \[\s+\"scikit-learn>=1\.2\",\s+\]",
        text,
    )


def test_benchmark_dependency_group_covers_intended_baselines() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    bench_block = re.search(r"\[dependency-groups\]\s+bench = \[(.*?)\]", text, re.S)
    assert bench_block is not None
    bench = bench_block.group(1)

    for dependency in [
        "catboost>=1.2",
        "darts>=0.30",
        "esda>=2.6",
        "gstools>=1.6",
        "libpysal>=4.10",
        "lightgbm>=4.0",
        "neuralforecast>=1.7",
        "pykrige>=1.7",
        "scikit-learn>=1.2",
        "spreg>=1.4",
        "xgboost>=2.0",
    ]:
        assert f'"{dependency}"' in bench


def test_model_suite_requires_requested_baseline_dependencies(monkeypatch) -> None:
    assert required_benchmark_dependencies(
        ["mean", "xgboost", "lightgbm", "catboost", "ridge"],
        ["diabetes"],
    ) == {
        "xgboost": "XGBRegressor",
        "lightgbm": "LGBMRegressor",
        "catboost": "CatBoostRegressor",
        "sklearn": None,
    }

    def fake_dependency_status(import_name, class_name=None, *, distribution_name=None):
        del class_name, distribution_name
        return {
            "package": import_name,
            "import_name": import_name,
            "version": None,
            "module_importable": import_name != "catboost",
            "required_class_available": import_name != "catboost",
        }

    monkeypatch.setattr(model_suite_module, "dependency_status", fake_dependency_status)

    with pytest.raises(RuntimeError, match=r"catboost.*uv sync --group bench --group dev"):
        validate_requested_benchmark_dependencies(["catboost"], [])


def test_non_forecast_required_baselines_are_concrete() -> None:
    baselines = load_config("required_baselines")

    assert baselines["tabular"] == [
        "cartoboost",
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
        "ridge",
        "mean",
        "deep_tabular_baseline",
    ]
    assert "intermittent" not in baselines
    assert baselines["spatial"] == [
        "cartoboost",
        "cartoboost_neural",
        "cartoboost_graph",
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
        "ridge",
        "mean",
        "pysal_spatial_regression",
        "pykrige",
        "gstools",
    ]
    assert baselines["graph"] == [
        "cartoboost",
        "cartoboost_graph",
        "node2vec_baseline",
        "graphsage_baseline",
        "tabularized_graph_baseline",
        "pytorch_geometric_temporal_baseline",
        "dcrnn_baseline",
        "mean",
    ]


def test_forecasting_benchmark_intermittent_roster_is_exposed() -> None:
    assert forecasting_benchmark_module.benchmark_model_names("intermittent") == [
        "croston",
        "sba",
        "tsb",
    ]


def test_forecasting_quality_summary_records_external_baseline_gate() -> None:
    metrics = {
        "cartoboost_auto_forecast": {"rmse": 10.0, "mae": 5.0, "wape": 0.2},
        "lightgbm_lag": {"rmse": 10.5, "mae": 5.5, "wape": 0.21},
        "xgboost_lag": {"rmse": 11.0, "mae": 5.8, "wape": 0.22},
        "functime_snaive": {"rmse": 8.0, "mae": 4.0, "wape": 0.18},
    }

    summary = forecasting_benchmark_module.quality_summary(metrics)

    assert summary["best_external_baseline"] == "lightgbm_lag"
    assert summary["rmse_ratio_vs_best_external_baseline"] == 10.0 / 10.5
    assert summary["external_baseline_rmse_gate_limit"] == 1.05
    assert summary["external_baseline_rmse_gate_passed"] is True


def test_forecasting_quality_gate_requires_three_leakage_safe_origins(tmp_path) -> None:
    artifact = tmp_path / "forecasting.json"
    artifact.write_text(
        json.dumps(
            {
                "rolling_origin": {
                    "folds": 3,
                    "splits": {"one": {}, "two": {}, "three": {}},
                },
                "quality": {
                    "best_external_baseline": "lightgbm_lag",
                    "rmse_ratio_vs_best_external_baseline": 1.02,
                    "external_baseline_rmse_gate_passed": True,
                },
                "comparability_audit": {
                    "same_forecast_rows": True,
                    "selection_uses_outer_test_labels": False,
                },
            }
        ),
        encoding="utf-8",
    )

    report = check_forecasting_quality_gate(artifact)

    assert report["passed"] is True
    assert report["checks"]["minimum_origin_count"] is True


def test_forecasting_quality_gate_rejects_ratio_over_five_percent(tmp_path) -> None:
    artifact = tmp_path / "forecasting.json"
    artifact.write_text(
        json.dumps(
            {
                "rolling_origin": {"folds": 3, "splits": {"one": {}, "two": {}, "three": {}}},
                "quality": {
                    "best_external_baseline": "lightgbm_lag",
                    "rmse_ratio_vs_best_external_baseline": 1.06,
                    "external_baseline_rmse_gate_passed": False,
                },
                "comparability_audit": {
                    "same_forecast_rows": True,
                    "selection_uses_outer_test_labels": False,
                },
            }
        ),
        encoding="utf-8",
    )

    report = check_forecasting_quality_gate(artifact)

    assert report["passed"] is False
    assert report["checks"]["external_rmse_ratio_within_limit"] is False


def test_non_forecast_dataset_identities_are_frozen() -> None:
    specs = {spec.name: spec for spec in load_all_tracks()}
    for track in ["tabular", "spatial", "graph"]:
        for dataset in specs[track].datasets["datasets"]:
            assert dataset["hash"].startswith("sha256:")
            assert dataset["hash"] != "to_be_frozen"
            assert dataset["source_url"]
            assert dataset["source_identity"]

    assert specs["tabular"].datasets["datasets"][0]["id"] == "sklearn_diabetes_regression_v1"
    assert (
        specs["tabular"].datasets["datasets"][1]["id"]
        == "sklearn_california_housing_regression_seed42_5000_v1"
    )
    assert specs["graph"].datasets["datasets"][0]["id"] == "zachary_karate_club_78_edge_v1"


def test_spatial_benchmark_splits_require_geo_manifests() -> None:
    spatial = {spec.name: spec for spec in load_all_tracks()}["spatial"]
    split_ids = {split["id"] for split in spatial.splits["splits"]}

    assert split_ids == {
        "pickup_zone_group_cv_manifest_v1",
        "pickup_zone_buffered_cv_manifest_v1",
        "epa_monitor_buffered_cv_manifest_v1",
        "synthetic_field_spatial_block_manifest_v1",
        "geo_causal_panel_rolling_origin_manifest_v1",
    }
    for split in spatial.splits["splits"]:
        assert split["kind"] in {
            "group_spatial_cv",
            "buffered_spatial_cv",
            "spatial_block_cv",
            "rolling_origin_panel_split",
        }
        assert split["random_row_split_allowed"] is False
        assert split["split_manifest_hash"].startswith("sha256:")
        assert "random" not in split["id"]


def test_official_geo_benchmark_claims_are_declared() -> None:
    specs = load_all_tracks()

    validate_official_geo_benchmark_suite(specs)

    tasks = [task for spec in specs for task in spec.tasks["tasks"] if "claim_family" in task]
    assert {task["claim_family"] for task in tasks} == {
        "nyc_tlc_zone_lane_demand",
        "metr_la_pems_graph_forecasting",
        "epa_air_quality_interpolation",
        "california_housing_sanity",
        "synthetic_spatial_fields",
        "synthetic_graph_diffusion",
        "synthetic_geo_causal_lift_panels",
    }
    for task in tasks:
        assert {"leaderboard_json", "leaderboard_markdown"} <= set(task["scorecard_outputs"])
        assert {
            "fit_wallclock_seconds",
            "predict_wallclock_seconds",
            "peak_memory_mb",
        } <= set(task["resource_metrics"])


def test_release_gates_cover_benchmark_dependencies_and_workflows() -> None:
    assert check_benchmark_dependencies()["passed"] is True
    assert check_external_baseline_install_metadata()["passed"] is True
    assert check_ci_release_gates()["passed"] is True
    assert check_publish_artifact_attestation()["passed"] is True


def test_performance_thresholds_are_enforced() -> None:
    report = check_performance_thresholds()

    assert report["passed"] is True
    assert report["missing_groups"] == []
    assert report["failed_benchmarks"] == []
    assert {row["benchmark"].split("/", 1)[0] for row in report["checked"]} >= {
        "data_loading",
        "prediction",
        "serialize",
        "training",
    }
    assert all(row["headroom_ratio"] > 1.0 for row in report["checked"])


def test_benchmark_freshness_empty_input_is_explicit_noop() -> None:
    report = check_benchmark_freshness([], root=ROOT)

    assert report["passed"] is True
    assert report["artifacts_requested"] == 0
    assert report["checks"] == []


def test_benchmark_freshness_accepts_artifact_from_current_commit(tmp_path) -> None:
    artifact = tmp_path / "benchmark.json"
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact.write_text(json.dumps({"git_commit": current}), encoding="utf-8")

    report = check_benchmark_freshness(
        [artifact],
        root=ROOT,
        current_commit=current,
        allow_dirty=True,
    )

    assert report["passed"] is True
    row = report["checks"][0]
    assert row["commit_is_ancestor"] is True
    assert row["changed_benchmark_files"] == []


def test_benchmark_freshness_rejects_missing_provenance(tmp_path) -> None:
    artifact = tmp_path / "benchmark.json"
    artifact.write_text(json.dumps({"metrics": {"rmse": 1.0}}), encoding="utf-8")

    report = check_benchmark_freshness([artifact], root=ROOT, current_commit="a" * 40)

    assert report["passed"] is False
    assert report["checks"][0]["reason"] == "artifact does not record a valid git_commit"


def test_artifact_compatibility_gate_rejects_unsupported_versions() -> None:
    report = check_artifact_compatibility()

    assert report["passed"] is True
    assert {row["key"] for row in report["cases"]} >= {
        "models.cartoboost_regressor",
        "geo.nngp",
        "geo.residual_nngp",
        "prob.conformal_interval",
        "prob.spatial_conformal",
    }
    for row in report["cases"]:
        assert row["version_markers"]
        assert row["roundtrip_max_abs_diff"] <= 1e-10
        assert row["unsupported_version_rejected"] is True
        assert "unsupported" in row["unsupported_version_error"].lower()


def test_official_geo_evidence_audit_defers_removed_selector_acceptance() -> None:
    report = official_geo_evidence_report()

    assert report["audit_passed"] is True
    assert report["acceptance_passed"] is True
    assert report["selector_shipped"] is False
    assert report["acceptance_scope"] == "deferred_until_native_autogeo_selector"
    assert report["real_autogeo_family_wins"] < report["required_real_autogeo_family_wins"]
    assert report["synthetic_autogeo_gate"]["counts_toward_final_acceptance"] is False
    assert "synthetic gates" in report["claim_policy"].lower()
    assert set(report["families"]) == {
        "nyc_tlc_zone_lane_demand",
        "metr_la_pems_graph_forecasting",
        "epa_air_quality_interpolation",
        "california_housing_sanity",
        "synthetic_spatial_fields",
        "synthetic_graph_diffusion",
        "synthetic_geo_causal_lift_panels",
    }


def test_autogeo_benchmark_gate_is_explicitly_deferred() -> None:
    with pytest.raises(RuntimeError, match="not shipped"):
        autogeo_gate_module.build_workloads(sample_size=60, seed=17)
    with pytest.raises(RuntimeError, match="not shipped"):
        autogeo_gate_module.run_workload(None)


def test_benchmark_navigation_links_resolve() -> None:
    docs = [
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "llms.txt",
        ROOT / "llms.txt",
    ]
    benchmark_links = set()
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        benchmark_links.update(re.findall(r"\((?:\./)?(docs/benchmarks/[^)]+\.md)\)", text))
        benchmark_links.update(re.findall(r": \./(docs/benchmarks/[^\s]+\.md)", text))
        benchmark_links.update(re.findall(r"\((benchmarks/[^)]+\.md)\)", text))

    resolved = []
    for link in benchmark_links:
        path = ROOT / "docs" / link if link.startswith("benchmarks/") else ROOT / link
        resolved.append(path)
        assert path.exists(), f"missing benchmark navigation target: {link}"

    assert ROOT / "docs" / "benchmarks" / "model-suite.md" in resolved
    assert ROOT / "docs" / "benchmarks" / "nyc-taxi.md" in resolved


def test_benchmark_docs_asset_paths_exist() -> None:
    docs = [
        ROOT / "docs" / "benchmarks" / "index.md",
        ROOT / "docs" / "benchmarks" / "model-suite.md",
        ROOT / "docs" / "benchmarks" / "nyc-taxi.md",
        ROOT / "docs" / "benchmarks" / "taxi-zone.md",
    ]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for asset in sorted(set(re.findall(r"`(docs/assets/[^`]+)`", text))):
            path = ROOT / asset
            assert path.exists(), f"{doc.relative_to(ROOT)} references missing asset: {asset}"


def test_benchmark_index_lists_maintained_regression_artifacts() -> None:
    text = (ROOT / "docs" / "benchmarks" / "index.md").read_text(encoding="utf-8")
    expected_paths = [
        "docs/assets/nyc_taxi_benchmarks/results.json",
        "docs/assets/nyc_taxi_benchmarks/results.jsonl",
        "docs/assets/nyc_taxi_benchmarks/results.md",
        "docs/assets/nyc_taxi_benchmarks/repeated_results.json",
        "docs/assets/nyc_taxi_benchmarks/repeated_results.md",
        "docs/assets/model_benchmarks_public/results.json",
        "docs/assets/model_benchmarks_public/results.jsonl",
        "docs/assets/model_benchmarks_public/results_aggregate.json",
        "docs/assets/model_benchmarks_public/results.md",
    ]
    for path in expected_paths:
        assert f"`{path}`" in text
        assert (ROOT / path).exists()


def test_v03_benchmark_quality_gate_is_documented() -> None:
    script = ROOT / "scripts" / "run_v02_modeling_benchmarks.py"
    methodology = (ROOT / "docs" / "benchmarks" / "methodology.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "benchmarks" / "index.md").read_text(encoding="utf-8")

    assert script.exists()
    assert "scripts/run_v02_modeling_benchmarks.py" not in methodology
    assert "scripts/run_v02_modeling_benchmarks.py" not in index
    assert "v0.3 Acceptance Gates" in methodology
    assert "scripts/check_forecasting_quality_gate.py" in methodology


def test_v02_public_python_apis_have_docstring_examples() -> None:
    public_modules = [
        ROOT / "python" / "cartoboost" / "classifier.py",
        ROOT / "python" / "cartoboost" / "ranker.py",
        ROOT / "python" / "cartoboost" / "evaluation.py",
        ROOT / "python" / "cartoboost" / "metrics.py",
    ]
    documented_classes = {"CartoBoostClassifier", "CartoBoostRanker"}
    missing = []

    for path in public_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in documented_classes:
                doc = ast.get_docstring(node) or ""
                if "Example:" not in doc:
                    missing.append(f"{path.relative_to(ROOT)}:{node.name}")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                        item_doc = ast.get_docstring(item) or ""
                        if "Example:" not in item_doc:
                            missing.append(f"{path.relative_to(ROOT)}:{node.name}.{item.name}")
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node) or ""
                if "Example:" not in doc:
                    missing.append(f"{path.relative_to(ROOT)}:{node.name}")

    assert missing == []


def test_geo_system_docs_examples_are_executable() -> None:
    from scripts import check_docs_examples

    assert check_docs_examples.check_docs_reference_contract()["passed"] is True
    assert check_docs_examples.run_model_choice_example()["passed"] is True
    assert check_docs_examples.run_geo_evaluation_example()["passed"] is True
    assert check_docs_examples.run_probabilistic_conformal_example()["passed"] is True


def test_non_forecast_benchmark_docs_use_public_evidence_language() -> None:
    docs = [
        ROOT / "docs" / "benchmarks" / "index.md",
        ROOT / "docs" / "benchmarks" / "lane-level.md",
        ROOT / "docs" / "benchmarks" / "model-suite.md",
        ROOT / "docs" / "benchmarks" / "nyc-taxi.md",
        ROOT / "docs" / "benchmarks" / "taxi-zone.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    assert "tail wins" not in combined
    assert "committed acceptance artifacts" not in combined
    assert "maintained acceptance artifacts" in combined


def test_non_forecast_public_artifacts_exist() -> None:
    paths = [
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results.json",
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results.jsonl",
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results_aggregate.json",
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results.md",
    ]
    for path in paths:
        assert path.exists(), f"missing maintained public model benchmark artifact: {path}"

    report = paths[3].read_text(encoding="utf-8")
    assert "deterministic public tabular workloads and embedded graph diagnostics" in report
    assert "deterministic synthetic workloads" not in report

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert set(payload["workloads"]) == {"diabetes", "california_housing", "karate"}
    assert payload["datasets_requested"] == ["diabetes", "california_housing", "karate"]
    assert payload["benchmark_integrity"]["hpo"] == "inner_train_validation_search"
    assert payload["selection_mode"] == "validation_search"
    assert payload["repeat_seeds"] == [42, 43, 44]
    assert len(payload["repeated_external_baseline_comparison"]) == 4
    assert "catboost" in payload["models_requested"]
    assert payload["trial_budget"]["equal_tunable_trial_budget"] is True
    assert payload["trial_budget"]["tunable_trial_count"] == 3
    assert payload["comparability_audit"]["equal_tunable_trial_budget"] is True
    assert payload["comparability_audit"]["selection_uses_outer_test_labels"] is False
    assert payload["comparability_audit"]["skipped_requested_external_baselines"] == []
    assert payload["resource_usage"]["python"]
    assert payload["baseline_environment"]["xgboost"]["required_class_available"] is True


def test_model_suite_defaults_to_validation_search(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_model_benchmark_suite.py"])
    args = model_suite_module.parse_args()

    assert args.selection_mode == "validation_search"


def test_model_suite_validation_search_grids_are_equal_budget() -> None:
    args = argparse.Namespace(
        n_estimators=24,
        learning_rate=0.08,
        max_depth=4,
        neural_dim=12,
        graph_dim=8,
        graph_epochs=8,
        validation_trials=3,
        min_samples_leaf=None,
    )

    tunable_models = [
        "cartoboost",
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
        "ridge",
        "cartoboost_neural",
        "neural_embedding_regressor",
        "cartoboost_graph_node2vec",
        "cartoboost_graph_graphsage",
        "cartoboost_graph_hetero_graphsage",
        "cartoboost_graph_hinsage",
        "node2vec_regressor",
        "graphsage_regressor",
        "hetero_graphsage_regressor",
        "hinsage_regressor",
        "node2vec_link_predictor",
        "graphsage_link_predictor",
        "hetero_graphsage_link_predictor",
        "hinsage_link_predictor",
    ]

    for model_name in tunable_models:
        grid = model_suite_module.validation_candidate_grid(model_name, args)
        assert len(grid) == 3, model_name

    assert model_suite_module.validation_candidate_grid("mean", args) == []


def test_model_suite_validation_search_uses_inner_validation(tmp_path: Path) -> None:
    pytest.importorskip("sklearn.datasets")
    output_dir = tmp_path / "model_suite_validation_search"
    script = ROOT / "scripts" / "run_model_benchmark_suite.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "diabetes",
            "--models",
            "mean,cartoboost,ridge",
            "--n-estimators",
            "4",
            "--selection-mode",
            "validation_search",
            "--validation-trials",
            "2",
            "--no-plots",
        ],
        check=True,
        cwd=ROOT,
    )

    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))

    assert payload["benchmark_integrity"]["hpo"] == "inner_train_validation_search"
    assert payload["benchmark_integrity"]["validation_trials"] == 2
    assert payload["trial_budget"]["equal_tunable_trial_budget"] is True
    assert payload["trial_budget"]["tunable_trial_count"] == 2
    assert payload["trial_budget"]["models"]["cartoboost"]["requested_trials"] == 2
    assert payload["trial_budget"]["models"]["ridge"]["requested_trials"] == 2
    assert payload["trial_budget"]["models"]["mean"]["tunable"] is False
    assert payload["comparability_audit"]["same_outer_splits"] is True
    assert payload["comparability_audit"]["selection_uses_outer_test_labels"] is False
    assert payload["comparability_audit"]["equal_tunable_trial_budget"] is True
    assert payload["comparability_audit"]["completed_external_baselines"] == ["mean", "ridge"]
    assert payload["model_status_summary"]["completed_external_baselines"] == ["mean", "ridge"]
    assert payload["resource_usage"]["python"]
    assert payload["baseline_environment"]["sklearn"]["module_importable"] is True
    assert payload["output_artifacts"]["results.md"]["size_bytes"] > 0
    report = (output_dir / "results.md").read_text(encoding="utf-8")
    assert "## Comparability Audit" in report
    assert "Equal tunable trial budget" in report
    workload = payload["workloads"]["diabetes"]
    assert workload["source"] == "sklearn.datasets.load_diabetes bundled public regression dataset."
    assert len(workload["fingerprint_sha256"]) == 64
    assert len(workload["splits"]["random"]["train_index_sha256"]) == 64
    assert len(workload["splits"]["random"]["test_index_sha256"]) == 64
    cartoboost = payload["workloads"]["diabetes"]["splits"]["random"]["models"]["cartoboost"]
    assert cartoboost["selection"]["mode"] == "validation_search"
    assert cartoboost["selection"]["inner_train_rows"] > 0
    assert cartoboost["selection"]["inner_validation_rows"] > 0
    assert len(cartoboost["selection"]["validation_rows"]) == 2
    assert cartoboost["selection"]["selected_config"]


def test_model_suite_validation_search_skip_reason_preserves_dependency_error() -> None:
    reason = failed_validation_search_reason(
        [
            {"status": "skipped", "reason": "lightgbm is not installed"},
            {"status": "skipped", "reason": "lightgbm is not installed"},
        ]
    )

    assert reason == "all validation-search candidates failed: lightgbm is not installed"


def test_model_suite_repeated_summary_reports_delta_intervals() -> None:
    payloads = [
        {
            "seed": 11,
            "external_baseline_comparison": [
                {
                    "workload": "california_housing",
                    "split": "random",
                    "cartoboost_wape": 0.22,
                    "best_external_baseline": "xgboost",
                    "best_external_wape": 0.20,
                    "rmse_delta_vs_external": 0.03,
                    "r2_delta_vs_external": -0.02,
                }
            ],
        },
        {
            "seed": 29,
            "external_baseline_comparison": [
                {
                    "workload": "california_housing",
                    "split": "random",
                    "cartoboost_wape": 0.21,
                    "best_external_baseline": "hist_gradient_boosting",
                    "best_external_wape": 0.19,
                    "rmse_delta_vs_external": 0.01,
                    "r2_delta_vs_external": -0.01,
                }
            ],
        },
    ]

    summary = repeated_external_comparison_summary(payloads)

    assert summary[0]["runs"] == 2
    assert summary[0]["seeds"] == [11, 29]
    assert summary[0]["best_external_baseline_counts"] == {
        "hist_gradient_boosting": 1,
        "xgboost": 1,
    }
    assert summary[0]["rmse_delta_mean"] == pytest.approx(0.02)
    assert summary[0]["result"] == "external_lower_rmse"


def test_aggregate_results_reports_confidence_intervals(tmp_path: Path) -> None:
    rows = [
        {"task_id": "fare", "model_family": "cartoboost", "metric": "rmse", "value": 1.0},
        {"task_id": "fare", "model_family": "cartoboost", "metric": "rmse", "value": 2.0},
        {"task_id": "fare", "model_family": "gbdt", "metric": "rmse", "value": 3.0},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = aggregate(read_jsonl(path))

    cartoboost = [
        row
        for row in summary["metrics"]
        if row["task_id"] == "fare" and row["model_family"] == "cartoboost"
    ][0]
    assert cartoboost["n"] == 2
    assert cartoboost["mean"] == 1.5
    assert cartoboost["ci95_low"] < cartoboost["mean"] < cartoboost["ci95_high"]


def test_aggregate_results_preserves_track_and_split_identity(tmp_path: Path) -> None:
    rows = [
        {
            "track": "spatial",
            "task_id": "fare",
            "split_id": "random",
            "model_family": "cartoboost",
            "metric": "rmse",
            "value": 1.0,
        },
        {
            "track": "spatial",
            "task_id": "fare",
            "split_id": "spatial_holdout",
            "model_family": "cartoboost",
            "metric": "rmse",
            "value": 2.0,
        },
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = aggregate(read_jsonl(path))

    grouped = {(row["track"], row["split_id"]): row for row in summary["metrics"]}
    assert grouped[("spatial", "random")]["mean"] == 1.0
    assert grouped[("spatial", "spatial_holdout")]["mean"] == 2.0


def test_paired_bootstrap_ci_uses_paired_deltas() -> None:
    observed, low, high = paired_bootstrap_ci(
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        iterations=100,
        seed=11,
    )

    assert observed == -1.0
    assert low == -1.0
    assert high == -1.0


def test_average_ranks_respects_metric_direction() -> None:
    ranks = average_ranks(
        [
            {"cartoboost": 1.0, "gbdt": 2.0},
            {"cartoboost": 3.0, "gbdt": 2.0},
        ],
        lower_is_better=True,
    )

    assert ranks == {"cartoboost": 1.5, "gbdt": 1.5}
