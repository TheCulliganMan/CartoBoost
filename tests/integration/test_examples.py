from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_example(*args: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def test_taxi_od_graph_example_runs() -> None:
    payload = run_example(
        "examples/03_taxi_od_graph_regression.py",
        "--sample-size",
        "600",
        "--graph-epochs",
        "1",
        "--graph-dim",
        "3",
        "--n-estimators",
        "8",
    )

    assert payload["rows"] == 600
    assert payload["graph_augmented"]["feature_count"] > 4
    assert np.isfinite(payload["dense"]["rmse"])
    assert np.isfinite(payload["graph_augmented"]["rmse"])


def test_dependency_free_quickstart_runs() -> None:
    payload = run_example("examples/quickstart.py")

    assert payload["rows"] == 240
    assert payload["train_rows"] == 192
    assert payload["validation_rows"] == 48
    assert np.isfinite(float(payload["baseline_rmse"]))
    assert np.isfinite(float(payload["structured_rmse"]))
    assert float(payload["roundtrip_max_abs_diff"]) < 1e-12


def test_taxi_pickup_zone_graph_example_runs_and_improves_spatial_holdout() -> None:
    payload = run_example("examples/04_taxi_pickup_zone_graph.py")

    assert payload["task"] == "pickup_zone_demand_spatial_holdout"
    assert payload["graph_augmented"]["graph_edges"] > 0
    assert payload["graph_augmented"]["feature_count"] > 3
    assert payload["graph_augmented"]["rmse"] < payload["dense"]["rmse"]


def test_neural_embedding_example_runs_and_guards_cold_ids() -> None:
    payload = run_example(
        "examples/05_neural_embedding_regression.py",
        "--rows",
        "500",
        "--ids",
        "50",
        "--n-estimators",
        "80",
        "--embedding-dim",
        "8",
    )

    assert payload["task"] == "neural_embedding_regression"
    assert payload["rows"] == 500
    assert payload["random"]["neural_embedding"]["rmse"] < payload["random"]["dense"]["rmse"]
    assert payload["random"]["guarded"]["selected"] == "neural_embedding"
    assert payload["cold_id_holdout"]["guarded"]["selected"] == "dense"
    assert np.isfinite(payload["cold_id_holdout"]["guarded"]["rmse"])


def test_arima_example_visualization_runs_without_plot() -> None:
    payload = run_example(
        "examples/forecasting/arima_example_visualization.py",
        "--hours",
        "48",
        "--train-hours",
        "36",
        "--horizon",
        "6",
    )

    assert payload["task"] == "example_taxi_lane_arima_forecast"
    assert payload["rows"] == 96
    assert payload["lanes"] == 2
    assert np.isfinite(payload["arima_2_1_1"]["rmse"])
    assert np.isfinite(payload["arima_2_1_1"]["bias"])
    assert np.isfinite(payload["auto_arima"]["mae"])
    assert payload["auto_arima_selected_label"].startswith("ARIMA(")
    assert len(payload["auto_arima_top_candidates"]) > 0
    assert payload["auto_arima_metadata"]["selected_order"] is not None
    assert payload["heldout_winner_by_rmse"] in {"arima_2_1_1", "auto_arima"}
    assert len(payload["residuals"]["arima_2_1_1"]) == 6
    assert len(payload["residuals"]["auto_arima"]) == 6


def test_kalman_diagnostics_visualization_example_runs_and_writes_plot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_example("examples/forecasting/kalman_diagnostics_visualization.py")
    plot_path = repo_root / str(payload["plot"])

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert np.isfinite(float(payload["final_level"]))
    assert np.isfinite(float(payload["final_trend"]))
    assert np.isfinite(float(payload["rmse"]))
    assert np.isfinite(float(payload["mae"]))


def test_cartoboost_lag_visualization_example_runs_and_writes_plot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_example(
        "examples/forecasting/cartoboost_lag_visualization.py",
        "--hours",
        "60",
        "--train-hours",
        "48",
        "--horizon",
        "6",
    )
    plot_path = repo_root / str(payload["plot"])

    assert payload["task"] == "example_taxi_zone_cartoboost_lag_forecast"
    assert payload["rows"] == 180
    assert payload["zones"] == 3
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert np.isfinite(payload["lag_model"]["rmse"])
    assert np.isfinite(payload["lag_model"]["mae"])


def test_weighted_ensemble_visualization_example_runs_and_writes_plot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_example(
        "examples/forecasting/weighted_ensemble_visualization.py",
        "--hours",
        "60",
        "--train-hours",
        "48",
        "--horizon",
        "6",
    )
    plot_path = repo_root / str(payload["plot"])

    assert payload["task"] == "example_taxi_lane_weighted_ensemble_forecast"
    assert payload["rows"] == 120
    assert payload["lanes"] == 2
    assert payload["weights"] == {"kalman": 0.15, "seasonal": 0.55, "theta": 0.3}
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert np.isfinite(payload["ensemble"]["rmse"])
    assert np.isfinite(payload["seasonal"]["mae"])


def test_ets_component_visualization_example_runs_and_writes_plot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_example(
        "examples/forecasting/ets_component_visualization.py",
        "--hours",
        "72",
        "--train-hours",
        "56",
        "--horizon",
        "8",
    )
    plot_path = repo_root / str(payload["plot"])

    assert payload["task"] == "example_taxi_zone_ets_components"
    assert payload["PULocationID"] == "132"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert np.isfinite(float(payload["metrics"]["rmse"]))
    assert np.isfinite(float(payload["metrics"]["mae"]))
    assert np.isfinite(float(payload["final_level"]))
    assert np.isfinite(float(payload["final_trend"]))
    assert float(payload["seasonal_range"]) > 0.0


def test_theta_optimized_visualization_example_runs_and_writes_plot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_example(
        "examples/forecasting/theta_optimized_visualization.py",
        "--hours",
        "72",
        "--train-hours",
        "48",
        "--horizon",
        "12",
    )
    plot_path = repo_root / str(payload["plot"])

    assert payload["task"] == "example_taxi_zone_theta_forecast"
    assert payload["rows"] == 144
    assert payload["zones"] == 2
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert np.isfinite(float(payload["manual_theta"]["rmse"]))
    assert np.isfinite(float(payload["optimized_theta"]["rmse"]))
    assert np.isfinite(float(payload["best_holdout_grid_candidate"]["rmse"]))


def test_kriging_example_visualization_runs_and_writes_assets(tmp_path: Path) -> None:
    payload = run_example(
        "examples/forecasting/kriging_example_visualization.py",
        "--output-dir",
        str(tmp_path),
    )

    assert payload["task"] == "example_taxi_zone_kriging"
    assert payload["zones"] == 8
    assert payload["selected_config"]["variogram_model"] in {
        "exponential",
        "spherical",
        "gaussian",
    }
    assert np.isfinite(float(payload["diagnostics"]["rmse"]))
    assert np.isfinite(float(payload["diagnostics"]["mae"]))
    assert np.isfinite(float(payload["diagnostics"]["average_variance"]))

    for asset_path in payload["assets"].values():
        path = Path(asset_path)
        assert path.exists()
        assert path.stat().st_size > 0

    summary_path = tmp_path / "kriging_example_summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text())["task"] == payload["task"]


def test_naive_seasonal_visualization_example_runs_and_writes_plot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_example(
        "examples/forecasting/naive_seasonal_visualization.py",
        "--hours",
        "96",
        "--train-hours",
        "72",
        "--horizon",
        "24",
    )
    plot_path = repo_root / str(payload["plot"])

    assert payload["task"] == "example_taxi_zone_naive_seasonal_forecast"
    assert payload["rows"] == 192
    assert payload["zones"] == 2
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert np.isfinite(float(payload["naive"]["rmse"]))
    assert np.isfinite(float(payload["seasonal_naive"]["rmse"]))
    assert payload["seasonal_naive"]["rmse"] < payload["naive"]["rmse"]
