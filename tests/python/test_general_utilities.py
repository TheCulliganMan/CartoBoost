import math

import pytest
from cartoboost.preview.forecasting import (
    ReferenceSignal,
    SequenceRow,
    SequenceSeries,
    forward_ekf,
    generate_group_oof_candidate_rows,
    reference_path_viterbi,
    sequence_blend,
    validate_oof_meta_training,
)
from cartoboost.utilities import (
    arima_forecast,
    auto_arima_forecast,
    croston_forecast,
    empirical_variogram,
    ets_forecast,
    fit_ordinary_kriging_variogram,
    kalman_filter,
    local_level_kalman_filter,
    local_linear_trend_kalman_forecast,
    naive_forecast,
    ordinary_kriging_leave_one_out,
    ordinary_kriging_leave_one_out_diagnostics,
    ordinary_kriging_predict,
    sba_forecast,
    seasonal_naive_forecast,
    series_forecast,
    theta_forecast,
    tsb_forecast,
)


def assert_finite_forecast(values, horizon):
    assert len(values) == horizon
    assert all(math.isfinite(value) for value in values)


def test_sequence_wrappers_are_native_backed():
    series = SequenceSeries(
        "pickup_zone_1",
        [
            SequenceRow("r0", 0.0, 0.0),
            SequenceRow("r1", 1.0, 1.0),
            SequenceRow("r2", 2.0, None),
        ],
    )
    reference = ReferenceSignal([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    ekf = forward_ekf(series, reference)
    path = reference_path_viterbi(series, reference)

    assert len(ekf["points"]) == 3
    assert math.isfinite(ekf["points"][-1]["predicted_signal"])
    assert path["points"][1]["axis"] == pytest.approx(1.0)

    blended = sequence_blend(
        [
            {
                "name": "a",
                "predictions": [{"series_id": "s", "row_id": "r", "value": 1.0}],
            },
            {
                "name": "b",
                "predictions": [{"series_id": "s", "row_id": "r", "value": 3.0}],
            },
        ],
        weights={"a": 0.25, "b": 0.75},
    )
    assert blended["predictions"][0]["value"] == pytest.approx(2.5)

    oof_rows = generate_group_oof_candidate_rows(
        validation_group_id="pickup_zone_1",
        train_group_ids=["pickup_zone_2"],
        actuals=[{"series_id": "pickup_zone_1", "row_id": "r1", "value": 1.0}],
        candidates=[
            {
                "name": "a",
                "predictions": [{"series_id": "pickup_zone_1", "row_id": "r1", "value": 1.1}],
            }
        ],
    )
    assert oof_rows[0]["candidate_predictions"]["a"] == pytest.approx(1.1)

    with pytest.raises(ValueError, match="leakage"):
        validate_oof_meta_training(
            [
                {
                    "group_id": "pickup_zone_1",
                    "row_id": "r1",
                    "actual": 1.0,
                    "candidate_predictions": {"a": 1.1},
                    "train_group_ids": ["pickup_zone_1"],
                }
            ]
        )


def test_general_series_forecast_wrappers_are_native_backed():
    assert naive_forecast([1.0, 2.0, 3.0], 2) == [3.0, 3.0]
    assert seasonal_naive_forecast([10.0, 20.0, 30.0, 40.0], 3, season_length=2) == [
        30.0,
        40.0,
        30.0,
    ]
    assert_finite_forecast(theta_forecast([10.0, 11.0, 13.0, 16.0], 2), 2)
    assert_finite_forecast(ets_forecast([10.0, 11.0, 13.0, 16.0], 2), 2)
    assert_finite_forecast(arima_forecast([10.0, 11.0, 13.0, 16.0, 20.0], 2, p=1, d=1), 2)
    assert_finite_forecast(auto_arima_forecast([10.0, 11.0, 13.0, 16.0, 20.0], 2), 2)
    assert_finite_forecast(
        local_linear_trend_kalman_forecast([10.0, 12.0, 14.0, 16.0], 2),
        2,
    )
    assert_finite_forecast(series_forecast("local_level_kalman", [10.0, 12.0, 14.0], 2), 2)


def test_general_intermittent_demand_utilities():
    values = [0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0]

    assert_finite_forecast(croston_forecast(values, 3), 3)
    assert_finite_forecast(sba_forecast(values, 3), 3)
    assert_finite_forecast(tsb_forecast(values, 3), 3)
    assert sba_forecast(values, 1)[0] < croston_forecast(values, 1)[0]

    with pytest.raises(ValueError, match="non-negative"):
        croston_forecast([1.0, -1.0], 1)


def test_general_local_level_kalman_filter():
    result = local_level_kalman_filter([12.0, 14.0, 16.0], horizon=2)

    assert set(result) == {
        "final_state",
        "estimates",
        "smoothed_states",
        "forecast",
        "forecast_distribution",
        "diagnostics",
    }
    assert len(result["estimates"]) == 2
    assert len(result["smoothed_states"]) == 3
    assert result["forecast"] == [result["final_state"]["level"]] * 2
    assert result["final_state"]["variance"] > 0.0
    assert result["estimates"][-1]["standardized_innovation"] == pytest.approx(
        result["estimates"][-1]["innovation"]
        / math.sqrt(result["estimates"][-1]["innovation_variance"])
    )
    assert len(result["forecast_distribution"]) == 2
    assert result["forecast_distribution"][0]["mean"] == result["forecast"][0]
    assert result["forecast_distribution"][0]["lower"] < result["forecast"][0]
    assert result["forecast_distribution"][0]["upper"] > result["forecast"][0]
    assert math.isfinite(result["diagnostics"]["log_likelihood"])
    assert result["diagnostics"]["fitted_count"] == 2
    assert result["diagnostics"]["rmse"] > 0.0


def test_general_kalman_filter_tracks_trend_and_forecasts():
    result = kalman_filter(
        [12.0, 14.0, 16.0, 18.0],
        level_process_variance=0.01,
        trend_process_variance=0.001,
        observation_variance=0.1,
        horizon=2,
    )

    assert set(result) == {
        "final_state",
        "estimates",
        "smoothed_states",
        "forecast",
        "forecast_distribution",
        "diagnostics",
    }
    assert result["final_state"]["trend"] > 0.0
    assert len(result["final_state"]["covariance"]) == 2
    assert len(result["estimates"]) == 3
    assert len(result["smoothed_states"]) == 4
    assert result["estimates"][-1]["innovation_variance"] > 0.0
    assert result["estimates"][-1]["fitted"] == result["estimates"][-1]["prior_level"]
    assert result["forecast"][1] > result["forecast"][0]
    assert result["forecast_distribution"][1]["mean"] == result["forecast"][1]
    assert result["forecast_distribution"][1]["lower"] < result["forecast"][1]
    assert result["forecast_distribution"][1]["upper"] > result["forecast"][1]
    assert result["diagnostics"]["mae"] >= 0.0


def test_general_kalman_filter_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least two observations"):
        kalman_filter([1.0])

    with pytest.raises(ValueError, match="must be positive and finite"):
        kalman_filter([1.0, 2.0], level_process_variance=0.0)


def test_general_ordinary_kriging_predicts_known_coordinate():
    predictions = ordinary_kriging_predict(
        observations=[(0.0, 0.0, 12.0), (10.0, 0.0, 42.0)],
        targets=[(0.0, 0.0), (10.0, 0.0)],
        range=1.0,
        nugget=1.0e-9,
    )

    assert len(predictions) == 2
    assert abs(predictions[0]["mean"] - 12.0) < 1.0e-4
    assert abs(sum(predictions[0]["weights"]) - 1.0) < 1.0e-8


def test_general_ordinary_kriging_detailed_reports_variance_and_neighbors():
    predictions = ordinary_kriging_predict(
        observations=[(0.0, 0.0, 12.0), (10.0, 0.0, 42.0), (20.0, 0.0, 50.0)],
        targets=[(10.0, 0.0)],
        range=5.0,
        nugget=1.0e-6,
        variogram_model="spherical",
        max_neighbors=2,
        min_neighbors=2,
        detailed=True,
    )

    assert len(predictions) == 1
    assert predictions[0]["variance"] >= 0.0
    assert len(predictions[0]["weights"]) == 2
    assert len(predictions[0]["neighbor_indices"]) == 2


def test_general_ordinary_kriging_applies_advanced_config_without_detailed_shape():
    predictions = ordinary_kriging_predict(
        observations=[(0.0, 0.0, 12.0), (10.0, 0.0, 42.0), (20.0, 0.0, 50.0)],
        targets=[(10.0, 0.0)],
        range=5.0,
        nugget=1.0e-6,
        variogram_model="spherical",
        max_neighbors=2,
        min_neighbors=2,
    )

    assert set(predictions[0]) == {"x", "y", "mean", "weights"}
    assert len(predictions[0]["weights"]) == 2


def test_general_ordinary_kriging_leave_one_out_reports_each_observation():
    diagnostics = ordinary_kriging_leave_one_out(
        observations=[(0.0, 0.0, 12.0), (10.0, 0.0, 42.0), (20.0, 0.0, 50.0)],
        range=5.0,
        nugget=1.0e-6,
    )

    assert len(diagnostics) == 3
    assert all("variance" in row for row in diagnostics)


def test_general_empirical_variogram_and_fit_report_statistics():
    observations = [(0.0, 0.0, 10.0), (1.0, 0.0, 12.0), (2.0, 0.0, 16.0), (3.0, 0.0, 20.0)]

    bins = empirical_variogram(observations, bin_count=3)
    fit = fit_ordinary_kriging_variogram(
        observations,
        variogram_models=["exponential", "spherical"],
        range_candidates=[1.0, 2.0],
        nugget_candidates=[0.0, 0.1],
        sill_candidates=[1.0, 5.0],
        bin_count=3,
    )
    diagnostics = ordinary_kriging_leave_one_out_diagnostics(
        observations,
        range=fit["config"]["range"],
        nugget=fit["config"]["nugget"],
        sill=fit["config"]["sill"],
        variogram_model=fit["config"]["variogram_model"],
    )

    assert bins
    assert fit["weighted_sse"] >= 0.0
    assert fit["config"]["variogram_model"] in {"exponential", "spherical"}
    assert diagnostics["diagnostics"]["observation_count"] == len(observations)
    assert diagnostics["diagnostics"]["rmse"] >= 0.0


def test_general_ordinary_kriging_rejects_bad_inputs():
    with pytest.raises(ValueError, match="observations must not be empty"):
        ordinary_kriging_predict([], [(0.0, 0.0)])

    with pytest.raises(ValueError, match="nugget"):
        ordinary_kriging_predict([(0.0, 0.0, 1.0)], [(0.0, 0.0)], nugget=-1.0)
