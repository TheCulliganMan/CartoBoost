import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

try:
    from cartoboost.forecasting.probabilistic import (
        ConformalIntervalRegressor,
        ForecastConformalCalibrator,
        QuantileCartoBoostRegressor,
        SpatialConformalRegressor,
        benchmark_calibration_report_fields,
        crps_approximation,
        group_conformal_residual_quantiles,
        interval_coverage,
        mean_interval_width,
        nearest_conformal_residual_quantiles,
        pinball_loss,
        pit_bins,
        weighted_conformal_residual_quantile,
        weighted_interval_score,
    )
except ImportError:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "cartoboost"
        / "forecasting"
        / "probabilistic.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cartoboost_probabilistic_under_test",
        module_path,
    )
    probabilistic = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = probabilistic
    spec.loader.exec_module(probabilistic)
    ConformalIntervalRegressor = probabilistic.ConformalIntervalRegressor
    ForecastConformalCalibrator = probabilistic.ForecastConformalCalibrator
    QuantileCartoBoostRegressor = probabilistic.QuantileCartoBoostRegressor
    SpatialConformalRegressor = probabilistic.SpatialConformalRegressor
    benchmark_calibration_report_fields = probabilistic.benchmark_calibration_report_fields
    crps_approximation = probabilistic.crps_approximation
    group_conformal_residual_quantiles = probabilistic.group_conformal_residual_quantiles
    interval_coverage = probabilistic.interval_coverage
    mean_interval_width = probabilistic.mean_interval_width
    nearest_conformal_residual_quantiles = probabilistic.nearest_conformal_residual_quantiles
    pinball_loss = probabilistic.pinball_loss
    pit_bins = probabilistic.pit_bins
    weighted_conformal_residual_quantile = probabilistic.weighted_conformal_residual_quantile
    weighted_interval_score = probabilistic.weighted_interval_score


class MeanEstimator:
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


class SerializableMeanEstimator(MeanEstimator):
    def save(self, path):
        Path(path).write_text(str(self.mean_), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.mean_ = float(Path(path).read_text(encoding="utf-8"))
        return obj


class LastValueForecaster:
    def fit(self, values, _unused_target=None):
        self.last_ = float(np.asarray(values, dtype=float)[-1])
        return self

    def predict(self, horizon):
        return np.full(int(horizon), self.last_)


class RecordingCartoBoostRegressor:
    calls = []

    def __init__(self, **params):
        self.params = params
        self.quantile_alpha = float(params["quantile_alpha"])
        self.__class__.calls.append(("init", params))

    def fit(self, x, y):
        self.__class__.calls.append(("fit", self.quantile_alpha, len(x), len(y)))
        self.center_ = float(np.mean(y))
        return self

    def predict(self, x):
        offset = (self.quantile_alpha - 0.5) * 10.0
        return np.full(len(x), self.center_ + offset)


def test_conformal_interval_regressor_covers_synthetic_holdout():
    x_train = np.arange(30).reshape(-1, 1)
    y_train = np.full(30, 10.0)
    x_cal = np.arange(30, 80).reshape(-1, 1)
    y_cal = 10.0 + np.array([(-1) ** idx * (idx % 5) for idx in range(50)])
    x_test = np.arange(80, 100).reshape(-1, 1)
    y_test = 10.0 + np.array([0.0, 1.0, -1.0, 2.0, -2.0] * 4)

    model = ConformalIntervalRegressor(MeanEstimator(), alpha=0.1).fit(
        x_train,
        y_train,
        x_cal,
        y_cal,
        train_end_exclusive=30,
        calibration_start=30,
        calibration_end_exclusive=80,
        test_start=80,
    )
    interval = model.predict_interval(x_test, test_start=80)

    assert interval_coverage(y_test, interval.lower, interval.upper) >= 0.9
    assert mean_interval_width(interval.lower, interval.upper) > 0.0


def test_conformal_interval_regressor_save_load_preserves_intervals(tmp_path):
    model = ConformalIntervalRegressor(SerializableMeanEstimator(), alpha=0.2).fit(
        [[0], [1]],
        [10.0, 10.0],
        [[2], [3], [4]],
        [9.0, 10.0, 12.0],
        train_end_exclusive=2,
        calibration_start=2,
        calibration_end_exclusive=5,
        test_start=5,
    )
    before = model.predict_interval([[5], [6]], test_start=5)

    path = tmp_path / "conformal.json"
    model.save(path)
    loaded = ConformalIntervalRegressor.load(path)
    after = loaded.predict_interval([[5], [6]], test_start=5)

    np.testing.assert_allclose(after.lower, before.lower)
    np.testing.assert_allclose(after.upper, before.upper)


def test_conformal_rejects_holdout_leakage_ordering():
    with pytest.raises(ValueError, match="calibration rows must end before test rows start"):
        ConformalIntervalRegressor(MeanEstimator()).fit(
            [[0]],
            [1.0],
            [[1]],
            [1.0],
            train_end_exclusive=1,
            calibration_start=1,
            calibration_end_exclusive=3,
            test_start=2,
        )


def test_spatial_conformal_uses_group_specific_widths():
    x_train = np.arange(4).reshape(-1, 1)
    y_train = np.full(4, 10.0)
    x_cal = np.arange(8).reshape(-1, 1)
    y_cal = np.array([9.0, 11.0, 9.0, 11.0, 5.0, 15.0, 6.0, 14.0])
    groups = np.array(["pickup_142"] * 4 + ["pickup_236"] * 4)

    model = SpatialConformalRegressor(MeanEstimator(), alpha=0.25).fit(
        x_train,
        y_train,
        x_cal,
        y_cal,
        groups=groups,
        train_end_exclusive=4,
        calibration_start=4,
        calibration_end_exclusive=12,
        test_start=12,
    )
    interval = model.predict_interval(
        [[0], [1]],
        test_start=12,
        groups=["pickup_142", "pickup_236"],
    )

    assert interval.upper[1] - interval.lower[1] > interval.upper[0] - interval.lower[0]


def test_forecast_conformal_uses_only_past_cutoff_residuals():
    calibrator = ForecastConformalCalibrator(alpha=0.1).fit(
        actual=[10.0, 11.0, 14.0, 50.0],
        prediction=[10.0, 10.0, 10.0, 10.0],
        cutoff_index=[1, 2, 3, 4],
    )

    assert calibrator.residual_quantile_for_cutoff(4) == 4.0
    with pytest.raises(ValueError, match="requires past cutoff residuals"):
        calibrator.residual_quantile_for_cutoff(1)


def test_conformal_interval_regressor_wraps_forecaster_shaped_predictor():
    model = ConformalIntervalRegressor(LastValueForecaster(), alpha=0.2).fit(
        [10.0, 11.0, 12.0],
        [0.0, 0.0, 0.0],
        4,
        [12.0, 14.0, 10.0, 13.0],
        train_end_exclusive=3,
        calibration_start=3,
        calibration_end_exclusive=7,
        test_start=7,
    )

    interval = model.predict_interval(3, test_start=7)

    assert interval.lower.shape == (3,)
    assert interval.upper.shape == (3,)
    assert interval.metadata["method"] == "split_conformal"


def test_quantile_cartoboost_regressor_trains_one_model_per_quantile(monkeypatch):
    RecordingCartoBoostRegressor.calls = []
    monkeypatch.setitem(
        sys.modules,
        "cartoboost",
        SimpleNamespace(CartoBoostRegressor=RecordingCartoBoostRegressor),
    )

    model = QuantileCartoBoostRegressor(
        quantiles=(0.1, 0.5, 0.9),
        n_estimators=3,
        splitters=["axis"],
    ).fit([[0.0], [1.0], [2.0]], [10.0, 12.0, 14.0])
    distribution = model.predict_distribution([[3.0], [4.0]])

    assert [call[0] for call in RecordingCartoBoostRegressor.calls].count("init") == 3
    assert [call[0] for call in RecordingCartoBoostRegressor.calls].count("fit") == 3
    assert np.all(distribution.quantiles[0.1] <= distribution.quantiles[0.5])
    assert np.all(distribution.quantiles[0.5] <= distribution.quantiles[0.9])
    assert distribution.calibration_metadata["method"] == "quantile_cartoboost"


def test_distributional_metrics_score_quantile_rows():
    actual = [1.0, 2.0]
    quantiles = [0.1, 0.5, 0.9]
    predictions = [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]

    assert crps_approximation(actual, quantiles, predictions) >= 0.0
    assert weighted_interval_score(actual, [1.0, 2.0], [(0.2, [0.0, 1.0], [2.0, 3.0])]) >= 0.0
    assert sum(pit_bins(actual, quantiles, predictions, bins=5)["counts"]) == 2


def test_distributional_metric_helpers_delegate_to_native_when_available(monkeypatch):
    package = ModuleType("cartoboost")
    package.__path__ = []
    native = ModuleType("cartoboost._native")
    calls = []

    def native_pinball(actual, prediction, quantile):
        calls.append((actual, prediction, quantile))
        return 123.0

    native.prob_pinball_loss_value = native_pinball
    monkeypatch.setitem(sys.modules, "cartoboost", package)
    monkeypatch.setitem(sys.modules, "cartoboost._native", native)

    assert pinball_loss([1.0], [0.5], 0.5) == 123.0
    assert calls == [([1.0], [0.5], 0.5)]


def test_spatial_calibration_helpers_cover_groups_weights_and_nearest_residuals():
    actual = [10.0, 20.0, 100.0]
    prediction = [9.0, 18.0, 90.0]

    weighted = weighted_conformal_residual_quantile(
        actual,
        prediction,
        weights=[1.0, 2.0, 10.0],
        alpha=0.1,
    )
    grouped = group_conformal_residual_quantiles(
        actual,
        prediction,
        groups=["h3:892a", "h3:892a", "s2:89c25"],
        alpha=0.1,
    )
    nearest = nearest_conformal_residual_quantiles(
        actual,
        prediction,
        calibration_coordinates=[[0.0, 0.0], [1.0, 1.0], [100.0, 100.0]],
        query_coordinates=[[0.1, 0.1], [99.0, 99.0]],
        neighbor_count=1,
        alpha=0.1,
    )

    assert weighted == 10.0
    assert grouped == {"h3:892a": 2.0, "s2:89c25": 10.0}
    assert nearest.tolist() == [1.0, 10.0]


def test_benchmark_calibration_report_fields_emit_required_breakdowns():
    fields = benchmark_calibration_report_fields(
        y_true=[10.0, 12.0, 20.0, 25.0],
        lower=[9.0, 11.0, 15.0, 24.0],
        upper=[11.0, 13.0, 22.0, 24.5],
        horizons=[1, 1, 2, 2],
        spatial_blocks=["pickup_142", "pickup_142", "pickup_236", "pickup_236"],
        residual_morans_i_after_calibration=0.05,
    )

    assert fields["coverage_by_horizon"] == {1: 1.0, 2: 0.5}
    assert fields["coverage_by_spatial_block"] == {"pickup_142": 1.0, "pickup_236": 0.5}
    assert fields["width_by_horizon"] == {1: 2.0, 2: 3.75}
    assert fields["residual_morans_i_after_calibration"] == 0.05
