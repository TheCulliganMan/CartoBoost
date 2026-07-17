import pytest
from cartoboost.forecasting.ensemble import (
    BacktestWeightedEnsembleForecaster,
    BottomUpReconciler,
    MinTraceReconciler,
    WeightedEnsembleForecaster,
)
from cartoboost.forecasting.local import KalmanForecaster, NaiveForecaster, SeasonalNaiveForecaster


def test_weighted_ensemble_requires_models():
    with pytest.raises(ValueError, match="requires at least one model"):
        WeightedEnsembleForecaster(models={})


def test_weighted_ensemble_validates_weights_match_models():
    with pytest.raises(ValueError, match="weights must match models exactly"):
        WeightedEnsembleForecaster(
            models={"last": NaiveForecaster()},
            weights={"other": 1.0},
        )


def test_weighted_ensemble_rejects_unsupported_intervals():
    with pytest.raises(NotImplementedError, match="prediction intervals are not supported"):
        WeightedEnsembleForecaster(
            models={"last": NaiveForecaster()},
            interval_level=0.9,
        )


def test_weighted_ensemble_fits_native_members_for_single_series():
    model = WeightedEnsembleForecaster(
        models={
            "last": NaiveForecaster(),
            "seasonal": SeasonalNaiveForecaster(season_length=2),
        },
        weights={"last": 1.0, "seasonal": 3.0},
    )

    model.fit([10.0, 12.0, 14.0])
    result = model.predict(2)

    assert [row[-1] for row in result.predictions()] == [12.5, 14.0]
    assert model.get_metadata()["weights"] == {"last": 0.25, "seasonal": 0.75}


def test_weighted_ensemble_aligns_panel_series():
    model = WeightedEnsembleForecaster(
        models={
            "last": NaiveForecaster(),
            "seasonal": SeasonalNaiveForecaster(season_length=2),
        }
    )

    model.fit({"PU1->DO2": [10.0, 12.0, 14.0], "PU9->DO8": [30.0, 28.0, 26.0]})
    result = model.predict(1)

    assert result.predictions() == [
        ("PU1->DO2", "1970-01-04T00:00:00", 1, "weighted_ensemble", 13.0),
        ("PU9->DO8", "1970-01-04T00:00:00", 1, "weighted_ensemble", 27.0),
    ]


def test_weighted_ensemble_accepts_kalman_member():
    model = WeightedEnsembleForecaster(
        models={
            "seasonal": SeasonalNaiveForecaster(season_length=2),
            "kalman": KalmanForecaster(
                level_process_variance=0.05,
                trend_process_variance=0.005,
                observation_variance=1.0,
            ),
        },
        weights={"seasonal": 0.6, "kalman": 0.4},
    )

    model.fit([10.0, 12.0, 14.0, 18.0, 20.0])
    predictions = model.predict(2).predictions()

    assert [row[3] for row in predictions] == ["weighted_ensemble", "weighted_ensemble"]
    assert [row[2] for row in predictions] == [1, 2]
    assert all(row[-1] > 0.0 for row in predictions)
    assert model.get_metadata()["weights"] == {"kalman": 0.4, "seasonal": 0.6}


def test_backtest_weighted_ensemble_fit_requires_rust_binding():
    model = BacktestWeightedEnsembleForecaster(
        models={"last": NaiveForecaster()},
        backend="cuda",
    )
    assert model.get_params()["backend"] == "cuda"
    with pytest.raises(
        NotImplementedError,
        match="Rust binding.*BacktestWeightedEnsembleForecaster",
    ):
        model.fit([1.0, 2.0])


def test_bottom_up_reconciler_requires_hierarchy_surface():
    with pytest.raises(ValueError, match="requires hierarchy"):
        BottomUpReconciler()


def test_bottom_up_reconciler_passes_hierarchy_to_native_binding(install_fake_native):
    native = install_fake_native("BottomUpReconciler")
    model = BottomUpReconciler(
        hierarchy={"all": ["PU1", "PU2"]},
        series_id_column="PULocationID",
        non_negative=True,
        backend="cuda",
    )

    model.fit([1.0, 2.0]).predict(1)

    assert native.calls[0] == (
        "init",
        {
            "hierarchy": {"all": ["PU1", "PU2"]},
            "summing_matrix": None,
            "series_id_column": "PULocationID",
            "parent_column": None,
            "child_column": None,
            "non_negative": True,
            "metadata": {},
            "backend": "cuda",
        },
    )


def test_min_trace_reconciler_passes_covariance_config_to_native_binding(
    install_fake_native,
):
    native = install_fake_native("MinTraceReconciler")
    model = MinTraceReconciler(
        parent_column="borough",
        child_column="PULocationID",
        covariance_method="sample",
        backend="webgpu",
    )

    model.fit([1.0, 2.0]).predict(1)

    assert native.calls[0][1]["parent_column"] == "borough"
    assert native.calls[0][1]["child_column"] == "PULocationID"
    assert native.calls[0][1]["covariance_method"] == "sample"
    assert native.calls[0][1]["backend"] == "webgpu"
