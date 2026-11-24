import json
import pickle
from base64 import b64encode
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from cartoboost.forecasting import (
    AutoForecaster,
    CartoBoostLagForecaster,
    ForecastFrame,
    LagConfig,
)
from cartoboost.forecasting.artifacts import (
    ForecastArtifact,
    ForecastArtifactManifest,
    build_manifest,
)


def test_forecast_artifact_round_trips_manifest_and_csv(tmp_path):
    manifest = ForecastArtifactManifest(
        model_name="weighted_ensemble",
        horizon=2,
        columns=("pickup_zone", "step", "mean", "lower", "upper"),
        forecast_path="forecast.csv",
        forecast_format="csv",
        freq="H",
        target_column="pickup_demand",
        time_column="pickup_hour",
        panel_columns=("pickup_zone",),
        lower_bound=0,
        feature_config={"lags": [1, 24]},
        params={"season_length": 24},
        backtest_metrics={"mae": 1.25},
        interval_metadata={"level": 0.8},
        ensemble_metadata={"weights": {"naive": 0.4, "seasonal_naive": 0.6}},
        reconciliation_metadata={
            "method": "bottom_up_reconciler",
            "hierarchy": {"Manhattan": ["Midtown"]},
        },
        metadata={"split": "taxi_january_holdout"},
    )
    artifact = ForecastArtifact(
        [
            {"pickup_zone": "Midtown", "step": 1, "mean": 12.5, "lower": 10.0, "upper": 15.0},
            {"pickup_zone": "Midtown", "step": 2, "mean": 13.5, "lower": 11.0, "upper": 16.0},
        ],
        manifest,
    )

    manifest_path = artifact.save(tmp_path)
    loaded = ForecastArtifact.load(tmp_path)

    assert manifest_path.name == "manifest.json"
    assert loaded.forecast == artifact.forecast
    assert loaded.manifest.model_name == "weighted_ensemble"
    assert loaded.manifest.feature_config == {"lags": [1, 24]}
    assert loaded.manifest.ensemble_metadata["weights"]["seasonal_naive"] == 0.6
    assert loaded.manifest.reconciliation_metadata["method"] == "bottom_up_reconciler"
    assert json.loads(manifest_path.read_text())["forecast_path"] == "forecast.csv"


def test_artifact_rejects_rows_missing_manifest_columns():
    manifest = build_manifest(
        model_name="naive",
        horizon=1,
        columns=("step", "mean"),
    )

    with pytest.raises(ValueError, match="missing required column"):
        ForecastArtifact([{"step": 1}], manifest)


def test_parquet_save_hard_fails_without_optional_pyarrow(monkeypatch, tmp_path):
    manifest = build_manifest(
        model_name="naive",
        horizon=1,
        columns=("step", "mean"),
        forecast_format="parquet",
    )
    artifact = ForecastArtifact([{"step": 1, "mean": 3.0}], manifest)

    def missing_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("no pyarrow")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", missing_import)

    with pytest.raises(ImportError, match="pyarrow"):
        artifact.save(tmp_path)


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (
            CartoBoostLagForecaster,
            {"lag_config": LagConfig(lags=[1, 2]), "n_estimators": 4, "max_depth": 2},
        ),
        (AutoForecaster, {"n_estimators": 4}),
    ],
)
def test_stable_forecaster_v2_artifact_roundtrip(model_cls, kwargs, tmp_path):
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=40, freq="D"),
                "fare": np.arange(40, dtype=float),
            }
        ),
        timestamp_col="timestamp",
        target_col="fare",
        freq="D",
    )
    model = model_cls(**kwargs).fit(frame)
    before = model.predict(3).predictions()
    path = tmp_path / f"{model_cls.__name__}.json"
    model.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "cartoboost.model"
    assert payload["artifact_version"] == 2
    assert payload["model_type"] in {"auto_forecaster", "cartoboost_lag"}
    assert {"library_version", "schema_hash", "training_config", "payload"} <= set(payload)

    restored = model_cls.load(path)
    assert restored.get_params() == model.get_params()
    assert restored.predict(3).predictions() == before

    weights_path = tmp_path / f"{model_cls.__name__}.weights.json"
    model.save_weights(weights_path)
    weights_restored = model_cls.load_weights(weights_path)
    assert weights_restored.predict(3).predictions() == before
    assert pickle.loads(pickle.dumps(model)).predict(3).predictions() == before


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("contiguous", [False, True])
def test_refit_artifact_preserves_numpy_and_primitive_fit_data(protocol, contiguous):
    from cartoboost.forecasting._native_wrappers import _decode_fit_artifact

    values = np.arange(12, dtype=np.float64).reshape(4, 3)[:, ::2]
    if contiguous:
        values = np.ascontiguousarray(values)
    args = (values, {"pickup_zone": [np.int64(3), np.float32(2.5)]})
    kwargs = {"start": datetime(2024, 1, 1, tzinfo=timezone.utc), "offset": timedelta(days=1)}
    payload = {
        "args": b64encode(pickle.dumps(args, protocol=protocol)).decode("ascii"),
        "kwargs": b64encode(pickle.dumps(kwargs, protocol=protocol)).decode("ascii"),
    }

    restored_args, restored_kwargs = _decode_fit_artifact(payload)

    np.testing.assert_array_equal(restored_args[0], values)
    assert restored_args[1] == args[1]
    assert restored_kwargs == kwargs


@pytest.mark.parametrize("field", ["args", "kwargs"])
def test_refit_artifact_rejects_executable_pickle_without_execution(tmp_path, field):
    from cartoboost.forecasting.local import ThetaForecaster

    marker = tmp_path / "executed.txt"

    class MaliciousFitData:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    fit = {
        "args": b64encode(pickle.dumps(([1.0, 2.0, 3.0],))).decode("ascii"),
        "kwargs": b64encode(pickle.dumps({})).decode("ascii"),
    }
    fit[field] = b64encode(pickle.dumps(MaliciousFitData())).decode("ascii")
    path = tmp_path / "forecast.json"
    path.write_text(
        json.dumps(
            {
                "format": "cartoboost.native_forecast_refit",
                "artifact_version": 1,
                "native_class_name": ThetaForecaster.native_class_name,
                "params": {},
                "fit": fit,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid fit inputs"):
        ThetaForecaster.load(path)
    assert not marker.exists()


def test_refit_artifact_rejects_cached_pickle_extensions(monkeypatch):
    import copyreg

    from cartoboost.forecasting._native_wrappers import _decode_fit_artifact

    # EXT1 can retrieve a callable from the process cache without find_class.
    monkeypatch.setitem(copyreg._extension_cache, 239, eval)
    payload = {
        "args": b64encode(b"\x80\x02\x82\xef.").decode("ascii"),
        "kwargs": b64encode(pickle.dumps({})).decode("ascii"),
    }
    with pytest.raises(ValueError, match="invalid fit inputs"):
        _decode_fit_artifact(payload)


def test_refit_artifact_preserves_forecast_frame_inputs():
    from cartoboost.forecasting._native_wrappers import _decode_fit_artifact, _encode_fit_artifact

    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "fare": [3, 4, 5],
                "trip_distance": [1.0, 2.0, 3.0],
            }
        ),
        timestamp_col="timestamp",
        target_col="fare",
        freq="D",
        historical_covariates=["trip_distance"],
    )

    args, kwargs = _decode_fit_artifact(_encode_fit_artifact((frame,), {}))

    pd.testing.assert_frame_equal(args[0].to_pandas(), frame.to_pandas())
    assert args[0].historical_covariates == frame.historical_covariates
    assert kwargs == {}


@pytest.mark.parametrize("model_name", ["theta", "arima"])
@pytest.mark.parametrize("input_kind", ["series", "dataframe", "panel"])
def test_refit_artifact_normalizes_supported_pandas_inputs(model_name, input_kind, tmp_path):
    from cartoboost.forecasting.local import ArimaForecaster, ThetaForecaster

    values = pd.Series([2.0, 3.0, 4.0, 3.0, 5.0, 6.0], name="fare")
    if input_kind == "series":
        inputs = values
    elif input_kind == "dataframe":
        inputs = pd.DataFrame({"fare": values, "duration": values + 5.0})
    else:
        # A raw panel id must not collide with the serialized frame envelope.
        inputs = {"forecast_frame": values}
    model = ThetaForecaster() if model_name == "theta" else ArimaForecaster(p=1, d=0, q=0)
    model.fit(inputs)
    expected = model.predict(2).predictions()
    values[:] = -100.0

    path = tmp_path / f"{model_name}-{input_kind}.json"
    model.save(path)
    restored = type(model).load(path)

    assert restored.predict(2).predictions() == expected


def test_lag_mixed_dataframe_snapshot_retains_schema_and_covariates(tmp_path):
    from cartoboost.forecasting._native_wrappers import _decode_fit_artifact

    frame = pd.DataFrame(
        {
            "lane_id": ["PU1-DO2"] * 64,
            "pickup_day": pd.date_range("2026-01-01 00:00:00.125", periods=64),
            "trips": np.arange(64, dtype=float) + 10.0,
            "distance_miles": np.full(64, 2.5),
        }
    )
    model = CartoBoostLagForecaster(
        time_col="pickup_day",
        target_col="trips",
        panel_cols=["lane_id"],
        covariate_features=["distance_miles"],
        lags=[1],
        rolling_windows=[],
        calendar_features=False,
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
    ).fit(frame)
    args, kwargs = _decode_fit_artifact(model._fit_artifact)
    assert kwargs == {}
    snapshot = args[0]
    assert snapshot.to_metadata() == model._training_frame.to_metadata()
    pd.testing.assert_frame_equal(snapshot.to_pandas(), model._training_frame.to_pandas())
    expected = model.predict(2).predictions()
    frame.loc[:, "trips"] = -1000.0
    path = tmp_path / "mixed-lag.json"
    model.save(path)
    restored = CartoBoostLagForecaster.load(path)
    assert restored.predict(2).predictions() == expected


@pytest.mark.parametrize(
    "model_name", ["naive", "seasonal_naive", "theta", "optimized_theta", "lag"]
)
def test_residual_gaussian_intervals_roundtrip_and_refit(model_name, tmp_path):
    from cartoboost.forecasting import (
        NaiveForecaster,
        OptimizedThetaForecaster,
        SeasonalNaiveForecaster,
        ThetaForecaster,
    )

    factories = {
        "naive": (NaiveForecaster, {}),
        "seasonal_naive": (SeasonalNaiveForecaster, {"season_length": 2}),
        "theta": (ThetaForecaster, {"theta": 2.0, "alpha": 0.4}),
        "optimized_theta": (
            OptimizedThetaForecaster,
            {"theta_grid": [1.0, 2.0], "alpha_grid": [0.2, 0.8]},
        ),
        "lag": (
            CartoBoostLagForecaster,
            {
                "lags": [1, 2],
                "rolling_windows": [],
                "calendar_features": False,
                "n_estimators": 3,
                "max_depth": 1,
                "min_samples_leaf": 1,
            },
        ),
    }
    cls, options = factories[model_name]
    values = [10.0 + index * 0.5 + (index * index % 7) for index in range(64)]

    def as_frame(targets):
        return ForecastFrame.from_pandas(
            pd.DataFrame(
                {"timestamp": pd.date_range("2026-01-01", periods=len(targets)), "fare": targets}
            ),
            timestamp_col="timestamp",
            target_col="fare",
            freq="D",
        )

    frame = as_frame(values)
    model = cls(**options, prediction_interval_levels=[0.95, 0.8]).fit(frame)
    point_model = cls(**options).fit(frame)
    expected = json.loads(model.predict(4).to_json())
    assert model.predict(4).predictions() == point_model.predict(4).predictions()
    assert model.get_metadata()["prediction_intervals"]["method"] == "residual_gaussian"
    assert model.get_metadata()["prediction_intervals"]["levels"] == [0.8, 0.95]
    for row in expected["records"]:
        for level in [80, 95]:
            lower, upper = row[f"prediction_lower_p{level}"], row[f"prediction_upper_p{level}"]
            assert lower < row["prediction"] < upper
            assert lower + upper == pytest.approx(2 * row["prediction"])
        assert row["prediction_upper_p95"] > row["prediction_upper_p80"]
    path = tmp_path / f"{model_name}-intervals.json"
    model.save(path)
    restored = cls.load(path)
    assert json.loads(restored.predict(4).to_json()) == expected
    refit_values = [value + 2 * (index % 3) for index, value in enumerate(values)]
    restored.fit(as_frame(refit_values))
    fresh = cls(**options, prediction_interval_levels=[0.8, 0.95]).fit(as_frame(refit_values))
    assert json.loads(restored.predict(4).to_json()) == json.loads(fresh.predict(4).to_json())
    assert json.loads(model.predict(4).to_json()) == expected


def test_naive_residual_intervals_known_values_and_history_requirements():
    from cartoboost.forecasting import NaiveForecaster, SeasonalNaiveForecaster

    result = NaiveForecaster(prediction_interval_levels=[0.8]).fit([1.0, 3.0, 2.0, 6.0]).predict(2)
    for row in json.loads(result.to_json())["records"]:
        width = 1.2815515655446004 * np.sqrt(7 * row["horizon"])
        assert row["prediction_lower_p80"] == pytest.approx(6.0 - width)
        assert row["prediction_upper_p80"] == pytest.approx(6.0 + width)
    with pytest.raises(ValueError, match="two finite residuals"):
        NaiveForecaster(prediction_interval_levels=[0.8]).fit([1.0, 2.0])
    with pytest.raises(ValueError, match="two finite residuals"):
        SeasonalNaiveForecaster(2, prediction_interval_levels=[0.8]).fit([1.0, 2.0, 3.0])
