import json

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
