from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cartoboost.preview.forecasting import (  # noqa: E402
    ForecastFrame,
    ForecastingConfig,
    ThetaForecaster,
)


def _panel_frame() -> ForecastFrame:
    rows = []
    for zone, base in [("PULocationID=161", 20.0), ("PULocationID=236", 35.0)]:
        for day in range(10):
            rows.append(
                {
                    "PULocationID": zone,
                    "pickup_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "pickup_demand": base + day * 0.5 + (day % 7) * 1.5,
                }
            )
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="pickup_date",
        target_col="pickup_demand",
        series_id_col="PULocationID",
        freq="D",
    )


def test_full_stack_uses_rust_theta_binding() -> None:
    frame = _panel_frame()
    model = ThetaForecaster(season_length=7, prediction_interval_levels=[0.8, 0.95])

    forecast = model.fit(frame).predict(2)

    assert len(forecast.predictions()) == 4
    assert {row[3] for row in forecast.predictions()} == {"theta"}


def test_documented_toml_shape_parses() -> None:
    config = ForecastingConfig.from_toml(
        """
        horizon = 14
        freq = "D"
        target_column = "pickup_demand"
        time_column = "pickup_date"
        panel_columns = ["PULocationID"]

        [[models]]
        name = "theta"

        [models.params]
        season_length = 7
        prediction_interval_levels = [0.8, 0.95]
        """
    )

    assert config.horizon == 14
    assert config.models[0].name == "theta"
