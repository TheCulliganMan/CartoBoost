from __future__ import annotations

import pandas as pd
from cartoboost.forecasting.global_models import CartoBoostLagForecaster
from cartoboost.forecasting.lag_features import (
    CalendarFeatureConfig,
    LagConfig,
    RollingFeatureConfig,
)


def build_pickup_demand() -> pd.DataFrame:
    rows = []
    for pickup_zone, base in [("zone_4", 42.0), ("zone_79", 65.0)]:
        for hour in range(36):
            timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
            commute_bump = 8.0 if timestamp.hour in {8, 17, 18} else 0.0
            weekend_discount = -5.0 if timestamp.dayofweek >= 5 else 0.0
            rows.append(
                {
                    "PULocationID": pickup_zone,
                    "pickup_hour": timestamp,
                    "pickup_trips": base + commute_bump + weekend_discount + 0.4 * hour,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    training = build_pickup_demand()

    forecaster = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_trips",
        panel_cols=["PULocationID"],
        frequency="H",
        lag_config=LagConfig(lags=[1, 2, 24]),
        rolling_config=RollingFeatureConfig(windows=[3, 6], aggregations=["mean"]),
        calendar_config=CalendarFeatureConfig(features=["dayofweek", "month", "day"]),
    ).fit(training)

    result = forecaster.predict(2)
    forecast = pd.DataFrame(
        result.predictions(),
        columns=["PULocationID", "pickup_hour", "horizon", "model", "forecast"],
    )
    print(forecast[["PULocationID", "pickup_hour", "horizon", "forecast"]].to_string(index=False))


if __name__ == "__main__":
    main()
