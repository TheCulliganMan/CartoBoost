from __future__ import annotations

import pandas as pd
from cartoboost.preview.forecasting import (
    ForecastConformalCalibrator,
    ForecastResult,
    PredictionInterval,
    interval_coverage,
    mean_interval_width,
)


def main() -> None:
    future_dates = pd.to_datetime(["2026-01-15", "2026-01-16", "2026-01-15", "2026-01-16"])
    pickup_zones = ["142", "142", "236", "236"]
    mean_forecast = [211.0, 219.0, 178.0, 185.0]
    interval = PredictionInterval(
        level=0.8,
        lower=[201.0, 208.0, 168.0, 174.0],
        upper=[221.0, 230.0, 188.0, 196.0],
    )
    result = ForecastResult.from_predictions(
        timestamps=future_dates,
        predictions=mean_forecast,
        series_id=pickup_zones,
        intervals=[interval],
        prediction_col="forecast",
        series_id_col="PULocationID",
    )

    print(result.to_pandas().to_string(index=False))

    calibrator = ForecastConformalCalibrator(alpha=0.2).fit(
        actual=[197.0, 214.0, 173.0, 181.0],
        prediction=[200.0, 212.0, 176.0, 180.0],
        cutoff_index=[1, 2, 3, 4],
    )
    calibrated = calibrator.predict_interval(mean_forecast, cutoff=5)
    print(
        {
            "coverage": interval_coverage(
                [213.0, 221.0, 180.0, 188.0],
                calibrated.lower,
                calibrated.upper,
            ),
            "mean_width": mean_interval_width(calibrated.lower, calibrated.upper),
            "metadata": calibrated.metadata,
        }
    )


if __name__ == "__main__":
    main()
