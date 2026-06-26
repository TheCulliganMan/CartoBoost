import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Neural Panel

`NeuralPanelForecaster` is the Rust-native neural panel forecaster for direct
multi-horizon forecasts across related taxi-demand series. It uses
`ForecastFrame` rows with `(series_id, timestamp, target, covariates)`, learns
from train-only normalized windows, and keeps panel series isolated while
sharing the same configured neural structure.

Use `LaneNeuralPanelForecaster` when `series_id` is a directional taxi lane such
as `PULocationID:DOLocationID`. The lane wrapper keeps `132:138` distinct from
`138:132`, records origin/destination/lane metadata, and can forecast requested
cold lane ids through `predict_for_lanes()`.

## Interactive Example

<ForecastModelExample title="Neural panel taxi-lane forecast" model="neural_panel" />

The embedded example runs the browser-local wasm forecast path for
`neural_panel` on taxi-lane rows. Use it as a quick syntax and shape check, not
as quality evidence. Quality claims still need the maintained split benchmark
against seasonal naive and `CartoBoostLagForecaster`.

## When To Use

Use this model when the hypothesis needs:

- direct multi-horizon panel forecasts rather than recursive local forecasts;
- fitted target-lag state through `n_lags`;
- future-known regressors, lagged regressors, event offsets, or Fourier terms;
- local/global component behavior by series id;
- quantile output with non-crossing repair;
- directional lane ids and explicit cold-lane fallback.

Prefer `CartoBoostLagForecaster` when tabular lag features and tree splits are
the main signal. Prefer local models such as theta, ETS, ARIMA, Kalman, or
piecewise linear seasonal forecasting when one series needs an inspectable
statistical baseline.

## Python Example

```python
from cartoboost.forecasting import ForecastFrame, LaneNeuralPanelForecaster

frame = ForecastFrame.from_pandas(
    hourly_lane_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_trips",
    series_id_col="pickup_dropoff_lane",
    freq="h",
    known_future_covariates=["is_airport_event"],
    historical_covariates=["avg_trip_distance"],
)

model = LaneNeuralPanelForecaster(
    n_lags=24,
    n_forecasts=6,
    quantiles=[0.1, 0.5, 0.9],
    weekly_fourier_order=3,
    future_regressors={"is_airport_event": "additive"},
    lagged_regressors={"avg_trip_distance": 24},
    trend_mode="glocal",
    local_l2=0.1,
    seed=42,
)
model.fit(frame)
forecast = model.predict(6)
cold_lane_forecast = model.predict_for_lanes(6, ["132:138", "132:999"])
```

## Benchmark Check

```bash
uv run --group dev python scripts/forecasting_library_benchmark.py \
  --source polars \
  --model-roster neural-panel \
  --neural-panel-splits \
  --lanes 36 \
  --days 180 \
  --horizon 14 \
  --suite-folds 1 \
  --output target/neural_panel_taxi_lane_split_suite.json
```

The JSON artifact records the command, split definitions, model settings,
RMSE/MAE/WAPE metrics, timing, resource usage, and output path for
rolling-origin, cold-lane, cold-origin, and sparse-tail checks.
