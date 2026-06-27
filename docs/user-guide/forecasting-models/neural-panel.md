import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Neural Panel

`NeuralPanelForecaster` is the neural panel forecaster for direct
multi-horizon forecasts across related series. It uses `ForecastFrame` rows
with `(series_id, timestamp, target, covariates)`, learns from train-only
normalized windows, and keeps panel series isolated while sharing the same
configured neural structure.

Use `LaneNeuralPanelForecaster` when `series_id` is a directional lane such as
`source_id:target_id`. The lane wrapper keeps `132:138` distinct from
`138:132`, appends origin/destination/lane embedding features plus directional
graph summary features into the training frame, and can forecast requested
cold lane ids through `predict_for_lanes()`.

In the interactive forecast example, `extra_regressors`, `events`, and
`holidays` flow into the same neural-panel feature contract, so known future
covariates stay part of the forecast input.

## Interactive Example

<ForecastModelExample title="Neural panel forecast" model="neural_panel" />

The embedded example runs locally. Use it as a quick syntax and shape check,
not as quality evidence. Quality claims still need the maintained split
benchmark against seasonal naive and `CartoBoostLagForecaster`.

## When To Use

Use this model when the hypothesis needs:

- direct multi-horizon panel forecasts rather than recursive local forecasts;
- fitted target-lag state through AR-Net with `n_lags`;
- future-known regressors, lagged regressors, event offsets, conditional
  seasonalities, or Fourier terms;
- global, local, or glocal trend, seasonality, event, and regressor behavior by
  series id;
- quantile output with non-crossing repair;
- directional lane ids and explicit cold-lane fallback.

Prefer `CartoBoostLagForecaster` when tabular lag features and tree splits are
the main signal. Prefer local models such as theta, ETS, ARIMA, Kalman, or
piecewise linear seasonal forecasting when one series needs an inspectable
statistical baseline.

## Model Math

Training builds leakage-safe windows from each regular `ForecastFrame` series.
Each window contains normalized time for `n_lags + n_forecasts`, target lags
with shape `B x n_lags`, lagged covariate histories by configured name, and
future component features over the full lag-plus-horizon span. Target scaling is
fit only on the training frame.

The forward path is:

1. compute the global trend and optional local per-series trend deviations;
2. evaluate Fourier seasonality, event offsets, holidays encoded as events, and
   future regressors as additive or multiplicative nonstationary components;
3. stationarize lagged targets against train-fit trend and component state;
4. run AR-Net on stationarized target lags and Covar-Net on concatenated lagged
   regressor histories;
5. add direct horizon stationary network output to the nonstationary forecast;
6. map the median-first internal quantile layout back to requested quantile
   levels and repair crossings around the median output.

AR-Net and Covar-Net use deterministic Kaiming-style initialization, ReLU
hidden layers from `ar_layers` and `lagged_reg_layers`, and an output width of
`n_forecasts * len(quantiles)`. Training uses an AdamW-style update loop. The
default loss is SmoothL1; `loss="mse"`, `loss="mae"`, and `loss="pinball"` are
also accepted. Set `newer_sample_weight=True` to use a monotone cosine recency
ramp.

Quantile heads use a median-first internal layout: the median residual lives at
index 0 for each horizon, and non-median quantiles use learned positive or
negative residual offsets. Public quantile tensors remain ordered by requested
quantile level after crossing repair.

When `future_regressors` are configured, prediction requires known-future
covariates through `model.predict(horizon, known_future=future_frame)`, unless
the regressor was proven constant per fitted series and stored as static future
state. Missing required dynamic future values hard-fail instead of silently
dropping the regressor.

`LaneNeuralPanelForecaster` generates static covariates named
`lane_origin_embedding_*`, `lane_destination_embedding_*`, `lane_embedding_*`,
and `lane_graph_*`. These generated covariates are added to the inner neural
panel as additive future regressors, so fitted nonstationary feature weights
learn from lane identity and directional graph summaries.

`holidays` and `country_holidays` are also accepted on the Python wrapper. When
you fit with a `ForecastFrame`, CartoBoost injects those holiday indicators as
known-future covariates so the native neural panel can train on them directly.
Use `make_future_dataframe(frame, periods)` to build the future scaffold and
reuse the same holiday injection logic for forecast-time frames.
You can also add regressors and events after construction with
`add_seasonality()`, `add_future_regressor()`, `add_lagged_regressor()`,
`add_events()`, and `add_country_holidays()`. For forecast inspection, use
`components_json()` or the browser `includeComponents` option to get a horizon
breakdown of trend, feature, AR, and lagged-regressor contributions. Use
`history_components_json()` or browser `includeHistoryComponents` to inspect
the fitted training rows.

Custom seasonalities can be gated with a `condition_name` covariate. When the
condition is false, the Fourier term is masked to zero at both fit and
prediction time. When the condition is constant per series, CartoBoost stores
it as static future state so prediction can proceed without a separate future
frame.

Use `seasonality_global_local`, `event_global_local`, and
`regressor_global_local` to choose `global`, `local`, or `glocal` parameter
sharing independently for Fourier terms, event offsets, and future regressors.


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

future_frame = ForecastFrame.from_pandas(
    future_hourly_lane_events,
    timestamp_col="pickup_hour",
    target_col="pickup_trips",
    series_id_col="pickup_dropoff_lane",
    freq="h",
    known_future_covariates=["is_airport_event"],
)

model = LaneNeuralPanelForecaster(
    n_lags=24,
    n_forecasts=6,
    quantiles=[0.1, 0.5, 0.9],
    weekly_fourier_order=3,
    future_regressors={"is_airport_event": "additive"},
    lagged_regressors={"avg_trip_distance": 24},
    ar_layers=[32],
    lagged_reg_layers=[16],
    trend_mode="glocal",
    seasonality_global_local="glocal",
    event_global_local="global",
    regressor_global_local="glocal",
    local_l2=0.1,
    loss="smooth_l1",
    epochs=80,
    learning_rate=0.01,
    weight_decay=0.001,
    newer_sample_weight=True,
    seed=42,
)
model.fit(frame)
forecast = model.predict(6, known_future=future_frame)
cold_lane_forecast = model.predict_for_lanes(6, ["132:138", "132:999"])
```

Do not use this model when only a handful of observations exist per lane, when
future-known regressors are unavailable at forecast time, or when the main need
is an easily audited local statistical baseline.

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
