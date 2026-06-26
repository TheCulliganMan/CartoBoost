import {ForecastModelRosterExample} from '@site/src/components/ModelingLabClient';

# Forecasting Model Guides

These guides explain the forecasting model classes. Use this section when you
need to pick, configure, or compare a model. Use
[Forecasting](../../forecasting.md) when you need `ForecastFrame`,
rolling-origin backtesting, artifacts, CLI workflows, or shared evidence rules.

Start with the model whose assumptions match the forecast question, then prove
it on the same rolling-origin split as the simpler baselines.

## Run Any Forecast Model

<ForecastModelRosterExample />

## Pick A Guide

| Model guide | Best first use | Notes |
| --- | --- | --- |
| [Naive And Seasonal Naive](naive-seasonal.md) | Establish transparent last-value and last-season baselines. | Start here for every forecast comparison. |
| [Theta](theta.md) | Extrapolate level and trend with a lightweight deterministic model. | Includes manual and optimized theta examples. |
| [ETS](ets.md) | Model additive level, trend, and seasonality. | Useful when the series has smooth components and repeatable seasonal structure. |
| [ARIMA And AutoARIMA](arima.md) | Use differencing and autocorrelation in a bounded search. | Covers fixed-order ARIMA, AutoARIMA candidate selection, visual smoke checks, and benchmark notes. |
| [Kalman](kalman.md) | Track noisy local level and local trend over time. | Includes state diagnostics and visualization examples. |
| [Piecewise Linear Seasonal](/docs/user-guide/forecasting-models/piecewise-linear-seasonal) | Fit interpretable trend, changepoint, seasonality, event, and regressor components. | Includes an interactive example for `piecewise_linear_seasonal`. |
| [Kriging](kriging.md) | Borrow signal across pickup-zone or route coordinates. | Useful for coordinate-aware panel forecasting. |
| [Spatial Piecewise Kriging](spatial-piecewise-kriging.md) | Combine interpretable temporal components with spatial borrowing. | Includes an interactive coordinate-panel example for `spatial_piecewise_kriging`. |
| [CartoBoost Lag](cartoboost-lag.md) | Learn one supervised lag model across many related series. | Use for pickup-zone, dropoff-zone, and lane-level panels. |
| `NeuralPairwiseForecaster` / `LaneNeuralPairwiseForecaster` | Fit a Rust-native neural panel forecaster with direct multi-horizon output for route/lane series. | Use when lane identity, lagged targets, future-known regressors, Fourier seasonality, and quantile output are part of the hypothesis. |
| `AutoStatsBank` | Validate a deterministic statistical expert bank. | Useful when a local statistical selector is the model being tested. |
| `CrostonForecaster`, `SbaForecaster`, `TsbForecaster` | Forecast sparse non-negative taxi-demand series with fixed intermittent-demand methods. | Use when zeros are meaningful no-pickup periods rather than missing rows. |
| [AutoForecaster](auto-forecaster.md) | Use the guarded default selector over lag, direct, residual-corrected, intermittent, and classical candidates. | Includes diagrams for validation, gating, prediction, and metadata inspection. |
| N-BEATS / N-HiTS wrappers | Train deterministic neural forecasting experts from regular panels. | Public Python classes are `NBeatsForecaster` and `NHiTSForecaster`; use them when windowed neural extrapolation is the model being tested. |
| [Weighted Ensembles](ensembles.md) | Combine fitted forecasters with explicit weights. | Components and weights must be named explicitly. |

## Scientific Choice Criteria

Choose the model whose assumptions match the signal you can defend:

| Signal in the taxi series | First model to try | Scientific reason |
| --- | --- | --- |
| The latest observed level is the best short-horizon summary. | Naive | Tests whether any model adds information beyond persistence. |
| The same hour yesterday or same weekday last week dominates. | Seasonal naive | Tests repeatable seasonality without estimated parameters. |
| Level and trend are smooth, with optional simple seasonality. | Theta or ETS | Estimates a low-dimensional local structure that is easy to inspect. |
| Recent autocorrelation and differencing explain the series. | ARIMA or AutoARIMA | Models local serial dependence after bounded non-seasonal differencing. |
| The measured series is noisy and the latent level/trend should update gradually. | Kalman | Separates observation noise from latent state movement. |
| You need interpretable changepoints, Fourier seasonalities, event windows, known future regressors, quantiles, and component decomposition in one local model. | [Piecewise linear seasonal](/docs/user-guide/forecasting-models/piecewise-linear-seasonal) | Estimates additive or multiplicative component paths and keeps the result inspectable. |
| You need a forecast figure that matches Prophet's plotting surface for a Prophet-shaped result. | [Plotting](../../plotting.md) | Uses the same observed-point, forecast-line, capacity, floor, interval, axis, and legend behavior as `prophet.plot.plot`. |
| Nearby zones, route midpoints, or residual surfaces should be spatially related. | Kriging | Uses coordinate distance and a variogram to borrow cross-series signal. |
| Pickup/dropoff zones have both temporal changepoints and spatial residual structure. | [Spatial Piecewise Kriging](spatial-piecewise-kriging.md) | Separates the temporal forecast, spatial correction, kriging variance, neighbors, metadata, and components so the spatial claim can be checked. |
| Many related zones or lanes share lag, rolling, calendar, or trend structure. | CartoBoost lag | Learns one supervised model from many aligned panel examples. |
| Pickup-dropoff lanes need direct multi-horizon neural forecasts with lane direction preserved. | NeuralPairwise | Builds leak-free lag windows from `ForecastFrame`, keeps `A:B` distinct from `B:A`, and stores component, normalization, quantile, series-id, and train-cutoff metadata. |
| Pickup demand is sparse with many true zero periods. | Croston, SBA, or TSB | Uses intermittent-demand smoothing instead of generic trend extrapolation. |
| A local statistical bank should choose among reusable non-benchmark candidates. | AutoStatsBank | Runs validation over a deterministic statistical expert bank. |
| A production taxi-demand panel needs a deterministic guarded default with auditable candidate weights. | AutoForecaster | Validates a fixed roster, protects the lag baseline, and stores global, horizon, and series weights. |
| Validated models capture complementary errors. | Weighted ensemble | Averages explicit components after each member proves useful. |

Do not choose a richer model only because it is available. A scientist should
be able to say which mechanism the model represents, what it ignores, and which
baseline threshold it must clear on a time-ordered holdout.

## Shared Input Patterns

For quick checks, local forecasters can fit a plain numeric sequence:

```python
from cartoboost.forecasting import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=24)
model.fit(zone_hourly_counts)
forecast = model.predict(12)
```

For production taxi demand or fare-duration workflows, prefer a validated
`ForecastFrame`:

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
)
```

`ForecastFrame` validates timestamps, duplicate rows within each series, finite
targets, regular frequency, panel ids, and covariate role metadata.

## Advanced Candidates

`AutoStatsBank` is a public wrapper for the reusable statistical expert bank.
`AutoForecaster` can also validate direct, rectified-recursive,
intermittent-demand, classical-expert, and decomposition-style candidates. Treat
those as roster members of guarded selectors unless a separate Python class is
part of the public API. Keep benchmark-specific names and competition scoring
labels in benchmark harnesses and reports.

Use [Piecewise Linear Seasonal](/docs/user-guide/forecasting-models/piecewise-linear-seasonal) when the forecast
claim depends on inspectable local structure: growth, changepoints, Fourier
seasonalities, event windows, known future regressors, uncertainty intervals,
quantiles, trend-belief adjustments, residual shock propagation, forecast
component contributions, and fitted historical trend/seasonality diagnostics.
The guide includes an interactive example for `piecewise_linear_seasonal`
so you can run a small taxi-lane forecast before writing a Python workflow.

Use `NeuralPairwiseForecaster` when the model under test is a Rust-native
neural panel forecaster rather than a local statistical model. Input rows come
from `ForecastFrame` with `(series_id, timestamp, target, covariates)`. For taxi
lanes, set `series_id` to a stable directional lane id such as
`PULocationID:DOLocationID`; the lane wrapper records origin, destination, lane,
directional graph-feature, and cold-lane fallback metadata. The model builds
direct windows with `n_lags + n_forecasts`, uses train-only target
normalization, supports Fourier seasonality, event offsets, known-future
regressors, lagged regressors, local/global component modes, direct
multi-horizon forecasts, and non-crossing quantiles. Do not use it for a public
quality claim until it beats seasonal naive and `CartoBoostLagForecaster` under
the same rolling-origin split.

Python lane example:

```python
from cartoboost.forecasting import ForecastFrame, LaneNeuralPairwiseForecaster

frame = ForecastFrame.from_pandas(
    hourly_lane_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_trips",
    series_id_col="pickup_dropoff_lane",
    freq="h",
    known_future_covariates=["is_airport_event"],
    historical_covariates=["avg_trip_distance"],
)

model = LaneNeuralPairwiseForecaster(
    n_lags=24,
    n_forecasts=6,
    quantiles=[0.1, 0.5, 0.9],
    daily_fourier_order=3,
    weekly_fourier_order=3,
    future_regressors={"is_airport_event": "additive"},
    lagged_regressors={"avg_trip_distance": 24},
    trend_mode="glocal",
    local_l2=0.1,
    seed=42,
)
model.fit(frame)
forecast = model.predict(6)
quantiles = model.quantiles_json(6)
```

Wasm/browser example:

```js
const response = await runForecast({
  model: "neural_pairwise",
  horizon: 6,
  frequency: "h",
  rows: laneRows,
  metadata: {
    timestampCol: "pickup_hour",
    targetCol: "pickup_trips",
    seriesIdCol: "pickup_dropoff_lane",
    knownFutureCovariates: ["is_airport_event"],
    historicalCovariates: ["avg_trip_distance"],
  },
  options: {
    nLags: 24,
    nForecasts: 6,
    quantileLevels: [0.1, 0.5, 0.9],
    dailyFourierOrder: 3,
    weeklyFourierOrder: 3,
    extraRegressors: ["is_airport_event"],
    regressorModes: {is_airport_event: "additive"},
    laggedRegressors: {avg_trip_distance: 24},
    trendMode: "glocal",
    localL2: 0.1,
    uncertaintySeed: 42,
  },
});
```

Use [Spatial Piecewise Kriging](spatial-piecewise-kriging.md) when that piecewise seasonal CartoBoost
base should borrow spatial signal across stable taxi coordinates. Configure
`mode="residual_kriging"` to fit the temporal base, compute in-sample
cutoff-safe residuals by series, and krige residual corrections for each
horizon. Use `mode="kriged_regressors"` when known spatial covariates such as
pickup-zone traffic density should be interpolated into piecewise seasonal
regressor columns. Use `mode="hybrid"` for both mechanisms. The result includes
`prediction`, `base_mean`, `spatial_correction`, `kriging_variance`,
`selected_neighbors`, component decomposition, and metadata with cutoffs and
variogram settings.

Deterministic synthetic benchmark example:

```bash
uv run --group dev python scripts/forecasting_benchmark.py \
  --days 180 \
  --horizon 7 \
  --folds 1 \
  --panel-series 6 \
  --output target/spatial_piecewise_kriging_benchmark.json
```

The `spatial_piecewise_kriging_panel` run compares Naive, SeasonalNaive,
PiecewiseLinearSeasonal, KrigingForecaster, and the hybrid under the same
rolling-origin split. Use the artifact's `aggregate` table for RMSE, MAE, WAPE,
fit time, prediction time, and baseline deltas. Use the forecast result JSON
columns `selected_neighbors`, `kriging_variance`, and `component_decomposition`
for spatial and component diagnostics before plotting metric comparisons or
coordinate maps.

## Shared Result Shape

Forecasting models return a `ForecastResult` object. Use
`predictions()` for row tuples:

```python
forecast = model.predict(3)
rows = forecast.predictions()

for series_id, timestamp, horizon, model_name, mean in rows:
    print(series_id, timestamp, horizon, model_name, mean)
```

The tuple columns are also available from `forecast.columns()`. Use
`forecast.to_json()` for portable artifact roundtrips and downstream reporting.

## Validation Order

For forecast claims, compare models under the same rolling-origin split:

1. Start with naive and seasonal naive baselines.
2. Add a local model that matches the series structure, such as theta, ETS,
   ARIMA, or Kalman.
3. Use `CartoBoostLagForecaster` when many related series should share lag,
   rolling, calendar, or trend features.
4. Use kriging when stable coordinates are part of the forecast signal.
5. Use weighted ensembles only after component models have been validated.

Report RMSE, MAE, horizon, split dates, training time, prediction time, model
settings, sample size, and whether the input data is real, generated acceptance
data, or synthetic.
