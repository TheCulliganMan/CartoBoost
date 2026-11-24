# Forecasting

CartoBoost's forecasting workflow starts with an explicit series and cadence,
fits a model using past observations, and checks the next horizon on rolling origins.

| Your next step | Read |
| --- | --- |
| Prepare hourly pickup counts or route aggregates | [ForecastFrame](#forecastframe) |
| Choose a forecasting model | [Model guide](user-guide/forecasting-models/index.md) |
| Measure performance at realistic forecast cutoffs | [Backtesting](#backtesting) |
| Save forecast rows and their settings | [Artifacts and CLI](#artifacts-and-cli) |
| Compare recorded results | [Forecasting benchmarks](benchmarks/forecasting.md) |

## Workflow

Define the prediction before fitting. An hourly pickup-demand panel has one
series per `PULocationID`; a route panel can use an ordered pickup/dropoff pair.
Choose a target such as trip count, average fare, or average duration and a
horizon that matches the decision you need to make.

Declare only genuinely available values as known-future covariates. Hour and
day of week are known at forecast creation; tomorrow's observed trip count is
not. History-derived features must stop at the forecast cutoff.

Start with seasonal naive and compare models on the same origins and horizon.
Save the forecast rows, settings, and evaluation alongside each other so the
result remains understandable after the fitting session ends.

## ForecastFrame

`ForecastFrame` declares what each row means: its timestamp, target, series ID,
frequency, and covariate roles. For hourly taxi demand, one row is one pickup
zone at one hour.

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="timestamp",
    target_col="pickup_demand",
    series_id_col="PULocationID",
    freq="h",
    known_future_covariates=["hour", "day_of_week"],
    static_covariates=["borough_id"],
)
```

`ForecastFrame` validation is deterministic: timestamps are sorted within each
series, duplicate series/timestamp rows are rejected, targets must be finite by
default, regular frequency is checked when provided, and covariate roles remain
explicit. Known-future covariates are values available at forecast creation
time; lagged targets, rolling summaries, and other history-derived features
must be built from rows before the forecast origin.

Irregular history is opt-in and model-scoped. Pass `allow_irregular=True` with
a forecast cadence such as `freq="D"` when the observed rows are not evenly
spaced but future horizon steps should use that cadence. Native irregular
fitting is supported by `PiecewiseLinearSeasonalForecaster`, `NaiveForecaster`,
and cadence-agnostic window averages; equal-step models such as ETS, ARIMA,
theta, Kalman, intermittent-demand, lag, direct, neural, and auto selectors
raise a clear error and should be fit on a regularized frame.

Missing target values are also opt-in and model-scoped. Pass
`allow_missing_targets=True` when the target column contains `NaN` values that
should be treated like Prophet treats missing `y`: the frame keeps the
timestamps, fitting uses only observed target rows, and forecast horizons start
after the latest timestamp in the input. Infinity is always rejected.
`PiecewiseLinearSeasonalForecaster`, `NaiveForecaster`, and non-seasonal window
averages support this path. Equal-step, seasonal, lag, direct, neural,
intermittent-demand, spatial kriging, and auto-selector models reject missing
targets with a model-level error.

Missing covariate values are frame-level opt-in. Pass
`allow_missing_covariates=True` when a declared static, known-future, or
historical covariate contains `NaN` values and you want model-level validation
instead of `ForecastFrame` construction failure. Infinity is always rejected.
Forecasters that do not consume covariates can fit normally. Forecasters that
do consume a missing covariate, such as lag-feature or piecewise-regressor
paths, raise a model-level error naming the covariate and series.

When raw observations have multiple rows at the same timestamp, aggregate them
before modeling or pass a positive `sample_weight_col`. With
`sample_weight_col`, CartoBoost collapses each duplicate series/timestamp group
into one forecast row: the target and numeric covariates become weighted
means, and the weight column becomes the total weight for that timestamp.

```python
frame = ForecastFrame.from_pandas(
    trip_observations,
    timestamp_col="timestamp",
    target_col="fare",
    series_id_col="PULocationID",
    freq="h",
    historical_covariates=["trip_distance"],
    sample_weight_col="trip_count",
)
```

If five trip aggregates share the same pickup zone and timestamp, the
forecasting model sees one row. With `trip_count` as the weight, the hourly target is the weighted
mean and the weight column is the sum of the five weights. Without
`sample_weight_col`, those five rows still fail as duplicate timestamps because
forecasting requires one target per series/timestamp.

## Results And Metrics

Forecast outputs use stable columns so model rows can be aligned across
candidates:

| Column | Meaning |
| --- | --- |
| `series_id` | Single-series id or panel id. |
| `timestamp` | Forecasted timestamp. |
| `model` | Model name or benchmark alias. |
| `horizon` | One-based horizon from the forecast origin. |
| `forecast` | Point forecast. |
| `lower_*`, `upper_*` | Optional interval bounds when the model emitted them. |

`ForecastMetricSet` covers MAE, RMSE, MAPE, sMAPE, MASE, WAPE, bias, pinball
loss, and interval metrics where bounds are present. For honest comparisons,
score aligned rows from the same origin, horizon, and series ids.

## Backtesting

Use rolling-origin validation for forecasting claims. A fold trains on rows
strictly before the origin and scores the next horizon only. Random row splits
leak future demand and should not be used for forecast claims.

An infeasible splitter configuration is an error: the splitter does not return
an empty fold list when the history cannot satisfy `min_train_size` and the
requested horizon. Training frames created for each fold retain the original
`ForecastFrame` metadata, including covariate roles and irregular/missing-value
policies.

```python
from cartoboost.forecasting import (
    RollingOriginBacktester,
    RollingOriginSplitter,
    SeasonalNaiveForecaster,
)

splitter = RollingOriginSplitter(
    horizon=24,
    n_splits=3,
    step=24,
    min_train_size=72,
)
result = RollingOriginBacktester(splitter=splitter).evaluate(
    SeasonalNaiveForecaster(season_length=24), frame
)
```

Comparable evidence means the same frame, origins, horizons, metric definitions,
and baseline roster are reused across candidates. Report aggregate metrics and,
for panels, horizon-level and series-level diagnostics when one zone or lane can
hide failures in the average.

## Artifacts And CLI

`ForecastArtifact` saves forecast rows with a manifest that records the model
settings, frame contract, metrics, interval metadata, and optional config. Use
CSV for portable tables and Parquet only when the optional dependency is
installed intentionally.

From a source checkout, use `scripts/forecast.py` to fit, predict, or backtest:

```sh
python scripts/forecast.py fit \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col pickup_demand \
  --series-id-col entity_id \
  --freq D \
  --model theta \
  --horizon 7 \
  --season-length 7 \
  --artifact-dir target/forecasting/theta \
  --output target/forecasting/theta_forecast.csv
```

| Command | Purpose |
| --- | --- |
| `fit` | Reads CSV history, writes model/config artifacts, and can emit forecast rows. |
| `predict` | Reads a saved forecast artifact directory and writes a forecast CSV. |
| `backtest` | Runs deterministic time-ordered validation and writes JSON metrics. |
| `compare` | Scores multiple model names on the same holdout. |

Invalid configs, missing columns, unknown model names, unavailable optional
bindings, and missing artifacts should fail clearly instead of silently changing
the algorithm.

## Choose A Model

Use the [forecasting model guide](user-guide/forecasting-models/index.md) to
compare assumptions, inputs, and outputs. A practical progression is:

| Pattern | Start with |
| --- | --- |
| The last hour or last daily cycle is informative | [Naive and Seasonal Naive](user-guide/forecasting-models/naive-seasonal.md) |
| Level, trend, or a recurring additive cycle dominates | [Theta](user-guide/forecasting-models/theta.md), [ETS](user-guide/forecasting-models/ets.md), or [Kalman](user-guide/forecasting-models/kalman.md) |
| Recent values and errors explain local dependence | [ARIMA](user-guide/forecasting-models/arima.md) |
| Trend, events, and regressors need separate explanations | [Piecewise Linear Seasonal](user-guide/forecasting-models/piecewise-linear-seasonal.md) |
| Related zones share lag and calendar patterns | [CartoBoost Lag](user-guide/forecasting-models/cartoboost-lag.md) |
| Nearby locations share residual signal | [Kriging](user-guide/forecasting-models/kriging.md) or [Spatial Piecewise Kriging](user-guide/forecasting-models/spatial-piecewise-kriging.md) |
| A directed network explains movement between zones | [Graph and Market Structure Forecasting](user-guide/forecasting-models/graph-spatiotemporal.md) |
| Regular windows support a neural comparison | [N-BEATS and N-HiTS](user-guide/forecasting-models/beats-hits.md) or [Neural Panel](user-guide/forecasting-models/neural-panel.md) |
| Validation should select the strategy | [AutoForecaster](user-guide/forecasting-models/auto-forecaster.md) |
| A decision needs calibrated bounds | [Probabilistic and Conformal](user-guide/forecasting-models/probabilistic-conformal.md) |

Model limitations are specific. ETS uses additive components; AutoARIMA searches
non-seasonal orders; weighted ensembles require explicit members and weights.
Neural panel models require dynamic known-future regressors at prediction time
unless those regressors were constant within each fitted series.

## Advanced Behavior

| Need | Available behavior |
| --- | --- |
| Predict horizons directly or correct recursive error | Direct and rectified-recursive strategies in [AutoForecaster](user-guide/forecasting-models/auto-forecaster.md) and supervised lag forecasting |
| Separate one or more repeating cycles | `stl_cartoboost` and `mstl_cartoboost` decompose seasonality, model the adjusted target, and add the fitted seasonal cycle back |
| Keep pickup, dropoff, route, and total forecasts coherent | Hierarchical reconciliation with settings recorded in forecast metadata |
| Model uncertainty | Quantile losses, non-crossing repair, conformal calibration, coverage, width, and crossing diagnostics |
| Update a residual state over time | `KalmanResidualCorrector`, `StateFilter`, and `StateCorrectedBooster` use predict-before-update residual states |
| Detect changing conditions | CUSUM, Page-Hinkley, volatility, and robust residual summaries with `RegimeIntervalPolicy` |
| Calibrate a threshold event | Bounded probability outputs with Brier score, log loss, ECE, and reliability data |

Decomposition needs enough history: STL requires two complete cycles of its
configured period, and MSTL requires two complete cycles of every period.
Interval widening and residual correction should be evaluated on future
origins just like the point forecaster.

## Reading The Evidence

A useful comparison identifies its data, series definition, cadence, horizon,
cutoffs, covariates, and model settings. Read point metrics alongside bias and
per-horizon errors; for interval models, also read coverage, width, and pinball
loss. Fit and prediction times describe the cost of the chosen model.

The [forecasting benchmark reports](benchmarks/forecasting.md) include exact
commands, artifact paths, model identifiers, metrics, and limitations. Browser
examples use small demonstration inputs and should be read as behavior checks,
not as evidence of better forecast accuracy.
