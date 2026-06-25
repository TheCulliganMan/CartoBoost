# Forecasting

CartoBoost forecasting is organized around two docs surfaces:

- this page, for `ForecastFrame`, validation, metrics, artifacts, CLI workflows,
  and benchmark evidence rules;
- the [forecasting model guides](user-guide/forecasting-models/index.md), for
  choosing an individual model type such as naive, theta, ETS, ARIMA, Kalman,
  piecewise linear seasonal, kriging, spatial piecewise kriging, CartoBoost
  lag, AutoForecaster, or fixed weighted ensembles.

The Python forecasting package is intentionally a wrapper layer. It gives users
dataframe ergonomics, explicit configuration, CLI entry points, and artifact
handling. Behavior that affects model results lives in Rust under `crates/`:
fitting, prediction, backtesting, metric evaluation, leakage checks, feature
generation, intervals, reconciliation, and serialization contracts. Python does
not provide fallback forecasting algorithms.

## Workflow

Start by making the scientific unit of analysis explicit. For taxi demand, this
is usually one time series per pickup zone (`PULocationID`) or per pickup to
dropoff lane (`PULocationID` and `DOLocationID`). The timestamp might be
`pickup_hour`, the target might be `pickup_trips`, `fare`, `duration`, or
`trip_distance`, and known-future covariates should be limited to values that
are genuinely known at forecast creation time, such as hour or day-of-week.

Then choose the validation protocol before choosing a winner. Forecasting
validation should answer, "At this origin timestamp, using only information
available up to the origin, how well did the model predict the next horizon?"
CartoBoost uses rolling-origin splitters for that reason. Random
cross-validation is not a forecasting protocol.

Finally, save the evidence. A forecast table without its panel contract,
features, bounds, and backtest settings is hard to audit. CartoBoost artifacts
store forecast rows beside a manifest so the result can be compared or reviewed
without hidden Python process state.

## ForecastFrame

`ForecastFrame` is the production input contract. It records the timestamp
column, target column, optional series id, frequency, static covariates,
known-future covariates, and historical-only covariates before a model sees the
data.

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_trips",
    series_id_col="PULocationID",
    freq="h",
    known_future_covariates=["hour", "day_of_week"],
    static_covariates=["borough_id", "airport_zone"],
)
```

`ForecastFrame` validation is deterministic: timestamps are sorted within each
series, duplicate series/timestamp rows are rejected, targets must be finite,
regular frequency is checked when provided, and covariate roles remain explicit.
Known-future covariates are values available at forecast creation time; lagged
targets, rolling summaries, and other history-derived features must be built
from rows before the forecast origin.

When raw taxi observations have multiple rows at the same timestamp, aggregate
them before modeling or pass a positive `sample_weight_col`. With
`sample_weight_col`, CartoBoost collapses each duplicate series/timestamp group
into one native forecast row: the target and numeric covariates become weighted
means, and the weight column becomes the total weight for that timestamp.

```python
frame = ForecastFrame.from_pandas(
    trip_observations,
    timestamp_col="pickup_hour",
    target_col="fare",
    series_id_col="pickup_dropoff_lane",
    freq="h",
    historical_covariates=["trip_distance"],
    sample_weight_col="trip_count",
)
```

If five taxi trips share the same lane and `pickup_hour`, the native
forecasting model sees one hourly lane row. With `trip_count` as the weight, the
hourly target is the trip-count-weighted mean fare, `trip_distance` is the
trip-count-weighted mean distance, and `trip_count` is the sum of the five
weights. Without `sample_weight_col`, those five rows still fail as duplicate
timestamps because the Rust forecasting core requires one target per
series/timestamp.

## Results And Metrics

Forecast outputs use stable columns so model rows can be aligned across
candidates:

| Column | Meaning |
| --- | --- |
| `series_id` | Single-series id or panel id. |
| `timestamp` | Forecasted timestamp. |
| `model` | Native model name or benchmark alias. |
| `horizon` | One-based horizon from the forecast origin. |
| `forecast` | Point forecast. |
| `lower_*`, `upper_*` | Optional interval bounds when the model emitted them. |

`ForecastMetricSet` covers MAE, RMSE, MAPE, sMAPE, MASE, WAPE, bias, pinball
loss, and interval metrics where bounds are present. For honest comparisons,
score aligned rows from the same origin, horizon, and series ids.

## Backtesting

Use rolling-origin validation for forecasting claims. A fold trains on rows
strictly before the origin and scores the next horizon only. Random row splits
leak future demand and should not be used for taxi pickup, dropoff, or lane
forecasts.

```python
from cartoboost.forecasting import RollingOriginBacktester, SeasonalNaiveForecaster

backtester = RollingOriginBacktester(
    horizon=24,
    origin_count=3,
    step=24,
)
result = backtester.run(
    SeasonalNaiveForecaster(season_length=24),
    frame,
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

The forecasting command scaffold is exposed through `scripts/forecast.py`:

```sh
python scripts/forecast.py fit \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col pickup_demand \
  --series-id-col PULocationID \
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

Invalid configs, missing columns, unknown model names, unavailable native
bindings, and missing artifacts should fail clearly instead of silently changing
the algorithm.

## Native Model Surface

Use the model guides for modeling decisions:

| Modeling type | Guide |
| --- | --- |
| Last-value and last-season baselines | [Naive And Seasonal Naive](user-guide/forecasting-models/naive-seasonal.md) |
| Lightweight trend extrapolation | [Theta](user-guide/forecasting-models/theta.md) |
| Additive level, trend, and seasonality | [ETS](user-guide/forecasting-models/ets.md) |
| Autocorrelation and differencing | [ARIMA And AutoARIMA](user-guide/forecasting-models/arima.md) |
| Noisy latent level and trend | [Kalman](user-guide/forecasting-models/kalman.md) |
| Interpretable trend, changepoints, seasonalities, events, and regressors | [Piecewise Linear Seasonal](user-guide/forecasting-models/piecewise-linear-seasonal) |
| Coordinate-aware panel borrowing | [Kriging](user-guide/forecasting-models/kriging.md) |
| Temporal components plus spatial residual or regressor kriging | [Spatial Piecewise Kriging](user-guide/forecasting-models/spatial-piecewise-kriging.md) |
| Shared supervised lag features across many panels | [CartoBoost Lag](user-guide/forecasting-models/cartoboost-lag.md) |
| Guarded default selector over native candidates | [AutoForecaster](user-guide/forecasting-models/auto-forecaster.md) |
| Fixed combinations of fitted native models | [Weighted Ensembles](user-guide/forecasting-models/ensembles.md) |

Benchmark scripts expose stable aliases such as `cartoboost_lag` and
`cartoboost_auto_forecast` for reproducible evidence tables. Keep competition or
benchmark-specific aliases in benchmark orchestration, not in generic model
names.

Rust ETS is additive-only in this version. Rust AutoARIMA searches bounded
ARIMA(p,d,q) candidates with residual-lag moving-average terms; seasonal
AutoARIMA is rejected explicitly. Weighted ensembles require explicit native
component models. Neural forecasting wrappers require compiled native bindings
and should not be used for quality claims without real benchmark evidence.

## Advanced Behavior

Several advanced behaviors are Rust-core utilities rather than separate public
docs pages:

| Behavior | Where it belongs |
| --- | --- |
| Direct and rectified-recursive supervised strategies | Internal candidates for [AutoForecaster](user-guide/forecasting-models/auto-forecaster.md) and shared lag forecasting. |
| STL/MSTL decomposition hybrids | Model roster entries described from the model guide index when exposed, with benchmark claims kept in [Forecasting Benchmarks](benchmarks/forecasting.md). |
| Hierarchical reconciliation | Forecast artifact metadata and benchmark orchestration when pickup, dropoff, lane, or total demand must be coherent. |
| Quantiles, conformal intervals, and rank probability score helpers | Metrics and interval evaluation; competition-specific scoring stays in benchmark adapters. |
| Neural forecasting experts | Optional native-bound wrappers with no public quality claim unless a real benchmark run records commands, settings, timing, and metrics. |

## Evidence Standard

When reporting a forecasting result, record:

- data source and filtering rules;
- panel definition, timestamp column, target column, frequency, and horizon;
- train/validation split boundaries or rolling-origin splitter settings;
- model name and relevant parameters;
- feature configuration and covariate roles;
- RMSE, MAE, R2 when applicable, bias, WAPE or MAPE family metrics, and any
  interval coverage or pinball-loss metrics;
- for M5/M6-style benchmark claims, the `official_metrics` artifact section:
  level-aware WRMSSE for M5 and rank-probability score plus decision rows for
  M6;
- training time and prediction time when comparing models.

For benchmark claims, keep the train/test split, task names, model list, metrics,
and acceptance gates stable across reruns. Compare against serious baselines
with the same split and comparable estimator settings.
