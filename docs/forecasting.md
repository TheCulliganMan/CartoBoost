# Forecasting

CartoBoost forecasting is organized around two docs surfaces:

- this page, for `ForecastFrame`, validation, metrics, artifacts, CLI workflows,
  and benchmark evidence rules;
- the [forecasting model guides](user-guide/forecasting-models/index.md), for
  choosing an individual model type such as naive, theta, ETS, ARIMA, Kalman,
  piecewise linear seasonal, kriging, spatial piecewise kriging, CartoBoost
  lag, NeuralPanel, AutoForecaster, or fixed weighted ensembles.

The Python forecasting package gives users dataframe ergonomics, explicit
configuration, source-checkout script entry points, and artifact handling.
Model behavior is shared across Python, source-checkout scripts, and interactive examples: fitting, prediction,
backtesting, metric evaluation, leakage checks, feature generation, intervals,
reconciliation, and serialization contracts follow the same rules. Python does
not provide fallback forecasting algorithms.

## Workflow

Start by making the scientific unit of analysis explicit. For panel forecasting,
this is usually one time series per entity, location, route, or product. The
timestamp might be an hour, day, or week field, the target might be counts,
revenue, duration, or demand, and known-future covariates should be limited to
values that are genuinely known at forecast creation time, such as hour or
day-of-week.

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
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="entity_id",
    freq="h",
    known_future_covariates=["hour", "day_of_week"],
    static_covariates=["group_id"],
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
    series_id_col="entity_id",
    freq="h",
    historical_covariates=["trip_distance"],
    sample_weight_col="trip_count",
)
```

If five events share the same entity and timestamp, the forecasting model sees
one row. With `trip_count` as the weight, the hourly target is the weighted
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

The forecasting command scaffold is exposed through `scripts/forecast.py`:

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
| Directed graph sequence forecasting for roads, lanes, sensors, and zone flows | [Graph Spatiotemporal Forecasting](user-guide/forecasting-models/graph-spatiotemporal.md) |
| Learned sparse relationships, hierarchy-aware smoothing, and analyst-visible kernels for directional markets | [Market Structure Forecasting](user-guide/forecasting-models/graph-spatiotemporal.md) |
| Compact neural window experts | [N-BEATS And N-HiTS](user-guide/forecasting-models/beats-hits.md) |
| Neural panel forecasting with directional ids | [Neural Panel](user-guide/forecasting-models/neural-panel.md) |
| Guarded default selector over reusable candidates | [AutoForecaster](user-guide/forecasting-models/auto-forecaster.md) |
| Fixed combinations of fitted models | [Weighted Ensembles](user-guide/forecasting-models/ensembles.md) |

Benchmark scripts expose stable aliases such as `cartoboost_lag` and
`cartoboost_auto_forecast` for reproducible evidence tables. Keep benchmark-
specific aliases in benchmark orchestration, not in generic model names.

ETS is additive-only in this version. AutoARIMA searches bounded ARIMA(p,d,q)
candidates with residual-lag moving-average terms; seasonal AutoARIMA is
rejected explicitly. Weighted ensembles require explicit component models.
`NeuralPanelForecaster` accepts `ForecastFrame` panel rows. It builds train-only normalized direct windows, stores quantiles with median
output first internally, learns residual offsets for non-median quantiles,
applies fitted target-tail AR state from `n_lags`, repairs non-crossing
quantiles on prediction, and records normalization, component parameters,
series ids, feature schema, lag config, seed, and train cutoff in metadata.
Fourier seasonality, event offsets, and future regressors each have
independent global/local/glocal parameter-sharing modes. Missing dynamic
known-future regressors fail at prediction time; values proven constant within
each fitted series are stored as static future covariates. Do not use it for
quality claims without a real rolling-origin benchmark against seasonal naive
and `CartoBoostLagForecaster`.
The lane wrapper injects generated origin/destination/lane embedding
covariates and directional graph summary covariates into the inner neural
panel, so lane identity participates in the fitted feature weights rather than
living only in metadata.
The maintained benchmark entry point can emit a NeuralPanel split artifact:

```sh
uv run --group dev python scripts/forecasting_library_benchmark.py \
  --source polars \
  --model-roster neural-panel \
  --neural-panel-splits \
  --output target/neural_panel_split_suite.json
```

That artifact records rolling-origin, cold-lane, cold-origin, and sparse-tail
splits with metrics, timing, command metadata, and artifact paths.

## Advanced Behavior

Several advanced behaviors are reusable utilities rather than separate public
docs pages:

| Behavior | Where it belongs |
| --- | --- |
| Direct and rectified-recursive supervised strategies | Internal candidates for [AutoForecaster](user-guide/forecasting-models/auto-forecaster.md) and shared lag forecasting. |
| STL/MSTL decomposition hybrids | `stl_cartoboost` uses cycle-subseries LOESS, STL low-pass filtering, and a model of the seasonally adjusted target; `mstl_cartoboost` iteratively re-estimates each configured seasonal component before fitting the adjusted-target model. Both require at least two complete cycles of every configured period and repeat the final fitted seasonal cycle when reseasoning forecasts. Benchmark claims stay in [Forecasting Benchmarks](benchmarks/forecasting.md). |
| Hierarchical reconciliation | Forecast artifact metadata and benchmark orchestration when pickup, dropoff, lane, or total demand must be coherent. |
| Quantiles and conformal intervals | `QuantileLoss`, `HuberQuantileLoss`, `CompositeQuantileLoss`, `QuantileRegressorSet`, non-crossing repair, interval coverage, interval width, crossing-rate diagnostics, and serializable conformal calibration. |
| Temporal residual correction | `KalmanResidualCorrector`, `StateFilter`, and `StateCorrectedBooster` apply predict-before-update residual states by origin, destination, corridor, segment, entity family, target family, or time bucket. |
| Regime-aware uncertainty | `CUSUM`, `PageHinkley`, EWMA volatility, rolling median residuals, rolling MAD residuals, and `RegimeIntervalPolicy` can widen intervals, raise process variance, or lower confidence during detected shifts. |
| Calibrated forecast events | Probability calibration helpers turn threshold, horizon, failure-risk, or escalation-risk events into bounded probability forecasts with Brier score, log loss, ECE, calibration buckets, and reliability-curve data. |
| Rank probability score helpers | Metrics and interval evaluation; competition-specific scoring stays in benchmark adapters. |
| NeuralPanel forecasting | Panel neural forecaster with directional ids, generated embedding/graph covariates, direct horizons, separate global/local/glocal seasonality/event/regressor modes, known-future regressors, lagged regressors, median-first internal quantile residuals, and serializable metadata. |

These primitives are generic. A panel forecast may call the state dimensions
origin, destination, corridor, segment, entity family, target family, and time
bucket. Benchmark-specific labels and competition scoring stay in benchmark
orchestration.

The interactive examples use the same primitive families through
`runGeotemporalDiagnostics(request)`. The request can include any combination of
`quantiles`, `residualCorrection`, `regime`, and `calibration` sections. The
response is JSON-compatible and returns repaired quantiles, pinball loss,
interval diagnostics, Kalman residual-state corrections, CUSUM/Page-Hinkley/EWMA
regime signals, regime-adjusted intervals, calibration metrics, calibrated
probabilities, and probability event labels.

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
