# Forecasting

CartoBoost forecasting is for scientific forecasting work where the question is
not only "what method can I call?" but "can another scientist reproduce the
panel, the validation protocol, the features, and the evidence?"

The Python forecasting surface is intentionally a wrapper layer. It gives you
dataframe ergonomics, explicit configuration, CLI entry points, and artifact
handling. Forecasting behavior that affects model results lives in Rust under
`crates/`: fitting, prediction, backtesting, metric evaluation, leakage checks,
feature generation, intervals, reconciliation, and serialization contracts.
Python does not provide fallback forecasting algorithms.

Use these wrappers when you need:

- reproducible taxi-zone or taxi-route panels with timestamp, target, frequency,
  and series identity recorded before model fitting;
- comparable rolling-origin evaluation instead of random validation splits that
  leak future trip demand into training;
- feature provenance for lag, rolling, trend, calendar, static, known-future,
  and historical-only columns;
- portable forecast artifacts whose manifests describe the data contract,
  feature configuration, model settings, backtest metrics, and interval
  metadata;
- CLI runs that fail clearly when a native binding or optional dependency is not
  available instead of silently changing the algorithm.

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

## Wrapper Surfaces

The forecasting docs are organized by scientific concern:

- [Forecasting API](forecasting_api.md): `ForecastFrame`, `ForecastResult`,
  prediction intervals, and base wrapper contracts.
- [Forecasting backtesting](forecasting_backtesting.md): rolling-origin
  splitters, keyed metrics, fold safety, and comparable evidence.
- [Forecasting lag features](forecasting_lag_features.md): leakage-safe lag,
  rolling, trend, and calendar feature construction for taxi panels.
- [Forecasting artifacts](forecasting_artifacts.md): portable forecast
  directories, manifests, registry names, strict config parsing, and optional
  formats.
- [Forecasting CLI](forecasting_cli.md): reproducible command-line fit,
  predict, backtest, and compare workflows.
- [Forecasting examples](forecasting_examples.md): taxi-domain examples that
  show command shape, wrapper use, and evidence outputs.

Model selection guidance belongs in the model guide. These wrapper docs explain
how CartoBoost preserves the experiment boundary around any supported model.

## Native Model Surface

The public Python names with native PyO3 training and prediction bindings are:

- `NaiveForecaster`
- `SeasonalNaiveForecaster`
- `ThetaForecaster`
- `OptimizedThetaForecaster`
- `ETSForecaster`
- `AutoARIMAForecaster`
- `CartoBoostLagForecaster`
- `AutoForecaster`
- `WeightedEnsembleForecaster`

Additional Rust-core forecasting behavior includes rolling-origin splitters,
forecast metrics, result serialization, artifact manifests, reconciliation
metadata, and leakage-safe lag features.

Benchmark scripts also expose stable roster aliases for reproducible committed
runs:

| Benchmark alias | Python surface | Purpose |
| --- | --- | --- |
| `cartoboost_lag` | `CartoBoostLagForecaster` | Existing supervised lag baseline used for continuity with prior forecasting evidence. |
| `cartoboost_auto_forecast` | `AutoForecaster` | Deterministic hybrid default that routes through AutoStats, direct CartoBoost, decomposition, intermittent, probabilistic, reconciliation, neural, and ensemble branches when their inputs are available. |

Rust ETS is additive-only in this version. Rust AutoARIMA searches bounded
ARIMA(p,d,q) candidates with residual-lag moving-average terms; seasonal
AutoARIMA is rejected explicitly. Weighted ensembles require explicit native
component models. Python never computes fallback forecasts.

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
