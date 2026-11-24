import {ForecastModelRosterExample} from '@site/src/components/ModelingLabClient';

# Forecasting Models

Yesterday's hourly pickup cycle, a
changing airport-demand trend, and spillover between taxi zones call for
different models. Start with seasonal naive, then add the simplest model that
captures what the baseline misses.

For data preparation, rolling-origin backtesting, output columns, and saved
results, use the [forecasting workflow](../../forecasting.md).

## Interactive Example

Compare model behavior on a small browser sample. These runs help you inspect
forecast shapes and configuration; use held-out real data to measure quality.

<ForecastModelRosterExample />

## Choose A Model

| Pattern or task | Model guide | What to inspect |
| --- | --- | --- |
| The latest value or previous cycle is a useful forecast | [Naive and Seasonal Naive](naive-seasonal.md) | Persistence versus recurrence |
| A short history has a clear level and trend | [Theta](theta.md) | Trend extrapolation and selected smoothing |
| Level, trend, and an additive cycle evolve gradually | [ETS](ets.md) | Fitted components and residual seasonality |
| Recent values and forecast errors carry signal | [ARIMA and AutoARIMA](arima.md) | Differencing, selected order, and residuals |
| Observations are noisy but the underlying state changes smoothly | [Kalman](kalman.md) | State movement, innovations, and variance settings |
| Trend changes, seasonalities, events, and regressors need separate explanations | [Piecewise Linear Seasonal](piecewise-linear-seasonal.md) | Named component contributions |
| Nearby coordinates share signal | [Kriging](kriging.md) | Variogram, neighbors, and spatial holdout error |
| Temporal components leave spatially related residuals | [Spatial Piecewise Kriging](spatial-piecewise-kriging.md) | Base forecast versus spatial correction |
| Many related series share lag and calendar behavior | [CartoBoost Lag](cartoboost-lag.md) | Per-series and per-horizon errors |
| Directed edges connect the forecast series | [Graph Spatiotemporal Forecasting](graph-spatiotemporal.md) | Graph contribution versus graph-free baselines |
| Directional pickup/dropoff markets have sparse learned relationships | [Market Structure Forecasting](graph-spatiotemporal.md#learned-market-structure) | Retained relationships and forecast components |
| A decision depends on tails or coverage | [Probabilistic and Conformal](probabilistic-conformal.md) | Coverage and width alongside point error |
| Regular history windows support a neural comparison | [N-BEATS and N-HiTS](beats-hits.md) | Error versus local and lag baselines |
| Related series need direct multi-horizon neural forecasts | [Neural Panel](neural-panel.md) | Components, quantiles, and cold-lane behavior |
| Validation should choose a local statistical expert | [AutoStatsBank](auto-stats-bank.md) | Selected expert and internal holdout score |
| Pickup counts contain many true zeros | [Intermittent Demand](intermittent-demand.md) | Zero fraction and demand-event count |
| Sparse demand needs separate size and spacing estimates | [Croston](croston.md) | The constant expected demand level |
| Croston smoothing needs its standard bias adjustment | [SBA](sba.md) | Bias and error relative to Croston |
| The chance of a demand event changes over time | [TSB](tsb.md) | Occurrence and size smoothing |
| Validation should choose among lag, direct, and statistical strategies | [AutoForecaster](auto-forecaster.md) | Selected members, safeguards, and weights |
| Validated members make complementary errors | [Weighted Ensembles](ensembles.md) | Each member's contribution and the best single model |

A richer model earns its place through a better time-ordered holdout or a
necessary output, such as calibrated intervals. Complexity alone is not a
reason to switch.

## Shared Input Patterns

Local forecasters accept a numeric sequence for a small check:

```python
from cartoboost.forecasting import SeasonalNaiveForecaster

# Two illustrative days of hourly pickup counts.
zone_hourly_counts = [12, 9, 7, 6, 8, 15, 28, 41, 38, 31, 29, 33,
                      36, 39, 42, 47, 59, 64, 52, 40, 32, 26, 21, 16] * 2
model = SeasonalNaiveForecaster(season_length=24)
model.fit(zone_hourly_counts)
forecast = model.predict(12)
```

For timestamped taxi panels, declare the series and cadence with `ForecastFrame`:

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="timestamp",
    target_col="pickup_demand",
    series_id_col="PULocationID",
    freq="h",
)
```

The frame checks timestamps, duplicate series/time rows, finite targets,
frequency, and covariate roles. Missing targets, missing covariates, and
irregular observations require explicit opt-ins and a model that supports the
chosen policy. See the [frame contract](../../forecasting.md#forecastframe)
before changing these defaults.

## Decomposition And Automatic Candidates

The browser roster includes `stl_cartoboost` and `mstl_cartoboost`. They use
LOESS-based decomposition, fit the downstream model to the seasonally adjusted
target, and repeat the final fitted cycle when adding seasonality back. STL
requires two complete cycles of its period; MSTL requires two complete cycles
of every configured period.

[AutoForecaster](auto-forecaster.md) includes direct, rectified-recursive,
intermittent, classical, and decomposition-style candidates. Its guide explains
eligibility, validation, and how to inspect the chosen forecast.

## Shared Result Shape

A forecast is a `ForecastResult`. Read row tuples with `predictions()`:

```python
forecast = model.predict(3)
for series_id, timestamp, horizon, model_name, mean in forecast.predictions():
    print(series_id, timestamp, horizon, model_name, mean)
```

Use `columns()` for column names and `to_json()` for a portable representation.
[Plotting](../../plotting.md) covers forecast and component figures.

## Validate The Next Decision

Evaluate every candidate on the same series, forecast origins, and horizons.
Keep seasonal naive in the comparison. Report errors by horizon and zone so a
busy airport does not hide poor performance elsewhere.

The [forecasting workflow](../../forecasting.md#backtesting) shows rolling-origin
backtesting. The [benchmark reports](../../benchmarks/forecasting.md) give the
recorded data, model settings, metrics, timings, and limits of maintained runs.
