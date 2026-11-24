import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Theta

Extend a recent level and trend with Theta. It is a compact local forecasting
baseline for taxi pickup counts, fare totals, or duration aggregates whose
recent direction may continue into the next few horizons. Choose fixed
parameters or let `OptimizedThetaForecaster` select them on held-out history.

## Interactive Example

<ForecastModelExample title="Optimized theta panel forecast" model="optimized_theta" />

## When To Use

Use theta forecasting for counts, fares, trip duration, or trip distance
aggregates when the recent history is short, the main signal is level plus
trend, and you want a fast deterministic baseline. Theta is often a practical
comparison point for single-series hourly demand and other short-horizon
settings where the next few horizons should continue a smooth recent movement.

Do not treat theta as evidence of spatial generalization. It does not know
zone geometry, route relationships, road distance, or graph structure. Compare
it against seasonal naive and lag-feature models on the same rolling-origin
folds before using it as a production benchmark.

## What The Model Learns {#scientific-role}

Theta combines a smoothed level with a controlled trend contribution. Optional
seasonal adjustment removes a repeating pattern before extrapolation.

This makes Theta useful when a short history contains a clear rise or fall but
too few examples for a larger supervised model. It will continue that recent
movement; it cannot anticipate an airport event that is absent from the series.

## Assumptions And Failure Modes

Theta assumes that extrapolated level and trend are meaningful over the chosen
horizon. It can fail when demand changes because of known future events,
weather, operational disruptions, or spatial spillover that is not present in
the univariate history. Seasonal theta also assumes the configured season length
matches the row cadence and that enough full cycles exist to estimate a stable
seasonal adjustment.

Failure is usually visible as a forecast that keeps climbing after a plateau,
stays too flat during a ramp, or reproduces the wrong seasonal phase. Treat
those patterns as evidence about model mismatch, not as reasons to tune by eye.

## Models

| Model | Use when |
| --- | --- |
| `ThetaForecaster` | You want to set `theta` and smoothing `alpha` directly. |
| `OptimizedThetaForecaster` | You want deterministic grid search over theta and alpha values. |

`OptimizedThetaForecaster` reserves a non-empty suffix from every series,
fits each candidate only on the earlier prefix, forecasts the suffix, and
selects the lowest mean squared holdout error. It refits the winning
`(theta, alpha)` pair on the complete history after selection.

For a non-seasonal series, the horizon-`h` forecast combines the simple
exponential-smoothing forecast with the theta drift term:

```text
forecast(T + h) = SES(T + h)
  + (1 - 1 / theta) * b0
    * (h - 1 + 1 / alpha - (1 - alpha)^T / alpha)
```

Here `b0` is the fitted linear slope and `T` is the prefix length. The
sample-length and smoothing adjustment matters most on short histories; it is
not equivalent to adding an unadjusted `h * slope` term.

## Prediction Intervals

Set `prediction_interval_levels=(0.8, 0.95)` for symmetric Gaussian intervals
using each series' fitted training-residual root mean square. Initialization
residuals are excluded, and at least two residuals are required. Optimized Theta
uses the winning model refitted on the full training history; requesting
intervals does not change candidate selection or point forecasts.

Width remains constant across horizons. This is a training-residual Gaussian
approximation, not a held-out conformal guarantee; long-horizon uncertainty can
be larger. Save/load and refit preserve the requested levels.

## Python Example

```python
from cartoboost.forecasting import OptimizedThetaForecaster, ThetaForecaster

hourly_airport_pickups = [
    88, 84, 79, 72, 91, 118, 146, 162, 155, 141, 130, 126,
    92, 89, 83, 75, 96, 123, 151, 168, 160, 146, 133, 129,
]

manual = ThetaForecaster(theta=2.0, alpha=0.2)
manual.fit(hourly_airport_pickups)
manual_forecast = manual.predict(6)

optimized = OptimizedThetaForecaster(
    theta_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
    alpha_grid=(0.1, 0.2, 0.4, 0.6),
)
optimized.fit(hourly_airport_pickups)
optimized_forecast = optimized.predict(6)

print(manual_forecast.predictions())
print(optimized_forecast.predictions())
```

## ForecastFrame Example

Use a `ForecastFrame` for panel data. Each `series_id` is fit as its own local
series.

```python
from cartoboost.forecasting import ForecastFrame, OptimizedThetaForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
)

model = OptimizedThetaForecaster(
    theta_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
    alpha_grid=(0.1, 0.2, 0.4, 0.6, 0.8),
)
model.fit(frame)
forecast = model.predict(12)
```

## Seasonal Theta

Hourly demand often has a daily cycle. Set both `seasonality` and
`season_length` when the training window contains enough full cycles.

```python
from cartoboost.forecasting import ThetaForecaster

model = ThetaForecaster(
    theta=2.0,
    alpha=0.25,
    seasonality="additive",
    season_length=24,
)
model.fit(hourly_zone_demand_values)
forecast = model.predict(24)
```

`seasonality` accepts `None`, `"additive"`, or `"multiplicative"` at the Python
validation layer. Additive seasonality is usually the first choice for pickup
counts because rush-hour lift is often a roughly fixed number of trips.
Multiplicative seasonality requires strictly positive values and fits series
where rush-hour effects scale with the level.

Seasonal theta requires at least two complete cycles per series. A configured
season length is never shortened or silently removed when the history is too
short; fitting fails with the required history length instead.

## Parameters

| Parameter | Notes |
| --- | --- |
| `theta` | Positive trend-control value for `ThetaForecaster`. Larger values preserve more drift; `theta=1.0` removes theta drift. |
| `alpha` | Smoothing value in `(0, 1]`. Larger values react faster to recent observations. |
| `theta_grid` | Non-empty positive candidate values for `OptimizedThetaForecaster`. |
| `alpha_grid` | Non-empty smoothing candidates in `(0, 1]`. |
| `seasonality` | Optional seasonal adjustment mode: `None`, `"additive"`, or `"multiplicative"`. |
| `season_length` | Required when `seasonality` is set. Use `24` for hourly daily cycles, `7` for daily weekly cycles. |
| `prediction_interval_levels` | Unique interval levels between `0` and `1`; uses the fitted residual Gaussian approximation. |

## Tuning Pattern

Start with a fixed manual model and a compact optimized grid:

```python
manual = ThetaForecaster(theta=2.0, alpha=0.25, seasonality="additive", season_length=24)

optimized = OptimizedThetaForecaster(
    theta_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
    alpha_grid=(0.1, 0.2, 0.4, 0.6, 0.8),
    seasonality="additive",
    season_length=24,
)
```

For benchmark claims, select settings with rolling-origin validation on the
same split used by the competing models. Keep the grid small enough that it is
easy to rerun and explain.

Interpretation:

| Pattern | Meaning | Typical next step |
| --- | --- | --- |
| Forecasts lag a real pickup-demand shift. | Smoothing is too stiff. | Increase `alpha` or include larger `alpha_grid` values. |
| Forecasts overreact to one unusual taxi hour. | Smoothing is too reactive. | Decrease `alpha` or compare against Kalman with higher observation noise. |
| Trend is too flat after a clear ramp. | Theta drift is too weak. | Try larger `theta` values. |
| Trend keeps climbing after demand plateaus. | Theta drift is too strong. | Try smaller `theta` values or `theta=1.0`. |
| Seasonal peaks are phase-shifted. | Seasonal length or data frequency is wrong. | Verify `freq`, sorted timestamps, and `season_length`. |

## Visual Example

Run the committed visualization example:

```bash
uv run python examples/forecasting/theta_optimized_visualization.py
```

It writes `target/examples/theta_optimized_visualization.png` and prints a JSON
summary with manual theta metrics, optimized theta metrics, and the best
holdout grid candidate from the example scoring loop. The example uses
synthetic panel counts and does not download data.

The core plotting pattern is:

```python
from pathlib import Path

import matplotlib.pyplot as plt
from cartoboost.forecasting import ForecastFrame, OptimizedThetaForecaster, ThetaForecaster

frame = ForecastFrame.from_pandas(
    train_pickups,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
)

manual = ThetaForecaster(theta=2.0, alpha=0.25, seasonality="additive", season_length=24)
manual.fit(frame)
manual_forecast = manual.predict(12).predictions()

optimized = OptimizedThetaForecaster(
    theta_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
    alpha_grid=(0.15, 0.25, 0.45, 0.65),
    seasonality="additive",
    season_length=24,
)
optimized.fit(frame)
optimized_forecast = optimized.predict(12).predictions()

plt.plot(observed["pickup_hour"], observed["demand"], label="observed counts")
plt.plot(manual_hours, manual_values, label="Theta theta=2.0 alpha=0.25")
plt.plot(optimized_hours, optimized_values, label="OptimizedTheta grid")
plt.axvline(observed["pickup_hour"].iloc[-12], color="gray", linestyle="--")
plt.xlabel("pickup hour")
plt.ylabel("pickup count")
plt.legend()

Path("target/examples").mkdir(parents=True, exist_ok=True)
plt.savefig("target/examples/theta_forecast_comparison.png", dpi=150)
```

Use the plot to explain model behavior, not to choose parameters by eye. The
validation metric should decide between candidate grids.

## Command-line Example

The existing single-series Python example exercises the forecasting command-line
workflow:

```bash
uv run python examples/forecasting/single_series_theta.py
```

It fits a theta model against `examples/forecasting/forecast_cli_input.csv` with
`timestamp`, `demand`, and `series_id` columns, then prints the saved
model artifact JSON.

## Validation

Theta models are local univariate models. They do not encode hour-of-week
calendar features beyond the configured seasonal pattern, and they do not share
information across pickup zones. If a taxi workflow depends on special-event
effects, airport disruption, location mix, or spatial spillover, compare
theta against CartoBoost lag models, graph/neural features, ARIMA, ETS, Kalman,
and seasonal naive on the same split.

When reporting results, include RMSE, MAE, R2 when available, training time,
prediction time, horizon, split dates, model settings, sample size, and whether
the data came from real taxi trips or generated acceptance data.

## Use When

| Your task | Start with |
| --- | --- |
| A short series has a local level and drift | Theta |
| A repeated additive cycle needs its own state | [ETS](ets.md) |
| Measured history is noisy and state uncertainty matters | [Kalman](kalman.md) |

## Limitations

- Theta is local and does not share information across related series.
- Strong multiple seasonalities need explicit preprocessing or another model.
- Events, spatial spillover, and external regressors are not modeled automatically.
- Optimized variants must choose parameters inside training folds.
