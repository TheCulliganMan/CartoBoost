# Theta

Theta models provide lightweight trend extrapolation for local forecasting.
They are useful when taxi demand has a clear local level and drift, but you do
not need the larger ARIMA or lag-model surface.

## Try It In The Modeling Lab

Open the browser-local example with the bundled taxi lane sample:
[Optimized Theta forecast](/modeling-lab?sample=lane&model=optimized_theta&run=forecast).
Run a holdout backtest from the same page to compare theta against seasonal
naive, ETS, ARIMA, Kalman, and lag models under one split.

## When To Use

Use theta forecasting for pickup counts, fares, trip duration, or trip distance
aggregates when the recent history is short, the main signal is level plus
trend, and you want a fast deterministic baseline. Theta is often a practical
comparison point for airport pickup demand, single-zone hourly demand, and
taxi-lane aggregates where the next few horizons should continue a smooth
recent movement.

Do not treat theta as evidence of spatial generalization. It does not know
pickup-zone geometry, dropoff-zone relationships, road distance, or graph
structure. Compare it against seasonal naive and lag-feature models on the same
rolling-origin folds before using it as a production benchmark.

## Scientific Role

Theta is a low-dimensional extrapolator. It asks whether the future can be
explained by a smoothed level plus a controlled trend component, optionally
after applying a simple seasonal adjustment. That makes it useful when a
scientist wants a transparent trend baseline before moving to richer
autocorrelation, state-space, or supervised lag models.

Choose theta when the research question is close to: "Does recent taxi demand
continue its local direction?" It is especially useful for short windows where
a large lag model would have too few examples, but where naive persistence is
too flat.

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

## Example

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

Use a `ForecastFrame` for taxi panels. Each `PULocationID` is fit as its own
local series.

```python
from cartoboost.forecasting import ForecastFrame, OptimizedThetaForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
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

Hourly taxi demand often has a daily cycle. Set both `seasonality` and
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

## Parameters

| Parameter | Notes |
| --- | --- |
| `theta` | Positive trend-control value for `ThetaForecaster`. Larger values preserve more drift; `theta=1.0` removes theta drift. |
| `alpha` | Smoothing value in `(0, 1]`. Larger values react faster to recent observations. |
| `theta_grid` | Non-empty positive candidate values for `OptimizedThetaForecaster`. |
| `alpha_grid` | Non-empty smoothing candidates in `(0, 1]`. |
| `seasonality` | Optional seasonal adjustment mode: `None`, `"additive"`, or `"multiplicative"`. |
| `season_length` | Required when `seasonality` is set. Use `24` for hourly daily cycles, `7` for daily weekly cycles. |
| `prediction_interval_levels` | Validated interval levels between `0` and `1`. |

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
JFK-style and Upper East Side-style pickup counts and does not download data.

The core plotting pattern is:

```python
from pathlib import Path

import matplotlib.pyplot as plt
from cartoboost.forecasting import ForecastFrame, OptimizedThetaForecaster, ThetaForecaster

frame = ForecastFrame.from_pandas(
    train_pickups,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
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

plt.plot(observed["pickup_hour"], observed["pickup_count"], label="observed pickups")
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

## CLI Example

The existing single-series theta example exercises the forecasting CLI:

```bash
uv run python examples/forecasting/single_series_theta.py
```

It fits a theta model against `examples/forecasting/forecast_cli_input.csv` with
`timestamp`, `pickup_demand`, and `PULocationID` columns, then prints the saved
model artifact JSON.

## Validation Notes

Theta models are local univariate models. They do not encode hour-of-week
calendar features beyond the configured seasonal pattern, and they do not share
information across pickup zones. If a taxi workflow depends on special-event
effects, airport disruption, dropoff-zone mix, or spatial spillover, compare
theta against CartoBoost lag models, graph/neural features, ARIMA, ETS, Kalman,
and seasonal naive on the same split.

When reporting results, include RMSE, MAE, R2 when available, training time,
prediction time, horizon, split dates, model settings, sample size, and whether
the data came from real taxi trips or generated acceptance data.
