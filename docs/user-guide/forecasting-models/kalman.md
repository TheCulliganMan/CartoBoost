import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Kalman

CartoBoost exposes Kalman forecasters for both level-only and local-linear-trend
state-space models. Use `LocalLevelKalmanForecaster` when the series has a
noisy stable level, `KalmanForecaster` when the latent level also has a slowly
changing trend, and the `Auto*` variants when you want a deterministic
variance-grid search based on held-out predictive likelihood before refitting
on all training rows.

## Interactive Example

<ForecastModelExample title="Auto Kalman panel forecast" model="auto_kalman" />

## When To Use

Use Kalman forecasting for demand or fare aggregates when recent
observations should update the level and trend without requiring a fixed
seasonal cycle. It is often a good baseline for short horizons and noisy
single-zone series.

## Scientific Role

Kalman forecasting is a state-space choice. It represents observed counts
as noisy measurements of an unobserved level and trend. The model is useful
when the research question is: "What latent demand state best explains these
noisy observations, and how should that state move into the next few horizons?"

Choose Kalman when measurement noise and gradual state movement are central to
the problem. It is more scientifically appropriate than naive when the last
point may be noisy, and more direct than ARIMA when you want explicit latent
level and trend diagnostics.

## Assumptions And Failure Modes

The local-level class has one latent level. The local-linear class has a latent
level and trend. In both cases, process variances control how quickly the
latent state moves, while observation variance controls how strongly the model
trusts noisy measurements.

Kalman can fail when a fixed seasonal cycle dominates, when known future
events drive the forecast, or when abrupt structural breaks are too large for
the process variance settings. A very reactive filter can chase noise; a very
stiff filter can miss real pickup-demand shifts. Use standardized innovations,
forecast intervals, and rolling-origin error to decide whether the variance
settings match the series.

## Example

```python
from cartoboost.forecasting import AutoKalmanForecaster

airport_pickups = [72, 75, 79, 82, 80, 86, 91, 96, 94, 99, 103, 108]

model = AutoKalmanForecaster(
    level_process_variance_grid=[0.01, 0.05, 0.10],
    trend_process_variance_grid=[0.001, 0.005, 0.010],
    observation_variance_grid=[0.5, 1.0, 2.0],
    validation_window=3,
)
model.fit(airport_pickups)
forecast = model.predict(6)

print(model.get_metadata())
print(forecast.predictions())
```

## Examples

Use these small examples to choose the right Kalman surface before moving to a
full panel.

| Example | Use | Why |
| --- | --- | --- |
| A single location sensor is noisy but the true demand level is stable. | `LocalLevelKalmanForecaster` or `cartoboost.local_level_kalman_filter` | One latent level is enough; there is no explicit slope. |
| Airport pickup demand is drifting upward through the evening rush. | `KalmanForecaster` or `cartoboost.utilities.kalman_filter` | The local-linear model estimates both a level and a trend. |
| You want the model to choose variance settings from a small grid. | `AutoLocalLevelKalmanForecaster` or `AutoKalmanForecaster` | Candidate settings are scored by predictive negative log likelihood on a time-ordered tail window and the winner is refit. |
| You need a normal forecast band for a dashboard. | `forecast_distribution` from either utility | It returns mean, variance, lower, and upper for each horizon. |
| You need to explain why a point looked unusual. | Per-step estimates and `diagnostics` | Innovations, standardized innovations, gains, and log likelihood show how surprising the observation was. |

Local-level example:

```python
import cartoboost as cb

zone_236_readings = [184.0, 187.0, 183.0, 186.0, 185.0, 188.0, 186.0]

state = cb.local_level_kalman_filter(
    zone_236_readings,
    level_process_variance=0.04,
    observation_variance=1.5,
    horizon=3,
)

print(state["final_state"])
print(state["forecast_distribution"])
```

Local-linear example:

```python
import cartoboost as cb

jfk_evening_pickups = [74.0, 76.0, 79.0, 78.0, 83.0, 86.0, 84.0, 90.0, 92.0]

state = cb.kalman_filter(
    jfk_evening_pickups,
    level_process_variance=0.08,
    trend_process_variance=0.01,
    observation_variance=2.0,
    horizon=4,
)

print(state["final_state"]["level"], state["final_state"]["trend"])
print(state["diagnostics"]["rmse"], state["diagnostics"]["mae"])
```

## ForecastFrame Example

```python
from cartoboost.forecasting import AutoKalmanForecaster, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("series_id == '132'"),
    timestamp_col="pickup_hour",
    target_col="demand",
    freq="h",
)

model = AutoKalmanForecaster(
    level_process_variance_grid=[0.02, 0.05, 0.10],
    trend_process_variance_grid=[0.002, 0.005, 0.010],
    observation_variance_grid=[1.0, 2.0, 4.0],
    validation_window=24,
)
model.fit(frame)
forecast = model.predict(12)
print(model.metadata_["selected_params"])
```

## Parameters

| Class | Parameters |
| --- | --- |
| `LocalLevelKalmanForecaster` | `level_process_variance`, `observation_variance`. |
| `KalmanForecaster` | `level_process_variance`, `trend_process_variance`, `observation_variance`. |
| `AutoLocalLevelKalmanForecaster` | `level_process_variance_grid`, `observation_variance_grid`, optional `validation_window`. |
| `AutoKalmanForecaster` | `level_process_variance_grid`, `trend_process_variance_grid`, `observation_variance_grid`, optional `validation_window`. |

Larger process variance lets the latent level or trend move faster. Larger
observation variance makes the model trust noisy observations less. Auto
variants report `selected_params` and per-candidate `mse` and
`negative_log_likelihood` values under `validation_scores` in `metadata_`.

## Self-Tuning Pattern

```python
from cartoboost.forecasting import AutoLocalLevelKalmanForecaster

model = AutoLocalLevelKalmanForecaster(
    level_process_variance_grid=[0.005, 0.02, 0.08],
    observation_variance_grid=[0.5, 1.0, 3.0],
    validation_window=6,
)
model.fit(zone_236_readings)
print(model.metadata_["validation_scores"])
```

The auto forecasters use a deterministic time-ordered tail validation window.
They choose the lowest mean predictive negative log likelihood, use MSE and the
parameter values as deterministic tie-breakers, then refit the selected model
on all training rows. Predictive likelihood scores both the forecast error and
the variance assigned to that error, so grids that differ only by a common
variance scale remain distinguishable.

Every series must contain a real holdout: `AutoKalmanForecaster` requires at
least two fitting rows plus one held-out row, while
`AutoLocalLevelKalmanForecaster` requires at least one fitting row plus one
held-out row. An explicit `validation_window` must leave that minimum fitting
history for every series; an oversized window is rejected instead of shortened.
When the window is omitted, CartoBoost uses a deterministic tail window based on
one fifth of the series length, bounded to 1--12 rows and by the available
fitting history.

## Validation Notes

Kalman models do not encode hour-of-day wraparound or zone geography. If those
effects matter, compare Kalman against seasonal naive and a lag model with
calendar features.

## Filtering Diagnostics

Use the plain utility when you need state diagnostics instead of only a
forecasting API result:

```python
import cartoboost as cb

state = cb.kalman_filter(
    airport_pickups,
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=1.0,
    horizon=6,
)

print(state["final_state"]["covariance"])
print(state["diagnostics"]["log_likelihood"])
print(state["smoothed_states"][-1])
print(state["forecast_distribution"][0])
```

`forecast_distribution` contains mean, variance, and normal-approximation bounds
for each future taxi pickup horizon. Per-step estimates include innovations,
standardized innovations, fitted values, residuals, innovation variance,
level/trend Kalman gains, covariance matrices, and Gaussian log likelihoods.
`smoothed_states` are fixed-interval backward-smoothed states over the observed
history, and `diagnostics` includes AIC, BIC, RMSE, MAE, and standardized
innovation summaries.

## Visual Diagnostics

The most useful Kalman plots show four things together:

- observed taxi counts,
- one-step fitted values,
- filtered and smoothed latent levels,
- forecast intervals for future horizons.

Run the committed example:

```bash
uv run python examples/forecasting/kalman_diagnostics_visualization.py
```

It writes `target/examples/kalman_diagnostics_visualization.png` and prints a
small JSON summary. The example uses synthetic panel counts and does
not download data.

The core plotting pattern is:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import cartoboost as cb

pickup_hours = list(range(18))
pickups = [
    74.0, 76.0, 79.0, 78.0, 83.0, 86.0,
    84.0, 90.0, 92.0, 91.0, 97.0, 101.0,
    99.0, 104.0, 108.0, 109.0, 114.0, 117.0,
]

state = cb.kalman_filter(
    pickups,
    level_process_variance=0.08,
    trend_process_variance=0.01,
    observation_variance=2.0,
    horizon=6,
)

estimate_hours = [row["step"] for row in state["estimates"]]
filtered = [row["level"] for row in state["estimates"]]
fitted = [row["fitted"] for row in state["estimates"]]
smoothed_hours = [row["step"] for row in state["smoothed_states"]]
smoothed = [row["level"] for row in state["smoothed_states"]]

future_hours = [pickup_hours[-1] + row["step"] for row in state["forecast_distribution"]]
forecast_mean = [row["mean"] for row in state["forecast_distribution"]]
forecast_lower = [row["lower"] for row in state["forecast_distribution"]]
forecast_upper = [row["upper"] for row in state["forecast_distribution"]]

plt.plot(pickup_hours, pickups, marker="o", label="observed")
plt.plot(estimate_hours, fitted, linestyle="--", label="one-step fitted")
plt.plot(estimate_hours, filtered, label="filtered level")
plt.plot(smoothed_hours, smoothed, label="smoothed level")
plt.plot(future_hours, forecast_mean, marker="o", label="forecast")
plt.fill_between(future_hours, forecast_lower, forecast_upper, alpha=0.18, label="95% interval")
plt.xlabel("hour index")
plt.ylabel("pickup count")
plt.legend()

Path("target/examples").mkdir(parents=True, exist_ok=True)
plt.savefig("target/examples/kalman_forecast_band.png", dpi=160)
```

Plot standardized innovations when you want to spot unusual observations:

```python
standardized = [row["standardized_innovation"] for row in state["estimates"]]

plt.figure()
plt.axhline(0.0, color="black", linewidth=1)
plt.axhline(1.96, color="gray", linestyle="--", linewidth=1)
plt.axhline(-1.96, color="gray", linestyle="--", linewidth=1)
plt.bar(estimate_hours, standardized)
plt.xlabel("hour index")
plt.ylabel("standardized innovation")
plt.savefig("target/examples/kalman_standardized_innovations.png", dpi=160)
```

Interpretation:

| Visual pattern | Meaning | Typical next step |
| --- | --- | --- |
| Filtered level chases every observation. | The model trusts observations too much. | Increase `observation_variance` or reduce process variances. |
| Smoothed level lags a real shift. | The latent state is too stiff. | Increase `level_process_variance`. |
| Forecast band is too narrow for recent errors. | Observation/process variance is too low. | Compare RMSE/MAE and raise variance settings. |
| Many standardized innovations cross +/-1.96. | Recent observations are surprising under the model. | Check missing calendar/zone effects or tune variances with backtesting. |

## Tuning With An Example Grid

For a real workflow, score a small grid on a fixed time-ordered split before
choosing parameters. The example below mirrors the auto forecaster's primary
predictive-likelihood score and MSE tie-breaker.

```python
import math

import cartoboost as cb

train = [74.0, 76.0, 79.0, 78.0, 83.0, 86.0, 84.0, 90.0, 92.0, 91.0, 97.0, 101.0]
validation = [99.0, 104.0, 108.0]

candidates = [
    {"level_process_variance": 0.03, "trend_process_variance": 0.003, "observation_variance": 1.0},
    {"level_process_variance": 0.08, "trend_process_variance": 0.010, "observation_variance": 2.0},
    {"level_process_variance": 0.15, "trend_process_variance": 0.020, "observation_variance": 3.0},
]

scores = []
for params in candidates:
    state = cb.kalman_filter(train, horizon=len(validation), **params)
    distribution = state["forecast_distribution"]
    errors = [point["mean"] - actual for point, actual in zip(distribution, validation)]
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    mse = rmse * rmse
    mean_nll = sum(
        0.5
        * (
            math.log(2.0 * math.pi * point["variance"])
            + error * error / point["variance"]
        )
        for point, error in zip(distribution, errors)
    ) / len(errors)
    scores.append((mean_nll, mse, params))

print(min(scores, key=lambda item: (item[0], item[1])))
```

Use the diagnostic plots after picking the best validation candidate; do not
choose variance settings by making the in-sample line look smooth.
