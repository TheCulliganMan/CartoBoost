import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# ETS

`ETSForecaster` is the exponential smoothing forecaster for level, additive
trend, and optional additive seasonality. It is a good fit when a series has a
stable seasonal pattern and recent observations should update the state
smoothly instead of forcing abrupt step changes.

## Interactive Example

<ForecastModelExample title="Auto ETS panel forecast" model="auto_ets" />

## When To Use

Use ETS for hourly counts, fare totals, duration aggregates, or trip distance
aggregates when the series has a clear repeating cycle. Common uses include
hour-of-day demand, evening demand, or a single panel with a daily rhythm.

ETS is strongest when the forecast can be explained as:

- a smoothed level for the current demand baseline,
- an additive trend for gradual drift,
- an additive seasonal adjustment for the current hour or day slot.

If the main signal is a sudden intervention, a moving event calendar, or
cross-panel spillover, compare ETS against a lag model with calendar features
and against spatial models where appropriate.

## Scientific Role

ETS is appropriate when the scientist can explain the series as an evolving
baseline, a gradual drift, and a repeatable additive seasonal effect. For panel
data, that means questions like: "How much of this count is the current
baseline, how much is a slow movement in that baseline, and how much is the
hour-of-day lift or drag?"

Choose ETS when component interpretability matters. The fitted level, trend,
seasonal component, fitted values, and residuals let you inspect whether the
model is explaining a stable cycle or merely smoothing over missing causes.

## Assumptions And Failure Modes

The current ETS surface is additive. It assumes seasonal effects add or
subtract roughly fixed amounts from the level, not fixed percentages. It also
requires enough complete seasonal cycles to estimate the seasonal state.

ETS can fail when demand is dominated by sudden shocks, sparse panels,
structural breaks, or effects that are known before the forecast but absent
from the model, such as holidays, weather, airport operations, or event
schedules. If residuals cluster by hour, zone, or route, compare against
CartoBoost lag/calendar features or a spatial model where appropriate.

## Example

```python
from cartoboost.forecasting import ETSForecaster

hourly_jfk_pickups = [
    64, 58, 52, 49, 55, 73, 98, 116, 121, 108, 95, 90,
    88, 92, 97, 103, 118, 132, 139, 135, 124, 102, 84, 70,
    66, 60, 54, 50, 57, 76, 102, 120, 126, 112, 99, 93,
    91, 95, 101, 107, 123, 137, 144, 140, 128, 106, 87, 73,
]

model = ETSForecaster(
    trend="additive",
    seasonal="additive",
    seasonal_periods=24,
    alpha=0.46,
    beta=0.08,
    gamma=0.24,
)
model.fit(hourly_jfk_pickups)
forecast = model.predict(6)

print(model.get_metadata())
print(forecast.columns())
print(forecast.predictions())
```

Forecast rows are `(series_id, timestamp, horizon, model, mean)`. When a plain
Python list is used, the wrapper builds a single-series frame. For production
panel data, prefer `ForecastFrame` so timestamps and series IDs stay explicit.

## ForecastFrame Example

```python
from cartoboost.forecasting import ETSForecaster, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("series_id == '132'"),
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
)

model = ETSForecaster(
    trend="additive",
    seasonal="additive",
    seasonal_periods=24,
    alpha=0.46,
    beta=0.08,
    gamma=0.24,
)
model.fit(frame)
forecast = model.predict(24)
```

## Parameters

| Parameter | Effect |
| --- | --- |
| `trend` | `None`, `"add"`, or `"additive"`. Additive trend lets the baseline drift each step. |
| `seasonal` | `None`, `"add"`, or `"additive"`. Additive seasonality adds a fixed-cycle adjustment. |
| `seasonal_periods` | Required and greater than `1` when `seasonal` is set. Use `24` for hourly daily seasonality and `168` for hourly weekly seasonality when enough history is available. |
| `alpha` | Level smoothing in `(0, 1]`. Larger values adapt the baseline faster. |
| `beta` | Trend smoothing in `[0, 1]`. Larger values let drift change faster. |
| `gamma` | Seasonal smoothing in `[0, 1]`; requires additive seasonality. Larger values update the seasonal pattern faster. |

Seasonal ETS requires at least two full cycles per series. For `seasonal_periods=24`,
each pickup zone needs at least 48 hourly observations before fitting.

## Smoothing Components

CartoBoost's ETS forecast is additive:

```text
forecast(t + h) = level(t) + h * trend(t) + seasonal((t + h) mod season_length)
```

During fitting, the model updates state with the observed taxi count,
the current seasonal slot, and the previous level/trend:

```text
fitted(t) = level(t-1) + trend(t-1) + seasonal(t)
level(t) = alpha * (y(t) - seasonal(t)) + (1 - alpha) * (level(t-1) + trend(t-1))
trend(t) = beta * (level(t) - level(t-1)) + (1 - beta) * trend(t-1)
seasonal(t) = gamma * (y(t) - level(t)) + (1 - gamma) * seasonal(t)
```

Interpret the components in series terms:

| Component | Interpretation | What to inspect |
| --- | --- | --- |
| Level | Current baseline count after removing hour-of-day effects. | Whether the baseline follows real demand shifts without chasing every observation. |
| Trend | Recent drift in the baseline. | Whether the model projects a plausible ramp into the next few hours. |
| Seasonal | Additive lift or drag for a repeating slot, such as each hour of day. | Whether peak periods, overnight lows, and evening demand have sensible signs and magnitudes. |
| Residual | Difference between observed values and one-step fitted values. | Whether missed events or missing calendar features dominate the error. |

## Visual Diagnostics

Run the committed example:

```bash
uv run python examples/forecasting/ets_component_visualization.py
```

It writes `target/examples/ets_component_visualization.png` and prints a JSON
summary with RMSE, MAE, model metadata, final level/trend, and seasonal range.
The example uses deterministic panel counts and does not download data.

The plot is designed to answer three practical questions:

- Does the one-step fitted line follow the training history without copying
  noise?
- Do the level and trend components look plausible for the panel?
- Does the seasonal component show the expected overnight drag and peak-hour
  lift?

The core pattern is:

```python
from cartoboost.forecasting import ETSForecaster, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
)

model = ETSForecaster(
    trend="additive",
    seasonal="additive",
    seasonal_periods=24,
    alpha=0.46,
    beta=0.08,
    gamma=0.24,
)
model.fit(frame)
forecast = model.predict(12)

levels = model.levels("132")
trends = model.trends("132")
seasonal = model.seasonal_components("132")
fitted = model.fitted_values("132")
residuals = model.residuals("132")
```

These diagnostic arrays have one value per training observation. The
visualization example plots them alongside the forecast.

Visual interpretation:

| Visual pattern | Meaning | Typical next step |
| --- | --- | --- |
| Fitted values lag a real demand shift. | Level smoothing is too low or the model needs a faster trend update. | Increase `alpha`, then test a modestly higher `beta`. |
| Fitted values chase every spike. | The model is reacting to noise or event outliers. | Lower `alpha` and compare rolling-origin error. |
| Trend keeps projecting an unrealistic ramp. | Trend smoothing is too high for the horizon. | Lower `beta` or compare against seasonal naive. |
| Seasonal component is nearly flat despite clear hour-of-day demand. | Seasonal updates are too weak or the season length is wrong. | Check `seasonal_periods`; then raise `gamma`. |
| Seasonal component flips sign unexpectedly across adjacent hours. | Seasonality may be overfit or history is too short. | Lower `gamma` or fit on more complete cycles. |

## Tuning Guidance

Start with a small grid rather than trying to infer settings from the smoothest
in-sample line:

```python
candidates = [
    {"alpha": 0.25, "beta": 0.03, "gamma": 0.10},
    {"alpha": 0.46, "beta": 0.08, "gamma": 0.24},
    {"alpha": 0.65, "beta": 0.12, "gamma": 0.35},
]
```

For hourly demand, use the same train/validation split for every
candidate. Score each setting against the validation horizon:

```python
scores = []
for params in candidates:
    model = ETSForecaster(
        trend="additive",
        seasonal="additive",
        seasonal_periods=24,
        **params,
    )
    model.fit(train_frame)
    forecast_rows = model.predict(len(validation_counts)).predictions()
    predictions = [row[-1] for row in forecast_rows]
    errors = [
        prediction - actual
        for prediction, actual in zip(predictions, validation_counts)
    ]
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    scores.append((rmse, params))

print(sorted(scores, key=lambda item: item[0])[0])
```

Prefer the simplest setting selected by validation. If ETS and seasonal naive
are close, keep the seasonal naive baseline in reporting; it is a useful guard
against overclaiming smoothing behavior on strongly repeating series.

## Validation Notes

Compare ETS against seasonal naive at the same season length. For hourly taxi
demand, try daily (`24`) and weekly (`168`) season lengths when the data window
is long enough for stable validation. Record RMSE, MAE, the train/test split,
the horizon, and the smoothing parameters used for any benchmark claim.
