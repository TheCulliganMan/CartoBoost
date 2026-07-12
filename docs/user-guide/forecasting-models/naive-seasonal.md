import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Naive And Seasonal Naive

Naive and seasonal naive models are the first baselines to run for forecasting.
They are intentionally simple and make leakage problems easier to spot.

## Interactive Example

<ForecastModelExample title="Seasonal naive panel forecast" model="seasonal_naive" />

## When To Use

Use `NaiveForecaster` when the next value should be compared against the last
observed value. Use `SeasonalNaiveForecaster` when the series has a stable
cycle, such as hourly demand with `season_length=24` or daily demand with
`season_length=7`.

| Model | Behavior |
| --- | --- |
| `NaiveForecaster` | Repeats the most recent observed value for each future horizon. |
| `SeasonalNaiveForecaster(season_length)` | Repeats values from the most recent completed seasonal cycle. |

Use both baselines before moving to ARIMA, ETS, Theta, Kalman, or lagged
CartoBoost models. The naive baseline compares against the latest known value.
The seasonal naive baseline compares against the prior cycle, which is often
the stronger control for recurring demand.

`NaiveForecaster` supports `ForecastFrame.from_pandas(...,
allow_missing_targets=True)` by fitting on observed target rows and forecasting
from the latest timestamp in the input frame. `SeasonalNaiveForecaster` still
requires observed finite targets at every regular step because its
`season_length` indexes previous rows in a complete seasonal cycle.

## Scientific Role

These models are not weak because they are simple; they are the control group.
They encode two clear hypotheses:

| Hypothesis | Model | What a scientist learns |
| --- | --- | --- |
| Demand persists from the most recent observation. | Naive | Whether short-horizon inertia explains the target. |
| Demand repeats by a fixed cycle. | Seasonal naive | Whether the calendar phase explains the target without learned parameters. |

Choose naive or seasonal naive when you need an auditable baseline, a leakage
check, or a minimum bar for a richer model. A model that does not clear
seasonal naive on recurring hourly demand may only be restating the cycle with
more machinery.

## Assumptions And Failure Modes

Naive assumes the level is locally stable over the forecast horizon. It fails
when demand is moving into or out of a peak, when a disruption shifts the level,
or when the last point is an outlier.

Seasonal naive assumes the last completed cycle is representative of the next
cycle. It fails when the same point in the prior cycle is not comparable
because of holidays, weather, event schedules, or a real regime change in the
series. If a series does not contain a complete configured cycle, fitting fails
explicitly; the season length is never shortened to make the model run. A
season length that is numerically feasible but does not match the data cadence
still produces the wrong comparison, so verify both the declared frequency and
cycle definition.

## Single-Series Example

```python
from cartoboost.forecasting import NaiveForecaster, SeasonalNaiveForecaster

hourly_demand = [42, 38, 35, 31, 44, 67, 91, 105, 98, 86, 73, 69]

last_value = NaiveForecaster().fit(hourly_demand)
last_cycle = SeasonalNaiveForecaster(season_length=6).fit(hourly_demand)

print(last_value.predict(3).predictions())
print(last_cycle.predict(3).predictions())
```

Interpret the result directly:

| Output pattern | Meaning | Typical next step |
| --- | --- | --- |
| Naive and seasonal naive are close. | The latest observation is already a good short-horizon summary. | Compare against Kalman or ETS before adding many lag features. |
| Seasonal naive has much lower error. | Hour-of-day or day-of-week repetition dominates. | Keep the seasonal baseline in every validation table. |
| Naive has lower error than seasonal naive. | The recent level shifted away from the prior cycle. | Check for events, holidays, weather disruption, or zone-level regime changes. |
| Both baselines miss the same periods. | Repeated calendar cycles are not enough. | Add exogenous features, graph features, or a lagged CartoBoost model. |

## Pickup-Zone Panel Example

```python
from cartoboost.forecasting import ForecastFrame, SeasonalNaiveForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="zone_id",
    freq="h",
)

model = SeasonalNaiveForecaster(season_length=24)
model.fit(frame)
forecast = model.predict(12)

for row in forecast.predictions()[:5]:
    print(row)
```

`ForecastFrame` keeps each `zone_id` separate. For a 24-hour seasonal naive
model, the next forecast for zone `132` uses zone `132` from 24 hours ago; it
does not borrow observations from zone `236` or any other panel.

## Visual Example

Run the committed visualization example:

```bash
uv run python examples/forecasting/naive_seasonal_visualization.py
```

It writes `target/examples/naive_seasonal_visualization.png` and prints a JSON
summary with rows, panels, train horizon, forecast horizon, MAE, RMSE, and the
seasonal-naive RMSE delta versus naive. The example generates deterministic
panel demand, so it does not download data or write tracked benchmark
artifacts.

The plot compares three lines:

- observed hourly counts,
- the flat naive forecast from the last observed hour,
- the seasonal naive forecast from the previous daily cycle.

The core plotting pattern is:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cartoboost.forecasting import ForecastFrame, NaiveForecaster, SeasonalNaiveForecaster

    train = hourly_zone_demand.groupby("zone_id", sort=False).head(96)
frame = ForecastFrame.from_pandas(
    train,
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="zone_id",
    freq="h",
)

naive = NaiveForecaster().fit(frame).predict(24).predictions()
seasonal = SeasonalNaiveForecaster(season_length=24).fit(frame).predict(24).predictions()

naive_forecast = pd.DataFrame(
    naive,
    columns=["zone_id", "timestamp", "horizon", "model", "prediction"],
)
seasonal_forecast = pd.DataFrame(
    seasonal,
    columns=["zone_id", "timestamp", "horizon", "model", "prediction"],
)

zone_id = "132"
observed = hourly_zone_demand[hourly_zone_demand["zone_id"] == zone_id]
naive_zone = naive_forecast[naive_forecast["zone_id"] == zone_id]
seasonal_zone = seasonal_forecast[seasonal_forecast["zone_id"] == zone_id]

plt.plot(observed["timestamp"], observed["demand"], label="observed counts")
plt.plot(naive_zone["timestamp"], naive_zone["prediction"], label="naive")
plt.plot(seasonal_zone["timestamp"], seasonal_zone["prediction"], label="seasonal naive")
plt.xlabel("timestamp")
plt.ylabel("count")
plt.legend()

Path("target/examples").mkdir(parents=True, exist_ok=True)
plt.savefig("target/examples/naive_seasonal_demand.png", dpi=160)
```

Interpretation:

| Visual pattern | Meaning | Typical next step |
| --- | --- | --- |
| Naive is a horizontal line. | This is expected: it repeats the last observed value. | Use it as a leakage and horizon-alignment smoke test. |
| Seasonal naive follows the prior day's shape. | The daily profile is stable enough to forecast from the last cycle. | Compare complex models against this, not only against naive. |
| Seasonal naive is shifted above or below actuals. | The daily shape is useful but the level moved. | Try Kalman, ETS, or lagged features that can adjust level. |
| Seasonal naive gets rush hours wrong. | The previous cycle did not capture the current rush-hour pattern. | Add holiday/event/weather features or validate separate zone groups. |

## Parameters

| Parameter | Applies to | Notes |
| --- | --- | --- |
| `season_length` | `SeasonalNaiveForecaster` | Required positive integer. Use `24` for hourly daily seasonality and `168` for hourly weekly seasonality. |
| `prediction_interval_levels` | Both wrappers | Validated as values between `0` and `1`; interval support depends on the model output. |

## Choosing `season_length`

Match `season_length` to the row spacing in the `ForecastFrame`.

| Data frequency | Typical question | Common `season_length` |
| --- | --- | --- |
| Hourly counts | Does this hour behave like the same hour yesterday? | `24` |
| Hourly counts | Does this hour behave like the same hour last week? | `168` |
| Daily counts | Does this date behave like the same weekday last week? | `7` |
| 15-minute counts | Does this interval behave like the same interval yesterday? | `96` |

Do not use `24` for daily data or `7` for hourly data unless the rows have been
aggregated to that cadence. A wrong season length can look plausible in a plot
while comparing the wrong historical period.

## Backtest Guidance

Score these baselines with the same split, horizon, and aggregation used for
candidate models:

```python
from cartoboost.forecasting import (
    NaiveForecaster,
    RollingOriginBacktester,
    RollingOriginSplitter,
    SeasonalNaiveForecaster,
)

splitter = RollingOriginSplitter(horizon=24, step=24, min_train_size=72)
backtester = RollingOriginBacktester(splitter=splitter)

naive_result = backtester.evaluate(NaiveForecaster(), frame)
seasonal_result = backtester.evaluate(SeasonalNaiveForecaster(season_length=24), frame)
```

Keep the seasonal naive score in model-selection reports for hourly demand. If a
richer model only clears naive, it may only be learning the daily cycle rather
than adding useful zone, graph, weather, or calendar signal.

## Validation Notes

Seasonal naive is the minimum meaningful baseline for strongly seasonal demand.
If a more complex model does not clear seasonal naive under rolling-origin
backtests, inspect feature leakage, horizon alignment, and whether the model is
overfitting repeated panels.

These models do not learn trend, holiday effects, disruption, zone spillover,
or graph structure. That limitation is useful: when they perform well, the
repeated cycle is strong; when they fail, the residuals show where richer
forecasting models need to explain the system.
