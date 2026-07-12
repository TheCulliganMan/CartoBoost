import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# ARIMA And AutoARIMA

ARIMA models are useful when a series is mostly explained by recent
values, recent forecast errors, and one or two rounds of differencing.
CartoBoost exposes bounded, non-seasonal ARIMA models:

- `ArimaForecaster` fits one fixed `(p, d, q)` order.
- `AutoARIMAForecaster` searches a bounded deterministic grid of `(p, d, q)`
  candidates and refits the selected order.
- `cartoboost.preview.utilities.arima_forecast` and
  `cartoboost.preview.utilities.auto_arima_forecast` provide the
  same model behavior for quick single-series utility calls.

## Interactive Example

<ForecastModelExample title="Auto ARIMA panel forecast" model="auto_arima" />

## When To Use It

Use ARIMA for one regular series when:

- the series needs first or second differencing to remove a local trend,
- the last few observations predict the next value,
- recent forecast errors carry useful signal, or
- you want a strong local baseline before trying a global lag model.

Prefer `CartoBoostLagForecaster` when many panels should share one model and
borrow cross-panel structure. Prefer `KrigingForecaster` when spatial
coordinates should smooth nearby locations. Prefer seasonal naive, ETS, or
Theta when daily or weekly seasonality dominates and the non-seasonal ARIMA
scope is too narrow.

## Scientific Role

ARIMA is a local serial-dependence model. It is the right scientific choice
when the hypothesis is that a single series can be forecast from its own recent
values, its recent errors, and a bounded amount of differencing. It does not
explain the series with geography, shared panel behavior, or known future
covariates.

Choose fixed ARIMA when the order is part of the experiment or has already been
validated. Choose AutoARIMA when you want a reproducible bounded search over
non-seasonal `(p, d, q)` candidates, then still choose the deployed model by
held-out or rolling-origin error.

## Assumptions And Failure Modes

ARIMA assumes the differenced series is stable enough for short-range
autoregressive and moving-average terms to be useful. It can fail when the
strongest signal is deterministic seasonality, known calendar effects, sudden
interventions, spatial spillover, or cross-series learning.

Common failure modes are easy to diagnose:

| Failure mode | Scientific interpretation | Comparison to run |
| --- | --- | --- |
| Forecast is too flat across a ramp. | Differencing/order choice is not preserving local movement. | Kalman, theta, or `d=1` candidates. |
| Residuals repeat by hour of day. | Non-seasonal ARIMA is missing a seasonal mechanism. | Seasonal naive, ETS, or lag features with calendar terms. |
| One lane fits well and another fails. | Local orders do not transfer across lane regimes. | Per-lane validation or a global lag model. |
| AutoARIMA selects an implausibly explosive order. | The configured search is too broad or the history is too short. | Inspect stationarity/invertibility fields and use a longer rolling-origin evaluation. |

## Model Surface

| Model | Import | Use when |
| --- | --- | --- |
| `ArimaForecaster` | `from cartoboost.preview.forecasting import ArimaForecaster` | You already know the non-seasonal `(p, d, q)` order. |
| `AutoARIMAForecaster` | `from cartoboost.preview.forecasting import AutoARIMAForecaster` | You want bounded candidate search over `(p, d, q)`. |
| `arima_forecast` | `from cartoboost.preview.utilities import arima_forecast` | You need a quick one-shot forecast for one numeric series. |
| `auto_arima_forecast` | `from cartoboost.preview.utilities import auto_arima_forecast` | You need one-shot bounded order selection for one numeric series. |

## Pickup-Demand Example

This example creates two hourly lanes. `lane_a` contains a
stronger airport-style morning ramp; `PU79->DO230` is flatter but still
autocorrelated.

```python
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from cartoboost.preview.forecasting import AutoARIMAForecaster, ForecastFrame


def example_lane_table(hours: int = 72) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    rows = []
    for lane, bias, ramp in [
        ("lane_a", 95.0, 18.0),
        ("lane_b", 64.0, 7.0),
    ]:
        for hour in range(hours):
            pickup_hour = start + timedelta(hours=hour)
            hour_of_day = pickup_hour.hour
            daily = 10.0 if 6 <= hour_of_day <= 9 else -4.0
            evening = 5.0 if 16 <= hour_of_day <= 19 else 0.0
            trend = hour * 0.12
            demand = bias + daily + evening + trend + ramp * (hour / hours)
            rows.append(
                {
                    "lane_id": lane,
                    "pickup_hour": pickup_hour,
                    "demand": demand,
                }
            )
    return pd.DataFrame(rows)


table = example_lane_table()
frame = ForecastFrame.from_pandas(
    table,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="lane_id",
    freq="h",
)

model = AutoARIMAForecaster(max_p=3, max_d=1, max_q=2)
model.fit(frame)
forecast = model.predict(12)

print(model.get_metadata())
print(forecast.predictions()[:3])
```

`ForecastFrame` validates the regular hourly timestamps, duplicate
series/timestamp pairs, finite pickup counts, and panel ids before fitting.

## Visualization Example

The runnable example `examples/forecasting/arima_example_visualization.py`
generates a deterministic taxi pickup/dropoff lane panel, fits fixed ARIMA and
AutoARIMA, prints held-out diagnostics as JSON, and optionally writes a forecast
and residual plot.

Run it from the repository root:

```bash
uv run python examples/forecasting/arima_example_visualization.py \
  --output target/examples/arima_example_visualization.png
```

The output includes RMSE, MAE, bias, maximum absolute error, the selected
AutoARIMA order, the top candidate scores, and per-horizon residuals for
`PU132->DO138`. The plot has two panels:

- observed pickup counts with the train/test split and both forecasts,
- held-out residual bars where residual means `prediction - actual`.

For docs or CI smoke checks where Matplotlib is not installed, omit
`--output`; the example still fits models and prints metrics.

The example is intentionally deterministic. It is useful for checking API
shape, plotting, and interpretation, but it is not evidence for model selection
on real demand.

The compact JSON fields are the first values to inspect:

```json
{
  "auto_arima_selected_label": "ARIMA(0,1,0)",
  "heldout_winner_by_rmse": "arima_2_1_1",
  "arima_2_1_1": {"mae": 6.02, "rmse": 7.54, "bias": -1.75},
  "auto_arima": {"mae": 5.95, "rmse": 7.72, "bias": -2.26}
}
```

Exact values can change if you pass different `--hours`, `--train-hours`, or
`--horizon` settings. Use `auto_arima_top_candidates` to see whether the
selected order was a clear prefix-holdout winner, and use per-horizon
`residuals` to check whether the held-out miss is directional or grows with
horizon.

## Visual Diagnostics

The most useful ARIMA plots show forecast shape and residual behavior together.
In a local-series workflow, inspect:

- whether the forecast follows the held-out pickup ramp or flattens too early,
- whether residuals are mostly centered around zero,
- whether residual magnitude grows with horizon,
- whether one order has lower RMSE but has a directional bias that matters for the
  operational decision.

The committed example writes that diagnostic view:

```bash
uv run python examples/forecasting/arima_example_visualization.py \
  --hours 96 \
  --train-hours 72 \
  --horizon 12 \
  --output target/examples/arima_example_visualization.png
```

The core plotting pattern is:

```python
fixed_evaluation = actual.merge(fixed_forecast, on=["lane_id", "pickup_hour"])
fixed_evaluation["residual"] = fixed_evaluation["prediction"] - fixed_evaluation["demand"]

auto_evaluation = actual.merge(auto_forecast, on=["lane_id", "pickup_hour"])
auto_evaluation["residual"] = auto_evaluation["prediction"] - auto_evaluation["demand"]

axis.plot(observed["pickup_hour"], observed["demand"], label="Observed values")
axis.plot(fixed_evaluation["pickup_hour"], fixed_evaluation["prediction"], label="ARIMA(2,1,1)")
axis.plot(auto_evaluation["pickup_hour"], auto_evaluation["prediction"], label="AutoARIMA")
residual_axis.axhline(0.0, color="black", linewidth=1)
residual_axis.bar(fixed_evaluation["horizon"], fixed_evaluation["residual"])
```

Interpretation:

| Visual pattern | Meaning | Typical next step |
| --- | --- | --- |
| Forecast is too flat after a pickup ramp. | The selected order did not preserve recent level/trend movement. | Compare against `d=1` candidates or a local-trend model such as Kalman. |
| Residuals are mostly positive. | The model is underpredicting the held-out tail. | Check recent trend, event hours, and whether differencing is needed. |
| Residuals are mostly negative. | The model is overpredicting the held-out tail. | Check whether a temporary pickup spike entered the training window. |
| Residual magnitude grows with horizon. | Short horizon is acceptable, but uncertainty grows quickly. | Report horizon-specific metrics and consider shorter operational horizons. |
| AutoARIMA's internal holdout winner loses on a later test tail. | One validation boundary did not represent the deployment period. | Confirm with multiple rolling origins and keep the final test tail untouched. |

## Fixed-Order ARIMA

Use fixed order when validation, domain knowledge, or a benchmark has already
chosen the model shape.

```python
from cartoboost.preview.forecasting import ArimaForecaster

hourly_pickups = [42, 38, 35, 31, 44, 67, 91, 105, 98, 86, 73, 69]

model = ArimaForecaster(p=2, d=1, q=1)
model.fit(hourly_pickups)
forecast = model.predict(6)

for series_id, timestamp, horizon, model_name, mean in forecast.predictions():
    print(series_id, timestamp, horizon, model_name, mean)
```

Order meaning:

- `p`: autoregressive lags from the differenced series.
- `d`: differencing order, currently `0`, `1`, or `2`.
- `q`: moving-average lags from recent fitted residuals.

Bounds are intentionally small (`p <= 8`, `d <= 2`, `q <= 8`) so repeated
local-model workflows stay deterministic and fast.

## Model-Order Interpretation

Treat `(p, d, q)` as a compact explanation of what the local series model is
allowed to remember:

| Order part | Interpretation | Risk when too small | Risk when too large |
| --- | --- | --- | --- |
| `p` | How many recent differenced pickup counts affect the next forecast. | Forecast ignores short local momentum. | Forecast can chase short spikes from a single unusual hour. |
| `d` | How many times the lane series is differenced before fitting AR/MA terms. | Forecast can lag a local trend or ramp. | Forecast can overreact and drift when the original level was already stable. |
| `q` | How many recent fitted errors affect the next forecast. | Systematic recent misses are not corrected. | Residual noise can be treated as signal. |

Examples:

- `ARIMA(0,0,0)` is a constant local mean baseline.
- `ARIMA(1,0,0)` uses the previous pickup count pattern without differencing.
- `ARIMA(0,1,0)` is a random-walk style forecast after first differencing.
- `ARIMA(2,1,1)` allows two recent differenced lags and one residual correction,
  which is often a useful fixed candidate for a short hourly series example.

AutoARIMA reports the selected order and all candidate scores:

```python
metadata = model.get_metadata()
print(metadata["selected_order"])
print(sorted(metadata["validation_scores"], key=lambda score: score["mse"])[:5])
```

Those scores are mean squared forecasts on a real suffix holdout. Differenced
candidate forecasts are reconstructed to the original target scale before
comparison, so orders with different `d` values are scored in the same units.
The selected candidate is then refit on the complete history.

## AutoARIMA

AutoARIMA searches all candidates in the bounded grid:

```python
from cartoboost.preview.forecasting import AutoARIMAForecaster

model = AutoARIMAForecaster(
    max_p=3,
    max_d=1,
    max_q=2,
)
model.fit(hourly_pickups)
forecast = model.predict(12)

metadata = model.get_metadata()
print(metadata["selected_order"])
print(metadata["validation_scores"][:5])
```

Candidate selection is deterministic. Every candidate is fitted only on the
prefix before the validation suffix, forecasts the suffix recursively, and is
scored on the original target scale. Autoregressive coefficients must satisfy
the stationarity check and moving-average coefficients must satisfy the
invertibility check. If two valid candidates tie, stable ordering keeps the
selection reproducible.

Read AutoARIMA metadata as an audit trail, not as a deployment decision. In the
visualization example, the internal suffix is used for order selection while a
later untouched tail measures deployment performance. The two boundaries can
still favor different models, which is why benchmark claims should use fixed
rolling origins or a separate final test period.

## ForecastFrame Panel Usage

ARIMA remains a local model family: each panel series is fitted independently.
That is useful for a small set of important pickup/dropoff lanes.

```python
from cartoboost.preview.forecasting import AutoARIMAForecaster, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_lane_demand,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="lane_id",
    freq="h",
)

model = AutoARIMAForecaster(max_p=4, max_d=1, max_q=2)
model.fit(frame)
forecast = model.predict(24)
```

For many lanes, use a rolling-origin backtest and compare against seasonal
naive, ETS, Theta, and `CartoBoostLagForecaster`. Do not claim a production
winner from the synthetic examples on this page.

## Many Local Series

When each lane should be modeled independently, schedule one local forecast per
lane and keep the evaluation table keyed by lane, horizon, and timestamp:

```python
from concurrent.futures import ThreadPoolExecutor

from cartoboost.preview.utilities import auto_arima_forecast


def forecast_lane(values: list[float]) -> list[float]:
    return auto_arima_forecast(values, horizon=12, max_p=3, max_d=1, max_q=2)


lanes = {
    "PU132->DO138": [92.0, 89.0, 101.0, 118.0, 126.0, 119.0],
    "PU79->DO230": [63.0, 61.0, 68.0, 73.0, 75.0, 72.0],
}

with ThreadPoolExecutor(max_workers=2) as pool:
    forecasts = dict(zip(lanes, pool.map(forecast_lane, lanes.values())))
```

Use this pattern for small batches of important lanes where each local order is
audited separately. For a large regular panel, also compare against
`CartoBoostLagForecaster`, which can share calendar and lag signal across
lanes instead of treating every lane as isolated.

## Parameters

| Parameter | Notes |
| --- | --- |
| `p`, `d`, `q` | Non-negative order values for `ArimaForecaster`; `p <= 8`, `d <= 2`, and `q <= 8`. |
| `max_p`, `max_d`, `max_q` | Non-negative AutoARIMA search bounds with the same upper limits. |

Fixed and automatic fits reject non-stationary AR recursions and non-invertible
MA recursions. These checks apply to the modeled series after differencing, not
only to level models with `d=0`.

## Validation Notes

- Input targets must be finite.
- Differencing must leave enough observations for the requested `p` and `q`
  lags.
- Timestamps in a `ForecastFrame` must be regular at the declared frequency.
- Duplicate `(series_id, timestamp)` pairs are rejected.

## Held-Out Evaluation Pattern

Use fixed train/test boundaries when comparing fixed ARIMA and AutoARIMA on a
series. Keep the boundary and horizon identical across candidates:

```python
train = lane_table.iloc[:72]
actual = lane_table.iloc[72:84]

frame = ForecastFrame.from_pandas(
    train,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="lane_id",
    freq="h",
)

model = ArimaForecaster(p=2, d=1, q=1)
model.fit(frame)
forecast = pd.DataFrame(
    [
        {
            "lane_id": series_id,
            "pickup_hour": pd.Timestamp(timestamp),
            "horizon": horizon,
            "model": model_name,
            "prediction": prediction,
        }
        for series_id, timestamp, horizon, model_name, prediction in model.predict(12).predictions()
    ]
)

joined = actual.merge(forecast, on=["lane_id", "pickup_hour"])
joined["residual"] = joined["prediction"] - joined["demand"]
rmse = (joined["residual"].pow(2).mean()) ** 0.5
mae = joined["residual"].abs().mean()
bias = joined["residual"].mean()
```

Report RMSE and MAE for comparability, and include bias when underprediction or
overprediction has a different cost. For benchmark claims, add R2, training
time, prediction time, model settings, sample size, task names, and split names
from real or clearly labeled benchmark data.

## Benchmark Notes

The repository includes a focused Criterion benchmark for ARIMA runtime paths:

```bash
cargo bench -p cartoboost-core --bench forecasting
```

The benchmark uses deterministic synthetic taxi pickup/dropoff lane demand and
measures fixed ARIMA fit+predict and bounded AutoARIMA fit+predict. Treat those
numbers as runtime checks, not modeling-quality evidence. Public quality claims
should come from rolling-origin backtests with fixed splits, recorded
RMSE/MAE/R2, and serious baselines on real or clearly labeled benchmark data.
