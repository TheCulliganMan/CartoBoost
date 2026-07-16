import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Piecewise Linear Seasonal

`PiecewiseLinearSeasonalForecaster` is the local model for interpretable trend,
changepoints, seasonalities, events, extra regressors, and component
decomposition. Use it when the forecast needs to show which named components
move the prediction, not just the final point estimate.

## When To Use

Use this model for one regular demand, fare, duration, or trip-distance series
when the forecast should be explained as:

- a piecewise trend with optional changepoints;
- one or more repeating seasonal components;
- known event windows or known future regressors;
- additive or multiplicative component effects;
- external trend belief adjustments for forecast horizons;
- recent residual shock propagation when the fitted trend has under- or
  over-predicted for several consecutive observations;
- residual intervals or quantile-style uncertainty summaries.

It is a good fit for demand with event windows, recurring weekday and hour
effects, or lane-level aggregate demand where a scientist needs to inspect the
trend and seasonal contribution separately.

Prefer seasonal naive, theta, ETS, ARIMA, or Kalman when the question is only a
small local baseline. Prefer `CartoBoostLagForecaster` or `AutoForecaster` when
many related zones or lanes should share supervised lag features.

## Interactive Example

<ForecastModelExample title="Piecewise linear seasonal panel forecast" model="piecewise_linear_seasonal" />

The embedded example runs in this page and returns forecast rows from the
interactive example. Use the full Python API when you need to persist component
decomposition, fitted history diagnostics, residuals, trend movement, and named
components for a report.

## Scientific Role

Piecewise linear seasonal forecasting is a component model. It asks whether the
future can be explained by a trend path plus named seasonal, event, and
regressor effects. That makes it useful when the model output must support a
claim such as "the weekday seasonality explains the recurring lift, while the
trend changed after this cutoff."

The component rows are part of the evidence. If a benchmark or report uses this
model, preserve the forecast table, component decomposition, changepoints,
input cadence, regressors, and validation split with the metrics.

## Assumptions And Failure Modes

The model assumes the configured seasonal periods and event/regressor columns
match the forecast cadence and are known at prediction time. It can fail when
the strongest signal is cross-zone borrowing, sparse intermittent demand,
unmodeled disruptions, or a supervised panel effect that a local component model
cannot see.

Common failure modes in panel data:

| Failure mode | Scientific interpretation | Comparison to run |
| --- | --- | --- |
| Trend changes too often. | Changepoint flexibility is fitting noise. | ETS, Kalman, or fewer changepoints. |
| Seasonal component is phase-shifted. | Season length or timestamp cadence is wrong. | Seasonal naive on the same cadence. |
| Event effect persists outside its window. | The event window is standing in for missing covariates. | Lag model with explicit calendar features. |
| One lane fits and another fails. | Local components do not transfer across panels. | CartoBoost lag or AutoForecaster. |

## Python Example

```python
from cartoboost.forecasting import ForecastFrame, PiecewiseLinearSeasonalForecaster

frame = ForecastFrame.from_pandas(
    hourly_airport_pickups,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
    known_future_covariates=["hour", "day_of_week", "holiday_event"],
)

model = PiecewiseLinearSeasonalForecaster(
    growth="linear",
    changepoints=8,
    weekly_fourier_order=3,
    daily_fourier_order=4,
    prediction_interval_levels=(0.8, 0.95),
)

forecast = model.fit(frame).predict(24)
components = model.components(24)
components_frame = model.components_frame(24)
history_components = model.history_components()
history_frame = model.history_components_frame()
```

## Irregular History

`PiecewiseLinearSeasonalForecaster` can fit Prophet-style irregular history
without filling missing dates. Use this when older aggregate rows are weekly
and recent operational rows are daily or business-daily. The model fits on the
actual timestamps and uses `freq` only as the default horizon cadence.

```python
from cartoboost.forecasting import ForecastFrame, PiecewiseLinearSeasonalForecaster

frame = ForecastFrame.from_pandas(
    taxi_route_training_rows,
    timestamp_col="pickup_day",
    target_col="demand",
    freq="D",
    allow_irregular=True,
)

model = PiecewiseLinearSeasonalForecaster(
    growth="linear",
    changepoint_prior_scale=0.05,
    weekly_fourier_order=3,
)

forecast = model.fit(frame).predict(
    5,
    future_timestamps=[
        "2026-06-30",
        "2026-07-01",
        "2026-07-02",
        "2026-07-06",
        "2026-07-07",
    ],
)
```

Pass explicit `future_timestamps` or `future_timestamps_by_series` when the
forecast calendar should skip weekends or holidays. Models whose training
logic depends on equal row spacing, including ETS, ARIMA, theta, Kalman,
intermittent demand, lag, neural, and auto selectors, still require a regular
`ForecastFrame`.

## Missing Targets

Prophet drops rows whose `y` value is missing before fitting, while still using
the full timestamp range as the history calendar. CartoBoost exposes the same
behavior with `allow_missing_targets=True`: `NaN` targets are allowed in
`ForecastFrame`, infinity is still rejected, and the piecewise model fits only
observed target rows. Horizon forecasts start after the latest timestamp in the
input, even when the final row has a missing target.
Historical diagnostics retain the missing-target timestamps:
`history_components_frame()` emits a finite `fitted` value for each missing
target row while leaving `actual` and `residual` null.

```python
frame = ForecastFrame.from_pandas(
    taxi_route_training_rows,
    timestamp_col="pickup_day",
    target_col="demand",
    freq="D",
    allow_missing_targets=True,
)

forecast = PiecewiseLinearSeasonalForecaster(
    growth="linear",
    weekly_fourier_order=3,
).fit(frame).predict(7)
```

This option is model-scoped. Use it for `PiecewiseLinearSeasonalForecaster` or
last-value baselines when upstream training data carries future or withheld
timestamps with missing targets. Do not use it to mask failed data joins or
missing covariates. If the piecewise model uses an extra regressor, capacity
regressor, floor regressor, or condition flag, that covariate must be present
and finite on the rows where the component is evaluated.

Prophet-shaped holiday and changepoint inputs are accepted as configuration
ergonomics for users migrating existing component-model workflows. The default
automatic changepoint count is 25, matching Prophet's public default.
CartoBoost's default
`changepoint_range` is 1.0 so lane-level nowcasts can place trend breaks across
the full training window when recent demand is moving; set
`changepoint_range=0.8` when you need Prophet's stricter default placement.
Use `n_changepoints` for automatic changepoint placement, pass explicit dates
through `changepoints=[...]` or `changepoint_timestamps=[...]`, and pass a
Prophet-style `holidays` table with `holiday`, `ds`, optional `lower_window`,
`upper_window`, and `prior_scale` columns. Holidays become named event windows,
and prior scales become per-event regularization settings before fitting.
Prophet-style
`changepoint_prior_scale`, `seasonality_prior_scale`, `seasonality_mode`, and
`holidays_mode` aliases map to changepoint penalty, seasonality penalty,
component mode, and event mode fields.
The interactive forecast example accepts the same modeling aliases in camelCase:
`nChangepoints`, `changepointPriorScale`, `seasonalityPriorScale`,
`holidaysPriorScale`, `seasonalityMode`, `holidaysMode`, `intervalWidth`, and
Prophet-shaped `holidays` rows with `holiday`, `ds`, `lowerWindow`,
`upperWindow`, and `priorScale`.

Built-in country holiday calendars are available before fitting through
`model.add_country_holidays("US")` or the constructor argument
`country_holidays="US"`. This path requires the optional `holidays` package:
install `holidays` when country calendars are needed. Explicit `holidays`
dataframes do not require the third-party package.

## Prophet-Style Plotting

The local plotting layer provides full public plotting-utility parity with the
`prophet.plot` module from `prophet==1.2.2`, the package version resolved by
CartoBoost's benchmark dependency range `prophet>=1.1,<1.3`. Use
`cartoboost.plotting.plot`, `plot_components`, `plot_forecast_component`,
`plot_weekly`, `plot_yearly`, `plot_seasonality`,
`add_changepoints_to_plot`, `plot_cross_validation_metric`, the matching
Plotly helpers, and the helper-prop functions when a report expects Prophet's
plotting API names.

The proof is maintained in [Plotting](../../plotting.md): it lists every public
`prophet.plot` 1.2.2 utility beside the matching local implementation and
points to the parity tests in `tests/python/test_plotting.py`. This parity is
limited to plotting utilities over Prophet-shaped forecast/component data; the
reusable model API remains `PiecewiseLinearSeasonalForecaster`, not a
`prophet` alias.

## Trend Beliefs And Residual Shocks

Use `trend_adjustments` when forecast-time market beliefs should move the local
trend path without changing the fitted historical coefficients. The mapping is
keyed by forecast horizon, with values interpreted as trend multipliers. For
example, `{1: 1.01, 2: 1.02}` raises the horizon-1 trend by 1 percent and the
horizon-2 trend by 2 percent before the final forecast is assembled. Panel
models can use `trend_adjustments_by_series` for route- or zone-specific
beliefs; per-series values override global horizon values.

```python
model = PiecewiseLinearSeasonalForecaster(
    changepoints=8,
    weekly_fourier_order=3,
    trend_adjustments={1: 1.01, 2: 1.02, 3: 1.03},
    trend_adjustments_by_series={
        "132": {1: 1.04, 2: 1.05},
    },
)
```

Use residual shocks when recent same-sign residuals indicate that the local
trend is persistently under- or over-predicting a market. Set
`residual_shock_window` to the required run length, `residual_shock_scale` to
the fraction of the recent average residual to pass through, and
`residual_shock_decay` to control how quickly the shock fades across horizons.
The default scale is zero, so shock propagation is opt-in.

```python
model = PiecewiseLinearSeasonalForecaster(
    changepoints=8,
    weekly_fourier_order=3,
    residual_shock_window=3,
    residual_shock_scale=0.5,
    residual_shock_decay=0.8,
)
```

Component records include `trend`, `adjusted_trend`,
`trend_adjustment_multiplier`, `trend_adjustment`, and `residual_shock` so a
report can separate fitted trend, external market belief, and recent residual
carry-forward.

## Historical Component Diagnostics

Use `history_components()` after fitting when rolling-origin backtests need to
explain why a cutoff won or lost. Use `components()` for future horizon
decomposition and `components_frame(horizon)` when future component records
should be flattened into the same dotted pandas columns as
`history_components_frame()`, such as `components.seasonal_total`,
`components.weekly`, or `components.events.airport_surge`.

Historical component methods return one row for every training observation and
are computed from the fitted coefficients and the original covariates. Each row
includes:

| Field | Meaning |
| --- | --- |
| `actual` | Observed target for the training timestamp. |
| `fitted` | In-sample fitted value from the same component equation used for prediction. |
| `residual` | `actual - fitted`, useful for diagnosing systematic under- or over-fit before a holdout. |
| `trend` | Fitted trend path before forecast-time external trend adjustments. |
| `trend_movement` | Change in fitted trend versus the previous training row for that series. |
| `fitted_movement` | Change in fitted total value versus the previous training row for that series. |
| `components` | Named component contributions, including `seasonal_total`, `yearly`, `weekly`, `daily`, custom seasonalities, event totals, and regressor totals. |

`history_components_frame()` and `components_frame(horizon)` keep top-level
fields and expand nested components with dotted column names:

| Column | Diagnostic use |
| --- | --- |
| `trend` | Plot the fitted level/trend path against `actual` to see whether the model is tracking the lane's current demand level. |
| `trend_movement` | Compare week-over-week trend movement with the last observed week-over-week target movement. Near-zero movement during a rising lane usually means the trend is too stiff. |
| `components.seasonal_total` | Check whether seasonality is doing most of the work or offsetting trend in the wrong direction. |
| `components.yearly`, `components.weekly`, `components.daily` | Inspect built-in Fourier seasonal effects separately when the aggregate seasonal total looks plausible but one cadence is wrong. |
| `components.events.*` | Verify that holiday or event windows explain known calendar shocks instead of forcing the trend to absorb them. |
| `components.regressors.*` | Verify known future or historical regressor contribution and sign. |

For whole-dataset transformations, concatenate the historical and future
component frames after adding any project-specific split label:

```python
import pandas as pd

history_frame = model.history_components_frame().assign(split="history")
future_frame = model.components_frame(14).assign(split="forecast")
component_frame = pd.concat([history_frame, future_frame], ignore_index=True)
```
| `residual` | Identify whether the model is already biased immediately before the holdout. |

For a weekly backtest with 12 cutoffs, run the model once per cutoff, holding
out one additional week each time, then persist both the holdout predictions
and `history_components_frame()` from that cutoff fit. When Prophet beats
CartoBoost with a negative bias, inspect the last several historical component
rows before each cutoff:

```python
diagnostics_by_cutoff = {}

for cutoff, train in weekly_cutoff_training_frames:
    model = PiecewiseLinearSeasonalForecaster(
        n_changepoints=25,
        weekly_fourier_order=3,
        changepoint_range=1.0,
    ).fit(train)
    diagnostics_by_cutoff[str(cutoff)] = {
        "history_components": model.history_components_frame(),
        "forecast_components": model.components(1)["records"],
        "forecast": model.predict(1).to_pandas(),
    }
```

To recreate a trend/seasonality comparison table, concatenate the stored
history frames with the cutoff label:

```python
import pandas as pd

history = pd.concat(
    frame.assign(cutoff=cutoff)
    for cutoff, frame in diagnostics_by_cutoff.items()
)

trend_columns = [
    "cutoff",
    "series_id",
    "timestamp",
    "actual",
    "fitted",
    "residual",
    "trend",
    "trend_movement",
    "components.seasonal_total",
]

trend_table = history[trend_columns].sort_values(
    ["cutoff", "series_id", "timestamp"]
)
```

If `trend_movement` is near zero while the last observed weeks are rising, the
trend is too stiff for the lane. If `seasonal_total` is large and offsetting the
trend in the wrong direction, inspect the cadence and Fourier order. If recent
residuals are consistently positive before the cutoff, the model is
under-predicting the lane before it ever reaches the holdout; consider fewer or
better placed changepoints, explicit event/regressor inputs, or an opt-in
residual shock rather than treating the holdout error as a pure forecast-horizon
problem.

CartoBoost defaults to `n_changepoints=25` and `changepoint_range=1.0`. This
keeps the automatic changepoint count aligned with Prophet's public default but
allows late trend breaks across the full training window, which matters for
one-week-ahead lane nowcasts where the most recent movement carries the holdout.
Use `changepoint_range=0.8` only when you intentionally want Prophet's stricter
placement window for an apples-to-apples tuning probe.

The embedded example exposes `piecewise_linear_seasonal` for a quick
forecast run. The full plotting workflow can request forecast components and
fitted history diagnostics, then plot trend, fitted values, residuals, built-in
seasonalities, custom seasonalities, event windows, regressors, aggregate
non-trend totals, fitted movement, and trend movement.

## Validation

Select changepoints, seasonalities, events, and regressors using training-side
rolling origins. Compare against seasonal naive and a simpler trend model on
identical rows. Report errors by horizon and inspect residuals around events.

## Limitations

- Additive components can miss interactions between location, event, and seasonality.
- Too many changepoints or Fourier terms can overfit short histories.
- Future regressors must be known at forecast time.
- Irregular timestamps and missing targets require explicit handling.
