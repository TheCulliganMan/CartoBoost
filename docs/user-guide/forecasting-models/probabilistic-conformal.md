import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Probabilistic And Conformal Models

Use probabilistic and conformal APIs when point forecasts are not enough. These
surfaces add quantiles, calibrated intervals, interval width, and coverage
checks around CartoBoost regressors, spatial models, residual hybrids, and
forecasting workflows.

Use them when the question is about risk, service levels, uncertainty, or
decision thresholds. Keep point metrics visible, but do not treat RMSE alone as
the quality claim for an interval model.

## Interactive Example

<ForecastModelExample title="CartoBoost lag forecast with interval-ready rows" model="cartoboost_lag" sample="spatial" />

The browser example runs a local forecast and returns forecast rows that can be
used as the base predictions for interval inspection. Use Python calibration
for coverage claims because calibration needs explicit train, calibration, and
holdout splits.

## Python Example

### Quantile Regressor

```python
from cartoboost.forecasting import QuantileCartoBoostRegressor

model = QuantileCartoBoostRegressor(
    quantiles=(0.1, 0.5, 0.9),
    n_estimators=160,
    learning_rate=0.05,
    max_depth=5,
    split_policy="structured",
)
model.fit(x_train, y_train)

distribution = model.predict_distribution(x_holdout)
median = model.predict(x_holdout)
quantile_matrix = model.predict_quantiles(x_holdout)
```

`predict_distribution` returns a `DistributionalForecastResult` with `mean`,
optional `median`, quantile columns, and interval bounds.

### Split Conformal Wrapper

```python
from cartoboost import CartoBoostRegressor
from cartoboost.forecasting import ConformalIntervalRegressor

base = CartoBoostRegressor(n_estimators=200, learning_rate=0.04)
model = ConformalIntervalRegressor(base, alpha=0.1)
model.fit(
    x_train,
    y_train,
    x_calibration,
    y_calibration,
    train_end_exclusive=50_000,
    calibration_start=50_000,
    calibration_end_exclusive=65_000,
    test_start=65_000,
)

interval = model.predict_interval(x_holdout, test_start=65_000)
```

The base estimator fits on train rows only. Calibration rows estimate residual
width. Holdout rows are not used for fitting or calibration.

### Spatial Group Conformal

```python
from cartoboost.forecasting import SpatialConformalRegressor

model = SpatialConformalRegressor(base_estimator, alpha=0.1)
model.fit(
    x_train,
    y_train,
    x_calibration,
    y_calibration,
    groups=calibration_region_id,
    train_end_exclusive=20_000,
    calibration_start=20_000,
    calibration_end_exclusive=25_000,
    test_start=25_000,
)

interval = model.predict_interval(
    x_holdout,
    test_start=25_000,
    groups=holdout_region_id,
)
```

Groups can be regions, cells, routes, stores, sensors, or other calibration
blocks. Unseen holdout groups use the global split-conformal width and record
the method in metadata.

### Rolling-Origin Forecast Calibration

```python
from cartoboost.forecasting import ForecastConformalCalibrator

calibrator = ForecastConformalCalibrator(alpha=0.2).fit(
    actual=backtest_actual,
    prediction=backtest_prediction,
    cutoff_index=backtest_cutoff_number,
)

interval = calibrator.predict_interval(
    next_forecast_prediction,
    cutoff=current_cutoff,
)
```

`ForecastConformalCalibrator` only uses residuals from cutoffs strictly before
the prediction cutoff.

## When To Use

- You need calibrated lower and upper bounds, not only a point prediction.
- Errors vary by horizon, location, route, customer, sensor, or product group.
- A decision depends on upper-tail or lower-tail risk.
- You can preserve a clean train, calibration, and holdout split.

## Use When

| Need | Better first choice |
| --- | --- |
| Conditional quantiles from a CartoBoost regressor. | `QuantileCartoBoostRegressor` |
| Distribution-free intervals around any `.fit/.predict` estimator. | `ConformalIntervalRegressor` |
| Wider or narrower intervals by region, route, block, or cell. | `SpatialConformalRegressor` |
| Forecast intervals from rolling-origin residuals. | `ForecastConformalCalibrator` |
| A portable interval result object. | `DistributionalForecastResult` |

## Metrics

Use interval metrics beside point metrics:

```python
from cartoboost.forecasting import (
    interval_coverage,
    mean_interval_width,
    pinball_loss,
    weighted_interval_score,
)

coverage = interval_coverage(y_holdout, interval.lower, interval.upper)
width = mean_interval_width(interval.lower, interval.upper)
p50_loss = pinball_loss(y_holdout, p50_prediction, 0.5)
wis = weighted_interval_score(y_holdout, p50_prediction, [(0.2, p10, p90)])
```

For forecast intervals, report coverage and width by horizon. For spatial or
grouped intervals, report coverage and width by group as well as globally.

## Validation

Keep splits explicit:

| Split | Used for |
| --- | --- |
| Train | Fit the base estimator. |
| Calibration | Estimate conformal residual widths or quantile behavior. |
| Holdout | Report final point and interval metrics. |

Do not calibrate on the holdout rows. For rolling-origin forecasts, use only
residuals from earlier cutoffs when predicting a later cutoff.

For the native split-conformal calibrator, the calibration arrays must contain
exactly the rows declared by the calibration bounds. A finite two-sided
residual interval at miscoverage `alpha` also requires
`alpha >= 1 / (n_calibration + 1)`. If that finite-sample rank does not exist,
calibration fails instead of substituting the largest observed residual and
overstating the requested coverage.

## Limitations

- Conformal intervals depend on the calibration split matching the holdout
  regime closely enough to be meaningful.
- Group-specific conformal widths need enough calibration rows per group.
- Quantile regressors can be useful without conformal calibration, but coverage
  claims still need holdout evidence.
- Report both coverage and width; wide intervals can cover well while being
  operationally useless.

The executable contract for the conformal examples and interval metrics is
checked by `scripts/check_docs_examples.py` in CI.
