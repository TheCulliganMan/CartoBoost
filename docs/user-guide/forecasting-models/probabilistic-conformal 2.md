import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Probabilistic And Conformal Models

CartoBoost exposes uncertainty as a model layer instead of a plotting afterthought.
Use these APIs when a geographic, spatial, residual-hybrid, or forecasting model
needs calibrated intervals, quantile scores, and spatial residual diagnostics.

Do not report geo model quality without calibration and spatial residual
diagnostics. Point accuracy is not enough for pickup demand, pricing, duration,
or interpolation claims.

## Interactive WASM Example

<ForecastModelExample title="CartoBoost lag forecast with interval fields" model="cartoboost_lag" sample="spatial" />

The browser Modeling Lab runs the bundled `cartoboost-wasm` forecast model
locally and returns forecast rows with prediction and interval-compatible
fields. Use it to inspect whether the model family is appropriate before
running full Python calibration with holdout residuals.

## Introduced Models

| Model | Role | Use when |
| --- | --- | --- |
| `QuantileCartoBoostRegressor` | Fits one quantile CartoBoost regressor per requested level and repairs crossing rows. | You need conditional quantiles for taxi fare, trip duration, or zone demand. |
| `ConformalIntervalRegressor` | Wraps any estimator with `.fit/.predict` and adds split-conformal intervals. | You have a train/calibration/test split and need coverage without changing the base estimator. |
| `SpatialConformalRegressor` | Adds group-specific conformal widths for H3, S2, block, or route groups. | Coverage differs by pickup zone, dropoff zone, spatial block, or lane. |
| `ForecastConformalCalibrator` | Calibrates forecast intervals from residuals available before each cutoff. | Rolling-origin forecast intervals must avoid future residual leakage. |
| `DistributionalForecastResult` | Carries mean, median, quantiles, standard deviation, interval bounds, and calibration metadata. | Downstream reports need a stable uncertainty schema. |

## Common Prediction Schema

Probabilistic rows should carry:

| Field | Meaning |
| --- | --- |
| `mean` | Expected prediction from the base model or calibrated forecast. |
| `median` | Central quantile when available. |
| `quantiles` | Mapping from quantile level to prediction vector. |
| `std` | Optional predictive standard deviation. |
| `interval_lower`, `interval_upper` | Calibrated interval bounds for the selected level. |
| `calibration_metadata` | Method, alpha, split boundaries, cutoff, group ids, and residual quantile details. |

## Demand Example

```python
from cartoboost.forecasting import ConformalIntervalRegressor

model = ConformalIntervalRegressor(base_demand_model, alpha=0.1)
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

The base model is trained only on `x_train` and `y_train`. Calibration rows are
used to estimate residual width. Holdout rows are never used for training or
calibration.

## Pricing Example

```python
from cartoboost.forecasting import QuantileCartoBoostRegressor

pricing = QuantileCartoBoostRegressor(
    quantiles=(0.1, 0.5, 0.9),
    n_estimators=160,
    learning_rate=0.05,
    max_depth=5,
    splitters=["axis", "periodic:24"],
)
pricing.fit(train_features, train_fare)
distribution = pricing.predict_distribution(holdout_features)
```

Use quantile regressors when the question is asymmetric, such as high-fare risk
for airport pickup lanes or upper-tail duration for congested routes.

## Spatial Interpolation Example

```python
from cartoboost.forecasting import SpatialConformalRegressor

spatial = SpatialConformalRegressor(base_interpolator, alpha=0.1)
spatial.fit(
    x_train,
    y_train,
    x_calibration,
    y_calibration,
    groups=calibration_h3_or_block_id,
    train_end_exclusive=20_000,
    calibration_start=20_000,
    calibration_end_exclusive=25_000,
    test_start=25_000,
)

interval = spatial.predict_interval(
    x_holdout,
    test_start=25_000,
    groups=holdout_h3_or_block_id,
)
```

Group ids can be H3 cells, S2 cells, manually assigned spatial blocks, pickup
zone ids, route ids, or nearest-calibration-residual neighborhoods. If a holdout
group was not seen during calibration, the wrapper falls back to the global
split-conformal residual width and records the conformal method in metadata.

## Forecast Cutoff Safety

```python
from cartoboost.forecasting import ForecastConformalCalibrator

calibrator = ForecastConformalCalibrator(alpha=0.2).fit(
    actual=backtest_actual,
    prediction=backtest_prediction,
    cutoff_index=backtest_cutoff_number,
)

interval = calibrator.predict_interval(next_forecast_prediction, cutoff=current_cutoff)
```

`ForecastConformalCalibrator` only uses residual rows whose cutoff index is
strictly less than the prediction cutoff. That keeps rolling-origin intervals
from training on future holdout residuals.

## Metrics

Use distributional metrics next to point metrics:

```python
from cartoboost.forecasting import (
    crps_approximation,
    interval_coverage,
    mean_interval_width,
    pinball_loss,
    pit_bins,
    weighted_interval_score,
)

coverage = interval_coverage(y_holdout, interval.lower, interval.upper)
width = mean_interval_width(interval.lower, interval.upper)
p50_loss = pinball_loss(y_holdout, p50_prediction, 0.5)
crps = crps_approximation(y_holdout, [0.1, 0.5, 0.9], quantile_matrix)
wis = weighted_interval_score(y_holdout, p50_prediction, [(0.2, p10, p90)])
pit = pit_bins(y_holdout, [0.1, 0.5, 0.9], quantile_matrix, bins=10)
```

Benchmark reports for probabilistic geographic models should include coverage
by horizon, coverage by spatial block, width by horizon, and residual Moran's I
after calibration. Keep these fields in benchmark artifacts even when the table
headline is still RMSE or MAE.

## Model Lab Workflow

1. Open the [Modeling Lab](../../../modeling-lab) or use the embedded WASM
   example above to run a forecast model on the taxi sample.
2. Export or reproduce the same model settings in Python.
3. Build train, calibration, and holdout splits in time order.
4. Fit the base model on train rows only.
5. Calibrate on calibration residuals only.
6. Report point metrics, interval coverage, interval width, PIT bins, and
   residual spatial autocorrelation on holdout rows.
