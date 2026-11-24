import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Croston Forecasting

Forecast sparse pickup demand by separating event size from event spacing.
`CrostonForecaster` smooths non-zero counts and the gaps between them, then
returns a constant expected demand across the requested horizon. Use it only
when zeros are real no-demand periods on a regular time grid.

## Interactive Example

<ForecastModelExample title="Croston sparse-demand forecast" model="croston" />

## Python Example

```python
from cartoboost.forecasting import CrostonForecaster

demand = [0, 0, 4, 0, 0, 7, 0, 3, 0, 0, 0, 5]

model = CrostonForecaster(alpha=0.2)
model.fit(demand)
forecast = model.predict(6)
```

## Use When

Use Croston when demand is non-negative, most periods are genuine zeros, and
you need an interpretable baseline before testing SBA, TSB, seasonal naive, or
`AutoForecaster`. Do not use it merely because observations are missing; first
restore the regular time grid and distinguish missing measurements from zero demand.

## Fit Contract

| Requirement | Detail |
| --- | --- |
| Target values | Finite and non-negative. |
| Zero values | Treated as real no-demand periods, not missing rows. |
| Non-zero demand | At least one non-zero observation is required. |
| Smoothing | `alpha` must be greater than 0 and at most 1. |

## Validation

Croston is transparent, but it is not bias-adjusted. Compare it against
`SbaForecaster` and `TsbForecaster` on the same rolling-origin split before
claiming it is the best sparse-demand model for a panel.

Report MAE or RMSE, WAPE where the aggregate denominator is non-zero, the zero
fraction, and results by forecast horizon. Include seasonal naive whenever a
calendar cycle is plausible.

## Limitations

- Forecasts are flat across the requested horizon.
- The method does not model calendar seasonality, covariates, or cross-series effects.
- Long runs of zeros caused by unavailable taxi service or missing collection violate the demand interpretation.
